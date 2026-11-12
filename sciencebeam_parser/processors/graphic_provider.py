import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Counter, Iterable, List, Optional, Sequence, Tuple

from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutGraphic,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutToken
)
from sciencebeam_parser.document.semantic_document import SemanticContentWrapper, SemanticGraphic


LOGGER = logging.getLogger(__name__)


class DocumentGraphicProvider(ABC):
    @abstractmethod
    def iter_semantic_graphic_for_layout_document(
        self,
        layout_document: LayoutDocument,
        extract_graphic_assets: bool
    ) -> Iterable[SemanticGraphic]:
        pass


def get_semantic_graphic_for_layout_graphic(
    layout_graphic: LayoutGraphic,
    extract_graphic_assets: bool
) -> SemanticGraphic:
    relative_path: Optional[str] = None
    if layout_graphic.local_file_path and extract_graphic_assets:
        relative_path = os.path.basename(layout_graphic.local_file_path)
    return SemanticGraphic(
        layout_graphic=layout_graphic,
        relative_path=relative_path
    )


def get_semantic_graphic_list_for_layout_graphic_list(
    layout_graphic_iterable: Iterable[LayoutGraphic],
    extract_graphic_assets: bool
) -> List[SemanticGraphic]:
    return [
        get_semantic_graphic_for_layout_graphic(
            layout_graphic,
            extract_graphic_assets=extract_graphic_assets
        )
        for layout_graphic in layout_graphic_iterable
        if layout_graphic.coordinates
    ]


def get_page_numbers_for_semantic_content_list(
    semantic_content_list: Sequence[SemanticContentWrapper]
) -> Sequence[int]:
    return sorted({
        coordinates.page_number
        for semantic_content in semantic_content_list
        for coordinates in semantic_content.merged_block.get_merged_coordinates_list()
    })


def get_all_page_numbers_of_layout_document(
    layout_document: LayoutDocument
) -> Sequence[int]:
    return sorted({
        page.meta.page_number
        for page in layout_document.pages
    })


def get_graphic_matching_candidate_page_numbers_for_semantic_content_list(
    semantic_content_list: Sequence[SemanticContentWrapper],
    layout_document: Optional[LayoutDocument] = None
) -> Sequence[int]:
    page_numbers = get_page_numbers_for_semantic_content_list(semantic_content_list)
    if layout_document:
        document_page_numbers = set(get_all_page_numbers_of_layout_document(layout_document))
        page_numbers = sorted(
            set(page_numbers).union({
                page_number + 1
                for page_number in page_numbers
                if page_number + 1 in document_page_numbers
            })
        )
    return page_numbers


def _get_rounded_page_dimension(bounding_box: BoundingBox) -> Tuple[int, int]:
    return (round(bounding_box.width), round(bounding_box.height))


def get_page_numbers_with_uncommon_page_dimension(
    layout_document: LayoutDocument
) -> Sequence[int]:
    page_dimension_counter = Counter((
        _get_rounded_page_dimension(page.meta.coordinates.bounding_box)
        for page in layout_document.pages
        if page.meta and page.meta.coordinates
    ))
    LOGGER.debug('page_dimension_counter: %r', page_dimension_counter)
    if len(page_dimension_counter) < 2:
        return []
    most_common_page_dimension = page_dimension_counter.most_common(1)[0][0]
    LOGGER.debug('most_common_page_dimension: %r', most_common_page_dimension)
    return sorted({
        page.meta.page_number
        for page in layout_document.pages
        if (
            page.meta
            and page.meta.coordinates
            and _get_rounded_page_dimension(page.meta.coordinates.bounding_box)
            != most_common_page_dimension
        )
    })


def is_page_with_mostly_bitmap_graphics(
    layout_page: LayoutPage
) -> bool:
    if not layout_page.meta or not layout_page.meta.coordinates:
        LOGGER.debug('page has no coordinates')
        return False
    page_area = layout_page.meta.coordinates.bounding_box.area
    if not page_area:
        LOGGER.debug('page has no area')
        return False
    bitmap_graphics_with_area_ratio = [
        (graphic, graphic.coordinates.bounding_box.area / page_area)
        for graphic in layout_page.graphics
        if (
            graphic.graphic_type != 'svg'
            and graphic.coordinates
        )
    ]
    LOGGER.debug('bitmap_graphics_with_area_ratio: %r', bitmap_graphics_with_area_ratio)
    if not bitmap_graphics_with_area_ratio:
        LOGGER.debug('no bitmap images')
        return False
    accepted_bitmap_graphics = [
        bitmap_graphics
        for bitmap_graphics, area_ratio in bitmap_graphics_with_area_ratio
        if area_ratio > 0.5
    ]
    if not accepted_bitmap_graphics:
        LOGGER.debug('no too small bitmap images')
        return False
    return True


def get_page_numbers_with_mostly_bitmap_graphics(
    layout_document: LayoutDocument
) -> Sequence[int]:
    return [
        page.meta.page_number
        for page in layout_document.pages
        if (
            page.meta
            and is_page_with_mostly_bitmap_graphics(page)
        )
    ]


def are_page_coordinates_within_bounding_box(
    page_coordinates: Optional[LayoutPageCoordinates],
    bounding_box: BoundingBox,
    min_area_ratio: float = 0.5
) -> bool:
    if not page_coordinates:
        return False
    item_bounding_box = page_coordinates.bounding_box
    item_bounding_box_area = item_bounding_box.area
    if not item_bounding_box_area:
        return False
    intersection_bounding_box = item_bounding_box.intersection(bounding_box)
    if not intersection_bounding_box:
        return False
    if intersection_bounding_box.area / item_bounding_box_area < min_area_ratio:
        return False
    return True


def is_layout_token_within_bounding_box(layout_token: LayoutToken, **kwargs) -> bool:
    return are_page_coordinates_within_bounding_box(layout_token.coordinates, **kwargs)


def is_layout_graphic_within_bounding_box(layout_graphic: LayoutGraphic, **kwargs) -> bool:
    return are_page_coordinates_within_bounding_box(layout_graphic.coordinates, **kwargs)


def _remove_tokens_within_bounding_box_flatmap_fn(
    layout_token: LayoutToken,
    **kwargs
) -> List[LayoutToken]:
    if not is_layout_token_within_bounding_box(layout_token, **kwargs):
        return [layout_token]
    return []


def get_layout_page_with_text_or_graphic_replaced_by_graphic(
    layout_page: LayoutPage,
    semantic_graphic: SemanticGraphic,
    is_only_semantic_graphic_on_page: bool,
    is_replace_overlapping_text: bool
) -> LayoutPage:
    layout_graphic = semantic_graphic.layout_graphic
    assert layout_graphic
    assert layout_graphic.coordinates
    graphic_bounding_box = layout_graphic.coordinates.bounding_box
    if is_only_semantic_graphic_on_page:
        layout_graphic = layout_graphic._replace(
            related_block=LayoutBlock.for_tokens(list(layout_page.iter_all_tokens()))
        )
    modified_layout_page = (
        layout_page.replace(
            graphics=[
                _layout_graphic
                for _layout_graphic in layout_page.graphics
                if not is_layout_graphic_within_bounding_box(
                    _layout_graphic,
                    bounding_box=graphic_bounding_box
                )
             ] + [layout_graphic]
        )
    )
    if is_replace_overlapping_text:
        modified_layout_page = (
            modified_layout_page
            .flat_map_layout_tokens(functools.partial(
                _remove_tokens_within_bounding_box_flatmap_fn,
                bounding_box=graphic_bounding_box
            )).remove_empty_blocks()
        )
    return modified_layout_page


def get_layout_document_with_text_or_graphic_replaced_by_graphics(
    layout_document: LayoutDocument,
    semantic_graphics: Iterable[SemanticGraphic],
    is_replace_overlapping_text: bool
) -> LayoutDocument:
    page_by_page_number = {
        page.meta.page_number: page
        for page in layout_document.pages
        if page.meta
    }
    LOGGER.debug('page_by_page_number.keys: %r', page_by_page_number.keys())
    has_changes = False
    semantic_graphics_list = list(semantic_graphics)
    semantic_graphic_count_by_page = Counter((
        semantic_graphic.layout_graphic.coordinates.page_number
        for semantic_graphic in semantic_graphics_list
        if (
            semantic_graphic.layout_graphic
            and semantic_graphic.layout_graphic.coordinates
        )
    ))
    for semantic_graphic in semantic_graphics_list:
        layout_graphic = semantic_graphic.layout_graphic
        assert layout_graphic
        if not layout_graphic.coordinates:
            continue
        page_number = layout_graphic.coordinates.page_number
        page_by_page_number[page_number] = (
            get_layout_page_with_text_or_graphic_replaced_by_graphic(
                page_by_page_number[page_number],
                semantic_graphic,
                is_only_semantic_graphic_on_page=(
                    semantic_graphic_count_by_page[page_number] < 2
                ),
                is_replace_overlapping_text=is_replace_overlapping_text
            )
        )
        has_changes = True
    if not has_changes:
        return layout_document
    pages = [
        (
            page_by_page_number[page.meta.page_number]
            if page.meta
            else page
        )
        for page in layout_document.pages
    ]
    return layout_document.replace(pages=pages)


def get_layout_document_with_text_and_graphics_replaced_by_graphics(
    layout_document: LayoutDocument,
    semantic_graphics: Iterable[SemanticGraphic]
) -> LayoutDocument:
    return get_layout_document_with_text_or_graphic_replaced_by_graphics(
        layout_document,
        semantic_graphics=semantic_graphics,
        is_replace_overlapping_text=True
    )


def get_layout_document_with_graphics_replaced_by_graphics(
    layout_document: LayoutDocument,
    semantic_graphics: Iterable[SemanticGraphic]
) -> LayoutDocument:
    return get_layout_document_with_text_or_graphic_replaced_by_graphics(
        layout_document,
        semantic_graphics=semantic_graphics,
        is_replace_overlapping_text=False
    )


class SimpleDocumentGraphicProvider(DocumentGraphicProvider):
    def iter_semantic_graphic_for_layout_document(
        self,
        layout_document: LayoutDocument,
        extract_graphic_assets: bool
    ) -> Iterable[SemanticGraphic]:
        return get_semantic_graphic_list_for_layout_graphic_list(
            layout_document.iter_all_graphics(),
            extract_graphic_assets=extract_graphic_assets
        )
