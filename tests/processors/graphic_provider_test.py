import logging
from typing import Optional

from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticFigure,
    SemanticGraphic
)
from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutGraphic,
    LayoutLine,
    LayoutLineMeta,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken
)
from sciencebeam_parser.processors.graphic_provider import (
    SimpleDocumentGraphicProvider,
    get_graphic_matching_candidate_page_numbers_for_semantic_content_list,
    get_layout_document_with_graphics_replaced_by_graphics,
    get_layout_document_with_text_and_graphics_replaced_by_graphics,
    get_page_numbers_with_mostly_bitmap_graphics,
    get_page_numbers_with_uncommon_page_dimension
)


LOGGER = logging.getLogger(__name__)


LAYOUT_PAGE_COORDINATES_1 = LayoutPageCoordinates(
    x=10, y=11, width=100, height=101, page_number=1
)

LAYOUT_PAGE_COORDINATES_2 = LayoutPageCoordinates(
    x=10, y=11, width=200, height=101, page_number=2
)


PAGE_META_1 = LayoutPageMeta(
    page_number=1,
    coordinates=LAYOUT_PAGE_COORDINATES_1
)

LINE_META_1 = LayoutLineMeta(
    line_id=1,
    page_meta=PAGE_META_1
)


def _get_semantic_content_for_page_coordinates(
    coordinates: LayoutPageCoordinates,
    line_meta: Optional[LayoutLineMeta] = None
) -> SemanticContentWrapper:
    if line_meta is None:
        line_meta = LINE_META_1._replace(
            page_meta=PAGE_META_1._replace(
                page_number=coordinates.page_number
            )
        )

    return SemanticFigure(
        layout_block=LayoutBlock.for_tokens([
            LayoutToken(
                text='dummy',
                coordinates=coordinates,
                line_meta=line_meta
            )
        ])
    )


def _get_layout_document_for_layout_graphic(
    layout_graphic: LayoutGraphic
) -> LayoutDocument:
    return LayoutDocument(pages=[LayoutPage(
        blocks=[],
        graphics=[layout_graphic]
    )])


class TestGetPageNumbersWithUncommonPageDimension:
    def test_should_provide_empty_list_for_empty_document(self):
        layout_document = LayoutDocument(pages=[])
        result = get_page_numbers_with_uncommon_page_dimension(layout_document)
        assert result == []

    def test_should_provide_empty_list_if_all_pages_have_same_dimension(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=1,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
            )),
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=2,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=2)
            )),
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=3,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=3)
            ))
        ])
        result = get_page_numbers_with_uncommon_page_dimension(layout_document)
        assert result == []

    def test_should_provide_page_number_with_uncomment_page_dimension(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=1,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
            )),
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=2,
                coordinates=LAYOUT_PAGE_COORDINATES_2._replace(page_number=2)
            )),
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=3,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=3)
            ))
        ])
        result = get_page_numbers_with_uncommon_page_dimension(layout_document)
        assert result == [2]


class TestGetPageNumbersWithMostlyBitmapGraphics:
    def test_should_provide_empty_list_for_empty_document(self):
        layout_document = LayoutDocument(pages=[])
        result = get_page_numbers_with_mostly_bitmap_graphics(layout_document)
        assert result == []

    def test_should_provide_empty_list_for_empty_pages(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=1,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
            ))
        ])
        result = get_page_numbers_with_mostly_bitmap_graphics(layout_document)
        assert result == []

    def test_should_provide_empty_list_for_pages_without_any_graphics(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock.for_text('test')], meta=LayoutPageMeta(
                page_number=1,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
            ))
        ])
        result = get_page_numbers_with_mostly_bitmap_graphics(layout_document)
        assert result == []

    def test_should_provide_empty_list_for_pages_with_svg_graphics_only(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[],
                graphics=[LayoutGraphic(
                    graphic_type='svg',
                    coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
                )],
                meta=LayoutPageMeta(
                    page_number=1,
                    coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
                )
            )
        ])
        result = get_page_numbers_with_mostly_bitmap_graphics(layout_document)
        assert result == []

    def test_should_provide_page_number_with_bitmap_graphics(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[],
                graphics=[LayoutGraphic(
                    graphic_type='image',
                    coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
                )],
                meta=LayoutPageMeta(
                    page_number=1,
                    coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
                )
            )
        ])
        result = get_page_numbers_with_mostly_bitmap_graphics(layout_document)
        assert result == [1]

    def test_should_ignore_small_bitmap(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[],
                graphics=[LayoutGraphic(
                    graphic_type='image',
                    coordinates=LAYOUT_PAGE_COORDINATES_1._replace(
                        page_number=1, width=1, height=1
                    )
                )],
                meta=LayoutPageMeta(
                    page_number=1,
                    coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
                )
            )
        ])
        result = get_page_numbers_with_mostly_bitmap_graphics(layout_document)
        assert result == []


class TestGetGraphicMatchingCandidatePageNumbersForSemanticContentList:
    def test_should_return_empty_list_for_empty_semantic_content_list(self):
        result = get_graphic_matching_candidate_page_numbers_for_semantic_content_list(
            []
        )
        assert not result

    def test_should_return_page_number_of_semantic_content(self):
        semantic_content = _get_semantic_content_for_page_coordinates(
            coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=1)
        )
        result = get_graphic_matching_candidate_page_numbers_for_semantic_content_list(
            [semantic_content]
        )
        assert result == [1]

    def test_should_return_subsequent_page_numbers_present_in_layout_document(self):
        semantic_content = _get_semantic_content_for_page_coordinates(
            coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=2)
        )
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock.for_text('test')], meta=LayoutPageMeta(
                page_number=2,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=2)
            )),
            LayoutPage(blocks=[LayoutBlock.for_text('test')], meta=LayoutPageMeta(
                page_number=3,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=3)
            )),
            LayoutPage(blocks=[LayoutBlock.for_text('test')], meta=LayoutPageMeta(
                page_number=4,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=4)
            )),
            LayoutPage(blocks=[LayoutBlock.for_text('test')], meta=LayoutPageMeta(
                page_number=5,
                coordinates=LAYOUT_PAGE_COORDINATES_1._replace(page_number=5)
            ))
        ])
        result = get_graphic_matching_candidate_page_numbers_for_semantic_content_list(
            [semantic_content],
            layout_document=layout_document
        )
        assert result == [2, 3]


class TestGetLayoutDocumentWithTextAndGraphicsReplacedByGraphics:
    def test_should_not_change_layout_document_if_semantic_graphics_is_empty(self):
        layout_document = LayoutDocument(pages=[])
        result = get_layout_document_with_text_and_graphics_replaced_by_graphics(
            layout_document,
            semantic_graphics=[]
        )
        assert result == layout_document

    def test_should_replace_text_and_graphics_within_bounding_box_of_semantic_graphics(
        self
    ):
        page_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(0, 0, 200, 200),
            page_number=1
        )
        semantic_graphic_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 90, 100, 50),
            page_number=1
        )
        keep_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 10, 100, 20),
            page_number=1
        )
        remove_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 100, 100, 20),
            page_number=1
        )
        empty_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 100, 0, 0),
            page_number=1
        )
        keep_token = LayoutToken('keep', coordinates=keep_coordinates)
        remove_token = LayoutToken('remove', coordinates=remove_coordinates)
        keep_graphic = LayoutGraphic(coordinates=keep_coordinates, graphic_type='keep-graphic')
        remove_graphic = LayoutGraphic(
            coordinates=remove_coordinates, graphic_type='remove-graphic'
        )
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[LayoutLine(tokens=[
                    keep_token,
                    remove_token
                ])])],
                graphics=[
                    keep_graphic,
                    remove_graphic
                ],
                meta=LayoutPageMeta(
                    page_number=page_coordinates.page_number,
                    coordinates=page_coordinates
                )
            )
        ])
        layout_graphic = LayoutGraphic(
            coordinates=semantic_graphic_coordinates,
            graphic_type='new-graphic'
        )
        no_coords_layout_graphic = LayoutGraphic(
            coordinates=empty_coordinates,
            graphic_type='empty-coords-graphic'
        )
        result = get_layout_document_with_text_and_graphics_replaced_by_graphics(
            layout_document,
            semantic_graphics=[
                SemanticGraphic(layout_graphic=layout_graphic),
                SemanticGraphic(layout_graphic=no_coords_layout_graphic)
            ]
        )
        LOGGER.debug('result.pages[0].graphics: %r', result.pages[0].graphics)
        assert result.pages[0].graphics[:-1] == [keep_graphic]
        LOGGER.debug('result.pages[0].graphics[-1]: %r', result.pages[0].graphics[-1])
        assert result.pages[0].graphics[-1].graphic_type == layout_graphic.graphic_type
        assert result.pages[0].graphics[-1].coordinates == layout_graphic.coordinates
        assert list(result.pages[0].blocks[0].iter_all_tokens()) == [keep_token]
        related_block = result.pages[0].graphics[-1].related_block
        assert related_block is not None
        assert list(related_block.iter_all_tokens()) == [keep_token, remove_token]


class TestGetLayoutDocumentWithGraphicsReplacedByGraphics:
    def test_should_not_change_layout_document_if_semantic_graphics_is_empty(self):
        layout_document = LayoutDocument(pages=[])
        result = get_layout_document_with_graphics_replaced_by_graphics(
            layout_document,
            semantic_graphics=[]
        )
        assert result == layout_document

    def test_should_replace_graphics_but_not_text_within_bounding_box_of_semantic_graphics(
        self
    ):
        page_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(0, 0, 200, 200),
            page_number=1
        )
        semantic_graphic_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 90, 100, 50),
            page_number=1
        )
        keep_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 10, 100, 20),
            page_number=1
        )
        remove_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 100, 100, 20),
            page_number=1
        )
        empty_coordinates = LayoutPageCoordinates.from_bounding_box(
            BoundingBox(10, 100, 0, 0),
            page_number=1
        )
        non_overlapping_token = LayoutToken('keep', coordinates=keep_coordinates)
        overlapping_token = LayoutToken('remove', coordinates=remove_coordinates)
        all_tokens = [non_overlapping_token, overlapping_token]
        keep_graphic = LayoutGraphic(coordinates=keep_coordinates, graphic_type='keep-graphic')
        remove_graphic = LayoutGraphic(
            coordinates=remove_coordinates, graphic_type='remove-graphic'
        )
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[LayoutLine(tokens=all_tokens)])],
                graphics=[
                    keep_graphic,
                    remove_graphic
                ],
                meta=LayoutPageMeta(
                    page_number=page_coordinates.page_number,
                    coordinates=page_coordinates
                )
            )
        ])
        layout_graphic = LayoutGraphic(
            coordinates=semantic_graphic_coordinates,
            graphic_type='new-graphic'
        )
        no_coords_layout_graphic = LayoutGraphic(
            coordinates=empty_coordinates,
            graphic_type='empty-coords-graphic'
        )
        result = get_layout_document_with_graphics_replaced_by_graphics(
            layout_document,
            semantic_graphics=[
                SemanticGraphic(layout_graphic=layout_graphic),
                SemanticGraphic(layout_graphic=no_coords_layout_graphic)
            ]
        )
        LOGGER.debug('result.pages[0].graphics: %r', result.pages[0].graphics)
        assert result.pages[0].graphics[:-1] == [keep_graphic]
        LOGGER.debug('result.pages[0].graphics[-1]: %r', result.pages[0].graphics[-1])
        assert result.pages[0].graphics[-1].graphic_type == layout_graphic.graphic_type
        assert result.pages[0].graphics[-1].coordinates == layout_graphic.coordinates
        assert list(result.pages[0].blocks[0].iter_all_tokens()) == all_tokens


class TestSimpleDocumentGraphicProvider:
    def test_should_provide_semantic_graphic_with_assets(self):
        layout_graphic = LayoutGraphic(
            local_file_path='/path/to/image.png',
            coordinates=LAYOUT_PAGE_COORDINATES_1
        )
        semantic_graphic_list = list(
            SimpleDocumentGraphicProvider()
            .iter_semantic_graphic_for_layout_document(
                _get_layout_document_for_layout_graphic(layout_graphic),
                extract_graphic_assets=True
            )
        )
        assert len(semantic_graphic_list) == 1
        assert semantic_graphic_list[0].layout_graphic == layout_graphic
        assert semantic_graphic_list[0].relative_path == 'image.png'

    def test_should_provide_semantic_graphic_without_assets(self):
        layout_graphic = LayoutGraphic(
            local_file_path='/path/to/image.png',
            coordinates=LAYOUT_PAGE_COORDINATES_1
        )
        semantic_graphic_list = list(
            SimpleDocumentGraphicProvider()
            .iter_semantic_graphic_for_layout_document(
                _get_layout_document_for_layout_graphic(layout_graphic),
                extract_graphic_assets=False
            )
        )
        assert len(semantic_graphic_list) == 1
        assert semantic_graphic_list[0].layout_graphic == layout_graphic
        assert semantic_graphic_list[0].relative_path is None

    def test_should_ignore_layout_graphic_without_coordinates(self):
        layout_graphic = LayoutGraphic(
            local_file_path='/path/to/image.png',
            coordinates=None
        )
        semantic_graphic_list = list(
            SimpleDocumentGraphicProvider()
            .iter_semantic_graphic_for_layout_document(
                _get_layout_document_for_layout_graphic(layout_graphic),
                extract_graphic_assets=False
            )
        )
        assert not semantic_graphic_list
