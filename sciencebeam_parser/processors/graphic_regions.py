import logging
import os
import re
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence

import PIL.Image
from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutGraphic,
    LayoutPage,
    LayoutPageCoordinates
)
from sciencebeam_parser.processors.document_page_image import (
    iter_pdf_document_page_images
)
from sciencebeam_parser.processors.graphic_provider import (
    get_page_numbers_with_uncommon_page_dimension
)


LOGGER = logging.getLogger(__name__)

SVG_GRAPHIC_TYPE = 'svg'
VECTOR_REGION_GRAPHIC_TYPE = 'vector-region'
PAGE_REGION_GRAPHIC_TYPE = 'page-region'
MERGED_GRAPHIC_TYPE_SUFFIX = '-region'

SVG_NS = 'http://www.w3.org/2000/svg'
PATH_COORDINATE_PATTERN = re.compile(r'-?\d+\.?\d*')


@dataclass(frozen=True)
class GraphicRegionConfig:
    enabled: bool = True
    merge_gap_ratio: float = 0.02
    min_area_ratio: float = 0.01
    vector_cluster_gap_ratio: float = 0.02
    min_vector_path_count: int = 8
    min_vector_shape_path_count: int = 2
    max_vector_path_count: int = 10000
    render_dpi: float = 150.0
    min_text_over_graphic_ratio: float = 0.5
    uncommon_page_as_one_region: bool = True


class _Box(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def union(self, other: '_Box') -> '_Box':
        return _Box(
            min(self.x1, other.x1), min(self.y1, other.y1),
            max(self.x2, other.x2), max(self.y2, other.y2)
        )

    def intersection(self, other: '_Box') -> '_Box':
        return _Box(
            max(self.x1, other.x1), max(self.y1, other.y1),
            min(self.x2, other.x2), min(self.y2, other.y2)
        )

    def overlaps(self, other: '_Box') -> bool:
        return (
            self.x2 >= other.x1 and self.x1 <= other.x2
            and self.y2 >= other.y1 and self.y1 <= other.y2
        )

    def is_within_gap_of(self, other: '_Box', gap: float) -> bool:
        return not (
            self.x1 - gap > other.x2 or other.x1 - gap > self.x2
            or self.y1 - gap > other.y2 or other.y1 - gap > self.y2
        )


class PageRegionRenderer:
    """Renders a region of a page, through the same renderer the detector uses.

    `pdf2image` runs `pdftoppm` as a process of its own, so a document that
    defeats the renderer cannot take the caller down with it.
    """

    def __init__(self, pdf_path: str, dpi: float, temp_dir: str):
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.scale = dpi / 72.0
        self.temp_dir = temp_dir
        self._page_number: Optional[int] = None
        self._page_image: Optional[PIL.Image.Image] = None

    def _get_page_image(self, page_number: int) -> Optional[PIL.Image.Image]:
        if self._page_number == page_number and self._page_image is not None:
            return self._page_image
        page_images = list(iter_pdf_document_page_images(
            pdf_path=self.pdf_path,
            output_dir=self.temp_dir,
            dpi=self.dpi,
            page_numbers=[page_number]
        ))
        if not page_images:
            return None
        page_image = PIL.Image.open(page_images[0].page_image_path).convert('RGB')
        page_image.load()
        self._page_number = page_number
        self._page_image = page_image
        return page_image

    def render(self, page_number: int, box: '_Box', image_path: str) -> Optional[str]:
        try:
            page_image = self._get_page_image(page_number)
            if page_image is None:
                return None
            cropped_image = page_image.crop((
                max(0, round(box.x1 * self.scale)),
                max(0, round(box.y1 * self.scale)),
                min(page_image.width, round(box.x2 * self.scale)),
                min(page_image.height, round(box.y2 * self.scale))
            ))
            if not cropped_image.width or not cropped_image.height:
                return None
            cropped_image.save(image_path)
            return image_path
        except Exception:  # pylint: disable=broad-except
            LOGGER.warning(
                'unable to render page %d of %r', page_number, self.pdf_path, exc_info=True
            )
            return None


def _get_box_for_coordinates(coordinates: LayoutPageCoordinates) -> _Box:
    return _Box(
        coordinates.x, coordinates.y,
        coordinates.x + coordinates.width, coordinates.y + coordinates.height
    )


def _get_coordinates_for_box(box: _Box, page_number: int) -> LayoutPageCoordinates:
    return LayoutPageCoordinates(
        x=box.x1, y=box.y1, width=box.width, height=box.height, page_number=page_number
    )


def _iter_svg_path_boxes(svg_path: str) -> List[_Box]:
    boxes: List[_Box] = []
    try:
        root = etree.parse(svg_path).getroot()
    except (OSError, etree.XMLSyntaxError):
        LOGGER.warning('unable to read svg: %r', svg_path)
        return boxes
    for group in root.iter('{%s}g' % SVG_NS):
        style = group.get('style') or ''
        if 'fill: #FFFFFF' in style:
            continue
        for path in group.iter('{%s}path' % SVG_NS):
            numbers = [
                float(value)
                for value in PATH_COORDINATE_PATTERN.findall(path.get('d') or '')
            ]
            xs = numbers[0::2]
            ys = numbers[1::2]
            if xs and ys:
                boxes.append(_Box(min(xs), min(ys), max(xs), max(ys)))
    return boxes


def _get_clusters(boxes: Sequence[_Box], gap: float) -> List[List[int]]:
    clusters: List[List[int]] = []
    cluster_boxes: List[_Box] = []
    for index, box in enumerate(boxes):
        merged_box = box
        merged_indices = [index]
        remaining_clusters: List[List[int]] = []
        remaining_boxes: List[_Box] = []
        for other_indices, other_box in zip(clusters, cluster_boxes):
            if merged_box.is_within_gap_of(other_box, gap):
                merged_box = merged_box.union(other_box)
                merged_indices = merged_indices + other_indices
            else:
                remaining_clusters.append(other_indices)
                remaining_boxes.append(other_box)
        clusters = remaining_clusters + [merged_indices]
        cluster_boxes = remaining_boxes + [merged_box]
    return clusters


def _get_boxes_clipped_to_page(boxes: Sequence[_Box], page_box: Optional[_Box]) -> List[_Box]:
    if page_box is None:
        return list(boxes)
    return [
        box.intersection(page_box)
        for box in boxes
        if box.overlaps(page_box)
    ]


def _is_shape_box(box: _Box, minimum: float = 1.0) -> bool:
    return box.width >= minimum and box.height >= minimum


def iter_vector_region_graphics(
    layout_graphic: LayoutGraphic,
    page_coordinates: Optional[LayoutPageCoordinates],
    config: GraphicRegionConfig
) -> List[LayoutGraphic]:
    local_file_path = layout_graphic.local_file_path
    if not local_file_path or not os.path.exists(local_file_path):
        LOGGER.debug('no readable svg file for graphic: %r', layout_graphic)
        return []
    assert layout_graphic.coordinates
    page_number = layout_graphic.coordinates.page_number
    boxes = _get_boxes_clipped_to_page(
        _iter_svg_path_boxes(local_file_path),
        _get_box_for_coordinates(page_coordinates) if page_coordinates else None
    )
    if not boxes:
        return []
    gap = config.vector_cluster_gap_ratio * (
        page_coordinates.height if page_coordinates else 842.0
    )
    if len(boxes) > config.max_vector_path_count:
        LOGGER.info(
            'reading %d paths as one region: %r', len(boxes), local_file_path
        )
        clusters: List[List[int]] = [list(range(len(boxes)))]
    else:
        clusters = _get_clusters(boxes, gap=gap)
    region_graphics: List[LayoutGraphic] = []
    for indices in clusters:
        cluster_boxes = [boxes[index] for index in indices]
        if len(cluster_boxes) < config.min_vector_path_count:
            continue
        if sum(1 for box in cluster_boxes if _is_shape_box(box)) \
                < config.min_vector_shape_path_count:
            continue
        region_box = cluster_boxes[0]
        for box in cluster_boxes[1:]:
            region_box = region_box.union(box)
        region_graphics.append(LayoutGraphic(
            coordinates=_get_coordinates_for_box(region_box, page_number=page_number),
            graphic_type=VECTOR_REGION_GRAPHIC_TYPE,
            page_meta=layout_graphic.page_meta
        ))
    return region_graphics


def _get_scale(layout_graphics: Sequence[LayoutGraphic]) -> Optional[float]:
    scales = []
    for layout_graphic in layout_graphics:
        assert layout_graphic.coordinates
        if not layout_graphic.local_file_path or not layout_graphic.coordinates.width:
            return None
        try:
            with PIL.Image.open(layout_graphic.local_file_path) as image:
                scales.append(image.width / layout_graphic.coordinates.width)
        except OSError:
            LOGGER.warning('unable to read image: %r', layout_graphic.local_file_path)
            return None
    if not scales:
        return None
    return max(scales)


def _get_composed_image_path(
    layout_graphics: Sequence[LayoutGraphic],
    box: _Box,
    temp_dir: str,
    name: str
) -> Optional[str]:
    scale = _get_scale(layout_graphics)
    if not scale:
        return None
    width = max(1, round(box.width * scale))
    height = max(1, round(box.height * scale))
    composed_image = PIL.Image.new('RGB', (width, height), 'white')
    for layout_graphic in layout_graphics:
        coordinates = layout_graphic.coordinates
        assert coordinates and layout_graphic.local_file_path
        try:
            with PIL.Image.open(layout_graphic.local_file_path) as image:
                resized_image = image.convert('RGB').resize((
                    max(1, round(coordinates.width * scale)),
                    max(1, round(coordinates.height * scale))
                ))
        except OSError:
            LOGGER.warning('unable to read image: %r', layout_graphic.local_file_path)
            return None
        composed_image.paste(resized_image, (
            round((coordinates.x - box.x1) * scale),
            round((coordinates.y - box.y1) * scale)
        ))
    composed_image_path = os.path.join(temp_dir, name)
    composed_image.save(composed_image_path)
    return composed_image_path


def _get_merged_graphic(
    layout_graphics: Sequence[LayoutGraphic],
    page_number: int,
    temp_dir: Optional[str],
    require_file: bool
) -> Optional[Sequence[LayoutGraphic]]:
    if len(layout_graphics) == 1:
        return layout_graphics
    assert layout_graphics[0].coordinates
    box = _get_box_for_coordinates(layout_graphics[0].coordinates)
    for layout_graphic in layout_graphics[1:]:
        assert layout_graphic.coordinates
        box = box.union(_get_box_for_coordinates(layout_graphic.coordinates))
    graphic_types = {layout_graphic.graphic_type for layout_graphic in layout_graphics}
    graphic_type = graphic_types.pop() if len(graphic_types) == 1 else 'mixed'
    local_file_path: Optional[str] = None
    if require_file:
        if not temp_dir:
            return layout_graphics
        local_file_path = _get_composed_image_path(
            layout_graphics, box, temp_dir,
            name='region-%d-%d-%d.png' % (page_number, round(box.x1), round(box.y1))
        )
        if not local_file_path:
            return layout_graphics
    return [LayoutGraphic(
        local_file_path=local_file_path,
        coordinates=_get_coordinates_for_box(box, page_number=page_number),
        graphic_type=(graphic_type or 'graphic') + MERGED_GRAPHIC_TYPE_SUFFIX,
        page_meta=layout_graphics[0].page_meta
    )]


def get_merged_layout_graphics(
    layout_graphics: Sequence[LayoutGraphic],
    page_coordinates: Optional[LayoutPageCoordinates],
    config: GraphicRegionConfig,
    temp_dir: Optional[str] = None,
    require_file: bool = False
) -> List[LayoutGraphic]:
    with_coordinates = [
        layout_graphic
        for layout_graphic in layout_graphics
        if layout_graphic.coordinates
    ]
    if len(with_coordinates) < 2:
        return list(with_coordinates)
    boxes = [
        _get_box_for_coordinates(layout_graphic.coordinates)  # type: ignore[arg-type]
        for layout_graphic in with_coordinates
    ]
    gap = config.merge_gap_ratio * (
        page_coordinates.height if page_coordinates else 842.0
    )
    page_number = with_coordinates[0].coordinates.page_number  # type: ignore[union-attr]
    merged: List[LayoutGraphic] = []
    for indices in _get_clusters(boxes, gap=gap):
        merged.extend(_get_merged_graphic(
            [with_coordinates[index] for index in indices],
            page_number=page_number,
            temp_dir=temp_dir,
            require_file=require_file
        ) or [])
    return merged


def _get_clipped_graphic(
    layout_graphic: LayoutGraphic,
    page_box: Optional[_Box]
) -> Optional[LayoutGraphic]:
    assert layout_graphic.coordinates
    if page_box is None:
        return layout_graphic
    box = _get_box_for_coordinates(layout_graphic.coordinates).intersection(page_box)
    if not box.area:
        return None
    return layout_graphic._replace(coordinates=_get_coordinates_for_box(
        box, page_number=layout_graphic.coordinates.page_number
    ))


def _iter_text_boxes(layout_page: LayoutPage) -> List[_Box]:
    text_boxes: List[_Box] = []
    for block in layout_page.blocks:
        for line in block.lines:
            coordinates_list = [
                token.coordinates for token in line.tokens if token.coordinates
            ]
            if not coordinates_list:
                continue
            box = _get_box_for_coordinates(coordinates_list[0])
            for coordinates in coordinates_list[1:]:
                box = box.union(_get_box_for_coordinates(coordinates))
            text_boxes.append(box)
    return text_boxes


def _is_text_drawn_over(
    box: _Box,
    text_boxes: Sequence[_Box],
    min_ratio: float
) -> bool:
    for text_box in text_boxes:
        text_area = text_box.area
        if not text_area:
            continue
        intersection = box.intersection(text_box)
        if intersection.width <= 0 or intersection.height <= 0:
            continue
        if intersection.area / text_area >= min_ratio:
            return True
    return False


def _is_large_enough(
    layout_graphic: LayoutGraphic,
    page_coordinates: Optional[LayoutPageCoordinates],
    config: GraphicRegionConfig
) -> bool:
    if not layout_graphic.coordinates:
        return False
    page_area = (
        _get_box_for_coordinates(page_coordinates).area if page_coordinates else 0.0
    )
    if not page_area:
        return True
    area_ratio = _get_box_for_coordinates(layout_graphic.coordinates).area / page_area
    return area_ratio >= config.min_area_ratio


def _get_graphic_with_rendered_file(
    layout_graphic: LayoutGraphic,
    renderer: PageRegionRenderer,
    temp_dir: str,
    text_boxes: Sequence[_Box],
    config: GraphicRegionConfig
) -> Optional[LayoutGraphic]:
    assert layout_graphic.coordinates
    box = _get_box_for_coordinates(layout_graphic.coordinates)
    is_rendered_needed = (
        not layout_graphic.local_file_path
        or layout_graphic.graphic_type == VECTOR_REGION_GRAPHIC_TYPE
        or _is_text_drawn_over(box, text_boxes, config.min_text_over_graphic_ratio)
    )
    if not is_rendered_needed:
        return layout_graphic
    page_number = layout_graphic.coordinates.page_number
    rendered_file_path = renderer.render(page_number, box, os.path.join(
        temp_dir, 'region-%d-%d-%d.png' % (page_number, round(box.x1), round(box.y1))
    ))
    if rendered_file_path:
        return layout_graphic._replace(local_file_path=rendered_file_path)
    if layout_graphic.local_file_path:
        return layout_graphic
    return None


def get_layout_page_with_graphic_regions(
    layout_page: LayoutPage,
    config: GraphicRegionConfig,
    temp_dir: Optional[str] = None,
    require_file: bool = False,
    renderer: Optional[PageRegionRenderer] = None
) -> LayoutPage:
    page_coordinates = layout_page.meta.coordinates if layout_page.meta else None
    page_box = _get_box_for_coordinates(page_coordinates) if page_coordinates else None
    expanded: List[LayoutGraphic] = []
    unchanged: List[LayoutGraphic] = []
    for layout_graphic in layout_page.graphics:
        if not layout_graphic.coordinates:
            continue
        if layout_graphic.graphic_type == SVG_GRAPHIC_TYPE:
            if require_file and renderer is None:
                unchanged.append(layout_graphic)
            else:
                expanded.extend(iter_vector_region_graphics(
                    layout_graphic, page_coordinates=page_coordinates, config=config
                ))
            continue
        expanded.append(layout_graphic)
    clipped = [
        clipped_graphic
        for clipped_graphic in (
            _get_clipped_graphic(layout_graphic, page_box)
            for layout_graphic in expanded
        )
        if clipped_graphic is not None
    ]
    merged = get_merged_layout_graphics(
        clipped, page_coordinates=page_coordinates, config=config,
        temp_dir=temp_dir, require_file=require_file and renderer is None
    )
    graphics = [
        layout_graphic
        for layout_graphic in merged
        if _is_large_enough(layout_graphic, page_coordinates, config)
    ]
    if require_file and renderer is not None and temp_dir:
        text_boxes = _iter_text_boxes(layout_page)
        graphics = [
            rendered_graphic
            for rendered_graphic in (
                _get_graphic_with_rendered_file(
                    layout_graphic, renderer, temp_dir, text_boxes, config
                )
                for layout_graphic in graphics
            )
            if rendered_graphic is not None
        ]
    return layout_page.replace(graphics=graphics + unchanged)


def get_layout_page_as_one_region(
    layout_page: LayoutPage,
    temp_dir: Optional[str],
    require_file: bool,
    renderer: Optional[PageRegionRenderer]
) -> LayoutPage:
    page_coordinates = layout_page.meta.coordinates if layout_page.meta else None
    if not page_coordinates or not layout_page.graphics:
        return layout_page
    page_number = layout_page.meta.page_number
    local_file_path: Optional[str] = None
    if require_file:
        if renderer is None or not temp_dir:
            return layout_page
        local_file_path = renderer.render(
            page_number,
            _get_box_for_coordinates(page_coordinates),
            os.path.join(temp_dir, 'page-region-%d.png' % page_number)
        )
        if not local_file_path:
            return layout_page
    return layout_page.replace(graphics=[LayoutGraphic(
        local_file_path=local_file_path,
        coordinates=page_coordinates,
        graphic_type=PAGE_REGION_GRAPHIC_TYPE,
        related_block=LayoutBlock.for_tokens(list(layout_page.iter_all_tokens())),
        page_meta=layout_page.meta
    )])


def get_layout_document_with_graphic_regions(
    layout_document: LayoutDocument,
    config: GraphicRegionConfig,
    temp_dir: Optional[str] = None,
    require_file: bool = False,
    pdf_path: Optional[str] = None
) -> LayoutDocument:
    if not config.enabled:
        return layout_document
    renderer = (
        PageRegionRenderer(pdf_path, config.render_dpi, temp_dir=temp_dir)
        if require_file and pdf_path and temp_dir and os.path.exists(pdf_path)
        else None
    )
    uncommon_page_numbers = (
        set(get_page_numbers_with_uncommon_page_dimension(layout_document))
        if config.uncommon_page_as_one_region
        else set()
    )
    return layout_document.replace(pages=[
        (
            get_layout_page_as_one_region(
                page, temp_dir=temp_dir, require_file=require_file, renderer=renderer
            )
            if page.meta and page.meta.page_number in uncommon_page_numbers
            else get_layout_page_with_graphic_regions(
                page, config=config, temp_dir=temp_dir,
                require_file=require_file, renderer=renderer
            )
        )
        for page in layout_document.pages
    ])
