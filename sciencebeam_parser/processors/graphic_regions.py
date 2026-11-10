import logging
import os
import re
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Sequence

import PIL.Image
from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutGraphic,
    LayoutPage,
    LayoutPageCoordinates
)


LOGGER = logging.getLogger(__name__)

SVG_GRAPHIC_TYPE = 'svg'
VECTOR_REGION_GRAPHIC_TYPE = 'vector-region'
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
    region_graphics: List[LayoutGraphic] = []
    for indices in _get_clusters(boxes, gap=gap):
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


def get_layout_page_with_graphic_regions(
    layout_page: LayoutPage,
    config: GraphicRegionConfig,
    temp_dir: Optional[str] = None,
    require_file: bool = False
) -> LayoutPage:
    page_coordinates = layout_page.meta.coordinates if layout_page.meta else None
    page_box = _get_box_for_coordinates(page_coordinates) if page_coordinates else None
    expanded: List[LayoutGraphic] = []
    for layout_graphic in layout_page.graphics:
        if not layout_graphic.coordinates:
            continue
        if layout_graphic.graphic_type == SVG_GRAPHIC_TYPE:
            if not require_file:
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
        temp_dir=temp_dir, require_file=require_file
    )
    return layout_page.replace(graphics=[
        layout_graphic
        for layout_graphic in merged
        if _is_large_enough(layout_graphic, page_coordinates, config)
    ])


def get_layout_document_with_graphic_regions(
    layout_document: LayoutDocument,
    config: GraphicRegionConfig,
    temp_dir: Optional[str] = None,
    require_file: bool = False
) -> LayoutDocument:
    if not config.enabled:
        return layout_document
    return layout_document.replace(pages=[
        get_layout_page_with_graphic_regions(
            page, config=config, temp_dir=temp_dir, require_file=require_file
        )
        for page in layout_document.pages
    ])
