import logging
import os
from pathlib import Path
from typing import Sequence

import PIL.Image

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutGraphic,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta
)
from sciencebeam_parser.processors.graphic_regions import (
    _Box,
    GraphicRegionConfig,
    PAGE_REGION_GRAPHIC_TYPE,
    PageRegionRenderer,
    VECTOR_REGION_GRAPHIC_TYPE,
    get_layout_document_with_graphic_regions
)


LOGGER = logging.getLogger(__name__)

PAGE_WIDTH = 600.0
PAGE_HEIGHT = 800.0

PAGE_COORDINATES_1 = LayoutPageCoordinates(
    x=0, y=0, width=PAGE_WIDTH, height=PAGE_HEIGHT, page_number=1
)

PAGE_META_1 = LayoutPageMeta(page_number=1, coordinates=PAGE_COORDINATES_1)

CONFIG = GraphicRegionConfig()

MINIMAL_EXAMPLE_PDF_PATH = 'test-data/minimal-example.pdf'

_BOX_FOR_RENDER = _Box(10, 10, 110, 60)


def _get_coordinates(x: float, y: float, width: float, height: float):
    return LayoutPageCoordinates(x=x, y=y, width=width, height=height, page_number=1)


def _get_layout_document(graphics: Sequence[LayoutGraphic]) -> LayoutDocument:
    return LayoutDocument(pages=[
        LayoutPage(blocks=[], graphics=list(graphics), meta=PAGE_META_1)
    ])


def _get_graphics(layout_document: LayoutDocument) -> Sequence[LayoutGraphic]:
    return list(layout_document.iter_all_graphics())


def _write_svg(path: Path, paths: Sequence[str], style: str = 'stroke: #000000;') -> str:
    groups = ''.join(
        '<g style="%s"><path d="%s"/></g>' % (style, d)
        for d in paths
    )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg">%s</svg>' % groups,
        encoding='utf-8'
    )
    return str(path)


def _get_rule_paths(count: int) -> Sequence[str]:
    return [
        'M50.0,%.1f L550.0,%.1f' % (100.0 + index * 20, 100.0 + index * 20)
        for index in range(count)
    ]


def _get_shape_paths(count: int) -> Sequence[str]:
    return [
        'M%.1f,%.1f L%.1f,%.1f L%.1f,%.1f Z' % (
            100.0 + index * 8, 100.0 + index * 8, 160.0 + index * 8, 100.0 + index * 8,
            160.0 + index * 8, 160.0 + index * 8
        )
        for index in range(count)
    ]


class TestGetLayoutDocumentWithGraphicRegions:
    def test_should_merge_adjacent_graphics_into_one_region(self):
        graphics = [
            LayoutGraphic(
                coordinates=_get_coordinates(100, 100, 150, 150),
                graphic_type='image', page_meta=PAGE_META_1
            ),
            LayoutGraphic(
                coordinates=_get_coordinates(255, 100, 150, 150),
                graphic_type='image', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert len(result) == 1
        coordinates = result[0].coordinates
        assert coordinates is not None
        assert (coordinates.x, coordinates.y) == (100, 100)
        assert (coordinates.width, coordinates.height) == (305, 150)

    def test_should_not_merge_graphics_that_are_far_apart(self):
        graphics = [
            LayoutGraphic(
                coordinates=_get_coordinates(50, 50, 150, 150),
                graphic_type='image', page_meta=PAGE_META_1
            ),
            LayoutGraphic(
                coordinates=_get_coordinates(50, 500, 150, 150),
                graphic_type='image', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert len(result) == 2

    def test_should_remove_a_graphic_below_the_area_floor(self):
        graphics = [
            LayoutGraphic(
                coordinates=_get_coordinates(100, 100, 10, 10),
                graphic_type='image', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert not result

    def test_should_keep_small_graphics_that_merge_into_a_large_region(self):
        graphics = [
            LayoutGraphic(
                coordinates=_get_coordinates(100 + column * 55, 100 + row * 55, 50, 50),
                graphic_type='image', page_meta=PAGE_META_1
            )
            for row in range(4)
            for column in range(4)
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert len(result) == 1
        assert result[0].coordinates is not None
        assert result[0].coordinates.width == 215

    def test_should_clip_a_graphic_to_the_page(self):
        graphics = [
            LayoutGraphic(
                coordinates=_get_coordinates(-100, -100, 400, 400),
                graphic_type='image', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert len(result) == 1
        coordinates = result[0].coordinates
        assert coordinates is not None
        assert (coordinates.x, coordinates.y) == (0, 0)
        assert (coordinates.width, coordinates.height) == (300, 300)

    def test_should_report_a_drawn_region_of_a_page_svg(self, tmp_path: Path):
        svg_path = _write_svg(tmp_path / 'page.svg', _get_shape_paths(20))
        graphics = [
            LayoutGraphic(
                local_file_path=svg_path,
                coordinates=_get_coordinates(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                graphic_type='svg', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert len(result) == 1
        assert result[0].graphic_type == VECTOR_REGION_GRAPHIC_TYPE
        assert result[0].coordinates is not None
        assert result[0].coordinates.width < PAGE_WIDTH

    def test_should_report_no_region_for_a_page_of_rules(self, tmp_path: Path):
        svg_path = _write_svg(tmp_path / 'rules.svg', _get_rule_paths(20))
        graphics = [
            LayoutGraphic(
                local_file_path=svg_path,
                coordinates=_get_coordinates(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                graphic_type='svg', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert not result

    def test_should_report_no_region_for_a_page_border(self, tmp_path: Path):
        svg_path = _write_svg(
            tmp_path / 'border.svg',
            ['M10.0,10.0 L590.0,10.0 L590.0,790.0 L10.0,790.0 Z']
        )
        graphics = [
            LayoutGraphic(
                local_file_path=svg_path,
                coordinates=_get_coordinates(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                graphic_type='svg', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert not result

    def test_should_ignore_a_white_fill(self, tmp_path: Path):
        svg_path = _write_svg(
            tmp_path / 'mask.svg', _get_shape_paths(20),
            style='fill: #FFFFFF;fill-opacity: 1;'
        )
        graphics = [
            LayoutGraphic(
                local_file_path=svg_path,
                coordinates=_get_coordinates(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                graphic_type='svg', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG
        ))
        assert not result

    def test_should_compose_a_merged_region_when_a_file_is_required(self, tmp_path: Path):
        graphics = []
        for index in range(2):
            image_path = tmp_path / ('tile%d.png' % index)
            PIL.Image.new('RGB', (100, 50), 'white').save(str(image_path))
            graphics.append(LayoutGraphic(
                local_file_path=str(image_path),
                coordinates=_get_coordinates(100, 100 + index * 50, 100, 50),
                graphic_type='image', page_meta=PAGE_META_1
            ))
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG,
            temp_dir=str(tmp_path), require_file=True
        ))
        assert len(result) == 1
        assert result[0].local_file_path
        with PIL.Image.open(result[0].local_file_path) as composed_image:
            assert (composed_image.width, composed_image.height) == (100, 100)

    def test_should_not_merge_when_a_member_has_no_file_and_a_file_is_required(
        self, tmp_path: Path
    ):
        image_path = tmp_path / 'tile.png'
        PIL.Image.new('RGB', (100, 50), 'white').save(str(image_path))
        graphics = [
            LayoutGraphic(
                local_file_path=str(image_path),
                coordinates=_get_coordinates(100, 100, 100, 50),
                graphic_type='image', page_meta=PAGE_META_1
            ),
            LayoutGraphic(
                coordinates=_get_coordinates(100, 150, 100, 50),
                graphic_type='image', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG,
            temp_dir=str(tmp_path), require_file=True
        ))
        assert len(result) == 2

    def test_should_keep_the_page_svg_when_a_file_is_required(self, tmp_path: Path):
        svg_path = _write_svg(tmp_path / 'page.svg', _get_shape_paths(20))
        graphics = [
            LayoutGraphic(
                local_file_path=svg_path,
                coordinates=_get_coordinates(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                graphic_type='svg', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG,
            temp_dir=str(tmp_path), require_file=True
        ))
        assert len(result) == 1
        assert result[0].graphic_type == 'svg'
        assert result[0].local_file_path == svg_path

    def test_should_read_a_dense_page_svg_as_one_region(self, tmp_path: Path):
        svg_path = _write_svg(tmp_path / 'dense.svg', _get_shape_paths(20))
        graphics = [
            LayoutGraphic(
                local_file_path=svg_path,
                coordinates=_get_coordinates(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                graphic_type='svg', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics),
            config=GraphicRegionConfig(max_vector_path_count=5)
        ))
        assert len(result) == 1
        assert result[0].graphic_type == VECTOR_REGION_GRAPHIC_TYPE

    def test_should_render_a_vector_region_when_a_file_is_required(self, tmp_path: Path):
        svg_path = _write_svg(tmp_path / 'page.svg', _get_shape_paths(20))
        graphics = [
            LayoutGraphic(
                local_file_path=svg_path,
                coordinates=_get_coordinates(0, 0, PAGE_WIDTH, PAGE_HEIGHT),
                graphic_type='svg', page_meta=PAGE_META_1
            )
        ]
        result = _get_graphics(get_layout_document_with_graphic_regions(
            _get_layout_document(graphics), config=CONFIG,
            temp_dir=str(tmp_path), require_file=True,
            pdf_path=MINIMAL_EXAMPLE_PDF_PATH
        ))
        assert len(result) == 1
        assert result[0].graphic_type == VECTOR_REGION_GRAPHIC_TYPE
        assert result[0].local_file_path
        assert os.path.exists(result[0].local_file_path)

    def test_should_read_a_page_of_its_own_size_as_one_region(self):
        other_page_meta = LayoutPageMeta(
            page_number=2,
            coordinates=LayoutPageCoordinates(
                x=0, y=0, width=300, height=400, page_number=2
            )
        )
        figure_page = LayoutPage(blocks=[], graphics=[
            LayoutGraphic(
                coordinates=LayoutPageCoordinates(
                    x=10, y=10, width=100, height=100, page_number=2
                ),
                graphic_type='image', page_meta=other_page_meta
            ),
            LayoutGraphic(
                coordinates=LayoutPageCoordinates(
                    x=180, y=10, width=100, height=100, page_number=2
                ),
                graphic_type='image', page_meta=other_page_meta
            )
        ], meta=other_page_meta)
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[], graphics=[], meta=PAGE_META_1),
            figure_page,
            LayoutPage(blocks=[], graphics=[], meta=PAGE_META_1._replace(page_number=3))
        ])
        result = _get_graphics(get_layout_document_with_graphic_regions(
            layout_document, config=CONFIG
        ))
        assert len(result) == 1
        assert result[0].graphic_type == PAGE_REGION_GRAPHIC_TYPE
        coordinates = result[0].coordinates
        assert coordinates is not None
        assert (coordinates.width, coordinates.height) == (300, 400)

    def test_should_leave_the_document_unchanged_when_disabled(self):
        graphics = [
            LayoutGraphic(
                coordinates=_get_coordinates(100, 100, 10, 10),
                graphic_type='image', page_meta=PAGE_META_1
            )
        ]
        layout_document = _get_layout_document(graphics)
        result = get_layout_document_with_graphic_regions(
            layout_document, config=GraphicRegionConfig(enabled=False)
        )
        assert result is layout_document


class TestPageRegionRenderer:
    def test_should_render_a_region_of_a_page(self, tmp_path: Path):
        renderer = PageRegionRenderer(
            MINIMAL_EXAMPLE_PDF_PATH, dpi=72.0, temp_dir=str(tmp_path)
        )
        image_path = renderer.render(1, _BOX_FOR_RENDER, str(tmp_path / 'region.png'))
        assert image_path
        with PIL.Image.open(image_path) as rendered_image:
            assert (rendered_image.width, rendered_image.height) == (100, 50)

    def test_should_return_none_for_a_page_that_is_not_there(self, tmp_path: Path):
        renderer = PageRegionRenderer(
            MINIMAL_EXAMPLE_PDF_PATH, dpi=72.0, temp_dir=str(tmp_path)
        )
        assert renderer.render(99, _BOX_FOR_RENDER, str(tmp_path / 'region.png')) is None

    def test_should_return_none_for_a_document_it_cannot_read(self, tmp_path: Path):
        renderer = PageRegionRenderer(
            str(tmp_path / 'not-there.pdf'), dpi=72.0, temp_dir=str(tmp_path)
        )
        assert renderer.render(1, _BOX_FOR_RENDER, str(tmp_path / 'region.png')) is None
