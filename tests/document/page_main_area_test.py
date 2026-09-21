from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta
)
from sciencebeam_parser.document.page_main_area import calculate_page_main_areas
from sciencebeam_parser.utils.bounding_box import BoundingBox


class TestCalculatePageMainAreas:
    def test_should_use_full_page_when_only_odd_pages(self):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        block_coords = LayoutPageCoordinates(x=50, y=50, width=400, height=600, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(page_coords)
            )
        ])
        result = calculate_page_main_areas(layout_document)
        assert result[1] == BoundingBox(0, 0, 500, 700)

    def test_should_use_full_page_when_no_page_coordinates(self):
        block_coords = LayoutPageCoordinates(x=50, y=50, width=400, height=600, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[
                LayoutLine.for_text('text', coordinates=block_coords)
            ])])
        ])
        result = calculate_page_main_areas(layout_document)
        assert result[0] is None

    def test_should_filter_small_blocks(self):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        # height=15 < 20: filtered out
        small_block_coords = LayoutPageCoordinates(x=50, y=50, width=400, height=15, page_number=1)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=small_block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(page_coords)
            )
        ])
        result = calculate_page_main_areas(layout_document)
        # filtered → fallback to full page
        assert result[1] == BoundingBox(0, 0, 500, 700)

    def test_should_compute_main_area_from_odd_and_even_pages(self):
        odd_page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=1)
        even_page_coords = LayoutPageCoordinates(x=0, y=0, width=500, height=700, page_number=2)
        odd_block_coords = LayoutPageCoordinates(x=50, y=60, width=400, height=580, page_number=1)
        even_block_coords = LayoutPageCoordinates(x=60, y=60, width=380, height=580, page_number=2)
        layout_document = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=odd_block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(odd_page_coords)
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    LayoutLine.for_text('text', coordinates=even_block_coords)
                ])],
                meta=LayoutPageMeta.for_coordinates(even_page_coords)
            )
        ])
        result = calculate_page_main_areas(layout_document)
        # odd page: min_left=50, max_right=450, max_bottom=640
        # width = 450 - 50 + 1 = 401, height = 640 - 60 + 1 = 581
        assert result[1] == BoundingBox(50, 60, 401, 581)
        # even page: min_left=60, max_right=440
        # width = 440 - 60 + 1 = 381, height = 581
        assert result[2] == BoundingBox(60, 60, 381, 581)
