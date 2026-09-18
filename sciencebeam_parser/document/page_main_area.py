from typing import Counter, Dict, Optional

from sciencebeam_parser.document.layout_document import LayoutDocument
from sciencebeam_parser.utils.bounding_box import BoundingBox


# based on:
# https://github.com/kermitt2/grobid/blob/0.9.0/grobid-core/src/main/java/org/grobid/core/document/Document.java#L463-L537
def calculate_page_main_areas(  # pylint: disable=too-many-locals,too-many-branches
    layout_document: LayoutDocument
) -> Dict[int, Optional[BoundingBox]]:
    left_even: Counter[int] = Counter()
    right_even: Counter[int] = Counter()
    left_odd: Counter[int] = Counter()
    right_odd: Counter[int] = Counter()
    top_counter: Counter[int] = Counter()
    bottom_counter: Counter[int] = Counter()

    for page in layout_document.pages:
        page_number = page.meta.page_number
        for block in page.blocks:
            coords_list = block.get_merged_coordinates_list()
            if not coords_list:
                continue
            coords = coords_list[0]
            # Filter small blocks (page numbers, journal headers, etc.)
            if (coords.x == 0 or coords.height < 20
                    or coords.width < 20 or coords.height * coords.width < 3000):
                continue
            x, y, width, height = coords.x, coords.y, coords.width, coords.height
            if page_number % 2 == 0:
                left_even[int(x)] += 1
                right_even[int(x + width)] += 1
            else:
                left_odd[int(x)] += 1
                right_odd[int(x + width)] += 1
            top_counter[int(y)] += 1
            bottom_counter[int(y + height)] += 1

    result: Dict[int, Optional[BoundingBox]] = {}

    if left_even and left_odd:
        page_y = min(top_counter)
        page_height = max(bottom_counter) - page_y + 1
        page_odd_x = min(left_odd)
        page_odd_width = max(right_odd) - page_odd_x + 1
        if len(layout_document.pages) > 1:
            page_even_x = min(left_even)
            page_even_width = max(right_even) - page_even_x + 1
        else:
            page_even_x = 0
            page_even_width = 0
        for page in layout_document.pages:
            page_number = page.meta.page_number
            if page_number % 2 == 0 and len(layout_document.pages) > 1:
                result[page_number] = BoundingBox(
                    page_even_x, page_y, page_even_width, page_height
                )
            else:
                result[page_number] = BoundingBox(
                    page_odd_x, page_y, page_odd_width, page_height
                )
    else:
        # Fallback: use full page dimensions when not enough data for both even/odd
        for page in layout_document.pages:
            page_number = page.meta.page_number
            if page.meta.coordinates:
                coords = page.meta.coordinates
                result[page_number] = BoundingBox(0, 0, coords.width, coords.height)
            else:
                result[page_number] = None

    return result
