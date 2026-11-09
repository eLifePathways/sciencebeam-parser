import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pytest


HARNESS_FILE = Path(__file__).parent / 'add_svg_view_box.mjs'


pytestmark = pytest.mark.skipif(
    shutil.which('node') is None,
    reason='node runs the view box the page applies in a browser'
)


def add_svg_view_box(svg: str) -> str:
    completed_process = subprocess.run(
        ['node', str(HARNESS_FILE)],
        input=svg,
        capture_output=True,
        text=True,
        check=True
    )
    return completed_process.stdout


def get_view_box(svg: str) -> Optional[str]:
    match = re.search(r'viewBox="([^"]*)"', svg)
    return match.group(1) if match else None


class TestAddSvgViewBox:
    def test_should_cover_the_coordinates_the_paths_use(self):
        svg = add_svg_view_box(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<g><path d="M41.2080,351.7260 L52.3320,211.4160 Z"/></g>'
            '</svg>'
        )
        assert get_view_box(svg) == '41.208 41.208 310.518 310.518'

    def test_should_leave_a_drawing_that_states_its_own_view_box(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            '<path d="M1,1 L99,99"/></svg>'
        )
        assert add_svg_view_box(svg) == svg

    def test_should_leave_a_drawing_that_states_a_width_and_a_height(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="30">'
            '<path d="M1,1 L99,99"/></svg>'
        )
        assert add_svg_view_box(svg) == svg

    def test_should_leave_a_drawing_with_no_paths(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><g/></svg>'
        assert add_svg_view_box(svg) == svg

    def test_should_leave_something_that_is_not_a_drawing(self):
        assert add_svg_view_box('not svg at all') == 'not svg at all'

    def test_should_keep_everything_the_drawing_carries(self):
        svg = add_svg_view_box(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<g style="fill: #BDBDBD;" id="p28_s11636">'
            '<path d="M41.2080,351.7260 L52.3320,211.4160 Z"/></g></svg>'
        )
        assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
        assert 'id="p28_s11636"' in svg
        assert 'style="fill: #BDBDBD;"' in svg
        assert svg.endswith('</svg>')

    def test_should_read_a_negative_coordinate(self):
        svg = add_svg_view_box(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<path d="M-5,-5 L10,10"/></svg>'
        )
        assert get_view_box(svg) == '-5 -5 15 15'

    def test_should_apply_to_a_self_closing_root(self):
        svg = add_svg_view_box(
            '<svg xmlns="http://www.w3.org/2000/svg" d="M0,0 L4,4"/>'
        )
        assert get_view_box(svg) == '0 0 4 4'
        assert svg.endswith('/>')
