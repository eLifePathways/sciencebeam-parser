from __future__ import annotations

import io
from pathlib import Path

import PIL.Image

from benchmarks.analyze_gold_failures._screenshots import (
    TeiCoords,
    render_region_screenshot
)


MINIMAL_EXAMPLE_PDF_PATH = Path('test-data/minimal-example.pdf')


class TestRenderRegionScreenshot:
    def test_should_render_the_region_a_coordinate_names(self):
        png_bytes = render_region_screenshot(
            MINIMAL_EXAMPLE_PDF_PATH, TeiCoords(page=1, x=10, y=10, w=100, h=50)
        )
        assert png_bytes
        with PIL.Image.open(io.BytesIO(png_bytes)) as image:
            assert image.width > 0 and image.height > 0

    def test_should_render_nothing_for_a_document_it_cannot_read(self, tmp_path: Path):
        bad_pdf_path = tmp_path / 'bad.pdf'
        bad_pdf_path.write_bytes(b'not a pdf at all')
        assert render_region_screenshot(
            bad_pdf_path, TeiCoords(page=1, x=10, y=10, w=100, h=50)
        ) is None

    def test_should_render_nothing_for_a_page_that_is_not_there(self):
        assert render_region_screenshot(
            MINIMAL_EXAMPLE_PDF_PATH, TeiCoords(page=99, x=10, y=10, w=100, h=50)
        ) is None
