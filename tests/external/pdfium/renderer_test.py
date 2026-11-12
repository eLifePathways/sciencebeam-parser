import logging
from pathlib import Path

import PIL.Image

from sciencebeam_parser.external.pdfium.renderer import (
    PdfRenderWorker,
    RenderWorkerLimit
)


LOGGER = logging.getLogger(__name__)

MINIMAL_EXAMPLE_PDF_PATH = 'test-data/minimal-example.pdf'


class TestPdfRenderWorker:
    def test_should_render_a_whole_page(self, tmp_path: Path):
        image_path = str(tmp_path / 'page.png')
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(1, 1.0, image_path) == image_path
        with PIL.Image.open(image_path) as rendered_image:
            assert (rendered_image.width, rendered_image.height) == (612, 792)

    def test_should_render_a_region_of_a_page(self, tmp_path: Path):
        image_path = str(tmp_path / 'region.png')
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(1, 1.0, image_path, crop=(10, 10, 110, 60)) == image_path
        with PIL.Image.open(image_path) as rendered_image:
            assert (rendered_image.width, rendered_image.height) == (100, 50)

    def test_should_render_several_pages_through_one_worker(self, tmp_path: Path):
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            first = worker.render(1, 1.0, str(tmp_path / 'a.png'), crop=(0, 0, 50, 50))
            second = worker.render(1, 1.0, str(tmp_path / 'b.png'), crop=(50, 50, 120, 120))
        assert first and second

    def test_should_report_the_page_count(self):
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.get_page_count() == 1

    def test_should_report_no_page_count_for_a_document_it_cannot_read(self, tmp_path: Path):
        bad_pdf_path = tmp_path / 'bad.pdf'
        bad_pdf_path.write_bytes(b'not a pdf at all')
        with PdfRenderWorker(str(bad_pdf_path), timeout=20) as worker:
            assert worker.get_page_count() is None

    def test_should_answer_a_second_request_after_one_it_could_not_read(self, tmp_path: Path):
        bad_pdf_path = tmp_path / 'bad.pdf'
        bad_pdf_path.write_bytes(b'not a pdf at all')
        with PdfRenderWorker(str(bad_pdf_path), timeout=20) as worker:
            assert worker.render(1, 1.0, str(tmp_path / 'page.png')) is None
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.get_page_count() == 1

    def test_should_return_none_for_a_document_it_cannot_read(self, tmp_path: Path):
        with PdfRenderWorker(str(tmp_path / 'not-there.pdf'), timeout=20) as worker:
            assert worker.render(1, 1.0, str(tmp_path / 'page.png')) is None

    def test_should_return_none_for_a_page_that_is_not_there(self, tmp_path: Path):
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(99, 1.0, str(tmp_path / 'page.png')) is None

    def test_should_answer_after_a_failure_through_a_worker_of_its_own(self, tmp_path: Path):
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(99, 1.0, str(tmp_path / 'missing.png')) is None
        with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(1, 1.0, str(tmp_path / 'page.png'))


class TestRenderWorkerLimit:
    def test_should_hold_a_slot_for_the_life_of_a_worker(self, tmp_path: Path):
        limit = RenderWorkerLimit(max_workers=1)
        worker = PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH, limit=limit)
        try:
            assert worker.render(1, 1.0, str(tmp_path / 'first.png'))
            assert not limit.acquire(timeout=0.1)
        finally:
            worker.close()
        assert limit.acquire(timeout=0.1)
        limit.release()

    def test_should_give_the_slot_back_where_a_worker_is_not_closed(self, tmp_path: Path):
        limit = RenderWorkerLimit(max_workers=1)
        worker = PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH, limit=limit)
        assert worker.render(1, 1.0, str(tmp_path / 'first.png'))
        del worker
        assert limit.acquire(timeout=5)
        limit.release()

    def test_should_render_nothing_where_no_slot_comes_free(self, tmp_path: Path):
        limit = RenderWorkerLimit(max_workers=1)
        assert limit.acquire(timeout=1)
        try:
            waiting = PdfRenderWorker(
                MINIMAL_EXAMPLE_PDF_PATH, timeout=0.5, limit=limit
            )
            assert waiting.render(1, 1.0, str(tmp_path / 'page.png')) is None
            waiting.close()
        finally:
            limit.release()
