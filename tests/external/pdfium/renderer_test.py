import logging
import signal
import threading
import time
from pathlib import Path
from unittest.mock import patch

import PIL.Image
import pytest

from sciencebeam_parser.external.pdfium.renderer import (
    RenderWorkerUnavailableError,
    PdfRenderWorker,
    PdfRenderWorkerPool
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

    def test_should_render_nothing_where_a_worker_stops_answering(self, tmp_path: Path):
        worker = PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH, timeout=2)
        try:
            assert worker.render(1, 1.0, str(tmp_path / 'first.png'))
            process = worker._process  # pylint: disable=protected-access
            assert process is not None
            process.send_signal(signal.SIGSTOP)
            assert worker.render(1, 1.0, str(tmp_path / 'second.png')) is None
        finally:
            worker.close()

    def test_should_render_nothing_where_a_worker_cannot_be_started(self, tmp_path: Path):
        with patch('subprocess.Popen', side_effect=OSError('cannot fork')):
            with PdfRenderWorker(MINIMAL_EXAMPLE_PDF_PATH, timeout=5) as worker:
                assert worker.render(1, 1.0, str(tmp_path / 'page.png')) is None
                assert worker.get_page_count() is None


class TestPdfRenderWorkerPool:
    def test_should_refuse_a_worker_while_the_only_one_is_in_use(self, tmp_path: Path):
        pool = PdfRenderWorkerPool(max_workers=1, timeout=1)
        with pool.worker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(1, 1.0, str(tmp_path / 'first.png'))
            with pytest.raises(RenderWorkerUnavailableError):
                with pool.worker(MINIMAL_EXAMPLE_PDF_PATH):
                    pass

    def test_should_hand_out_a_worker_once_one_is_given_back(self, tmp_path: Path):
        pool = PdfRenderWorkerPool(max_workers=1, timeout=30)
        with pool.worker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(1, 1.0, str(tmp_path / 'first.png'))
        with pool.worker(MINIMAL_EXAMPLE_PDF_PATH) as worker:
            assert worker.render(1, 1.0, str(tmp_path / 'second.png'))

    def test_should_give_a_worker_back_where_rendering_raises(self):
        pool = PdfRenderWorkerPool(max_workers=1, timeout=1)
        try:
            with pool.worker(MINIMAL_EXAMPLE_PDF_PATH):
                raise RuntimeError('while rendering')
        except RuntimeError:
            pass
        with pool.worker(MINIMAL_EXAMPLE_PDF_PATH):
            pass

    def test_should_wait_for_a_worker_rather_than_go_without(self, tmp_path: Path):
        pool = PdfRenderWorkerPool(max_workers=1, timeout=30)
        rendered = []

        def _render_once_free():
            with pool.worker(MINIMAL_EXAMPLE_PDF_PATH) as waiting:
                rendered.append(waiting.render(1, 0.5, str(tmp_path / 'waited.png')))

        with pool.worker(MINIMAL_EXAMPLE_PDF_PATH):
            waiting_thread = threading.Thread(target=_render_once_free)
            waiting_thread.start()
            time.sleep(0.5)
            assert not rendered
        waiting_thread.join(30)
        assert rendered and rendered[0]
