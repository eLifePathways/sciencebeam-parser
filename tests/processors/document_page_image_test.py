import logging
from pathlib import Path

import pytest

from sciencebeam_parser.external.pdfium.renderer import PdfRenderWorkerPool
from sciencebeam_parser.processors.document_page_image import (
    iter_pdf_document_page_images
)


LOGGER = logging.getLogger(__name__)

MINIMAL_EXAMPLE_PDF_PATH = 'test-data/minimal-example.pdf'


@pytest.fixture(name='pool')
def _pool() -> PdfRenderWorkerPool:
    return PdfRenderWorkerPool(max_workers=1, timeout=20)


class TestIterPdfDocumentPageImages:
    def test_should_render_every_page_where_none_is_named(
        self, tmp_path: Path, pool: PdfRenderWorkerPool
    ):
        page_images = list(iter_pdf_document_page_images(
            MINIMAL_EXAMPLE_PDF_PATH, str(tmp_path), pool, dpi=36
        ))
        assert [page_image.page_number for page_image in page_images] == [1]

    def test_should_render_no_page_where_an_empty_selection_is_given(
        self, tmp_path: Path, pool: PdfRenderWorkerPool
    ):
        page_images = list(iter_pdf_document_page_images(
            MINIMAL_EXAMPLE_PDF_PATH, str(tmp_path), pool, dpi=36, page_numbers=[]
        ))
        assert not page_images

    def test_should_render_the_pages_named(
        self, tmp_path: Path, pool: PdfRenderWorkerPool
    ):
        page_images = list(iter_pdf_document_page_images(
            MINIMAL_EXAMPLE_PDF_PATH, str(tmp_path), pool, dpi=36, page_numbers=[1]
        ))
        assert [page_image.page_number for page_image in page_images] == [1]

    def test_should_render_nothing_where_no_worker_is_free(self, tmp_path: Path):
        pool = PdfRenderWorkerPool(max_workers=1, timeout=1)
        with pool.worker(MINIMAL_EXAMPLE_PDF_PATH):
            page_images = list(iter_pdf_document_page_images(
                MINIMAL_EXAMPLE_PDF_PATH, str(tmp_path), pool, dpi=36
            ))
        assert not page_images

    def test_should_render_nothing_for_a_document_it_cannot_read(
        self, tmp_path: Path, pool: PdfRenderWorkerPool
    ):
        bad_pdf_path = tmp_path / 'bad.pdf'
        bad_pdf_path.write_bytes(b'not a pdf at all')
        page_images = list(iter_pdf_document_page_images(
            str(bad_pdf_path), str(tmp_path), pool, dpi=36
        ))
        assert not page_images
