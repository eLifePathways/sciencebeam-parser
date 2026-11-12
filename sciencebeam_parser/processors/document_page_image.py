import logging
import os
from time import monotonic
from typing import Iterable, NamedTuple, Optional, Sequence

from sciencebeam_parser.external.pdfium.renderer import (
    DEFAULT_RENDER_TIMEOUT,
    PdfRenderWorker
)


LOGGER = logging.getLogger(__name__)


DEFAULT_PDF_RENDER_DPI = 200


class DocumentPageImage(NamedTuple):
    page_number: int
    page_image_path: str


def iter_pdf_document_page_images(
    pdf_path: str,
    output_dir: str,
    dpi: float = DEFAULT_PDF_RENDER_DPI,
    page_numbers: Optional[Sequence[int]] = None,
    timeout: float = DEFAULT_RENDER_TIMEOUT
) -> Iterable[DocumentPageImage]:
    render_start = monotonic()
    rendered_count = 0
    with PdfRenderWorker(pdf_path, timeout=timeout) as worker:
        for page_number in _iter_page_numbers(pdf_path, page_numbers):
            image_path = worker.render(
                page_number=page_number,
                scale=dpi / 72.0,
                image_path=os.path.join(output_dir, 'page-%d.png' % page_number)
            )
            if image_path is None:
                continue
            rendered_count += 1
            yield DocumentPageImage(
                page_number=page_number,
                page_image_path=image_path
            )
    LOGGER.info(
        'rendered PDF pages, took=%.3fs, page_count=%d, dpi=%s',
        monotonic() - render_start, rendered_count, dpi
    )


def _iter_page_numbers(
    pdf_path: str,
    page_numbers: Optional[Sequence[int]]
) -> Iterable[int]:
    if page_numbers:
        return sorted(page_numbers)
    return range(1, get_pdf_page_count(pdf_path) + 1)


def get_pdf_page_count(pdf_path: str) -> int:
    import pypdfium2  # noqa pylint: disable=import-outside-toplevel
    pdf_document = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(pdf_document)
    finally:
        pdf_document.close()
