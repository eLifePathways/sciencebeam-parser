import logging
import os
from time import monotonic
from typing import Iterable, NamedTuple, Optional, Sequence

from sciencebeam_parser.external.pdfium.renderer import (
    PdfRenderWorkerPool,
    RenderWorkerUnavailableError
)


LOGGER = logging.getLogger(__name__)


DEFAULT_PDF_RENDER_DPI = 200


class DocumentPageImage(NamedTuple):
    page_number: int
    page_image_path: str


def iter_pdf_document_page_images(
    pdf_path: str,
    output_dir: str,
    pool: PdfRenderWorkerPool,
    dpi: float = DEFAULT_PDF_RENDER_DPI,
    page_numbers: Optional[Sequence[int]] = None
) -> Iterable[DocumentPageImage]:
    render_start = monotonic()
    rendered_count = 0
    try:
        with pool.worker(pdf_path) as worker:
            if page_numbers is None:
                page_count = worker.get_page_count()
                selected_page_numbers: Sequence[int] = (
                    range(1, page_count + 1) if page_count else []
                )
            else:
                selected_page_numbers = sorted(page_numbers)
            for page_number in selected_page_numbers:
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
    except RenderWorkerUnavailableError:
        LOGGER.warning('rendering no page image', exc_info=True)
        return
    LOGGER.info(
        'rendered PDF pages, took=%.3fs, page_count=%d, dpi=%s',
        monotonic() - render_start, rendered_count, dpi
    )
