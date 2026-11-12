"""Renders pages of one PDF, as a process of its own.

pdfium parses the document in the process that calls it, so a document it
cannot survive takes that process with it. Run as `python -m` against one
document and the file descriptor of a socket to answer on, this is the
process that can be lost.

Requests and replies are dictionaries over a `multiprocessing.connection`:

    {'page_count': True}
    {'page_number': 7, 'scale': 1.39, 'image_path': '/tmp/p7.png'}
    {'page_number': 7, 'scale': 1.39, 'image_path': '/tmp/r7.png',
     'crop': (12.0, 30.0, 220.0, 190.0)}

The document is opened on the first request, so one it cannot read is a
reply rather than a death, and stays open across the rest; the last page
rendered is kept, so several regions of one page cost one render.
"""

import multiprocessing.connection
import sys
from typing import Any, Dict, Optional, Tuple

import PIL.Image
import pypdfium2


T_Crop = Tuple[float, float, float, float]


class PageRenderer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self._pdf_document: Optional[pypdfium2.PdfDocument] = None
        self._page_number: Optional[int] = None
        self._scale: Optional[float] = None
        self._page_image: Optional[PIL.Image.Image] = None

    @property
    def pdf_document(self) -> pypdfium2.PdfDocument:
        if self._pdf_document is None:
            self._pdf_document = pypdfium2.PdfDocument(self.pdf_path)
        return self._pdf_document

    def get_page_count(self) -> int:
        return len(self.pdf_document)

    def get_page_image(self, page_number: int, scale: float) -> PIL.Image.Image:
        if (
            self._page_image is None
            or self._page_number != page_number
            or self._scale != scale
        ):
            self._page_image = (
                self.pdf_document[page_number - 1]
                .render(scale=scale)
                .to_pil()
                .convert('RGB')
            )
            self._page_number = page_number
            self._scale = scale
        return self._page_image

    def render(
        self,
        page_number: int,
        scale: float,
        image_path: str,
        crop: Optional[T_Crop] = None
    ) -> Dict[str, Any]:
        page_image = self.get_page_image(page_number, scale)
        image = page_image
        if crop is not None:
            image = page_image.crop((
                max(0, round(crop[0] * scale)),
                max(0, round(crop[1] * scale)),
                min(page_image.width, round(crop[2] * scale)),
                min(page_image.height, round(crop[3] * scale))
            ))
        if not image.width or not image.height:
            return {'ok': False, 'error': 'empty image'}
        image.save(image_path)
        return {'ok': True, 'image_path': image_path}


def get_reply(renderer: PageRenderer, request: Dict[str, Any]) -> Dict[str, Any]:
    if request.get('page_count'):
        return {'ok': True, 'page_count': renderer.get_page_count()}
    return renderer.render(
        page_number=request['page_number'],
        scale=request['scale'],
        image_path=request['image_path'],
        crop=request.get('crop')
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    renderer = PageRenderer(argv[0])
    connection = multiprocessing.connection.Connection(int(argv[1]))
    while True:
        try:
            request = connection.recv()
        except (EOFError, OSError):
            return
        try:
            reply = get_reply(renderer, request)
        except Exception as exc:  # pylint: disable=broad-except
            reply = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
        try:
            connection.send(reply)
        except (EOFError, OSError):
            return


if __name__ == '__main__':
    main()
