"""Building a laid-out page, so a test can say what the page prints.

The segmentation deriver reads coordinates, blocks and type as much as it
reads the JATS, so a fixture that leaves them out tests nothing: these put a
document on a page the way a renderer would.
"""
from typing import Optional

from sciencebeam_parser.document.layout_document import (
    EMPTY_FONT,
    LayoutBlock,
    LayoutDocument,
    LayoutFont,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken,
)
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.segmentation import (
    SegmentationConfig,
    SegmentationLabelDeriver,
)


BOLD_FONT = LayoutFont(font_id='bold', font_size=10.0, is_bold=True)


def make_page_meta(page_number: int = 1, height: float = 1000.0) -> LayoutPageMeta:
    return LayoutPageMeta(
        page_number=page_number,
        coordinates=LayoutPageCoordinates(
            x=0, y=0, width=600, height=height, page_number=page_number
        ),
    )


# A token is as wide as it has characters, which is what lets a fixture say
# that a heading does not fill the measure the page sets its text to.
_CHARACTER_WIDTH = 5.0
_SPACE_WIDTH = 5.0


def make_token(
    text: str,
    page_number: int = 1,
    y: float = 500.0,
    x: float = 10.0,
    bold: bool = False,
    font_size: Optional[float] = None,
    height: float = 12.0,
) -> LayoutToken:
    font = BOLD_FONT if bold else EMPTY_FONT
    if font_size is not None:
        font = LayoutFont(font_id=f'size-{font_size}', font_size=font_size, is_bold=bold)
    return LayoutToken(
        text=text,
        font=font,
        coordinates=LayoutPageCoordinates(
            x=x, y=y, width=max(len(text), 1) * _CHARACTER_WIDTH,
            height=height, page_number=page_number,
        ),
    )


def make_line(
    *texts: str,
    y: float = 500.0,
    page_number: int = 1,
    x: float = 10.0,
    bold: bool = False,
    font_size: Optional[float] = None,
    height: float = 12.0,
) -> LayoutLine:
    tokens = []
    left = x
    for text in texts:
        token = make_token(
            text, page_number=page_number, y=y, x=left, bold=bold,
            font_size=font_size, height=height,
        )
        tokens.append(token)
        assert token.coordinates is not None
        left += token.coordinates.width + _SPACE_WIDTH
    return LayoutLine(tokens=tokens)


def page_number_of(line: LayoutLine) -> int:
    coordinates = line.tokens[0].coordinates
    assert coordinates is not None
    return coordinates.page_number


def make_doc_with_page(
    *blocks: LayoutBlock, page_height: float = 1000.0, page_number: int = 1
) -> LayoutDocument:
    page_meta = make_page_meta(page_number=page_number, height=page_height)
    page = LayoutPage(blocks=list(blocks), meta=page_meta)
    return LayoutDocument(pages=[page])


def annotate(doc: LayoutDocument, field_by_line_index: dict) -> JatsAnnotatedLayoutDocument:
    annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
    lines = list(doc.iter_all_lines())
    for line_idx, field_name in field_by_line_index.items():
        for token in lines[line_idx].tokens:
            annotated.set_token_label(token, field_name)
    return annotated


def derive_labels(doc, annotated, **config_kwargs):
    config = SegmentationConfig(**config_kwargs) if config_kwargs else None
    return SegmentationLabelDeriver(config).derive_labels(doc, annotated)
