"""The per-span record a generated corpus carries, for reviewing it on the page.

One file per document, one JSON object per labelled span, in the order the TEI
writes them.  A span is a run of consecutive tokens inside one element and one
line: it is what the TEI marks, and it is the largest thing that can be drawn as
a box and labelled truthfully, because a line of a header document is often part
one label and part another.

Position survives nowhere else in a corpus: the features carry relative bins, and
the TEI carries no coordinate at all.  Spans come from the trace the TEI writer
leaves rather than from reading the written file back, so a span knows which
element instance it sits in and which tokens it was built from; that the spans of
a line still reproduce that line of the TEI is then checked rather than assumed.

Whatever reads it cannot import this package, so the format name, its version,
the field names and the model's labels travel with the data.  A span the layout
could not place keeps its place in the file, with its label and text and no
coordinates.
"""
import json
from typing import (
    Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple
)

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutPageCoordinates,
    LayoutToken,
    get_merged_coordinates_list
)
from sciencebeam_parser.models.data import LayoutModelData
from sciencebeam_parser.utils.labels import OTHER_LABELS
from sciencebeam_parser.utils.xml_writer import TracedElement, TracedItem, TracedText


SPANS_FORMAT = 'sciencebeam-training-spans'
SPANS_FORMAT_VERSION = 1

LINE_BREAK_TAG = 'lb'


class LabelledSpan(NamedTuple):
    line_index: int
    label: str
    text: str
    coordinates: Optional[LayoutPageCoordinates]


def _rounded(value: float) -> float:
    return round(value, 2)


def get_element_descriptor(element: etree.ElementBase) -> str:
    """How `training_xml_element_path_by_label` spells this element."""
    local_name = etree.QName(element).localname
    return local_name + ''.join(
        f'[@{name}="{value}"]' for name, value in element.attrib.items()
    )


def is_line_break(item: TracedItem) -> bool:
    return (
        isinstance(item, TracedElement)
        and etree.QName(item.element).localname == LINE_BREAK_TAG
    )


OTHER_LABEL = '<other>'


def get_label_by_element_path(
    training_xml_element_path_by_label: Mapping[str, Sequence[str]],
    root_training_xml_element_path: Sequence[str]
) -> Dict[Tuple[str, ...], str]:
    """The label of each element path, as the parser reading the TEI back would say it.

    Two labels can share an element path, and then the TEI cannot tell them apart;
    the first of them stands for both.  Text sitting directly in the root element
    is the case that matters: `header` maps `<note>` there as well as `<other>`,
    and the training TEI parser reads all of it as other text, so that is what the
    record says rather than a label no model is taught.
    """
    label_by_element_path: Dict[Tuple[str, ...], str] = {
        tuple(root_training_xml_element_path): OTHER_LABEL
    }
    for label, element_path in training_xml_element_path_by_label.items():
        if label in OTHER_LABELS:
            label_by_element_path[tuple(element_path)] = OTHER_LABEL
            continue
        label_by_element_path.setdefault(tuple(element_path), label)
    return label_by_element_path


def get_model_labels(
    training_xml_element_path_by_label: Mapping[str, Sequence[str]],
    root_training_xml_element_path: Sequence[str]
) -> List[str]:
    """Every label a span of this model can carry, in the order its table declares them.

    A label the TEI cannot tell from another is not one of them, so it is left out
    rather than given a colour nothing will use.  Position in this list is what a
    reader picks a colour by, so appending a label upstream leaves the labels
    before it where they were.
    """
    return list(get_label_by_element_path(
        training_xml_element_path_by_label, root_training_xml_element_path
    ).values())


def iter_model_data_tokens(model_data: LayoutModelData) -> Iterable[LayoutToken]:
    if model_data.layout_token is not None:
        yield model_data.layout_token
        return
    if model_data.layout_line is not None:
        yield from model_data.layout_line.tokens


def get_tokens_coordinates(
    layout_tokens: Iterable[LayoutToken]
) -> Optional[LayoutPageCoordinates]:
    merged_coordinates_list = get_merged_coordinates_list([
        layout_token.coordinates
        for layout_token in layout_tokens
        if layout_token.coordinates
    ])
    if not merged_coordinates_list:
        return None
    return merged_coordinates_list[0]


def get_coordinates_json_dict(
    coordinates: Optional[LayoutPageCoordinates]
) -> Dict[str, Any]:
    if coordinates is None:
        return {}
    return {
        'page': coordinates.page_number,
        'x': _rounded(coordinates.x),
        'y': _rounded(coordinates.y),
        'width': _rounded(coordinates.width),
        'height': _rounded(coordinates.height)
    }


class _PendingSpan:
    def __init__(self, element: etree.ElementBase, path: Tuple[str, ...]):
        self.element = element
        self.path = path
        self.text_parts: List[str] = []
        self.layout_tokens: List[LayoutToken] = []

    def add(self, text: str, source: Optional[Any]) -> None:
        self.text_parts.append(text)
        if isinstance(source, LayoutModelData):
            self.layout_tokens.extend(iter_model_data_tokens(source))


def iter_labelled_spans(
    trace: Iterable[TracedItem],
    training_xml_element_path_by_label: Mapping[str, Sequence[str]],
    root_training_xml_element_path: Sequence[str]
) -> Iterator[LabelledSpan]:
    """Every labelled span of the written TEI, in the order it was written.

    Text between two spans belongs to the one that follows it, so concatenating a
    line's spans reproduces that line: the whitespace separating two elements sits
    at their common ancestor and is part of neither on its own.

    A line is what a ``<lb/>`` ends, so the whitespace the writer leaves after the
    last of them is not one; anything else left there is, and is yielded so that
    the check against the TEI sees it rather than losing it quietly.
    """
    label_by_element_path = get_label_by_element_path(
        training_xml_element_path_by_label, root_training_xml_element_path
    )

    def _get_label(path: Tuple[str, ...]) -> str:
        return label_by_element_path.get(path, '/'.join(path))

    line_index = 0
    pending: Optional[_PendingSpan] = None
    pending_text = ''

    def _completed(span: _PendingSpan) -> LabelledSpan:
        return LabelledSpan(
            line_index=line_index,
            label=_get_label(span.path),
            text=''.join(span.text_parts),
            coordinates=get_tokens_coordinates(span.layout_tokens)
        )

    for item in trace:
        if is_line_break(item):
            if pending is None:
                pending = _PendingSpan(item.element, item.path)
            pending.add(pending_text, None)
            pending_text = ''
            yield _completed(pending)
            pending = None
            line_index += 1
            continue
        if not isinstance(item, TracedText):
            continue
        if item.source is None:
            pending_text += item.text
            continue
        if pending is not None and pending.element is not item.element:
            yield _completed(pending)
            pending = None
        if pending is None:
            pending = _PendingSpan(item.element, item.path)
        pending.add(pending_text + item.text, item.source)
        pending_text = ''
    if pending is None and not pending_text.strip():
        return
    if pending is None:
        pending = _PendingSpan(None, ())
    pending.add(pending_text, None)
    yield _completed(pending)


def iter_tei_line_texts(
    training_tei_root: etree.ElementBase,
    root_training_xml_element_path: Sequence[str]
) -> Iterator[str]:
    """The text between each pair of `<lb/>` elements, over every root element.

    A document's training TEI holds one root element per model data iterable, and
    a line can run from the end of one into the start of the next, so the text is
    read across them rather than restarted at each.  The whitespace left after the
    final ``<lb/>`` ends no line and is not one.
    """
    parent = training_tei_root
    for local_name in root_training_xml_element_path[:-1]:
        parent = parent.find(local_name)
        if parent is None:
            return
    root_local_name = root_training_xml_element_path[-1]
    parts: List[str] = []

    def _iter_text(element: etree.ElementBase) -> Iterator[Optional[str]]:
        if element.text:
            yield element.text
        for child in element:
            if not isinstance(child.tag, str):
                continue
            if etree.QName(child).localname == LINE_BREAK_TAG:
                yield None
            else:
                yield from _iter_text(child)
            if child.tail:
                yield child.tail

    for root_element in parent.findall(root_local_name):
        for text in _iter_text(root_element):
            if text is None:
                yield ''.join(parts)
                parts = []
                continue
            parts.append(text)
    trailing_text = ''.join(parts)
    if trailing_text.strip():
        yield trailing_text


def check_spans_against_tei(
    document_id: str,
    spans: Sequence[LabelledSpan],
    tei_line_texts: Sequence[str]
) -> None:
    """Refuse a record whose spans do not say what the TEI beside it says."""
    text_by_line_index: Dict[int, str] = {}
    for span in spans:
        text_by_line_index[span.line_index] = (
            text_by_line_index.get(span.line_index, '') + span.text
        )
    if len(text_by_line_index) != len(tei_line_texts):
        raise ValueError(
            f'{document_id}: spans cover {len(text_by_line_index)} lines against'
            f' {len(tei_line_texts)} in the training tei'
        )
    for line_index, tei_line_text in enumerate(tei_line_texts):
        span_text = text_by_line_index.get(line_index, '')
        if span_text != tei_line_text:
            raise ValueError(
                f'{document_id}: line {1 + line_index} is {span_text!r} in the spans'
                f' and {tei_line_text!r} in the training tei'
            )


def iter_page_json_dicts(
    layout_document: LayoutDocument,
    page_numbers: Iterable[int]
) -> Iterable[Dict[str, Any]]:
    """Geometry for the pages a span sits on, and no others."""
    wanted = set(page_numbers)
    for page in layout_document.pages:
        if page.meta.page_number not in wanted:
            continue
        page_json_dict: Dict[str, Any] = {'page': page.meta.page_number}
        coordinates = page.meta.coordinates
        if coordinates:
            page_json_dict['width'] = _rounded(coordinates.width)
            page_json_dict['height'] = _rounded(coordinates.height)
        yield page_json_dict


def format_spans_record(
    document_id: str,
    model_name: str,
    layout_document: LayoutDocument,
    spans: Sequence[LabelledSpan],
    labels: Sequence[str]
) -> str:
    span_json_dicts = [
        {
            'line': 1 + span.line_index,
            **get_coordinates_json_dict(span.coordinates),
            'label': span.label,
            'text': span.text
        }
        for span in spans
    ]
    page_numbers = sorted({
        span.coordinates.page_number for span in spans if span.coordinates
    })
    header_json_dict = {
        'format': SPANS_FORMAT,
        'version': SPANS_FORMAT_VERSION,
        'document_id': document_id,
        'model': model_name,
        'line_count': 1 + max((span.line_index for span in spans), default=-1),
        'span_count': len(span_json_dicts),
        'page_count': len(layout_document.pages),
        'labels': list(labels),
        'pages': list(iter_page_json_dicts(layout_document, page_numbers))
    }
    return ''.join(
        json.dumps(json_dict) + '\n'
        for json_dict in [header_json_dict] + span_json_dicts
    )
