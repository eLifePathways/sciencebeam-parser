"""The per-line record a generated corpus carries, for reviewing it on the page.

One file per document, one JSON object per text line, in the order the TEI writes
its ``<lb/>`` elements and the raw file writes its feature rows.  Each row holds
where the line sits, what it says and what it is labelled -- everything needed to
put a label back on the page, and nothing that has to be looked up elsewhere.

Position survives nowhere else in a corpus: the features carry relative bins, and
the TEI carries no coordinate at all.  The label and the text are read back out of
the training TEI this run just built, rather than recomputed from the model data,
so the record cannot describe a different document to the one beside it.

Whatever reads it cannot import this package, so the format name, its version and
the field names travel with the data.  A line the layout could not place keeps its
place in the file, with its label and text and no coordinates.
"""
import json
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Sequence

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutLine,
    LayoutPageCoordinates,
    get_merged_coordinates_list
)
from sciencebeam_parser.models.data import LayoutModelData


LINES_FORMAT = 'sciencebeam-training-lines'
LINES_FORMAT_VERSION = 1

LINE_BREAK_TAG = 'lb'


class LabelledLine(NamedTuple):
    label: str
    text: str


def _rounded(value: float) -> float:
    return round(value, 2)


def get_line_coordinates(layout_line: LayoutLine) -> Optional[LayoutPageCoordinates]:
    merged_coordinates_list = get_merged_coordinates_list([
        token.coordinates
        for token in layout_line.tokens
        if token.coordinates
    ])
    if not merged_coordinates_list:
        return None
    return merged_coordinates_list[0]


def get_coordinates_json_dict(layout_line: Optional[LayoutLine]) -> Dict[str, Any]:
    coordinates = get_line_coordinates(layout_line) if layout_line is not None else None
    if coordinates is None:
        return {}
    return {
        'page': coordinates.page_number,
        'x': _rounded(coordinates.x),
        'y': _rounded(coordinates.y),
        'width': _rounded(coordinates.width),
        'height': _rounded(coordinates.height)
    }


def iter_page_json_dicts(layout_document: LayoutDocument) -> Iterable[Dict[str, Any]]:
    for page in layout_document.pages:
        page_json_dict: Dict[str, Any] = {'page': page.meta.page_number}
        coordinates = page.meta.coordinates
        if coordinates:
            page_json_dict['width'] = _rounded(coordinates.width)
            page_json_dict['height'] = _rounded(coordinates.height)
        yield page_json_dict


def get_element_descriptor(element: etree.ElementBase) -> str:
    """How `training_xml_element_path_by_label` spells this element."""
    local_name = etree.QName(element).localname
    return local_name + ''.join(
        f'[@{name}="{value}"]' for name, value in element.attrib.items()
    )


def get_label_by_element_path(
    training_xml_element_path_by_label: Mapping[str, Sequence[str]]
) -> Dict[tuple, str]:
    """The label of each element path, keeping the first label a path is named by."""
    label_by_element_path: Dict[tuple, str] = {}
    for label, element_path in training_xml_element_path_by_label.items():
        label_by_element_path.setdefault(tuple(element_path), label)
    return label_by_element_path


def _iter_text_and_line_breaks(
    element: etree.ElementBase,
    element_path: tuple
) -> Iterator[tuple]:
    if element.text:
        yield (element_path, element.text, False)
    for child in element:
        if not isinstance(child.tag, str):
            continue
        if etree.QName(child).localname == LINE_BREAK_TAG:
            yield (element_path, '', True)
        else:
            yield from _iter_text_and_line_breaks(
                child, element_path + (get_element_descriptor(child),)
            )
        if child.tail:
            yield (element_path, child.tail, False)


def iter_labelled_lines(
    training_tei_root: etree.ElementBase,
    root_training_xml_element_path: Sequence[str],
    training_xml_element_path_by_label: Mapping[str, Sequence[str]]
) -> Iterator[LabelledLine]:
    """Every `<lb/>`-terminated line of the training TEI, with the label of its element."""
    label_by_element_path = get_label_by_element_path(training_xml_element_path_by_label)
    root_path = tuple(root_training_xml_element_path)
    root_element = training_tei_root
    for local_name in root_training_xml_element_path:
        root_element = root_element.find(local_name)
        if root_element is None:
            return

    def _get_label(element_path: tuple) -> str:
        return label_by_element_path.get(element_path, '/'.join(element_path))

    parts: List[str] = []
    element_path: Optional[tuple] = None
    for current_path, text, is_line_break in _iter_text_and_line_breaks(
        root_element, root_path
    ):
        if not is_line_break:
            if element_path is None and text.strip():
                element_path = current_path
            parts.append(text)
            continue
        yield LabelledLine(
            label=_get_label(element_path if element_path is not None else current_path),
            text=' '.join(''.join(parts).split())
        )
        parts = []
        element_path = None


def format_lines_record(
    document_id: str,
    model_name: str,
    layout_document: LayoutDocument,
    model_data_list_list: Sequence[Sequence[LayoutModelData]],
    labelled_lines: Sequence[LabelledLine]
) -> str:
    coordinates_json_dicts = [
        get_coordinates_json_dict(model_data.layout_line)
        for model_data_list in model_data_list_list
        for model_data in model_data_list
    ]
    if len(coordinates_json_dicts) != len(labelled_lines):
        raise ValueError(
            f'{document_id}: {len(coordinates_json_dicts)} lines of model data against'
            f' {len(labelled_lines)} lines in the training tei'
        )
    line_json_dicts = [
        {**coordinates_json_dict, 'label': labelled_line.label, 'text': labelled_line.text}
        for coordinates_json_dict, labelled_line in zip(coordinates_json_dicts, labelled_lines)
    ]
    header_json_dict = {
        'format': LINES_FORMAT,
        'version': LINES_FORMAT_VERSION,
        'document_id': document_id,
        'model': model_name,
        'line_count': len(line_json_dicts),
        'pages': list(iter_page_json_dicts(layout_document))
    }
    return ''.join(
        json.dumps(json_dict) + '\n'
        for json_dict in [header_json_dict] + line_json_dicts
    )
