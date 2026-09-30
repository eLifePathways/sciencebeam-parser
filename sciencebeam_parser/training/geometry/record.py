"""The per-line geometry a generated corpus carries.

One file per document, one JSON object per text line, in the order the TEI writes
its ``<lb/>`` elements and the raw file writes its feature rows.  Position is what
makes a segmentation label judgeable, and it survives nowhere else in a corpus:
the features carry relative bins, and the TEI carries no coordinate at all.

The record holds geometry alone.  Labels stay in the TEI beside it, so a
regeneration that only moves labels leaves this file untouched, and the line
counts of the two are what tells a reader they still line up.

Whatever reads it cannot import this package, so the format name, its version and
the field names travel with the data.  A line the layout could not place is an
object with no geometry, keeping its position in the file.
"""
import json
from typing import Any, Dict, Iterable, Optional, Sequence

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutLine,
    LayoutPageCoordinates,
    get_merged_coordinates_list
)
from sciencebeam_parser.models.data import LayoutModelData


GEOMETRY_FORMAT = 'sciencebeam-training-geometry'
GEOMETRY_FORMAT_VERSION = 1


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


def get_line_json_dict(layout_line: Optional[LayoutLine]) -> Dict[str, Any]:
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


def format_geometry_record(
    document_id: str,
    model_name: str,
    layout_document: LayoutDocument,
    model_data_list_list: Sequence[Sequence[LayoutModelData]]
) -> str:
    line_json_dicts = [
        get_line_json_dict(model_data.layout_line)
        for model_data_list in model_data_list_list
        for model_data in model_data_list
    ]
    header_json_dict = {
        'format': GEOMETRY_FORMAT,
        'version': GEOMETRY_FORMAT_VERSION,
        'document_id': document_id,
        'model': model_name,
        'line_count': len(line_json_dicts),
        'pages': list(iter_page_json_dicts(layout_document))
    }
    return ''.join(
        json.dumps(json_dict) + '\n'
        for json_dict in [header_json_dict] + line_json_dicts
    )
