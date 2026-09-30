import json
from typing import List, Sequence

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken
)
from sciencebeam_parser.models.data import LayoutModelData
from sciencebeam_parser.training.geometry.record import (
    GEOMETRY_FORMAT,
    GEOMETRY_FORMAT_VERSION,
    format_geometry_record,
    get_line_coordinates,
    get_line_json_dict
)


PAGE_COORDINATES_1 = LayoutPageCoordinates(x=0, y=0, width=595.0, height=842.0, page_number=1)


def _get_layout_line(
    token_coordinates_list: Sequence[LayoutPageCoordinates]
) -> LayoutLine:
    return LayoutLine(tokens=[
        LayoutToken(text=f'token{index}', coordinates=coordinates)
        for index, coordinates in enumerate(token_coordinates_list)
    ])


def _get_json_dicts(record: str) -> List[dict]:
    return [json.loads(line) for line in record.splitlines()]


class TestGetLineCoordinates:
    def test_should_merge_the_coordinates_of_the_tokens_of_a_line(self):
        coordinates = get_line_coordinates(_get_layout_line([
            LayoutPageCoordinates(x=10, y=20, width=30, height=10, page_number=1),
            LayoutPageCoordinates(x=50, y=20, width=20, height=10, page_number=1)
        ]))
        assert coordinates == LayoutPageCoordinates(
            x=10, y=20, width=60, height=10, page_number=1
        )

    def test_should_ignore_tokens_without_coordinates(self):
        line = LayoutLine(tokens=[
            LayoutToken(text='token1'),
            LayoutToken(
                text='token2',
                coordinates=LayoutPageCoordinates(x=10, y=20, width=30, height=10, page_number=1)
            )
        ])
        coordinates = get_line_coordinates(line)
        assert coordinates == LayoutPageCoordinates(
            x=10, y=20, width=30, height=10, page_number=1
        )

    def test_should_return_none_without_any_token_coordinates(self):
        assert get_line_coordinates(LayoutLine(tokens=[LayoutToken(text='token1')])) is None


class TestGetLineJsonDict:
    def test_should_round_the_coordinates(self):
        assert get_line_json_dict(_get_layout_line([
            LayoutPageCoordinates(x=10.123, y=20.456, width=30.5, height=10.0, page_number=2)
        ])) == {'page': 2, 'x': 10.12, 'y': 20.46, 'width': 30.5, 'height': 10.0}

    def test_should_return_empty_dict_for_a_line_without_coordinates(self):
        assert not get_line_json_dict(LayoutLine(tokens=[LayoutToken(text='token1')]))

    def test_should_return_empty_dict_without_a_layout_line(self):
        assert not get_line_json_dict(None)


class TestFormatGeometryRecord:
    def test_should_describe_the_format_and_the_document(self):
        json_dicts = _get_json_dicts(format_geometry_record(
            document_id='document1',
            model_name='segmentation',
            layout_document=LayoutDocument(pages=[]),
            model_data_list_list=[]
        ))
        assert json_dicts[0]['format'] == GEOMETRY_FORMAT
        assert json_dicts[0]['version'] == GEOMETRY_FORMAT_VERSION
        assert json_dicts[0]['document_id'] == 'document1'
        assert json_dicts[0]['model'] == 'segmentation'

    def test_should_describe_the_pages_lines_are_placed_on(self):
        layout_document = LayoutDocument(pages=[
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=1, coordinates=PAGE_COORDINATES_1
            ))
        ])
        json_dicts = _get_json_dicts(format_geometry_record(
            document_id='document1',
            model_name='segmentation',
            layout_document=layout_document,
            model_data_list_list=[]
        ))
        assert json_dicts[0]['pages'] == [{'page': 1, 'width': 595.0, 'height': 842.0}]

    def test_should_write_one_row_per_model_data_in_order(self):
        lines = [
            _get_layout_line([
                LayoutPageCoordinates(x=10, y=y, width=30, height=10, page_number=1)
            ])
            for y in [20, 40]
        ]
        model_data_list_list = [[
            LayoutModelData(data_line='', layout_line=line)
            for line in lines
        ]]
        json_dicts = _get_json_dicts(format_geometry_record(
            document_id='document1',
            model_name='segmentation',
            layout_document=LayoutDocument(pages=[LayoutPage(
                blocks=[LayoutBlock(lines=list(lines))],
                meta=LayoutPageMeta(page_number=1, coordinates=PAGE_COORDINATES_1)
            )]),
            model_data_list_list=model_data_list_list
        ))
        assert json_dicts[0]['line_count'] == 2
        assert [json_dict['y'] for json_dict in json_dicts[1:]] == [20, 40]

    def test_should_keep_the_position_of_a_line_it_could_not_place(self):
        model_data_list_list = [[
            LayoutModelData(
                data_line='',
                layout_line=LayoutLine(tokens=[LayoutToken(text='token1')])
            ),
            LayoutModelData(
                data_line='',
                layout_line=_get_layout_line([
                    LayoutPageCoordinates(x=10, y=40, width=30, height=10, page_number=1)
                ])
            )
        ]]
        json_dicts = _get_json_dicts(format_geometry_record(
            document_id='document1',
            model_name='segmentation',
            layout_document=LayoutDocument(pages=[]),
            model_data_list_list=model_data_list_list
        ))
        assert json_dicts[0]['line_count'] == 2
        assert json_dicts[1] == {}
        assert json_dicts[2]['y'] == 40
