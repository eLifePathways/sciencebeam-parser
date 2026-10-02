import json
from typing import List, Sequence

import pytest

from lxml import etree

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
from sciencebeam_parser.models.segmentation.training_data import (
    ROOT_TRAINING_XML_ELEMENT_PATH,
    TRAINING_XML_ELEMENT_PATH_BY_LABEL
)
from sciencebeam_parser.training.lines.record import (
    LINES_FORMAT,
    LINES_FORMAT_VERSION,
    LabelledLine,
    format_lines_record,
    get_coordinates_json_dict,
    get_line_coordinates,
    iter_labelled_lines
)


PAGE_COORDINATES_1 = LayoutPageCoordinates(x=0, y=0, width=595.0, height=842.0, page_number=1)


def _get_layout_line(
    token_coordinates_list: Sequence[LayoutPageCoordinates]
) -> LayoutLine:
    return LayoutLine(tokens=[
        LayoutToken(text=f'token{index}', coordinates=coordinates)
        for index, coordinates in enumerate(token_coordinates_list)
    ])


def _get_model_data_list_list(
    layout_lines: Sequence[LayoutLine]
) -> Sequence[Sequence[LayoutModelData]]:
    return [[
        LayoutModelData(data_line='', layout_line=layout_line)
        for layout_line in layout_lines
    ]]


def _get_labelled_lines(xml_text: str) -> List[LabelledLine]:
    return list(iter_labelled_lines(
        training_tei_root=etree.fromstring(xml_text),
        root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
        training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL
    ))


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
        assert get_line_coordinates(line) == LayoutPageCoordinates(
            x=10, y=20, width=30, height=10, page_number=1
        )

    def test_should_return_none_without_any_token_coordinates(self):
        assert get_line_coordinates(LayoutLine(tokens=[LayoutToken(text='token1')])) is None


class TestGetCoordinatesJsonDict:
    def test_should_round_the_coordinates(self):
        assert get_coordinates_json_dict(_get_layout_line([
            LayoutPageCoordinates(x=10.123, y=20.456, width=30.5, height=10.0, page_number=2)
        ])) == {'page': 2, 'x': 10.12, 'y': 20.46, 'width': 30.5, 'height': 10.0}

    def test_should_return_empty_dict_for_a_line_without_coordinates(self):
        assert not get_coordinates_json_dict(LayoutLine(tokens=[LayoutToken(text='token1')]))


class TestIterLabelledLines:
    def test_should_read_the_label_and_text_of_every_line(self):
        assert _get_labelled_lines(
            '<tei><text><front>Title line<lb/>\n'
            'Author line<lb/>\n'
            '</front><body>First body line<lb/>\n'
            '</body></text></tei>'
        ) == [
            LabelledLine(label='<header>', text='Title line'),
            LabelledLine(label='<header>', text='Author line'),
            LabelledLine(label='<body>', text='First body line')
        ]

    def test_should_label_a_line_outside_any_element_as_other(self):
        assert _get_labelled_lines(
            '<tei><text>Loose line<lb/>\n</text></tei>'
        ) == [LabelledLine(label='<other>', text='Loose line')]

    def test_should_read_the_label_of_an_element_with_attributes(self):
        assert [line.label for line in _get_labelled_lines(
            '<tei><text><div type="annex">Annex line<lb/>\n</div></text></tei>'
        )] == ['<annex>']

    def test_should_name_an_element_it_does_not_know_rather_than_guess(self):
        assert [line.label for line in _get_labelled_lines(
            '<tei><text><note type="other">A line<lb/>\n</note></text></tei>'
        )] == ['text/note[@type="other"]']

    def test_should_keep_an_empty_line_with_the_label_of_its_element(self):
        assert _get_labelled_lines(
            '<tei><text><body><lb/>A line<lb/></body></text></tei>'
        ) == [
            LabelledLine(label='<body>', text=''),
            LabelledLine(label='<body>', text='A line')
        ]

    def test_should_return_nothing_without_the_root_element(self):
        assert not _get_labelled_lines('<tei><teiHeader/></tei>')


class TestFormatLinesRecord:
    def test_should_describe_the_format_the_document_and_its_pages(self):
        json_dicts = _get_json_dicts(format_lines_record(
            document_id='document1',
            model_name='segmentation',
            layout_document=LayoutDocument(pages=[
                LayoutPage(blocks=[], meta=LayoutPageMeta(
                    page_number=1, coordinates=PAGE_COORDINATES_1
                ))
            ]),
            model_data_list_list=[],
            labelled_lines=[]
        ))
        assert json_dicts[0]['format'] == LINES_FORMAT
        assert json_dicts[0]['version'] == LINES_FORMAT_VERSION
        assert json_dicts[0]['document_id'] == 'document1'
        assert json_dicts[0]['model'] == 'segmentation'
        assert json_dicts[0]['pages'] == [{'page': 1, 'width': 595.0, 'height': 842.0}]

    def test_should_write_the_position_label_and_text_of_every_line_in_order(self):
        layout_lines = [
            _get_layout_line([
                LayoutPageCoordinates(x=10, y=y, width=30, height=10, page_number=1)
            ])
            for y in [20, 40]
        ]
        json_dicts = _get_json_dicts(format_lines_record(
            document_id='document1',
            model_name='segmentation',
            layout_document=LayoutDocument(pages=[LayoutPage(
                blocks=[LayoutBlock(lines=list(layout_lines))],
                meta=LayoutPageMeta(page_number=1, coordinates=PAGE_COORDINATES_1)
            )]),
            model_data_list_list=_get_model_data_list_list(layout_lines),
            labelled_lines=[
                LabelledLine(label='<header>', text='Title line'),
                LabelledLine(label='<body>', text='First body line')
            ]
        ))
        assert json_dicts[0]['line_count'] == 2
        assert [
            (json_dict['y'], json_dict['label'], json_dict['text'])
            for json_dict in json_dicts[1:]
        ] == [(20, '<header>', 'Title line'), (40, '<body>', 'First body line')]

    def test_should_keep_the_label_and_text_of_a_line_it_could_not_place(self):
        layout_lines = [LayoutLine(tokens=[LayoutToken(text='token1')])]
        json_dicts = _get_json_dicts(format_lines_record(
            document_id='document1',
            model_name='segmentation',
            layout_document=LayoutDocument(pages=[]),
            model_data_list_list=_get_model_data_list_list(layout_lines),
            labelled_lines=[LabelledLine(label='<body>', text='A line')]
        ))
        assert json_dicts[1] == {'label': '<body>', 'text': 'A line'}

    def test_should_refuse_a_training_tei_with_another_number_of_lines(self):
        layout_lines = [_get_layout_line([
            LayoutPageCoordinates(x=10, y=20, width=30, height=10, page_number=1)
        ])]
        with pytest.raises(ValueError):
            format_lines_record(
                document_id='document1',
                model_name='segmentation',
                layout_document=LayoutDocument(pages=[]),
                model_data_list_list=_get_model_data_list_list(layout_lines),
                labelled_lines=[]
            )
