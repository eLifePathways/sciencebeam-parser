import json
from typing import List, Optional, Sequence, Tuple

import pytest

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutLine,
    LayoutLineMeta,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken
)
from sciencebeam_parser.models.data import LabeledLayoutModelData, LayoutModelData
from sciencebeam_parser.models.header.training_data import (
    HeaderTeiTrainingDataGenerator,
    ROOT_TRAINING_XML_ELEMENT_PATH as HEADER_ROOT_ELEMENT_PATH,
    TRAINING_XML_ELEMENT_PATH_BY_LABEL as HEADER_ELEMENT_PATH_BY_LABEL
)
from sciencebeam_parser.models.segmentation.training_data import (
    SegmentationTeiTrainingDataGenerator,
    TRAINING_XML_ELEMENT_PATH_BY_LABEL as SEGMENTATION_ELEMENT_PATH_BY_LABEL
)
from sciencebeam_parser.training.spans.record import (
    SPANS_FORMAT,
    SPANS_FORMAT_VERSION,
    LabelledSpan,
    check_spans_against_tei,
    format_spans_record,
    get_labelled_content_hash,
    get_model_labels,
    iter_labelled_spans,
    iter_tei_line_texts
)


PAGE_COORDINATES_1 = LayoutPageCoordinates(x=0, y=0, width=595.0, height=842.0, page_number=1)


def _get_model_data(
    text: str,
    label: str,
    line_id: int,
    x: float,
    whitespace: str = ' ',
    placed: bool = True,
    layout_line_by_id: Optional[dict] = None
) -> LabeledLayoutModelData:
    layout_token = LayoutToken(
        text=text,
        whitespace=whitespace,
        line_meta=LayoutLineMeta(line_id=line_id),
        coordinates=LayoutPageCoordinates(
            x=x, y=100.0 + 20 * line_id, width=5.0 * len(text), height=10.0, page_number=1
        ) if placed else None
    )
    layout_line = (layout_line_by_id if layout_line_by_id is not None else {}).setdefault(
        line_id, LayoutLine(tokens=[])
    )
    layout_line.tokens.append(layout_token)
    return LabeledLayoutModelData.from_model_data(
        LayoutModelData(data_line='', layout_token=layout_token, layout_line=layout_line),
        label=label
    )


def _get_header_model_data_list(
    token_label_line_list: Sequence[Tuple[str, str, int]]
) -> List[LabeledLayoutModelData]:
    layout_line_by_id: dict = {}
    model_data_list = []
    for index, (text, label, line_id) in enumerate(token_label_line_list):
        is_last_of_line = (
            index + 1 == len(token_label_line_list)
            or token_label_line_list[index + 1][2] != line_id
        )
        model_data_list.append(_get_model_data(
            text=text,
            label=label,
            line_id=line_id,
            x=50.0 + 30 * index,
            whitespace='\n' if is_last_of_line else ' ',
            layout_line_by_id=layout_line_by_id
        ))
    return model_data_list


def _get_header_spans(
    token_label_line_list: Sequence[Tuple[str, str, int]]
) -> Tuple[List[LabelledSpan], List[str]]:
    generator = HeaderTeiTrainingDataGenerator()
    training_tei_root, trace = generator.get_training_tei_xml_and_trace([
        _get_header_model_data_list(token_label_line_list)
    ])
    spans = list(iter_labelled_spans(
        trace, HEADER_ELEMENT_PATH_BY_LABEL, generator.root_training_xml_element_path
    ))
    tei_line_texts = list(iter_tei_line_texts(
        training_tei_root, generator.root_training_xml_element_path
    ))
    return spans, tei_line_texts


AUTHOR_THEN_AFFILIATION_LINE = [
    ('Poul', 'B-<author>', 1),
    ('Holm', 'I-<author>', 1),
    ('1', 'B-<affiliation>', 1),
    ('Centre', 'I-<affiliation>', 2)
]


class TestIterLabelledSpans:
    def test_should_split_a_line_carrying_two_labels_into_two_spans(self):
        spans, _ = _get_header_spans(AUTHOR_THEN_AFFILIATION_LINE)
        first_line_spans = [span for span in spans if span.line_index == 0]
        assert [span.label for span in first_line_spans] == ['<author>', '<affiliation>']
        assert [span.text.strip() for span in first_line_spans] == ['Poul Holm', '1']

    def test_should_give_each_span_the_box_of_its_own_tokens(self):
        spans, _ = _get_header_spans(AUTHOR_THEN_AFFILIATION_LINE)
        author_span, affiliation_span = [span for span in spans if span.line_index == 0]
        assert author_span.coordinates is not None
        assert affiliation_span.coordinates is not None
        assert affiliation_span.coordinates.x > author_span.coordinates.x

    def test_should_number_the_line_a_span_sits_on(self):
        spans, _ = _get_header_spans(AUTHOR_THEN_AFFILIATION_LINE)
        assert [span.line_index for span in spans] == [0, 0, 1]

    def test_should_attribute_whitespace_between_two_spans_to_the_one_after_it(self):
        spans, tei_line_texts = _get_header_spans(AUTHOR_THEN_AFFILIATION_LINE)
        first_line_spans = [span for span in spans if span.line_index == 0]
        assert first_line_spans[1].text.startswith(' ')
        assert ''.join(span.text for span in first_line_spans) == tei_line_texts[0]

    def test_should_keep_two_sibling_elements_of_one_label_apart(self):
        spans, _ = _get_header_spans([
            ('A', 'B-<affiliation>', 1),
            ('B', 'B-<affiliation>', 1)
        ])
        assert [(span.label, span.text.strip()) for span in spans] == [
            ('<affiliation>', 'A'), ('<affiliation>', 'B')
        ]

    def test_should_label_a_span_in_the_root_element_as_the_parser_reads_it(self):
        """`<note>` and unlabelled text both sit bare under `<front>`, and the
        training TEI parser reads both as other text."""
        assert [span.label for span in _get_header_spans([('Note', 'B-<note>', 1)])[0]] == [
            '<other>'
        ]
        assert [span.label for span in _get_header_spans([('Loose', 'O', 1)])[0]] == [
            '<other>'
        ]

    def test_should_keep_a_span_the_layout_could_not_place(self):
        generator = HeaderTeiTrainingDataGenerator()
        _, trace = generator.get_training_tei_xml_and_trace([[
            _get_model_data('Title', 'B-<title>', line_id=1, x=0, placed=False)
        ]])
        spans = list(iter_labelled_spans(
            trace, HEADER_ELEMENT_PATH_BY_LABEL,
            generator.root_training_xml_element_path
        ))
        assert [(span.label, span.text, span.coordinates) for span in spans] == [
            ('<title>', 'Title', None)
        ]

    def test_should_not_end_a_line_that_no_line_break_ends(self):
        spans, tei_line_texts = _get_header_spans([('Title', 'B-<title>', 1)])
        assert len(tei_line_texts) == 1
        assert [span.line_index for span in spans] == [0]

    def test_should_give_a_segmentation_line_a_single_span(self):
        generator = SegmentationTeiTrainingDataGenerator()
        layout_line_by_id: dict = {}
        model_data_list = [
            LabeledLayoutModelData.from_model_data(
                LayoutModelData(
                    data_line='',
                    layout_line=_get_model_data(
                        text, label, line_id, x=50.0, layout_line_by_id=layout_line_by_id
                    ).layout_line
                ),
                label=label
            )
            for text, label, line_id in [
                ('Title', '<header>', 1), ('Body', '<body>', 2)
            ]
        ]
        _, trace = generator.get_training_tei_xml_and_trace([model_data_list])
        spans = list(iter_labelled_spans(
            trace, SEGMENTATION_ELEMENT_PATH_BY_LABEL,
            generator.root_training_xml_element_path
        ))
        assert [(span.line_index, span.label, span.text.strip()) for span in spans] == [
            (0, '<header>', 'Title'), (1, '<body>', 'Body')
        ]


class TestIterTeiLineTexts:
    def test_should_read_every_root_element_rather_than_only_the_first(self):
        generator = HeaderTeiTrainingDataGenerator()
        training_tei_root, _ = generator.get_training_tei_xml_and_trace([
            _get_header_model_data_list([('First', 'B-<title>', 1)]),
            _get_header_model_data_list([('Second', 'B-<title>', 1)])
        ])
        assert len(training_tei_root.xpath('text/front')) == 2
        tei_line_texts = list(iter_tei_line_texts(
            training_tei_root, generator.root_training_xml_element_path
        ))
        assert [text.strip() for text in tei_line_texts] == ['First', 'Second']

    def test_should_number_lines_across_a_root_boundary(self):
        generator = HeaderTeiTrainingDataGenerator()
        training_tei_root, trace = generator.get_training_tei_xml_and_trace([
            _get_header_model_data_list([('First', 'B-<title>', 1)]),
            _get_header_model_data_list([('Second', 'B-<title>', 1)])
        ])
        spans = list(iter_labelled_spans(
            trace, HEADER_ELEMENT_PATH_BY_LABEL,
            generator.root_training_xml_element_path
        ))
        assert [span.line_index for span in spans] == [0, 1]
        check_spans_against_tei('document1', spans, list(iter_tei_line_texts(
            training_tei_root, generator.root_training_xml_element_path
        )))

    def test_should_return_nothing_without_the_root_element(self):
        assert not list(iter_tei_line_texts(
            etree.fromstring('<tei><teiHeader/></tei>'), ['text', 'front']
        ))


class TestCheckSpansAgainstTei:
    def test_should_accept_spans_reproducing_every_line(self):
        spans, tei_line_texts = _get_header_spans(AUTHOR_THEN_AFFILIATION_LINE)
        check_spans_against_tei('document1', spans, tei_line_texts)

    def test_should_refuse_spans_that_lose_text(self):
        spans, tei_line_texts = _get_header_spans(AUTHOR_THEN_AFFILIATION_LINE)
        with pytest.raises(ValueError, match='line 1'):
            check_spans_against_tei('document1', spans[1:], tei_line_texts)

    def test_should_refuse_spans_covering_another_number_of_lines(self):
        spans, tei_line_texts = _get_header_spans(AUTHOR_THEN_AFFILIATION_LINE)
        with pytest.raises(ValueError, match='lines'):
            check_spans_against_tei('document1', spans, tei_line_texts[:1])


class TestGetModelLabels:
    def test_should_keep_the_first_label_of_two_sharing_an_element_path(self):
        labels = get_model_labels(
            HEADER_ELEMENT_PATH_BY_LABEL, HEADER_ROOT_ELEMENT_PATH
        )
        assert '<affiliation>' in labels
        assert '<institution>' not in labels
        assert '<address>' in labels
        assert '<location>' not in labels

    def test_should_not_list_a_label_the_tei_cannot_tell_from_other_text(self):
        labels = get_model_labels(
            HEADER_ELEMENT_PATH_BY_LABEL, HEADER_ROOT_ELEMENT_PATH
        )
        assert '<note>' not in labels
        assert '<other>' in labels
        assert '<submission>' in labels

    def test_should_keep_the_order_the_table_declares(self):
        labels = get_model_labels(
            HEADER_ELEMENT_PATH_BY_LABEL, HEADER_ROOT_ELEMENT_PATH
        )
        assert labels.index('<title>') < labels.index('<author>')


class TestGetLabelledContentHash:
    def _get_span(
        self,
        line_index: int = 0,
        label: str = '<title>',
        text: str = 'Title',
        coordinates: Optional[LayoutPageCoordinates] = None
    ) -> LabelledSpan:
        return LabelledSpan(
            line_index=line_index, label=label, text=text, coordinates=coordinates
        )

    def test_should_name_the_version_and_the_algorithm_it_used(self):
        assert get_labelled_content_hash([self._get_span()]).startswith('v1:sha256:')

    def test_should_stay_at_the_value_a_corpus_has_already_committed(self):
        """A corpus commits these, so changing how one is taken invalidates them all."""
        assert get_labelled_content_hash([
            LabelledSpan(line_index=0, label='<title>', text='A title', coordinates=None),
            LabelledSpan(line_index=1, label='<author>', text='\nJo Bloggs', coordinates=None)
        ]) == (
            'v1:sha256:'
            'b6208f0c3b9b25f77e2da405101a7e5aeacc8ac5f365ee38a604b6345f73f354'
        )

    def test_should_not_change_where_a_span_only_moved_on_the_page(self):
        assert get_labelled_content_hash([
            self._get_span(coordinates=LayoutPageCoordinates(
                x=10, y=20, width=30, height=10, page_number=1
            ))
        ]) == get_labelled_content_hash([
            self._get_span(coordinates=LayoutPageCoordinates(
                x=99, y=88, width=30, height=10, page_number=2
            ))
        ])

    def test_should_change_where_a_span_was_relabelled(self):
        assert get_labelled_content_hash([
            self._get_span(label='<title>')
        ]) != get_labelled_content_hash([
            self._get_span(label='<author>')
        ])

    def test_should_change_where_the_text_a_label_covers_changed(self):
        assert get_labelled_content_hash([
            self._get_span(text='Title')
        ]) != get_labelled_content_hash([
            self._get_span(text='Titel')
        ])

    def test_should_change_where_a_line_break_moved(self):
        assert get_labelled_content_hash([
            self._get_span(line_index=0, text='Poul'),
            self._get_span(line_index=0, text=' Holm')
        ]) != get_labelled_content_hash([
            self._get_span(line_index=0, text='Poul'),
            self._get_span(line_index=1, text=' Holm')
        ])

    def test_should_change_where_a_label_boundary_moved_inside_a_line(self):
        assert get_labelled_content_hash([
            self._get_span(label='<author>', text='Poul Holm 1')
        ]) != get_labelled_content_hash([
            self._get_span(label='<author>', text='Poul Holm'),
            self._get_span(label='<affiliation>', text=' 1')
        ])

    def test_should_change_where_one_element_became_two_under_the_same_label(self):
        """A span ends where its element does, and two siblings train differently."""
        assert get_labelled_content_hash([
            self._get_span(label='<affiliation>', text='A B')
        ]) != get_labelled_content_hash([
            self._get_span(label='<affiliation>', text='A'),
            self._get_span(label='<affiliation>', text=' B')
        ])


class TestFormatSpansRecord:
    def _get_json_dicts(self, record: str) -> List[dict]:
        return [json.loads(line) for line in record.splitlines()]

    def _get_layout_document(self, page_count: int) -> LayoutDocument:
        return LayoutDocument(pages=[
            LayoutPage(blocks=[], meta=LayoutPageMeta(
                page_number=page_number,
                coordinates=LayoutPageCoordinates(
                    x=0, y=0, width=595.0, height=842.0, page_number=page_number
                )
            ))
            for page_number in range(1, 1 + page_count)
        ])

    def test_should_describe_the_format_the_document_and_its_labels(self):
        json_dicts = self._get_json_dicts(format_spans_record(
            document_id='document1',
            model_name='header',
            layout_document=self._get_layout_document(3),
            spans=[],
            labels=['<title>', '<author>']
        ))
        assert json_dicts[0]['format'] == SPANS_FORMAT
        assert json_dicts[0]['version'] == SPANS_FORMAT_VERSION
        assert json_dicts[0]['document_id'] == 'document1'
        assert json_dicts[0]['model'] == 'header'
        assert json_dicts[0]['labels'] == ['<title>', '<author>']
        assert json_dicts[0]['page_count'] == 3

    def test_should_carry_geometry_only_for_the_pages_a_span_sits_on(self):
        json_dicts = self._get_json_dicts(format_spans_record(
            document_id='document1',
            model_name='header',
            layout_document=self._get_layout_document(3),
            spans=[LabelledSpan(
                line_index=0,
                label='<title>',
                text='Title',
                coordinates=LayoutPageCoordinates(
                    x=10, y=20, width=30, height=10, page_number=2
                )
            )],
            labels=[]
        ))
        assert json_dicts[0]['page_count'] == 3
        assert json_dicts[0]['pages'] == [{'page': 2, 'width': 595.0, 'height': 842.0}]

    def test_should_write_the_line_position_label_and_text_of_every_span(self):
        json_dicts = self._get_json_dicts(format_spans_record(
            document_id='document1',
            model_name='header',
            layout_document=self._get_layout_document(1),
            spans=[
                LabelledSpan(
                    line_index=0, label='<author>', text='Poul Holm',
                    coordinates=LayoutPageCoordinates(
                        x=10.123, y=20.456, width=30.0, height=10.0, page_number=1
                    )
                ),
                LabelledSpan(line_index=0, label='<affiliation>', text=' 1', coordinates=None)
            ],
            labels=[]
        ))
        assert json_dicts[0]['line_count'] == 1
        assert json_dicts[0]['span_count'] == 2
        assert json_dicts[1] == {
            'line': 1, 'page': 1, 'x': 10.12, 'y': 20.46, 'width': 30.0, 'height': 10.0,
            'label': '<author>', 'text': 'Poul Holm'
        }
        assert json_dicts[2] == {'line': 1, 'label': '<affiliation>', 'text': ' 1'}
