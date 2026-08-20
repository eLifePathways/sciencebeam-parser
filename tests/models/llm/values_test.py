import json

import pytest

from sciencebeam_parser.models.llm.decode import LlmResponseError
from sciencebeam_parser.models.llm.tasks import get_citation_labels
from sciencebeam_parser.models.llm.values import (
    decode_values_response,
    get_values_response_schema,
    iter_words
)


LABELS = get_citation_labels()

TOKENS = [
    '1', '.', 'Fleming', 'PS', ',', 'Koletsi', 'D', ':',
    'High', 'quality', 'of', 'the', 'evidence', '.',
    'J', 'Clin', 'Epidemiol', '.', '2016', ';', '78', ':', '34', '-', '42', '.',
]


def response(fields) -> str:
    return json.dumps({'fields': [
        {'label': label, 'text': text} for label, text in fields
    ]})


def get_labels(content: str):
    return [label for _, label in decode_values_response(content, TOKENS, LABELS)]


class TestIterWords:
    def test_should_drop_punctuation(self):
        assert iter_words('Fleming PS , Koletsi D') == ['Fleming', 'PS', 'Koletsi', 'D']

    def test_should_split_a_hyphenated_surname(self):
        assert iter_words('Treble-Barna A') == ['Treble', 'Barna', 'A']


class TestGetValuesResponseSchema:
    def test_should_constrain_the_label_to_the_model_vocabulary(self):
        schema = get_values_response_schema(LABELS)
        enum = schema['properties']['fields']['items']['properties']['label']['enum']
        assert 'pubnum' in enum
        assert 'other' not in enum


class TestDecodeValuesResponse:
    def test_should_label_a_located_value(self):
        labels = get_labels(response([('author', 'Fleming PS , Koletsi D')]))
        assert labels[2:7] == [
            'B-<author>', 'I-<author>', 'I-<author>', 'I-<author>', 'I-<author>'
        ]

    def test_should_leave_unclaimed_tokens_as_other(self):
        labels = get_labels(response([('author', 'Fleming PS , Koletsi D')]))
        assert labels[0] == 'O'
        assert labels[-1] == 'O'

    def test_should_return_the_input_tokens_unchanged(self):
        labeled = decode_values_response(
            response([('author', 'Fleming PS')]), TOKENS, LABELS
        )
        assert [token for token, _ in labeled] == TOKENS

    def test_should_treat_a_page_range_as_two_separate_entities(self):
        labels = get_labels(response([('pages', '34'), ('pages', '42')]))
        assert labels[22] == 'B-<pages>'
        assert labels[23] == 'O'
        assert labels[24] == 'B-<pages>'

    def test_should_match_a_value_whose_punctuation_differs_from_the_tokens(self):
        labels = get_labels(response([('author', 'Fleming PS, Koletsi D')]))
        assert labels[2] == 'B-<author>'

    def test_should_match_a_value_returned_out_of_document_order(self):
        labels = get_labels(response([('date', '2016'), ('author', 'Fleming PS')]))
        assert labels[18] == 'B-<date>'
        assert labels[2] == 'B-<author>'

    def test_should_drop_a_field_whose_tokens_an_earlier_field_claimed(self, caplog):
        with caplog.at_level('WARNING'):
            labels = get_labels(response([('pages', '34'), ('volume', '34')]))
        assert labels[22] == 'B-<pages>'
        assert 'B-<volume>' not in labels
        assert 'already claimed' in caplog.text

    def test_should_keep_the_other_fields_when_one_overlaps(self):
        labels = get_labels(response([
            ('pages', '34'), ('volume', '34'), ('date', '2016')
        ]))
        assert labels[18] == 'B-<date>'

    def test_should_still_raise_when_a_value_is_absent_from_the_source(self):
        with pytest.raises(LlmResponseError, match='not found in source'):
            get_labels(response([('title', 'text that is not in the input at all')]))

    def test_should_match_a_repeated_value_at_its_next_occurrence(self):
        labels = get_labels(response([('pages', '.'), ('title', 'High quality')]))
        assert labels[8] == 'B-<title>'

    def test_should_raise_when_a_value_is_absent_from_the_source(self):
        with pytest.raises(LlmResponseError, match='not found in source'):
            get_labels(response([('title', 'A title that was never in the input')]))

    def test_should_raise_when_a_value_is_paraphrased(self):
        with pytest.raises(LlmResponseError, match='not found in source'):
            get_labels(response([('journal', 'Journal of Clinical Epidemiology')]))

    def test_should_raise_for_an_unknown_label(self):
        with pytest.raises(LlmResponseError, match='unknown label'):
            get_labels(response([('nonsense', 'Fleming')]))

    def test_should_raise_for_malformed_json(self):
        with pytest.raises(LlmResponseError, match='not json'):
            get_labels('{"fields": [')

    def test_should_raise_for_a_truncated_response_without_fields(self):
        with pytest.raises(LlmResponseError, match='no "fields"'):
            get_labels('{}')

    def test_should_raise_for_a_malformed_field_entry(self):
        with pytest.raises(LlmResponseError, match='malformed'):
            get_labels(json.dumps({'fields': [{'label': 'author'}]}))

    def test_should_skip_an_empty_value_rather_than_claiming_tokens(self):
        labels = get_labels(response([('issue', ''), ('date', '2016')]))
        assert labels[18] == 'B-<date>'
        assert 'B-<issue>' not in labels


class TestCharacterLevelMatching:
    def test_should_match_a_value_the_model_joined_differently(self):
        tokens = ['Moreno', '-', 'San', 'Segundo', 'J', ',', 'Casado', 'C']
        labeled = decode_values_response(
            json.dumps({'fields': [
                {'label': 'author', 'text': 'Moreno SanSegundo J , Casado C'}
            ]}), tokens, LABELS
        )
        assert [label for _, label in labeled][0] == 'B-<author>'
        assert [label for _, label in labeled][-1] == 'I-<author>'

    def test_should_match_a_value_the_model_split_differently(self):
        tokens = ['SanSegundo', 'J']
        labeled = decode_values_response(
            json.dumps({'fields': [{'label': 'author', 'text': 'San Segundo J'}]}),
            tokens, LABELS
        )
        assert [label for _, label in labeled] == ['B-<author>', 'I-<author>']

    def test_should_ignore_case_differences(self):
        labeled = decode_values_response(
            json.dumps({'fields': [{'label': 'author', 'text': 'FLEMING PS'}]}),
            TOKENS, LABELS
        )
        assert [label for _, label in labeled][2] == 'B-<author>'

    def test_should_not_locate_a_value_starting_mid_token(self):
        tokens = ['Fleming', 'PS']
        with pytest.raises(LlmResponseError, match='not found in source'):
            decode_values_response(
                json.dumps({'fields': [{'label': 'author', 'text': 'leming'}]}),
                tokens, LABELS
            )
