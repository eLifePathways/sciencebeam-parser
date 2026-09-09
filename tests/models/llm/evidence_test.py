import json

import pytest

from sciencebeam_parser.models.llm.config import LlmEngineConfig
from sciencebeam_parser.models.llm.decode import (
    LlmResponseError,
    count_evidence_mismatches,
    decode_evidence_response,
    get_line_numbers,
    get_lines
)
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.model_impl import LlmModelImpl


# line 0: "1 ."   line 1: reference text   line 2: continuation
# line 3: "2 ."   line 4: reference text
TOKENS = [
    '1', '.', 'Fleming', 'PS', ',', 'Koletsi', 'D', 'High', 'quality',
    '2', '.', 'Howick', 'J',
]
LINE_STATUS = [
    'LINESTART', 'LINEEND',
    'LINESTART', 'LINEIN', 'LINEIN', 'LINEIN', 'LINEEND',
    'LINESTART', 'LINEEND',
    'LINESTART', 'LINEEND',
    'LINESTART', 'LINEEND',
]

LINES = get_lines(TOKENS, get_line_numbers(LINE_STATUS))
LINE_STATUS_INDEX = get_feature_column_index('reference_segmenter', 'line_status')

CONFIG = {
    'task': 'reference_segmenter',
    'response_shape': 'evidence',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'evidence-v1',
}


def response(*entries) -> str:
    return json.dumps({'references': [
        {'line': line, 'starts_with': starts_with} for line, starts_with in entries
    ]})


def feature_rows():
    return [
        ['x'] * LINE_STATUS_INDEX + [status] + ['y'] for status in LINE_STATUS
    ]


class FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.prompts: list = []

    def validate_configuration(self) -> None:
        pass

    def get_completion(self, prompt, response_schema):
        assert response_schema['required'] == ['references']
        self.prompts.append(prompt)
        return {'choices': [{'message': {'content': self.content}}]}


def get_model_impl(content: str, **overrides):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config({**CONFIG, **overrides}),
        client=FakeClient(content)
    )


class TestCountEvidenceMismatches:
    def test_should_accept_a_quote_from_the_named_line(self):
        assert count_evidence_mismatches([1], ['Fleming PS Koletsi'], LINES) == 0

    def test_should_accept_a_quote_from_the_line_below_the_named_one(self):
        assert count_evidence_mismatches([0], ['Fleming PS Koletsi'], LINES) == 0

    def test_should_count_a_quote_from_neither_line(self):
        assert count_evidence_mismatches([0], ['Howick J'], LINES) == 1

    def test_should_ignore_punctuation_in_the_quote(self):
        assert count_evidence_mismatches([1], ['Fleming PS, Koletsi'], LINES) == 0

    def test_should_ignore_an_empty_quote(self):
        assert count_evidence_mismatches([1], [''], LINES) == 0


class TestDecodeEvidenceResponse:
    def test_should_label_like_the_lines_shape(self):
        labeled, _ = decode_evidence_response(
            response((0, '1'), (3, '2')), TOKENS, LINE_STATUS
        )
        assert [label for _, label in labeled][:3] == [
            'B-<label>', 'I-<label>', 'B-<reference>'
        ]

    def test_should_snap_a_content_line_onto_its_label_line(self):
        labeled, _ = decode_evidence_response(
            response((1, 'Fleming PS Koletsi'), (4, 'Howick J')), TOKENS, LINE_STATUS
        )
        assert [label for _, label in labeled][:2] == ['B-<label>', 'I-<label>']

    def test_should_return_the_input_tokens_unchanged(self):
        labeled, _ = decode_evidence_response(response((0, '1')), TOKENS, LINE_STATUS)
        assert [token for token, _ in labeled] == TOKENS

    def test_should_report_the_mismatch_count(self):
        _, mismatches = decode_evidence_response(
            response((0, 'Howick J'), (3, '2')), TOKENS, LINE_STATUS
        )
        assert mismatches == 1

    def test_should_raise_for_a_line_out_of_range(self):
        with pytest.raises(LlmResponseError, match='out of range'):
            decode_evidence_response(response((99, 'x')), TOKENS, LINE_STATUS)

    def test_should_raise_for_non_ascending_lines(self):
        with pytest.raises(LlmResponseError, match='ascending'):
            decode_evidence_response(response((3, '2'), (0, '1')), TOKENS, LINE_STATUS)

    def test_should_raise_for_a_malformed_entry(self):
        with pytest.raises(LlmResponseError, match='malformed'):
            decode_evidence_response(
                json.dumps({'references': [{'starts_with': 'x'}]}), TOKENS, LINE_STATUS
            )

    def test_should_raise_for_malformed_json(self):
        with pytest.raises(LlmResponseError, match='not json'):
            decode_evidence_response('{"references": [', TOKENS, LINE_STATUS)


class TestEvidenceShapeInModelImpl:
    def test_should_resolve_the_line_status_column(self):
        assert get_model_impl(response((0, '1'))).line_status_index == LINE_STATUS_INDEX

    def test_should_ask_for_the_line_number_and_the_words(self):
        model_impl = get_model_impl(response((0, '1')))
        model_impl.predict_labels([TOKENS], [feature_rows()])
        prompt = model_impl.client.prompts[0]
        assert 'first three words' in prompt
        assert '0\t1 .' in prompt

    def test_should_warn_rather_than_raise_on_a_mismatch_by_default(self, caplog):
        model_impl = get_model_impl(response((0, 'Howick J')))
        with caplog.at_level('WARNING'):
            assert model_impl.predict_labels([TOKENS], [feature_rows()])
        assert 'quoted words' in caplog.text

    def test_should_raise_on_a_mismatch_when_configured_to(self):
        model_impl = get_model_impl(
            response((0, 'Howick J')), evidence_mismatch_raises=True
        )
        with pytest.raises(LlmResponseError, match='quoted words'):
            model_impl.predict_labels([TOKENS], [feature_rows()])
