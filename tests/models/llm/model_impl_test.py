import json
from typing import Any, List, Mapping, Optional

import pytest

from sciencebeam_parser.models.llm.config import LlmConfigError, LlmEngineConfig
from sciencebeam_parser.models.llm.decode import (
    LlmInputTooLargeError,
    LlmResponseError
)
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.model_impl import LlmModelImpl


CONFIG = {
    'task': 'reference_segmenter',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'lines-v1',
    'provider': 'siliconflow',
}

TOKENS = ['1', '.', 'Fleming', 'PS', 'High', 'quality']
LINE_STATUS = ['LINESTART', 'LINEEND', 'LINESTART', 'LINEIN', 'LINEIN', 'LINEEND']

LINE_STATUS_INDEX = get_feature_column_index('reference_segmenter', 'line_status')


def feature_rows():
    return [
        ['x'] * LINE_STATUS_INDEX + [status] + ['y']
        for status in LINE_STATUS
    ]


class FakeClient:
    """Stands in for the network. No test in this module reaches an endpoint."""
    def __init__(self, content: Optional[str] = None, error: Optional[Exception] = None):
        self.content = content
        self.error = error
        self.prompts: List[str] = []

    def validate_configuration(self) -> None:
        if self.error:
            raise self.error

    def get_completion(self, prompt: str, response_schema: Mapping[str, Any]):
        assert response_schema['type'] == 'object'
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return {'choices': [{'message': {'content': self.content}}], 'provider': 'SiliconFlow'}


def get_model_impl(content: Optional[str] = None, error: Optional[Exception] = None):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config(CONFIG),
        client=FakeClient(content=content, error=error)
    )


class TestLlmModelImpl:
    def test_should_resolve_the_line_status_column_by_name(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        assert model_impl.line_status_index == LINE_STATUS_INDEX

    def test_should_return_labels_for_every_input_token(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        result = model_impl.predict_labels([TOKENS], [feature_rows()])
        assert len(result) == 1
        assert [token for token, _ in result[0]] == TOKENS

    def test_should_include_numbered_lines_and_the_prompt_version_text(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        model_impl.predict_labels([TOKENS], [feature_rows()])
        prompt = model_impl.client.prompts[0]
        assert '0\t1 .' in prompt
        assert '1\tFleming PS High quality' in prompt
        assert 'line number' in prompt

    def test_should_predict_for_each_sequence(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        result = model_impl.predict_labels([TOKENS, TOKENS], [feature_rows(), feature_rows()])
        assert len(result) == 2

    def test_should_raise_rather_than_fall_back_on_an_invalid_response(self):
        model_impl = get_model_impl(json.dumps({'starts': [99]}))
        with pytest.raises(LlmResponseError):
            model_impl.predict_labels([TOKENS], [feature_rows()])

    def test_should_raise_for_an_unsupported_output_format(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        with pytest.raises(NotImplementedError):
            model_impl.predict_labels([TOKENS], [feature_rows()], output_format='json')

    def test_should_reject_an_unsupported_response_shape(self):
        with pytest.raises(LlmConfigError, match='response_shape'):
            LlmModelImpl(LlmEngineConfig.from_model_config({
                **CONFIG, 'response_shape': 'spans'
            }), client=FakeClient())

    def test_preload_should_surface_a_configuration_failure(self):
        model_impl = get_model_impl(error=RuntimeError('endpoint unreachable'))
        with pytest.raises(RuntimeError, match='unreachable'):
            model_impl.preload()


CITATION_CONFIG = {
    'task': 'citation',
    'response_shape': 'values',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'values-v3',
}

CITATION_TOKENS = ['Fleming', 'PS', ',', 'Koletsi', 'D', ':', 'High', 'quality']
SECOND_REFERENCE = ['Rada', 'G', ':', 'What', 'is', 'best']


def batched(*per_reference) -> str:
    return json.dumps({'references': [
        {'index': index, 'fields': [
            {'label': label, 'text': text} for label, text in fields
        ]}
        for index, fields in enumerate(per_reference)
    ]})


def get_citation_model_impl(content: str, **overrides):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config({**CITATION_CONFIG, **overrides}),
        client=FakeClient(content=content)
    )


def no_features(token_lists):
    return [[[]] * len(tokens) for tokens in token_lists]


class TestLlmModelImplValuesShape:
    def test_should_not_need_a_line_status_column(self):
        assert get_citation_model_impl(batched([])).line_status_index == -1

    def test_should_label_from_located_values(self):
        model_impl = get_citation_model_impl(
            batched([('author', 'Fleming PS , Koletsi D')])
        )
        result = model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert [label for _, label in result[0]][:5] == [
            'B-<author>', 'I-<author>', 'I-<author>', 'I-<author>', 'I-<author>'
        ]

    def test_should_return_the_input_tokens_unchanged(self):
        model_impl = get_citation_model_impl(batched([('title', 'High quality')]))
        result = model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert [token for token, _ in result[0]] == CITATION_TOKENS

    def test_should_drop_a_value_that_is_not_in_the_source(self, caplog):
        model_impl = get_citation_model_impl(batched([('title', 'a paraphrased title')]))
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels(
                [CITATION_TOKENS], no_features([CITATION_TOKENS])
            )
        assert [label for _, label in result[0]] == ['O'] * len(CITATION_TOKENS)
        assert 'could not be located' in caplog.text

    def test_should_raise_for_a_dropped_field_when_configured_to(self):
        model_impl = get_citation_model_impl(
            batched([('title', 'a paraphrased title')]), dropped_field_raises=True
        )
        with pytest.raises(LlmResponseError, match='could not be located'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))

    def test_should_send_the_conventions_and_the_numbered_references(self):
        model_impl = get_citation_model_impl(batched([]))
        model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        prompt = model_impl.client.prompts[0]
        assert 'REFERENCE 0\nFleming PS , Koletsi D : High quality' in prompt
        assert 'page range is TWO' in prompt


class TestCitationBatching:
    def test_should_send_several_references_in_one_call(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(
            batched([('author', 'Fleming PS')], [('author', 'Rada G')])
        )
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 1
        assert len(result) == 2
        assert result[1][0] == ('Rada', 'B-<author>')

    def test_should_number_each_reference_in_the_prompt(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(batched([], []))
        model_impl.predict_labels(token_lists, no_features(token_lists))
        prompt = model_impl.client.prompts[0]
        assert 'REFERENCE 0\nFleming PS' in prompt
        assert 'REFERENCE 1\nRada G' in prompt

    def test_should_split_into_calls_at_the_configured_bound(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE, CITATION_TOKENS]
        # no fields, so the same stubbed response is valid for every reference
        model_impl = get_citation_model_impl(batched([]), max_references_per_request=1)
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 3
        assert len(result) == 3

    def test_should_locate_values_within_their_own_reference_only(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(
            batched([('author', 'Rada G')], [('author', 'Rada G')])
        )
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        # reference 0 does not contain "Rada G", so that claim is dropped rather
        # than matched against the other reference
        assert [label for _, label in result[0]] == ['O'] * len(CITATION_TOKENS)
        assert result[1][0] == ('Rada', 'B-<author>')

    def test_should_raise_when_a_reference_has_no_answer(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(batched([('author', 'Fleming PS')]))
        with pytest.raises(LlmResponseError, match=r'no answer for reference\(s\) \[1\]'):
            model_impl.predict_labels(token_lists, no_features(token_lists))

    def test_should_raise_when_a_reference_index_appears_twice(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        content = json.dumps({'references': [
            {'index': 0, 'fields': []}, {'index': 0, 'fields': []}
        ]})
        model_impl = get_citation_model_impl(content)
        with pytest.raises(LlmResponseError, match='appears twice'):
            model_impl.predict_labels(token_lists, no_features(token_lists))

    def test_should_raise_when_a_reference_index_is_out_of_range(self):
        content = json.dumps({'references': [{'index': 5, 'fields': []}]})
        model_impl = get_citation_model_impl(content)
        with pytest.raises(LlmResponseError, match='out of range'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))


BIG_LINE_STATUS = ['LINESTART', 'LINEEND'] * 400


def get_big_input():
    tokens = [str(index) for index in range(len(BIG_LINE_STATUS))]
    features = [
        ['x'] * LINE_STATUS_INDEX + [status] + ['y'] for status in BIG_LINE_STATUS
    ]
    return tokens, features


class TestInputSize:
    def test_should_warn_above_the_warn_threshold(self, caplog):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        tokens, features = get_big_input()
        with caplog.at_level('WARNING'):
            model_impl.predict_labels([tokens], [features])
        assert 'segmentation model' in caplog.text

    def test_should_not_warn_for_an_ordinary_reference_list(self, caplog):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        with caplog.at_level('WARNING'):
            model_impl.predict_labels([TOKENS], [feature_rows()])
        assert 'segmentation model' not in caplog.text

    def test_should_raise_when_a_hard_limit_is_configured_and_exceeded(self):
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({**CONFIG, 'max_input_lines': 10}),
            client=FakeClient(content=json.dumps({'starts': [0]}))
        )
        tokens, features = get_big_input()
        with pytest.raises(LlmInputTooLargeError, match='max_input_lines=10'):
            model_impl.predict_labels([tokens], [features])

    def test_should_not_raise_when_no_hard_limit_is_configured(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        tokens, features = get_big_input()
        assert model_impl.predict_labels([tokens], [features])


class TestCitationReferenceMarker:
    def test_should_not_use_a_bracketed_number_as_the_marker(self):
        model_impl = get_citation_model_impl(batched([]))
        model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        prompt = model_impl.client.prompts[0]
        assert '[0]' not in prompt

    def test_should_tell_the_model_the_marker_is_not_content(self):
        model_impl = get_citation_model_impl(batched([]))
        model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert 'not part of the reference' in model_impl.client.prompts[0]

    def test_should_name_the_reference_and_field_when_a_value_is_dropped(self, caplog):
        model_impl = get_citation_model_impl(batched([('note', '1')]))
        with caplog.at_level('WARNING'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert 'reference index 0' in caplog.text
        assert 'label=note' in caplog.text
        assert 'Fleming' in caplog.text
