import json
from typing import Any, List, Mapping, Optional

import pytest

from sciencebeam_parser.models.llm.config import LlmConfigError, LlmEngineConfig
from sciencebeam_parser.models.llm.decode import LlmResponseError
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
    'prompt_version': 'values-v1',
}

CITATION_TOKENS = ['Fleming', 'PS', ',', 'Koletsi', 'D', ':', 'High', 'quality']


def get_citation_model_impl(content: str):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config(CITATION_CONFIG),
        client=FakeClient(content=content)
    )


class TestLlmModelImplValuesShape:
    def test_should_not_need_a_line_status_column(self):
        model_impl = get_citation_model_impl(json.dumps({'fields': []}))
        assert model_impl.line_status_index == -1

    def test_should_label_from_located_values(self):
        model_impl = get_citation_model_impl(json.dumps({'fields': [
            {'label': 'author', 'text': 'Fleming PS , Koletsi D'}
        ]}))
        result = model_impl.predict_labels([CITATION_TOKENS], [[[]] * len(CITATION_TOKENS)])
        assert [label for _, label in result[0]][:5] == [
            'B-<author>', 'I-<author>', 'I-<author>', 'I-<author>', 'I-<author>'
        ]

    def test_should_return_the_input_tokens_unchanged(self):
        model_impl = get_citation_model_impl(json.dumps({'fields': [
            {'label': 'title', 'text': 'High quality'}
        ]}))
        result = model_impl.predict_labels([CITATION_TOKENS], [[[]] * len(CITATION_TOKENS)])
        assert [token for token, _ in result[0]] == CITATION_TOKENS

    def test_should_raise_rather_than_fall_back_when_a_value_is_not_in_the_source(self):
        model_impl = get_citation_model_impl(json.dumps({'fields': [
            {'label': 'title', 'text': 'a paraphrased title'}
        ]}))
        with pytest.raises(LlmResponseError, match='not found in source'):
            model_impl.predict_labels([CITATION_TOKENS], [[[]] * len(CITATION_TOKENS)])

    def test_should_send_the_reference_text_and_the_conventions(self):
        model_impl = get_citation_model_impl(json.dumps({'fields': []}))
        model_impl.predict_labels([CITATION_TOKENS], [[[]] * len(CITATION_TOKENS)])
        prompt = model_impl.client.prompts[0]
        assert 'Fleming PS , Koletsi D : High quality' in prompt
        assert 'page range is TWO' in prompt
