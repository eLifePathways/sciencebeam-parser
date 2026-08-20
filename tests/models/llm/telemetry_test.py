import json
from typing import Any, Dict, List, Tuple

import pytest

from sciencebeam_parser.models.llm.config import LlmEngineConfig
from sciencebeam_parser.models.llm.telemetry import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    OPENINFERENCE_INPUT_VALUE,
    OPENINFERENCE_MODEL_NAME,
    OPENINFERENCE_OUTPUT_VALUE,
    OPENINFERENCE_SPAN_KIND,
    OPENINFERENCE_TOKEN_COUNT_COMPLETION,
    get_configured_endpoint,
    get_invocation_parameters,
    is_configured,
    llm_span,
    set_response_attributes
)


CONFIG = LlmEngineConfig.from_model_config({
    'task': 'citation',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'values-v1',
    'provider': 'siliconflow',
    'response_shape': 'values',
})

RESPONSE: Dict[str, Any] = {
    'choices': [{'message': {'content': '{"fields": []}'}, 'finish_reason': 'stop'}],
    'provider': 'SiliconFlow',
    'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'cost': 0.0001},
}


class RecordingSpan:
    def __init__(self):
        self.attributes: List[Tuple[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes.append((key, value))

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.attributes)


@pytest.fixture(name='no_endpoint')
def _no_endpoint(monkeypatch: pytest.MonkeyPatch):
    for name in ('OTEL_EXPORTER_OTLP_ENDPOINT', 'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT'):
        monkeypatch.delenv(name, raising=False)


class TestIsConfigured:
    @pytest.mark.usefixtures('no_endpoint')
    def test_should_be_false_without_an_endpoint(self):
        assert is_configured() is False

    def test_should_be_true_with_an_otlp_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')
        assert is_configured() is True

    @pytest.mark.usefixtures('no_endpoint')
    def test_should_ignore_a_backend_specific_endpoint_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv('PHOENIX_COLLECTOR_ENDPOINT', 'http://localhost:6006')
        assert is_configured() is False


class TestGetConfiguredEndpoint:
    @pytest.mark.usefixtures('no_endpoint')
    def test_should_be_none_without_any_endpoint(self):
        assert get_configured_endpoint() is None

    @pytest.mark.usefixtures('no_endpoint')
    def test_should_prefer_the_traces_endpoint(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://base:4318')
        monkeypatch.setenv(
            'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT', 'http://traces:4318/v1/traces'
        )
        assert get_configured_endpoint() == 'http://traces:4318/v1/traces'


class TestLlmSpan:
    @pytest.mark.usefixtures('no_endpoint')
    def test_should_be_a_no_op_without_an_endpoint(self):
        with llm_span(CONFIG, 'prompt') as span:
            span.set_attribute('anything', 1)
        assert type(span).__name__ == 'NoOpSpan'


class TestGetInvocationParameters:
    def test_should_record_the_parameters_that_change_a_response(self):
        parameters = json.loads(get_invocation_parameters(CONFIG))
        assert parameters['temperature'] == 0.0
        assert parameters['response_shape'] == 'values'
        assert parameters['provider_routing']['zdr'] is True


class TestSetResponseAttributes:
    def test_should_record_token_counts_and_cost(self):
        span = RecordingSpan()
        set_response_attributes(span, RESPONSE, '{"fields": []}')
        attributes = span.as_dict()
        assert attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert attributes[OPENINFERENCE_TOKEN_COUNT_COMPLETION] == 20
        assert attributes['sciencebeam.cost_usd'] == 0.0001

    def test_should_record_the_resolved_provider(self):
        span = RecordingSpan()
        set_response_attributes(span, RESPONSE, '{"fields": []}')
        assert span.as_dict()['sciencebeam.resolved_provider'] == 'SiliconFlow'

    def test_should_record_the_finish_reason(self):
        span = RecordingSpan()
        set_response_attributes(span, RESPONSE, '{"fields": []}')
        assert span.as_dict()[GEN_AI_RESPONSE_FINISH_REASONS] == ['stop']

    def test_should_record_the_content_when_asked(self):
        span = RecordingSpan()
        set_response_attributes(span, RESPONSE, 'the response')
        assert span.as_dict()[OPENINFERENCE_OUTPUT_VALUE] == 'the response'

    def test_should_omit_the_content_when_not_asked(self):
        span = RecordingSpan()
        set_response_attributes(span, RESPONSE, 'the response', record_content=False)
        assert OPENINFERENCE_OUTPUT_VALUE not in span.as_dict()

    def test_should_tolerate_a_response_without_usage(self):
        span = RecordingSpan()
        set_response_attributes(span, {'choices': []}, None)
        assert GEN_AI_USAGE_OUTPUT_TOKENS not in span.as_dict()


class TestSpanAttributeNames:
    def test_should_use_the_otel_genai_conventions(self):
        assert GEN_AI_OPERATION_NAME == 'gen_ai.operation.name'
        assert GEN_AI_REQUEST_MODEL == 'gen_ai.request.model'
        assert GEN_AI_USAGE_OUTPUT_TOKENS == 'gen_ai.usage.output_tokens'

    def test_should_also_use_openinference_names(self):
        assert OPENINFERENCE_SPAN_KIND == 'openinference.span.kind'
        assert OPENINFERENCE_MODEL_NAME == 'llm.model_name'
        assert OPENINFERENCE_INPUT_VALUE == 'input.value'
