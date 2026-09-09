import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

from sciencebeam_parser.utils.telemetry import NoOpSpan, SpanLike, get_tracer


LOGGER = logging.getLogger(__name__)

OPERATION_NAME = 'chat'

# OpenTelemetry GenAI semantic conventions.
GEN_AI_OPERATION_NAME = 'gen_ai.operation.name'
GEN_AI_SYSTEM = 'gen_ai.system'
GEN_AI_REQUEST_MODEL = 'gen_ai.request.model'
GEN_AI_REQUEST_TEMPERATURE = 'gen_ai.request.temperature'
GEN_AI_REQUEST_MAX_TOKENS = 'gen_ai.request.max_tokens'
GEN_AI_RESPONSE_MODEL = 'gen_ai.response.model'
GEN_AI_RESPONSE_FINISH_REASONS = 'gen_ai.response.finish_reasons'
GEN_AI_USAGE_INPUT_TOKENS = 'gen_ai.usage.input_tokens'
GEN_AI_USAGE_OUTPUT_TOKENS = 'gen_ai.usage.output_tokens'
GEN_AI_PROMPT = 'gen_ai.prompt'
GEN_AI_COMPLETION = 'gen_ai.completion'

# Emitted in addition, so a backend reading OpenInference rather than the GenAI
# conventions still renders this as an LLM call. Duplicating a handful of
# attributes is cheaper than depending on one backend's reading of the standard.
OPENINFERENCE_SPAN_KIND = 'openinference.span.kind'
OPENINFERENCE_MODEL_NAME = 'llm.model_name'
OPENINFERENCE_PROVIDER = 'llm.provider'
OPENINFERENCE_INVOCATION_PARAMETERS = 'llm.invocation_parameters'
OPENINFERENCE_TOKEN_COUNT_PROMPT = 'llm.token_count.prompt'
OPENINFERENCE_TOKEN_COUNT_COMPLETION = 'llm.token_count.completion'
OPENINFERENCE_INPUT_VALUE = 'input.value'
OPENINFERENCE_OUTPUT_VALUE = 'output.value'


def get_invocation_parameters(config) -> str:
    return json.dumps({
        'temperature': config.temperature,
        'max_tokens': config.max_output_tokens,
        'reasoning': config.reasoning or 'default',
        'response_shape': config.response_shape,
        'provider_routing': config.provider_routing,
    }, sort_keys=True)


@contextmanager
def llm_span(config, prompt: str, record_content: bool = True) -> Iterator[SpanLike]:
    tracer = get_tracer(__name__)
    if tracer is None:
        yield NoOpSpan()
        return
    with tracer.start_as_current_span(f'{OPERATION_NAME} {config.model}') as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, OPERATION_NAME)
        span.set_attribute(GEN_AI_REQUEST_MODEL, config.model)
        span.set_attribute(GEN_AI_REQUEST_TEMPERATURE, config.temperature)
        span.set_attribute(GEN_AI_REQUEST_MAX_TOKENS, config.max_output_tokens)
        span.set_attribute(OPENINFERENCE_SPAN_KIND, 'LLM')
        span.set_attribute(OPENINFERENCE_MODEL_NAME, config.model)
        span.set_attribute(
            OPENINFERENCE_INVOCATION_PARAMETERS, get_invocation_parameters(config)
        )
        span.set_attribute('sciencebeam.task', config.task)
        span.set_attribute('sciencebeam.response_shape', config.response_shape)
        span.set_attribute('sciencebeam.prompt_version', config.prompt_version)
        if config.provider:
            span.set_attribute(GEN_AI_SYSTEM, config.provider)
            span.set_attribute(OPENINFERENCE_PROVIDER, config.provider)
        if record_content:
            span.set_attribute(GEN_AI_PROMPT, prompt)
            span.set_attribute(OPENINFERENCE_INPUT_VALUE, prompt)
        yield span


def set_response_attributes(
    span: SpanLike,
    response_json: Mapping[str, Any],
    content: Optional[str],
    record_content: bool = True
) -> None:
    usage: Dict[str, Any] = dict(response_json.get('usage') or {})
    token_attributes = (
        (
            'prompt_tokens',
            (GEN_AI_USAGE_INPUT_TOKENS, OPENINFERENCE_TOKEN_COUNT_PROMPT)
        ),
        (
            'completion_tokens',
            (GEN_AI_USAGE_OUTPUT_TOKENS, OPENINFERENCE_TOKEN_COUNT_COMPLETION)
        ),
    )
    for key, attributes in token_attributes:
        if usage.get(key) is not None:
            for attribute in attributes:
                span.set_attribute(attribute, usage[key])
    if usage.get('cost') is not None:
        span.set_attribute('sciencebeam.cost_credits', usage['cost'])
    response_model = response_json.get('model')
    if response_model:
        span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
    resolved_provider = response_json.get('provider')
    if resolved_provider:
        span.set_attribute('sciencebeam.resolved_provider', resolved_provider)
    choices = response_json.get('choices') or []
    if choices:
        finish_reason = (
            choices[0].get('finish_reason') or choices[0].get('native_finish_reason')
        )
        if finish_reason:
            span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
            span.set_attribute('sciencebeam.finish_reason', finish_reason)
    if record_content and content:
        span.set_attribute(GEN_AI_COMPLETION, content)
        span.set_attribute(OPENINFERENCE_OUTPUT_VALUE, content)
