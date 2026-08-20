import importlib
import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional, Protocol


LOGGER = logging.getLogger(__name__)

# Standard OTLP configuration, read by the exporter itself. Nothing here names a
# backend; Phoenix is only what happens to listen in development.
OTLP_ENDPOINT_ENV_NAMES = (
    'OTEL_EXPORTER_OTLP_ENDPOINT',
    'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT',
)

SERVICE_NAME = 'sciencebeam-parser'

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


class SpanLike(Protocol):
    def set_attribute(self, key: str, value: Any) -> None:
        ...


class NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass


def is_configured() -> bool:
    return any(os.environ.get(name) for name in OTLP_ENDPOINT_ENV_NAMES)


def get_configured_endpoint() -> Optional[str]:
    return (
        os.environ.get('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT')
        or os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
    )


def _import_optional(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _get_tracer():
    """None unless opentelemetry is installed and an OTLP endpoint is set.

    Absent either, the engine emits nothing and behaves identically — tracing is
    an optional extra, not a dependency of the default install.
    """
    if not is_configured():
        return None
    trace = _import_optional('opentelemetry.trace')
    if trace is None:
        LOGGER.info(
            'an otlp endpoint is set but opentelemetry is not installed;'
            ' install the "telemetry" extra to emit spans'
        )
        return None
    _ensure_tracer_provider(trace)
    return trace.get_tracer(__name__)


def _ensure_tracer_provider(trace) -> None:
    current = trace.get_tracer_provider()
    if type(current).__name__ not in ('DefaultTracerProvider', 'ProxyTracerProvider'):
        return
    resources = _import_optional('opentelemetry.sdk.resources')
    sdk_trace = _import_optional('opentelemetry.sdk.trace')
    export = _import_optional('opentelemetry.sdk.trace.export')
    otlp = _import_optional(
        'opentelemetry.exporter.otlp.proto.http.trace_exporter'
    )
    if not all((resources, sdk_trace, export, otlp)):
        LOGGER.info('no opentelemetry sdk or otlp exporter; not configuring a provider')
        return
    provider = sdk_trace.TracerProvider(
        resource=resources.Resource.create({'service.name': SERVICE_NAME})
    )
    provider.add_span_processor(export.BatchSpanProcessor(otlp.OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    LOGGER.info(
        'configured otlp tracing for %r, exporting to %s',
        SERVICE_NAME, get_configured_endpoint()
    )


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
    tracer = _get_tracer()
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
        span.set_attribute('sciencebeam.cost_usd', usage['cost'])
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
