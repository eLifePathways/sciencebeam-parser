"""OpenTelemetry tracing, for any part of the parser rather than one model engine.

Optional throughout: without the `telemetry` extra installed, and without an
OTLP endpoint set, nothing is emitted and every caller behaves identically.
"""
import importlib
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Protocol


LOGGER = logging.getLogger(__name__)

# Standard OTLP configuration, read by the exporter itself. Nothing here names a
# backend; Phoenix is only what happens to listen in development.
OTLP_ENDPOINT_ENV_NAMES = (
    'OTEL_EXPORTER_OTLP_ENDPOINT',
    'OTEL_EXPORTER_OTLP_TRACES_ENDPOINT',
)

SERVICE_NAME = 'sciencebeam-parser'


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


def get_tracer(name: str = __name__):
    """None unless opentelemetry is installed and an OTLP endpoint is set.

    Absent either, callers emit nothing and behave identically — tracing is an
    optional extra, not a dependency of the default install.
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
    return trace.get_tracer(name)


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


def get_trace_id(active_span: SpanLike) -> Optional[str]:
    """The span's trace id, so a log line can name the trace holding its content.

    What a span carries and what the log carries are different halves of the
    same call, and this is what joins them. `None` when tracing is off, when the
    log line has nothing to point at anyway.
    """
    get_span_context = getattr(active_span, 'get_span_context', None)
    if get_span_context is None:
        return None
    span_context = get_span_context()
    trace_id = getattr(span_context, 'trace_id', None)
    if not trace_id:
        return None
    return format(trace_id, '032x')


@contextmanager
def span(
    name: str,
    attributes: Optional[Mapping[str, Any]] = None,
    tracer_name: str = __name__
) -> Iterator[SpanLike]:
    """A current span, or a no-op one when tracing is off.

    Made current rather than merely started, so work underneath it nests without
    every layer having to pass a parent. A span crossing into a thread pool is
    the exception: OpenTelemetry keeps the current span in a context variable,
    which a worker thread does not inherit, so the pool has to carry the context
    over itself.
    """
    tracer = get_tracer(tracer_name)
    if tracer is None:
        yield NoOpSpan()
        return
    with tracer.start_as_current_span(name) as started:
        for key, value in (attributes or {}).items():
            if value is not None:
                started.set_attribute(key, value)
        yield started
