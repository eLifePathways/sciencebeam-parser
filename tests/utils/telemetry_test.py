from typing import Any, Dict, List, Tuple

import pytest

from sciencebeam_parser.utils.telemetry import (
    get_configured_endpoint,
    get_trace_id,
    get_tracer,
    is_configured,
    span
)


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


class TestGetTracer:
    @pytest.mark.usefixtures('no_endpoint')
    def test_should_be_none_without_an_endpoint(self):
        assert get_tracer() is None


class TestSpan:
    @pytest.mark.usefixtures('no_endpoint')
    def test_should_be_a_no_op_without_an_endpoint(self):
        with span('process_document', {'sciencebeam.document.name': 'a.pdf'}) as started:
            started.set_attribute('anything', 1)
        assert type(started).__name__ == 'NoOpSpan'


class TestGetTraceId:
    @pytest.mark.usefixtures('no_endpoint')
    def test_should_be_none_for_the_no_op_span(self):
        with span('process_document') as started:
            assert get_trace_id(started) is None

    def test_should_be_none_for_a_span_without_a_trace(self):
        class SpanWithoutATrace:
            def get_span_context(self):
                return None

        assert get_trace_id(SpanWithoutATrace()) is None  # type: ignore[arg-type]

    def test_should_be_the_zero_padded_hex_of_the_trace_id(self):
        class SpanContext:
            trace_id = 0x0af7651916cd43dd8448eb211c80319c

        class TracedSpan(RecordingSpan):
            def get_span_context(self):
                return SpanContext()

        assert get_trace_id(TracedSpan()) == '0af7651916cd43dd8448eb211c80319c'
