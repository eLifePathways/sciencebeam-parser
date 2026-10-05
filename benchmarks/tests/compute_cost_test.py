from __future__ import annotations

import socket
from typing import Optional

import pytest

from benchmarks import compute_cost
from benchmarks.compute_cost import (
    aggregate_latency_ms,
    get_machine_record,
    is_local_parser_url,
    read_busy_cpu_seconds,
    start_cpu_measurement,
)


def _entry(
    corpus: str = "biorxiv",
    record_id: str = "doc1",
    status: str = "ok",
    elapsed_ms: Optional[int] = 1000,
) -> dict:
    entry: dict = {"corpus": corpus, "record_id": record_id, "status": status}
    if elapsed_ms is not None:
        entry["elapsed_ms"] = elapsed_ms
    return entry


class TestReadBusyCpuSeconds:
    def test_should_read_a_positive_increasing_figure(self):
        first = read_busy_cpu_seconds()
        assert first is not None and first > 0
        assert sum(i * i for i in range(2_000_000)) > 0
        second = read_busy_cpu_seconds()
        assert second is not None and second >= first

    def test_should_return_none_where_proc_stat_is_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(compute_cost, "PROC_STAT", tmp_path / "absent")
        assert read_busy_cpu_seconds() is None

    def test_should_not_count_guest_time_twice_or_count_idle(self, tmp_path, monkeypatch):
        clock_ticks = 100
        # user nice system idle iowait irq softirq steal guest guest_nice
        stat = tmp_path / "stat"
        stat.write_text(
            "cpu  1000 200 300 900000 5000 10 20 400 500 100\ncpu0 1 2 3 4 5 6 7 8 9 10\n"
        )
        monkeypatch.setattr(compute_cost, "PROC_STAT", stat)
        assert read_busy_cpu_seconds() == (1000 + 200 + 300 + 10 + 20) / clock_ticks


class TestIsLocalParserUrl:
    @pytest.mark.parametrize("url", [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
        "http://[::1]:8080",
    ])
    def test_should_accept_this_machine(self, url: str):
        assert is_local_parser_url(url) is True

    def test_should_accept_this_machines_own_hostname(self):
        assert is_local_parser_url(f"http://{socket.gethostname()}:8080") is True

    @pytest.mark.parametrize("url", ["http://parser.example:8080", "http://10.0.0.5", ""])
    def test_should_reject_another_host(self, url: str):
        assert is_local_parser_url(url) is False


class TestMachineRecord:
    def test_should_record_no_cpu_figure_for_a_remote_parser(self):
        started = start_cpu_measurement("http://parser.example:8080")
        assert started is None
        record = get_machine_record(started)
        assert "cpu_seconds" not in record
        assert record["cpu_count"] > 0

    def test_should_record_what_the_machine_spent_locally(self):
        started = start_cpu_measurement("http://localhost:8080")
        assert started is not None
        assert sum(i * i for i in range(2_000_000)) > 0
        record = get_machine_record(started)
        assert record["cpu_seconds"] > 0
        assert record["cpu_count"] > 0

    def test_should_describe_the_machine_even_without_a_measurement(self):
        record = get_machine_record(None)
        assert "cpu_seconds" not in record
        assert record["cpu_count"] > 0


class TestAggregateLatencyMs:
    def test_should_return_none_without_any_prediction(self):
        assert aggregate_latency_ms([]) is None
        assert aggregate_latency_ms([_entry(status="error")]) is None

    def test_should_report_the_median_and_p90(self):
        entries = [
            _entry(record_id=f"doc{i}", elapsed_ms=value)
            for i, value in enumerate([100, 200, 300, 400, 500, 600, 700, 800, 900, 10000])
        ]
        assert aggregate_latency_ms(entries) == {"n": 10, "median": 500, "p90": 900}

    def test_should_exclude_documents_without_a_prediction(self):
        entries = [
            _entry(record_id="doc1", elapsed_ms=100),
            _entry(record_id="doc2", status="error", elapsed_ms=60000),
        ]
        assert aggregate_latency_ms(entries) == {"n": 1, "median": 100, "p90": 100}

    def test_should_count_every_attempt_of_a_retried_document(self):
        entries = [
            _entry(record_id="doc1", status="error", elapsed_ms=60000),
            _entry(record_id="doc1", elapsed_ms=100),
            _entry(record_id="doc1", elapsed_ms=300),
        ]
        assert aggregate_latency_ms(entries) == {"n": 2, "median": 100, "p90": 300}

    def test_should_only_cover_the_named_corpora(self):
        entries = [
            _entry(corpus="biorxiv", elapsed_ms=100),
            _entry(corpus="ore", elapsed_ms=900),
        ]
        assert aggregate_latency_ms(entries, ["biorxiv"]) == {
            "n": 1, "median": 100, "p90": 100
        }
