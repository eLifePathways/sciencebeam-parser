from __future__ import annotations

import socket
from typing import Optional

import pytest

from benchmarks import compute_cost
from benchmarks.compute_cost import (
    aggregate_cost,
    aggregate_latency_ms,
    get_machine_record,
    is_local_parser_url,
    read_busy_cpu_seconds,
    run_manifest_entry,
    start_cpu_measurement,
)


INVOCATION = "2026-10-05T10:00:00.123456+00:00"


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


def _run_entry(
    started_at: str = INVOCATION,
    concurrency: int = 4,
    n_processed: int = 10,
    elapsed_s: float = 100.0,
    cpu_seconds: Optional[float] = 300.0,
    cpu_model: Optional[str] = "AMD EPYC 7763",
) -> dict:
    machine: dict = {"cpu_count": 4}
    if cpu_model is not None:
        machine["cpu_model"] = cpu_model
    if cpu_seconds is not None:
        machine["cpu_seconds"] = cpu_seconds
    return run_manifest_entry(
        run_started_at=started_at, concurrency=concurrency, n_processed=n_processed,
        elapsed_s=elapsed_s, machine=machine,
    )


class TestRunManifestEntry:
    def test_should_be_ignored_by_every_reader_that_selects_documents(self):
        entry = _run_entry()
        assert entry.get("status") is None
        assert entry.get("corpus") is None
        assert entry.get("record_id") is None


class TestAggregateCost:
    def _cost(self, entries: list) -> dict:
        cost = aggregate_cost(entries)
        assert cost is not None
        return cost

    def test_should_return_none_for_an_empty_manifest(self):
        assert aggregate_cost([]) is None

    def test_should_report_latency_alone_where_no_run_recorded_itself(self):
        cost = self._cost([_entry(elapsed_ms=400)])
        assert cost == {"latency_ms": {"n": 1, "median": 400, "p90": 400}}

    def test_should_sum_every_run_that_generated_any_of_the_documents(self):
        cost = self._cost([
            _entry(record_id="doc1", elapsed_ms=100),
            _run_entry(n_processed=10, elapsed_s=100.0, cpu_seconds=300.0),
            _run_entry(started_at="2026-10-06T10:00:00+00:00", n_processed=5,
                       elapsed_s=50.0, cpu_seconds=150.0),
        ])
        assert cost["n_runs"] == 2
        assert cost["n_processed"] == 15
        assert cost["elapsed_s"] == 150.0
        assert cost["cpu_seconds"] == 450.0
        assert cost["cpu_n_processed"] == 15

    def test_should_report_latency_over_every_document_whichever_run_made_it(self):
        cost = self._cost([
            _entry(record_id="stored", elapsed_ms=900),
            _entry(record_id="fresh", elapsed_ms=100),
            _run_entry(n_processed=1),
        ])
        assert cost["latency_ms"]["n"] == 2

    def test_should_cover_cpu_only_over_the_runs_that_measured_it(self):
        cost = self._cost([
            _run_entry(n_processed=10, cpu_seconds=300.0),
            _run_entry(started_at="2026-10-06T10:00:00+00:00", n_processed=5,
                       cpu_seconds=None),
        ])
        assert cost["n_processed"] == 15
        assert cost["cpu_seconds"] == 300.0
        assert cost["cpu_n_processed"] == 10

    def test_should_omit_cpu_where_no_run_measured_it(self):
        cost = self._cost([_run_entry(cpu_seconds=None)])
        assert "cpu_seconds" not in cost
        assert cost["n_processed"] == 10

    def test_should_collect_the_machines_and_concurrencies_it_was_measured_at(self):
        cost = self._cost([
            _run_entry(concurrency=4, cpu_model="AMD EPYC 7763"),
            _run_entry(started_at="2026-10-06T10:00:00+00:00", concurrency=2,
                       cpu_model="Intel Xeon Platinum 8370C"),
        ])
        assert cost["concurrency"] == [2, 4]
        assert cost["machines"] == [
            {"cpu_model": "AMD EPYC 7763", "cpu_count": 4},
            {"cpu_model": "Intel Xeon Platinum 8370C", "cpu_count": 4},
        ]
