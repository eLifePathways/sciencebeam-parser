from __future__ import annotations

from typing import Optional

import pytest

from benchmarks.report import _render_comparison_report
from benchmarks.report_cost import (
    COMPUTE_METRICS,
    compute_metric,
    estimated_cost_components,
)


def _title_summary(f1: float, method: str = "levenshtein") -> dict:
    """One field over one corpus: enough to hang a cost block on."""
    return {
        "fields": ["title"],
        "field_measures": {"title": [method]},
        "field_scoring_types": {"title": "string"},
        "corpora": {"biorxiv": {"n": 10, "aggregated": [{
            "scoring_type": "string",
            "scoring_method": method,
            "summary_scores": {"by-field": {"title": {"scores": {"f1": f1}}}},
        }]}},
    }


def _cost(
    n_predicted: Optional[int] = 60,
    n_attempted: Optional[int] = None,
    elapsed_s: Optional[float] = 600.0,
    n_runs: int = 1,
    concurrency: Optional[list] = None,
    cpu_seconds: Optional[float] = 1800.0,
    cpu_n_predicted: Optional[int] = None,
    machines: Optional[list] = None,
    median: int = 2400,
    p90: int = 9100,
) -> dict:
    cost: dict = {"latency_ms": {"n": 60, "median": median, "p90": p90}}
    if n_predicted is not None:
        cost.update({
            "n_runs": n_runs,
            "n_predicted": n_predicted,
            "n_attempted": n_predicted if n_attempted is None else n_attempted,
            "elapsed_s": elapsed_s,
            "concurrency": [4] if concurrency is None else concurrency,
            "machines": (
                [{"cpu_model": "AMD EPYC 7763", "cpu_count": 4}]
                if machines is None else machines
            ),
        })
    if cpu_seconds is not None:
        cost["cpu_seconds"] = cpu_seconds
        cost["cpu_n_predicted"] = (
            n_predicted if cpu_n_predicted is None else cpu_n_predicted
        )
    return cost


def _summary_with_cost(f1: float = 0.85, cost: Optional[dict] = None) -> dict:
    return {**_title_summary(f1), "cost": _cost() if cost is None else cost}


class TestComputeCostSection:
    def _report(self, *labeled_costs) -> str:
        return _render_comparison_report(
            [(label, _summary_with_cost(cost=cost)) for label, cost in labeled_costs]
        )

    def test_should_be_absent_where_nothing_was_recorded(self):
        assert "Compute cost" not in _render_comparison_report([("SB", _title_summary(0.85))])

    def test_should_be_collapsible(self):
        report = self._report(("SB", _cost()))
        assert "<summary><b>Compute cost</b></summary>" in report
        assert report.count("<details>") == report.count("</details>")

    def test_should_report_throughput_at_the_concurrency_it_was_measured_at(self):
        assert "60 docs in 10m00s at concurrency 4 — 360 docs/hour" in self._report(
            ("SB", _cost())
        )

    def test_should_report_latency_as_a_distribution(self):
        assert "2.4s median latency, 9.1s p90 over 60 docs" in self._report(("SB", _cost()))

    def test_should_report_cpu_per_document_and_the_cores_it_used(self):
        report = self._report(("SB", _cost()))
        assert "30.0 CPU-seconds per document, 3.0 of 4 cores busy" in report
        assert "AMD EPYC 7763, 4 cores" in report

    def test_should_state_how_many_runs_produced_the_predictions(self):
        report = self._report(("SB", _cost(n_runs=3, n_predicted=60, elapsed_s=600.0)))
        assert "60 docs in 10m00s over 3 runs at concurrency 4" in report

    def test_should_report_cpu_over_the_part_of_the_set_that_was_measured(self):
        report = self._report(
            ("SB", _cost(n_predicted=60, cpu_seconds=900.0, cpu_n_predicted=30))
        )
        assert "30.0 CPU-seconds per document over the 30 of 60 measured" in report

    def test_should_report_what_was_attempted_where_some_failed(self):
        report = self._report(("SB", _cost(n_predicted=50, n_attempted=60)))
        assert "50 docs in 10m00s (60 attempted) at concurrency 4 — 300 docs/hour" in report

    def test_should_omit_cpu_where_no_run_measured_it(self):
        report = self._report(("SB", _cost(cpu_seconds=None)))
        assert "CPU-seconds per document" not in report
        assert "360 docs/hour" in report

    def test_should_report_latency_alone_for_predictions_that_came_from_the_store(self):
        report = self._report(
            ("stored", {"latency_ms": {"n": 12, "median": 900, "p90": 2100}})
        )
        assert "900ms median latency" in report
        assert "docs/hour" not in report
        assert "CPU-seconds per document" not in report

    def test_should_warn_where_variants_ran_on_different_cpus(self):
        report = self._report(
            ("GROBID", _cost(machines=[
                {"cpu_model": "Intel Xeon Platinum 8370C", "cpu_count": 4}
            ])),
            ("SB", _cost()),
        )
        assert "CPU (Intel Xeon Platinum 8370C, AMD EPYC 7763)" in report

    def test_should_not_claim_busy_cores_for_a_set_measured_on_two_machines(self):
        report = self._report(
            ("SB", _cost(n_runs=2, machines=[
                {"cpu_model": "AMD EPYC 7763", "cpu_count": 4},
                {"cpu_model": "Intel Xeon Platinum 8370C", "cpu_count": 12},
            ]))
        )
        assert "30.0 CPU-seconds per document" in report
        assert "cores busy" not in report
        assert "AMD EPYC 7763, 4 cores" in report
        assert "Intel Xeon Platinum 8370C, 12 cores" in report

    def test_should_warn_where_one_set_was_itself_measured_on_two_machines(self):
        report = self._report(
            ("SB", _cost(n_runs=2, machines=[
                {"cpu_model": "AMD EPYC 7763", "cpu_count": 4},
                {"cpu_model": "Intel Xeon Platinum 8370C", "cpu_count": 12},
            ]))
        )
        assert "Measured differently" in report
        assert "CPU (AMD EPYC 7763/Intel Xeon Platinum 8370C)" in report

    def test_should_warn_where_a_variant_states_no_machine(self):
        report = self._report(
            ("stored", {"latency_ms": {"n": 12, "median": 900, "p90": 2100}}),
            ("SB", _cost()),
        )
        assert "CPU (unrecorded, AMD EPYC 7763)" in report

    def test_should_not_warn_where_the_measurement_matched(self):
        report = self._report(("GROBID", _cost()), ("SB", _cost()))
        assert "Measured differently" not in report


class TestComputeMetric:
    _COST = {
        "cpu_seconds": 1200.0, "cpu_n_predicted": 60,
        "latency_ms": {"median": 2400, "p90": 9100},
        "n_predicted": 60, "elapsed_s": 1800.0,
    }

    def test_should_divide_cpu_seconds_by_the_documents_it_measured(self):
        assert compute_metric(self._COST, "cpu_seconds_per_doc") == 20.0

    def test_should_report_latency_in_seconds(self):
        assert compute_metric(self._COST, "latency_median") == 2.4
        assert compute_metric(self._COST, "latency_p90") == 9.1

    def test_should_report_throughput_an_hour(self):
        assert compute_metric(self._COST, "docs_per_hour") == 120.0

    def test_should_give_nothing_where_the_run_recorded_nothing(self):
        assert all(
            compute_metric({}, metric) is None for metric in COMPUTE_METRICS
        )

    def test_should_give_nothing_rather_than_zero_for_an_unmeasured_part(self):
        assert compute_metric({"cpu_seconds": 10.0}, "cpu_seconds_per_doc") is None

    def test_should_reject_a_metric_it_does_not_know(self):
        with pytest.raises(ValueError, match="nope"):
            compute_metric(self._COST, "nope")


class TestEstimatedCost:
    _CPU = {"cpu_seconds": 720.0, "cpu_n_predicted": 100}

    def test_should_price_the_cpu_at_the_rate_given(self):
        # 7.2 CPU-seconds a document at $0.03 an hour is $0.00006, or $0.06 a thousand.
        assert compute_metric(
            self._CPU, "estimated_cost_per_1k", cpu_usd_per_hour=0.03
        ) == pytest.approx(0.06)

    def test_should_scale_with_the_rate(self):
        cheap = compute_metric(self._CPU, "estimated_cost_per_1k", cpu_usd_per_hour=0.03)
        dear = compute_metric(self._CPU, "estimated_cost_per_1k", cpu_usd_per_hour=0.06)
        assert cheap is not None and dear == pytest.approx(cheap * 2)

    def test_should_add_what_the_provider_charged(self):
        usage = {"cost_credits": 0.5, "n_with_usage": 100}
        with_llm = compute_metric(
            self._CPU, "estimated_cost_per_1k", usage, cpu_usd_per_hour=0.03
        )
        assert with_llm == pytest.approx(0.06 + 5.0)

    def test_should_price_a_provider_charge_without_any_cpu(self):
        usage = {"cost_credits": 0.5, "n_with_usage": 100}
        assert compute_metric({}, "estimated_cost_per_1k", usage) == pytest.approx(5.0)

    def test_should_divide_each_part_by_what_it_measured(self):
        # The provider charged over 50 documents, not the 100 the CPU covered.
        usage = {"cost_credits": 0.5, "n_with_usage": 50}
        assert compute_metric(
            self._CPU, "estimated_cost_per_1k", usage, cpu_usd_per_hour=0.03
        ) == pytest.approx(0.06 + 10.0)

    def test_should_give_nothing_where_neither_part_is_known(self):
        assert compute_metric({}, "estimated_cost_per_1k") is None


class TestCostComponents:
    def test_should_name_each_part_it_priced(self):
        parts = estimated_cost_components(
            {"cpu_seconds": 720.0, "cpu_n_predicted": 100},
            {"cost_credits": 0.5, "n_with_usage": 100},
        )
        assert sorted(parts) == ["LLM provider", "rented CPU"]

    def test_should_leave_out_a_part_nothing_was_spent_on(self):
        parts = estimated_cost_components(
            {"cpu_seconds": 720.0, "cpu_n_predicted": 100}, {}
        )
        assert list(parts) == ["rented CPU"]

    def test_should_sum_to_the_total(self):
        cost = {"cpu_seconds": 720.0, "cpu_n_predicted": 100}
        usage = {"cost_credits": 0.5, "n_with_usage": 100}
        parts = estimated_cost_components(cost, usage)
        assert sum(parts.values()) == pytest.approx(
            compute_metric(cost, "estimated_cost_per_1k", usage)
        )

    def test_should_be_empty_where_nothing_was_recorded(self):
        assert not estimated_cost_components({}, {})
