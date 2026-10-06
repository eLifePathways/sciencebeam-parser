from __future__ import annotations

from typing import Optional

from benchmarks.report import _render_comparison_report


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
