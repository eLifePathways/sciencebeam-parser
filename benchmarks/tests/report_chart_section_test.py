from __future__ import annotations

from benchmarks.report_chart_section import render_declared_chart
from benchmarks.report_charts import ChartOutput
from benchmarks.report_grid import ComputeChartConfig


def _summary(latency_ms: float | None) -> dict:
    return {
        "corpora": {"biorxiv": {"n": 5}},
        **({"cost": {"latency_ms": {"median": latency_ms}}} if latency_ms else {}),
    }


def _lines(summaries: list, out_dir=None) -> list:
    return render_declared_chart(
        ComputeChartConfig(metric="latency_median"),
        [label for label, _ in summaries], summaries, {}, [], ["biorxiv"],
        ChartOutput(out_dir=out_dir),
    )


class TestComputeChartWithoutAnOutputDirectory:
    """The comment report is rendered without one, so it must link what was drawn."""

    def test_links_nothing_where_one_variant_recorded_the_figure(self):
        assert not _lines([("A", _summary(100.0)), ("B", _summary(None))])

    def test_links_the_chart_where_two_did(self, tmp_path):
        summaries = [("A", _summary(100.0)), ("B", _summary(200.0))]
        assert _lines(summaries) == _lines(summaries, out_dir=tmp_path)
        assert any(".png" in line for line in _lines(summaries))
