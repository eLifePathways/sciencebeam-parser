"""Turning a comparison's declared charts into images and the links that reach them.

Separate from the report so that the one place that knows about every kind of chart is
not also the place that renders the tables.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from benchmarks.llm_usage import usage_for_corpora
from benchmarks.report_charts import (
    ChartOutput,
    chart_markdown,
    chart_spec,
    compute_chart_specs,
    fields_chart_specs,
    render_charts,
    render_compute_charts,
)
from benchmarks.report_grid import (
    ComputeChartConfig,
    FieldsChartConfig,
    GridRow,
)


def render_declared_chart(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    declared,
    labels: Sequence[str],
    labeled_summaries: List[Tuple[str, dict]],
    corpus_grids: Dict[str, List[GridRow]],
    overall_rows: Sequence[GridRow],
    common: Sequence[str],
    charts: ChartOutput,
) -> List[str]:
    """One declared chart, drawn and linked, or nothing where it has too little to say."""
    if isinstance(declared, ComputeChartConfig):
        specs: Sequence = compute_chart_specs(
            declared,
            [(label, summary.get("cost") or {}) for label, summary in labeled_summaries],
            [
                (label, usage_for_corpora(summary, list(summary.get("corpora", {}))) or {})
                for label, summary in labeled_summaries
            ],
        )
        if charts.out_dir is not None:
            specs = render_compute_charts(specs, charts.out_dir, charts.prefix)
    elif isinstance(declared, FieldsChartConfig):
        # No corpus axis, so no two-corpus floor: the question is which fields a
        # difference reaches rather than where it lives.
        specs = fields_chart_specs(declared, labels, overall_rows)
        if charts.out_dir is not None:
            render_charts(specs, charts.out_dir, charts.prefix)
    else:
        if len(common) < 2:
            return []
        specs = [
            spec for spec in [
                chart_spec(declared, labels, corpus_grids, overall_rows, common)
            ] if spec is not None
        ]
        if charts.out_dir is not None:
            render_charts(specs, charts.out_dir, charts.prefix)
    return chart_markdown(specs, charts.rel_dir, charts.prefix, charts.base_url)[2:]
