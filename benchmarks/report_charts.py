"""Draws a comparison row as a grouped bar chart, one bar per variant per corpus.

The image is written beside the report and linked from it, so a chart renders where the
report is read without the reader opening anything. Drawing is local and offline;
publishing the files somewhere a PR comment can reach is the caller's business.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from benchmarks.report_cost import COMPUTE_METRICS
from benchmarks.report_grid import ChartConfig, GridRow, Selection, SelectionError

# Validated as a set for adjacent marks in both colour-vision and normal-vision terms;
# the order is the safety mechanism rather than a preference, so slots are taken in turn
# and never cycled.
SERIES_COLOURS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"

# Enough to stay sharp pasted into a presentation, without making the PR comment slow.
DPI = 200

SCOPE_CAPTIONS = {
    "all": "over all {n} documents",
    "gold": "over the {n} documents whose gold records it",
}


class LinkableChart(Protocol):
    """All the markdown needs of a chart: what it is called and what it says."""

    @property
    def filename(self) -> str: ...

    @property
    def alt_text(self) -> str: ...


@dataclass(frozen=True)
class ChartSpec:
    """One chart: a single row of the comparison, drawn across the corpora."""
    field: str
    method: str
    scope: str
    n_docs: int
    corpora: Tuple[str, ...]
    series: Tuple[str, ...]
    # Per series, per corpus. None where that variant scored nothing there.
    values: Tuple[Tuple[Optional[float], ...], ...]
    # What the comparison called it, where it said.
    title_override: Optional[str] = None

    @property
    def filename(self) -> str:
        return f"{self.field}-{self.method}-{self.scope}.png"

    @property
    def title(self) -> str:
        return self.title_override or f"{self.field} ({self.method}) — f1 by corpus"

    @property
    def caption(self) -> str:
        return SCOPE_CAPTIONS[self.scope].format(n=self.n_docs)

    @property
    def alt_text(self) -> str:
        return f"{self.title}, {self.caption}"


def render_chart(spec: ChartSpec, out_dir: Path, prefix: str = "") -> Path:
    """Write one PNG and return its path. Raises if more variants than the palette holds."""
    if len(spec.series) > len(SERIES_COLOURS):
        raise ValueError(
            f"{len(spec.series)} variants is more than the {len(SERIES_COLOURS)} colours"
            " that stay distinguishable; compare fewer at a time"
        )
    # Agg explicitly: this runs on a CI runner and over SSH, where inferring a backend
    # finds none and fails at import rather than at draw time.
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    n_groups = len(spec.corpora)
    n_series = len(spec.series)
    figure, axes = plt.subplots(figsize=(max(7.0, 1.1 * n_groups + 1.6), 4.0), dpi=DPI)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    labels = _draw_bars(axes, spec)
    _style_axes(axes, spec, n_groups)
    axes.legend(
        loc="upper left", bbox_to_anchor=(0, -0.14), ncol=min(n_series, 4),
        frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY,
    )
    figure.suptitle(spec.title, x=0.012, y=0.98, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="medium")
    axes.set_title(spec.caption, loc="left", fontsize=8.5, color=TEXT_SECONDARY, pad=10)
    _drop_colliding_labels(figure, labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}{spec.filename}"
    figure.savefig(
        path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.22,
        # Without this the PNG carries the time it was drawn, so two identical runs
        # produce different bytes.
        metadata={"Software": None, "Creation Time": None},
    )
    plt.close(figure)
    return path


def _draw_bars(axes, spec: ChartSpec) -> List:
    n_groups = len(spec.corpora)
    n_series = len(spec.series)
    # A 2px gap at this dpi, so adjacent bars read as separate marks rather than a block.
    slot = 0.6 / n_series
    bar_width = max(slot - 2 / DPI, slot * 0.6)
    labels = []
    for index, (label, row) in enumerate(zip(spec.series, spec.values)):
        offsets = (
            position + (index - (n_series - 1) / 2) * slot
            for position in range(n_groups)
        )
        # A variant that scored nothing here leaves a gap; a bar at zero would state a
        # failure to extract, which is a different claim.
        drawn = [(x, value) for x, value in zip(offsets, row) if value is not None]
        axes.bar(
            [x for x, _ in drawn], [value for _, value in drawn],
            width=bar_width, label=label, color=SERIES_COLOURS[index], linewidth=0,
        )
        labels += [
            axes.text(
                x, value + 0.015, f"{value:.3f}", ha="center", va="bottom",
                fontsize=6.5, color=TEXT_SECONDARY,
            )
            for x, value in drawn
        ]
    return labels


def _drop_colliding_labels(figure, labels: List) -> int:
    """All of them or none, measured rather than guessed from a count.

    Two numbers running into each other are worse than no numbers, and keeping only the
    ones that happen to fit would read as a selection rather than as the space available.
    The table beside the chart carries every value either way.
    """
    if not labels:
        return 0
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = sorted(
        (label.get_window_extent(renderer) for label in labels),
        key=lambda box: box.x0,
    )
    crowded = any(
        later.x0 - earlier.x1 < 2
        for earlier, later in zip(boxes, boxes[1:])
        if abs(later.y0 - earlier.y0) < earlier.height
    )
    if not crowded:
        return len(labels)
    for label in labels:
        label.remove()
    return 0


def _style_axes(axes, spec: ChartSpec, n_groups: int) -> None:
    axes.set_ylim(0, 1.06)
    axes.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.set_ylabel("f1", fontsize=8.5, color=TEXT_SECONDARY)
    axes.set_xticks(range(n_groups))
    longest = max((len(corpus) for corpus in spec.corpora), default=0)
    axes.set_xticklabels(
        spec.corpora, fontsize=8, color=TEXT_SECONDARY,
        rotation=20 if longest > 10 else 0,
        ha="right" if longest > 10 else "center",
    )
    axes.tick_params(axis="both", length=0, colors=TEXT_SECONDARY, labelsize=8)
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.grid(axis="x", visible=False)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)


def render_charts(
    specs: Sequence[ChartSpec], out_dir: Path, prefix: str = ""
) -> List[Path]:
    return [render_chart(spec, out_dir, prefix) for spec in specs]


def chart_markdown(
    specs: Sequence[LinkableChart], rel_dir: str, prefix: str = "", base_url: str = ""
) -> List[str]:
    """The section that links the images.

    A relative path by default, which renders in an editor preview and in the
    repository's own view of the report; a base URL replaces it once the files have been
    published somewhere a PR comment can reach.
    """
    if not specs:
        return []
    lines = ["### Charts", ""]
    for spec in specs:
        name = f"{prefix}{spec.filename}"
        src = base_url.rstrip("/") + "/" + name if base_url else f"{rel_dir}/{name}"
        lines += [f"![{spec.alt_text}]({src})", ""]
    return lines


def render_compute_charts(
    specs: Sequence[ComputeChartSpec], out_dir: Path, prefix: str = ""
) -> List[ComputeChartSpec]:
    """The specs that produced an image, so the report links only those."""
    return [
        spec for spec in specs
        if render_compute_chart(spec, out_dir, prefix) is not None
    ]


@dataclass(frozen=True)
class ChartOutput:
    """Where a chart's image is written, and what the report should link it as."""
    out_dir: Optional[Path] = None
    rel_dir: str = "charts"
    prefix: str = ""
    base_url: str = ""


def _declared_charts(
    selection: Selection, overall_rows: Sequence[GridRow]
) -> List[ChartConfig]:
    """What to draw: each chart a comparison file named, plus every row of a field asked
    for by name alone, which is what `--chart` means."""
    declared = list(selection.chart_configs)
    for field in selection.charts:
        for row in overall_rows:
            if row.field != field:
                continue
            if selection.chart_methods is not None and row.method not in selection.chart_methods:
                continue
            declared.append(ChartConfig(field=field, method=row.method, scope=row.scope))
    return declared


def _chart_spec(
    chart: ChartConfig,
    labels: Sequence[str],
    corpus_grids: Dict[str, List[GridRow]],
    overall_rows: Sequence[GridRow],
    common: Sequence[str],
) -> Optional[ChartSpec]:
    """The per-corpus tables' own cells, so a chart cannot state anything its corpus
    section does not. A row a corpus has no cells for contributes nothing rather than a
    zero."""
    corpora = [
        corpus for corpus in common
        if chart.corpora is None or corpus in chart.corpora
    ]
    key = (chart.field, chart.method, chart.scope)
    overall = next(
        (row for row in overall_rows if (row.field, row.method, row.scope) == key), None
    )
    if overall is None or not corpora:
        return None
    per_corpus = [
        next(
            (
                found for found in corpus_grids.get(corpus, [])
                if (found.field, found.method, found.scope) == key
            ),
            None,
        )
        for corpus in corpora
    ]
    return ChartSpec(
        field=chart.field, method=chart.method, scope=chart.scope,
        n_docs=overall.n_docs, corpora=tuple(corpora), series=tuple(labels),
        values=tuple(
            tuple(found.values[index] if found else None for found in per_corpus)
            for index in range(len(labels))
        ),
        title_override=chart.title,
    )


def chart_specs(
    selection: Selection,
    labels: Sequence[str],
    corpus_grids: Dict[str, List[GridRow]],
    overall_rows: Sequence[GridRow],
    common: Sequence[str],
) -> List[ChartSpec]:
    declared = _declared_charts(selection, overall_rows)
    missing = [
        f"{chart.field} ({chart.method}, {chart.scope})"
        for chart in declared
        if not any(
            (row.field, row.method, row.scope) == (chart.field, chart.method, chart.scope)
            for row in overall_rows
        )
    ]
    if missing:
        raise SelectionError(
            "Nothing to chart for " + "; ".join(missing)
            + ". A chart draws a row the tables show."
        )
    specs = [
        _chart_spec(chart, labels, corpus_grids, overall_rows, common)
        for chart in declared
    ]
    return [spec for spec in specs if spec is not None]


@dataclass(frozen=True)
class ComputeChartSpec:
    """What a run spent, one bar per variant.

    A different shape from a score chart: one number per variant rather than one per
    corpus, so the variants go on the axis and there is no second dimension to colour.
    Each keeps the colour it has as a series elsewhere, so the two read as one set.
    """
    metric: str
    series: Tuple[str, ...]
    values: Tuple[Optional[float], ...]
    title_override: Optional[str] = None

    @property
    def filename(self) -> str:
        return f"compute-{self.metric}.png"

    @property
    def axis_label(self) -> str:
        return COMPUTE_METRICS[self.metric][1]

    @property
    def title(self) -> str:
        return self.title_override or COMPUTE_METRICS[self.metric][0]

    @property
    def alt_text(self) -> str:
        return f"{self.title}, {self.axis_label}, one bar per variant"


def _compute_value_format(largest: float) -> str:
    return "{:,.0f}" if largest >= 100 else "{:,.1f}"


def render_compute_chart(
    spec: ComputeChartSpec, out_dir: Path, prefix: str = ""
) -> Optional[Path]:
    """Nothing to draw until two variants recorded the figure.

    One bar is a number with a rectangle around it, and the cost section states it
    already. The figure only became a thing runs record recently, so a comparison of
    older stored predictions regularly has just the one.
    """
    drawn = [
        (label, value)
        for label, value, in zip(spec.series, spec.values) if value is not None
    ]
    if len(drawn) < 2:
        return None
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    figure, axes = plt.subplots(
        figsize=(8.0, max(2.2, 0.52 * len(drawn) + 1.4)), dpi=DPI,
    )
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    # Top to bottom in the order the comparison declares its variants, and each keeps
    # the colour it carries as a series in the score charts.
    positions = list(range(len(drawn)))[::-1]
    colours = [
        SERIES_COLOURS[index % len(SERIES_COLOURS)]
        for index, value in enumerate(spec.values) if value is not None
    ]
    axes.barh(
        positions, [value for _, value in drawn],
        height=0.58, color=colours, linewidth=0,
    )
    largest = max(value for _, value in drawn)
    template = _compute_value_format(largest)
    for position, (_, value) in zip(positions, drawn):
        axes.text(
            value + largest * 0.015, position, template.format(value),
            va="center", ha="left", fontsize=7.5, color=TEXT_SECONDARY,
        )

    _style_compute_axes(axes, spec, positions, [label for label, _ in drawn], largest)
    figure.suptitle(spec.title, x=0.012, y=0.985, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="medium")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}{spec.filename}"
    figure.savefig(
        path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.22,
        metadata={"Software": None, "Creation Time": None},
    )
    plt.close(figure)
    return path


def _style_compute_axes(axes, spec, positions, labels, largest: float) -> None:
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8, color=TEXT_SECONDARY)
    axes.set_xlabel(spec.axis_label, fontsize=8.5, color=TEXT_SECONDARY)
    axes.set_xlim(0, largest * 1.16)
    axes.tick_params(axis="both", length=0, colors=TEXT_SECONDARY, labelsize=8)
    axes.set_axisbelow(True)
    axes.grid(axis="x", color=GRID, linewidth=0.8)
    axes.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)


def compute_chart_specs(
    selection: Selection, labeled_costs: Sequence[Tuple[str, dict]]
) -> List[ComputeChartSpec]:
    from benchmarks.report_cost import compute_metric  # pylint: disable=import-outside-toplevel
    return [
        ComputeChartSpec(
            metric=chart.metric,
            series=tuple(label for label, _ in labeled_costs),
            values=tuple(compute_metric(cost, chart.metric) for _, cost in labeled_costs),
            title_override=chart.title,
        )
        for chart in selection.compute_charts
    ]
