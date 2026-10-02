"""Draws a comparison row as a grouped bar chart, one bar per variant per corpus.

The image is written beside the report and linked from it, so a chart renders where the
report is read without the reader opening anything. Drawing is local and offline;
publishing the files somewhere a PR comment can reach is the caller's business.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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

    @property
    def filename(self) -> str:
        return f"{self.field}-{self.method}-{self.scope}.png"

    @property
    def title(self) -> str:
        return f"{self.field} ({self.method}) — f1 by corpus"

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

    _draw_bars(axes, spec)
    _style_axes(axes, spec, n_groups)
    axes.legend(
        loc="upper left", bbox_to_anchor=(0, -0.14), ncol=min(n_series, 4),
        frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY,
    )
    figure.suptitle(spec.title, x=0.012, y=0.98, ha="left", fontsize=11,
                    color=TEXT_PRIMARY, fontweight="medium")
    axes.set_title(spec.caption, loc="left", fontsize=8.5, color=TEXT_SECONDARY, pad=10)

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


def _draw_bars(axes, spec: ChartSpec) -> None:
    n_groups = len(spec.corpora)
    n_series = len(spec.series)
    # A 2px gap at this dpi, so adjacent bars read as separate marks rather than a block.
    slot = 0.6 / n_series
    bar_width = max(slot - 2 / DPI, slot * 0.6)
    label_values = n_series * n_groups <= 21
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
        if not label_values:
            continue
        for x, value in drawn:
            axes.text(
                x, value + 0.015, f"{value:.3f}", ha="center", va="bottom",
                fontsize=6.5, color=TEXT_SECONDARY,
            )


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
    specs: Sequence[ChartSpec], rel_dir: str, prefix: str = "", base_url: str = ""
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
