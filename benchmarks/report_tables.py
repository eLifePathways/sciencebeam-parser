"""The score tables a comparison is made of."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from benchmarks.report_grid import GridRow
from benchmarks.report_movement import split_rows


def fmt_f1(f1: Optional[float]) -> str:
    return f"{f1:.3f}" if f1 is not None else "—"


def fmt_delta(delta: Optional[float]) -> str:
    return f"{delta:+.3f}" if delta is not None else "—"


def score_cells(values: Sequence[Optional[float]], primary: int = -1) -> List[str]:
    """Every column in the order it is declared, then a delta per other column.

    Which column the deltas measure against is stated rather than taken from the order,
    so that moving a variant for the sake of reading moves nothing else.
    """
    primary_f1 = values[primary]
    others = [value for index, value in enumerate(values) if index != primary % len(values)]
    deltas = [
        fmt_delta(primary_f1 - f1 if primary_f1 is not None and f1 is not None else None)
        for f1 in others
    ]
    return [fmt_f1(f1) for f1 in values] + deltas


def render_field_table(
    labeled_summaries: List[Tuple[str, dict]],
    rows: Sequence[GridRow],
    field_scoring_types: dict,
    primary: int = -1,
) -> List[str]:
    """Render the comparison table from cells that were already computed.

    A field some document's gold records nothing for gets a second row, over the documents
    that can answer the extraction question. `Docs` says which documents each row covers,
    so neither is read as the other.
    """
    labels = [label for label, _ in labeled_summaries]
    other_labels = [
        label for index, label in enumerate(labels)
        if index != primary % len(labels)
    ]

    col_labels = labels + [f"Δ {label}" for label in other_labels]
    n_cols = 2 * len(other_labels) + 3
    lines = [
        "| Field (method, type) | Docs | " + " | ".join(col_labels) + " |",
        "|" + "|".join(["---"] * n_cols) + "|",
    ]

    for row in rows:
        field_type = field_scoring_types.get(row.field, "string")
        cells = score_cells(row.values, primary)
        lines.append(
            f"| {row.field} ({row.method}, {field_type}) | {row.docs_label} | "
            + " | ".join(cells) + " |"
        )

    return lines


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def render_split_tables(  # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    labeled_summaries: List[Tuple[str, dict]],
    rows: Sequence[GridRow],
    field_scoring_types: dict,
    primary: int = -1,
    baseline: Optional[int] = None,
) -> List[str]:
    """What moved against the baseline, then what did not, collapsed.

    By field rather than by row, so a field whose `exact` moved and whose `levenshtein`
    did not is read in one place. One table where no column answers for movement, and
    one that still states the count where every field moved or none did. What moved is
    expanded and what did not is collapsed, whichever of them is all of it.
    """
    if baseline is None or not rows:
        return render_field_table(labeled_summaries, rows, field_scoring_types, primary)
    moved, unmoved = split_rows(rows, baseline, primary)
    reference = labeled_summaries[baseline][0]
    n_moved = len({row.field for row in moved})
    n_unmoved = len({row.field for row in unmoved})
    lines = [
        f"{n_moved} of {n_moved + n_unmoved} fields moved against **{reference}**.",
        "",
    ]
    if moved:
        lines += render_field_table(
            labeled_summaries, moved, field_scoring_types, primary
        )
    if unmoved:
        lines += [
            *([""] if moved else []),
            "<details>",
            f"<summary>Did not move ({_plural(n_unmoved, 'field')},"
            f" {_plural(len(unmoved), 'row')})</summary>",
            "",
            *render_field_table(
                labeled_summaries, unmoved, field_scoring_types, primary
            ),
            "",
            "</details>",
        ]
    return lines
