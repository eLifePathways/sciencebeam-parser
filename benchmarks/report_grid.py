from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

SCOPE_ALL = "all"
SCOPE_GOLD = "gold"

GetF1 = Callable[[dict, str, str], Optional[float]]
ScopeFn = Callable[[str], Tuple[int, Optional[int]]]


class SelectionError(ValueError):
    """A row, method or corpus that no summary can answer for."""


@dataclass(frozen=True)
class GridRow:
    """One table row's computed cells, before any of them are formatted.

    `split` is whether the field earned a second row here; it decides how the `Docs`
    cell reads, while `scope` identifies the row across corpora whether it split or not.
    """
    field: str
    method: str
    scope: str
    split: bool
    n_docs: int
    values: Tuple[Optional[float], ...]

    @property
    def docs_label(self) -> str:
        return f"{self.scope} {self.n_docs}" if self.split else str(self.n_docs)


def build_grid(
    labeled_summaries: List[Tuple[str, dict]],
    field_names: Sequence[str],
    field_measures: dict,
    get_f1_fn: GetF1,
    get_gold_f1_fn: Optional[GetF1] = None,
    scope_fn: Optional[ScopeFn] = None,
) -> List[GridRow]:
    """The cells a comparison is made of, one row per field, method and scope.

    Built once and rendered twice, so a chart cannot disagree with the table beside it,
    and so whatever decides which documents a column covers is applied in one place.
    """
    rows: List[GridRow] = []
    for field in field_names:
        n_docs, n_gold = scope_fn(field) if scope_fn else (0, None)
        split = n_gold is not None and get_gold_f1_fn is not None
        for method in field_measures.get(field, []):
            rows.append(GridRow(
                field, method, SCOPE_ALL, split, n_docs,
                tuple(get_f1_fn(summary, field, method) for _, summary in labeled_summaries),
            ))
            if split and get_gold_f1_fn is not None and n_gold is not None:
                rows.append(GridRow(
                    field, method, SCOPE_GOLD, split, n_gold,
                    tuple(
                        get_gold_f1_fn(summary, field, method)
                        for _, summary in labeled_summaries
                    ),
                ))
    return rows


@dataclass(frozen=True)
class ChartConfig:
    """One declared chart: the row it draws, and how it is presented."""
    field: str
    method: str
    scope: str = SCOPE_ALL
    title: Optional[str] = None
    corpora: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class Selection:
    """What the reader asked to see, as opposed to what was scored."""
    fields: Optional[Tuple[str, ...]] = None
    methods: Optional[Tuple[str, ...]] = None
    corpora: Optional[Tuple[str, ...]] = None
    charts: Tuple[str, ...] = ()
    # Which of a charted field's methods get an image. None draws every one it carries.
    chart_methods: Optional[Tuple[str, ...]] = None
    # Per field, the (method, scope) pairs asked for; an empty method means every one.
    # None keeps whatever the summaries carry.
    row_filter: Optional[Dict[str, Tuple[Tuple[str, str], ...]]] = None
    # Per field, the scoring type the comparison says it should have been measured with.
    expected_types: Optional[Dict[str, str]] = None
    # Charts named one at a time, as a comparison file declares them.
    chart_configs: Tuple[ChartConfig, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.fields or self.methods or self.corpora or self.charts
            or self.chart_methods or self.row_filter or self.chart_configs
        )


def _union(labeled_summaries: List[Tuple[str, dict]], key: str) -> List[str]:
    seen: List[str] = []
    for _, summary in labeled_summaries:
        for name in summary.get(key, []):
            if name not in seen:
                seen.append(name)
    return seen


def resolve_fields(
    labeled_summaries: List[Tuple[str, dict]], selection: Selection
) -> List[str]:
    """The fields to render, in the order they were asked for.

    Checked against every summary rather than the primary, so a field only a baseline
    scored can be asked for; what a run that did not score it shows is unchanged.
    """
    available = _union(labeled_summaries, "fields")
    wanted = list(selection.fields or []) + list(selection.charts)
    unknown = [field for field in wanted if field not in available]
    if unknown:
        raise SelectionError(
            f"No summary scored {_quoted(unknown)}. Scored: {_quoted(available)}"
        )
    if selection.fields is None:
        _, primary = labeled_summaries[-1]
        return list(primary.get("fields", []))
    # A chart carries no numbers to quote and the contrast its colours are chosen against
    # is relieved by the table, so a charted field the table leaves out is a mistake
    # rather than a chart on its own.
    unshown = [field for field in selection.charts if field not in selection.fields]
    if unshown:
        raise SelectionError(
            f"Cannot chart {_quoted(unshown)} without showing it; add it to --field"
        )
    return list(selection.fields)


def resolve_measures(
    labeled_summaries: List[Tuple[str, dict]],
    field_names: Sequence[str],
    selection: Selection,
) -> Dict[str, List[str]]:
    """Each field's methods, narrowed to those asked for and kept in the field's order.

    A method no selected field measures is an error rather than a silent narrowing: it
    is almost always a typo, and a second selected method would otherwise hide it.
    """
    _, primary = labeled_summaries[-1]
    primary_measures: dict = primary.get("field_measures") or {}
    if selection.methods is None:
        # The primary's own list, as the report has always used: a baseline measuring
        # something the run under test does not is a difference between the runs rather
        # than a row to add.
        return {field: list(primary_measures.get(field, [])) for field in field_names}

    measures: Dict[str, List[str]] = {}
    for _, summary in labeled_summaries:
        for field, methods in (summary.get("field_measures") or {}).items():
            known = measures.setdefault(field, [])
            known += [method for method in methods if method not in known]

    narrowed = {
        field: [method for method in measures.get(field, []) if method in selection.methods]
        for field in field_names
    }
    measured = {method for field in field_names for method in measures.get(field, [])}
    unknown = [method for method in selection.methods if method not in measured]
    if unknown:
        raise SelectionError(
            f"No selected field is measured by {_quoted(unknown)}."
            f" Measured: {_quoted(sorted(measured))}"
        )
    empty = [field for field in field_names if not narrowed[field]]
    if empty:
        raise SelectionError(
            f"{_quoted(empty)} is not measured by {_quoted(selection.methods)}"
            if len(empty) == 1 else
            f"{_quoted(empty)} are not measured by {_quoted(selection.methods)}"
        )
    return narrowed


def filter_grid_rows(
    rows: Sequence[GridRow],
    row_filter: Optional[Dict[str, Tuple[Tuple[str, str], ...]]],
) -> List[GridRow]:
    """Keep the rows a comparison asked for, in the order the grid built them."""
    if row_filter is None:
        return list(rows)
    return [
        row for row in rows
        if any(
            (not method or method == row.method) and (not scope or scope == row.scope)
            for method, scope in row_filter.get(row.field, ())
        )
    ]


def check_row_filter(
    rows: Sequence[GridRow],
    row_filter: Optional[Dict[str, Tuple[Tuple[str, str], ...]]],
) -> None:
    """A declared row the summaries cannot answer for is an error naming it.

    A scope is the common case: asking for the gold row of a field whose gold answers
    every document gets a row that does not exist, and an empty table would read as a
    score of nothing rather than as a question that cannot be asked.
    """
    present = {(row.field, row.method, row.scope) for row in rows}
    missing = [
        f"{field} ({method or 'any method'}, {scope or 'any scope'})"
        for field, wanted in (row_filter or {}).items()
        for method, scope in wanted
        if not any(
            found_field == field and (not method or method == found_method)
            and (not scope or scope == found_scope)
            for found_field, found_method, found_scope in present
        )
    ]
    if missing:
        raise SelectionError(
            "No summary has " + "; ".join(missing)
            + ". A gold row exists only where some variant produced a value the gold"
            " has none of."
        )


def check_expected_types(
    labeled_summaries: List[Tuple[str, dict]],
    expected_types: Optional[Dict[str, str]],
) -> None:
    """A declared scoring type the summary contradicts is an error stating both.

    A summary gives a field exactly one type, so this asserts what was measured rather
    than selecting among measurements -- and catches a run that re-typed a field, which
    the report otherwise only mentions in prose.
    """
    for field, expected in (expected_types or {}).items():
        for label, summary in labeled_summaries:
            actual = (summary.get("field_scoring_types") or {}).get(field)
            if actual is not None and actual != expected:
                raise SelectionError(
                    f"{label} scored '{field}' as '{actual}', not the '{expected}'"
                    " the comparison declares"
                )


def resolve_corpora(
    labeled_summaries: List[Tuple[str, dict]], selection: Selection
) -> List[str]:
    """The corpora to render, in the order they were asked for."""
    available: List[str] = []
    for _, summary in labeled_summaries:
        for corpus in summary.get("corpora", {}):
            if corpus not in available:
                available.append(corpus)
    if selection.corpora is None:
        _, primary = labeled_summaries[-1]
        return list(primary.get("corpora", {}).keys())
    unknown = [corpus for corpus in selection.corpora if corpus not in available]
    if unknown:
        raise SelectionError(
            f"No summary scored {_quoted(unknown)}. Scored: {_quoted(available)}"
        )
    return list(selection.corpora)


def _quoted(names: Sequence[str]) -> str:
    return ", ".join(f"'{name}'" for name in names)
