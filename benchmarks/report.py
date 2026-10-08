from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from benchmarks.comparison_config import (
    load_comparison,
    primary_index,
    resolve_variants,
    to_selection,
)
from benchmarks.report_charts import (
    ChartOutput,
    chart_markdown,
    chart_specs,
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
    Selection,
    SelectionError,
    build_grid,
    check_expected_types,
    check_row_filter,
    filter_grid_rows,
    resolve_corpora,
    resolve_fields,
    resolve_measures,
)
from benchmarks.gold_presence import (
    GOLD_PRESENCE_KEY,
    GOLD_PRESENT_AGGREGATED_KEY,
    has_gold,
    is_split_worth_reporting,
    merge_presence,
    produced_row,
)
from benchmarks.llm_usage import usage_for_corpora
from benchmarks.report_cost import render_cost_section
from benchmarks.variant_match import (
    VARIANT_MATCH_KEY,
    concatenation_row,
    merge_variant_matches,
    variant_match_row,
)

LOGGER = logging.getLogger(__name__)


def _get_f1(
    summary: dict,
    corpus: str,
    field: str,
    method: str,
    aggregated_key: str = "aggregated",
) -> Optional[float]:
    aggregated = summary.get("corpora", {}).get(corpus, {}).get(aggregated_key, [])
    field_type = summary.get("field_scoring_types", {}).get(field, "string")
    for entry in aggregated:
        if entry.get("scoring_type") == field_type and entry.get("scoring_method") == method:
            f1 = entry.get("summary_scores", {}).get("by-field", {}).get(
                field, {}
            ).get("scores", {}).get("f1")
            if f1 is not None:
                return float(f1)
    return None


def _get_overall_f1(
    summary: dict, field: str, method: str, corpora: Optional[List[str]] = None
) -> Optional[float]:
    """Doc-count-weighted mean F1, over the named corpora or all of them."""
    total_n = 0
    weighted = 0.0
    wanted = set(corpora) if corpora is not None else None
    for corpus, corpus_data in summary.get("corpora", {}).items():
        if wanted is not None and corpus not in wanted:
            continue
        n = corpus_data.get("n", 0)
        f1 = _get_f1(summary, corpus, field, method)
        if f1 is not None and n > 0:
            total_n += n
            weighted += n * f1
    return weighted / total_n if total_n > 0 else None


def _get_overall_gold_f1(
    summary: dict, field: str, method: str, corpora: List[str]
) -> Optional[float]:
    """Weighted by the documents it covers, which are the gold ones.

    Weighting it by every document would give a corpus full weight for a figure computed
    from a few of its documents -- `scielo_br` records five acknowledgements in forty. A
    corpus recording none drops out, which is why this figure does not carry the
    structural zeros the combined one does.
    """
    weighted = 0.0
    total = 0
    for corpus in corpora:
        presence = _gold_presence(summary, corpus, field)
        f1 = _get_f1(summary, corpus, field, method, GOLD_PRESENT_AGGREGATED_KEY)
        if not presence or f1 is None or not presence["n_gold"]:
            continue
        weighted += presence["n_gold"] * f1
        total += presence["n_gold"]
    return weighted / total if total else None


def _fmt_f1(f1: Optional[float]) -> str:
    return f"{f1:.3f}" if f1 is not None else "—"


def _fmt_delta(delta: Optional[float]) -> str:
    return f"{delta:+.3f}" if delta is not None else "—"


def _gold_presence(summary: dict, corpus: str, field: str) -> Optional[dict]:
    """None for a summary written before the split, which then reads as unknown."""
    return summary.get("corpora", {}).get(corpus, {}).get(GOLD_PRESENCE_KEY, {}).get(field)


def _corpus_f1_getter(corpus: str) -> Callable[[dict, str, str], Optional[float]]:
    """Dashes a field the corpus records nothing for, where an f1 of 0.000 would read as a
    failure to extract. Only here: the Overall row means what `_get_f1` returns, and
    dropping a corpus from that mean would move a published figure."""
    def get_f1(summary: dict, field: str, method: str) -> Optional[float]:
        if not has_gold(_gold_presence(summary, corpus, field)):
            return None
        return _get_f1(summary, corpus, field, method)
    return get_f1


def _fmt_count(value: Optional[float]) -> str:
    return f"{round(value):,}" if value else "—"


def _fmt_credits(value: Optional[float], places: int = 4) -> str:
    return f"{value:.{places}f}" if value is not None else "—"


def _all_calls(usage: Optional[Dict[str, Any]]) -> int:
    """Every completion the engine made, replayed ones included. The token and
    credit figures beside it are what was *spent*, which excludes them."""
    if not usage:
        return 0
    return (usage.get("calls") or 0) + ((usage.get("replayed") or {}).get("calls") or 0)


def _usage_bullets(usage: Dict[str, Any]) -> List[str]:
    """Calls, tokens, credits and tasks, a line each.

    Absent figures are left out rather than dashed: most of a line would
    otherwise be dashes for a backend that reports only calls.
    """
    recorded = usage.get("n_with_usage") or 0
    attempted = usage.get("n_attempted") or 0
    bullets = [
        f"{_all_calls(usage):,} calls"
        + (f" over {recorded} of {attempted} docs" if recorded else "")
    ]

    tokens = [
        f"{usage[key]:,} tokens {name}"
        for key, name in (("input_tokens", "in"), ("output_tokens", "out"))
        if usage.get(key)
    ]
    if tokens:
        bullets.append(", ".join(tokens))
    if usage.get("cost_credits") is not None:
        bullets.append(f"{usage['cost_credits']:.4f} credits")

    by_task = usage.get("by_task") or {}
    if len(by_task) > 1:
        for task, entry in sorted(by_task.items()):
            line = f"{task}: {entry.get('calls', 0):,} calls"
            if entry.get("output_tokens"):
                line += f", {entry['output_tokens']:,} tokens out"
            bullets.append(line)
    return bullets


def _render_usage_lines(labeled_usage: List[Tuple[str, Optional[dict]]]) -> List[str]:
    """A block per variant that spent something.

    Only those: a row per variant made most of the old table dashes, since the
    CRF tools spend nothing.
    """
    lines: List[str] = []
    for label, usage in labeled_usage:
        if not usage or not _all_calls(usage):
            continue
        if lines:
            lines.append("")
        lines.append(f"**{label}**")
        lines += [f"* {bullet}" for bullet in _usage_bullets(usage)]
    return lines


def _cached_input_note(labeled_usage: List[Tuple[str, Optional[dict]]]) -> str:
    """Only where it happened, since a cost difference it explains is otherwise
    attributed to the variant."""
    cached = sum((usage or {}).get("cached_input_tokens") or 0 for _, usage in labeled_usage)
    if not cached:
        return ""
    return (
        f" Input tokens include {cached:,} served from the provider's own prefix"
        " cache, which is where a credit difference over the same input comes from."
    )


def _replayed_note(labeled_usage: List[Tuple[str, Optional[dict]]]) -> str:
    """Only where it happened, so a cold report is byte-for-byte unchanged.

    The credits named are what those responses cost when they were generated, not
    a forecast of a cold run: prices and providers move, and generation is not
    reproducible, so a fresh run would not make exactly the same calls.
    """
    replayed = [(usage or {}).get("replayed") or {} for _, usage in labeled_usage]
    replayed_calls = sum(entry.get("calls") or 0 for entry in replayed)
    if not replayed_calls:
        return ""
    total_calls = sum(_all_calls(usage) for _, usage in labeled_usage)
    costs = [
        entry["cost_credits"] for entry in replayed
        if entry.get("cost_credits") is not None
    ]
    when_generated = (
        f" Those responses cost {sum(costs):.4f} credits when they were generated,"
        " at whatever prices and providers applied then, which is a record of one"
        " cold run rather than a forecast of another."
        if costs else ""
    )
    return (
        f" Of {total_calls:,} calls, {replayed_calls:,} were replayed from the"
        " on-disk response cache and spent nothing, so the tokens and credits here"
        f" are what these runs spent rather than what a cold run would cost.{when_generated}"
    )


def _render_usage_section(
    labeled_summaries: List[Tuple[str, dict]],
    corpora: List[str],
    note: str,
) -> List[str]:
    """Empty unless some variant recorded usage, so a CRF-only report is unchanged."""
    labeled_usage = [
        (label, usage_for_corpora(summary, corpora))
        for label, summary in labeled_summaries
    ]
    if not any(_all_calls(usage) for _, usage in labeled_usage):
        return []
    return [
        note + _cached_input_note(labeled_usage) + _replayed_note(labeled_usage),
        "",
        *_render_usage_lines(labeled_usage),
    ]


def _score_cells(values: Sequence[Optional[float]], primary: int = -1) -> List[str]:
    """Every column in the order it is declared, then a delta per other column.

    Which column the deltas measure against is stated rather than taken from the order,
    so that moving a variant for the sake of reading moves nothing else.
    """
    primary_f1 = values[primary]
    others = [value for index, value in enumerate(values) if index != primary % len(values)]
    deltas = [
        _fmt_delta(primary_f1 - f1 if primary_f1 is not None and f1 is not None else None)
        for f1 in others
    ]
    return [_fmt_f1(f1) for f1 in values] + deltas


def _render_field_table(
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
    n_cols = 2 * len(other_labels) + 4
    lines = [
        "| Field (method) | Type | Docs | " + " | ".join(col_labels) + " |",
        "|" + "|".join(["---"] * n_cols) + "|",
    ]

    for row in rows:
        field_type = field_scoring_types.get(row.field, "string")
        cells = _score_cells(row.values, primary)
        lines.append(
            f"| {row.field} ({row.method}) | {field_type} | {row.docs_label} | "
            + " | ".join(cells) + " |"
        )

    return lines


def _gold_totals(
    summary: dict, corpora: List[str], field: str
) -> Optional[Tuple[int, int]]:
    """This summary's gold and scored document counts for a field, or None where it
    predates the split."""
    presences = [_gold_presence(summary, corpus, field) for corpus in corpora]
    known = [presence for presence in presences if presence]
    if not known:
        return None
    return sum(p["n_gold"] for p in known), sum(p["n"] for p in known)


def _scope_getter(
    labeled_summaries: List[Tuple[str, dict]], corpora: List[str]
) -> Callable[[str], Tuple[int, Optional[int]]]:
    """How many documents a field's rows cover: every scored one, and the gold ones where
    the field earns a second row."""
    primary = labeled_summaries[-1][1]

    def scope(field: str) -> Tuple[int, Optional[int]]:
        n_docs = sum(
            primary.get("corpora", {}).get(corpus, {}).get("n", 0) for corpus in corpora
        )
        presences = [
            _gold_presence(summary, corpus, field)
            for _, summary in labeled_summaries for corpus in corpora
        ]
        if not is_split_worth_reporting(presences):
            return n_docs, None
        for _, summary in reversed(labeled_summaries):
            totals = _gold_totals(summary, corpora, field)
            # A second row over no documents would only repeat the first one's dashes.
            if totals and totals[0]:
                return n_docs, totals[0]
        return n_docs, None

    return scope


def _unequal_gold_note(
    labeled_summaries: List[Tuple[str, dict]], corpora: List[str], fields: List[str]
) -> List[str]:
    """Call out paired rows whose columns were scored over different documents.

    The `Docs` cell states one denominator for the row, so where the runs covered
    different documents it belongs to one column only, and each column's figure is over
    its own gold documents rather than over the stated ones.
    """
    for field in fields:
        counts = [
            (label, _gold_totals(summary, corpora, field))
            for label, summary in labeled_summaries
        ]
        known = [(label, totals) for label, totals in counts if totals]
        if len({totals for _, totals in known}) <= 1:
            continue
        listed = ", ".join(f"{label} {gold}/{n}" for label, (gold, n) in known)
        return [
            f"> ⚠️ **Unequal gold document sets** ({listed}). The `Docs` cell states one "
            "of them; each column's figure is over its own gold documents.",
            "",
        ]
    return []


def _variant_match(summary: dict, corpus: str, field: str) -> Optional[dict]:
    return summary.get("corpora", {}).get(corpus, {}).get(VARIANT_MATCH_KEY, {}).get(field)


def _render_variant_match_section(
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    corpora: List[str],
) -> List[str]:
    """How often each run was credited a language other than the article's own, and how
    often it returned several languages as one value.

    A run that changed which language it reads moves these and not the score, so they
    belong together. Empty where no gold carries a field in more than one language, and so
    where a summary was written before the counts existed.
    """
    labels = [label for label, _ in labeled_summaries]
    merged = {
        field: [
            merge_variant_matches(
                _variant_match(summary, corpus, field) for corpus in corpora
            )
            for _, summary in labeled_summaries
        ]
        for field in field_names
    }
    return _render_counts_section(
        "Credited a translation rather than the article's own language",
        labels, [variant_match_row(field, merged[field]) for field in field_names],
    ) + _render_counts_section(
        "Returned several languages as a single value",
        labels, [concatenation_row(field, merged[field]) for field in field_names],
    )


def _render_counts_section(
    summary_text: str, labels: List[str], rows: List[Optional[List[str]]]
) -> List[str]:
    present = [row for row in rows if row]
    if not present:
        return []
    return [
        "<details>",
        f"<summary>{summary_text} ({len(present)} fields)</summary>",
        "",
        "| Field | " + " | ".join(labels) + " |",
        "|" + "|".join(["---"] * (1 + len(labels))) + "|",
        *["| " + " | ".join(row) + " |" for row in present],
        "",
        "</details>",
        "",
    ]


def _render_produced_section(
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    corpora: List[str],
) -> List[str]:
    """The documents whose gold records no value for a field, and what each variant
    produced on them, over one corpus or over several.

    Collapsed, and not a score: it carries no delta, because nothing in the PDF says
    whether a publisher recorded the field. Empty unless some field earns it, so a report
    over corpora that record everything is unchanged, as is one against a summary written
    before the split.
    """
    labels = [label for label, _ in labeled_summaries]
    rows = []
    for field in field_names:
        presences = [
            _gold_presence(summary, corpus, field)
            for _, summary in labeled_summaries for corpus in corpora
        ]
        if not is_split_worth_reporting(presences):
            continue
        row = produced_row(field, [
            merge_presence(_gold_presence(summary, corpus, field) for corpus in corpora)
            for _, summary in labeled_summaries
        ])
        if row:
            rows.append(row)
    if not rows:
        return []
    return [
        "<details>",
        "<summary>Produced where the gold records nothing"
        f" ({len(rows)} fields)</summary>",
        "",
        "| Field | No gold | " + " | ".join(labels) + " |",
        "|" + "|".join(["---"] * (2 + len(labels))) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
        "",
        "</details>",
        "",
    ]


def _coverage_lines(
    labeled_run_records: List[Tuple[str, Optional[dict]]]
) -> List[str]:
    """Per column, what it had to retry and what it never got.

    Neither is visible in a score: a document without a prediction leaves the
    denominator rather than scoring zero, and a retried one scores like any other.
    This is what says a column is short because a run failed rather than because it
    covers less, and it is why an unequal comparison is unequal.
    """
    stated = []
    for label, run_record in labeled_run_records:
        if not run_record:
            continue
        recovered = run_record.get("n_recovered") or 0
        errors = run_record.get("n_errors") or 0
        if not recovered and not errors:
            continue
        parts = []
        if recovered:
            parts.append(f"{recovered} recovered on retry")
        if errors:
            parts.append(f"{errors} without a prediction, and so not scored")
        stated.append(f"> **{label}**: {', '.join(parts)}.")
    if not stated:
        return []
    return [*stated, ""]


def _unequal_docs_note(counts_by_label: List[Tuple[str, int]]) -> List[str]:
    """Call out a comparison whose columns do not cover the same documents.

    The counts are printed either way, but they are easy to read past, and a delta
    between columns of unequal length reflects which documents each run covered as
    much as how it behaved. A baseline that was stored at a smaller mode, or that
    predates a corpus, is short by construction rather than by failing.
    """
    if len({n for _, n in counts_by_label}) <= 1:
        return []
    listed = ", ".join(f"{label} {n}" for label, n in counts_by_label)
    return [
        f"> ⚠️ **Unequal document sets** ({listed}). Deltas below reflect which "
        f"documents each run covered as well as how it performed.",
        "",
    ]


def _common_corpora(
    labeled_summaries: List[Tuple[str, dict]], corpora: List[str]
) -> List[str]:
    """Corpora every run scored at least one document of, in the given order."""
    return [
        corpus for corpus in corpora
        if all(
            s.get("corpora", {}).get(corpus, {}).get("n", 0) > 0
            for _, s in labeled_summaries
        )
    ]


def _corpus_grid(
    corpus: str,
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
) -> List[GridRow]:
    return build_grid(
        labeled_summaries, field_names, field_measures,
        _corpus_f1_getter(corpus),
        lambda s, f, m: _get_f1(s, corpus, f, m, GOLD_PRESENT_AGGREGATED_KEY),
        _scope_getter(labeled_summaries, [corpus]),
    )


def _render_corpus_section(  # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    corpus: str,
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    rows: Sequence[GridRow],
    field_scoring_types: dict,
    primary: int = -1,
) -> List[str]:
    counts_by_label = [
        (label, s.get("corpora", {}).get(corpus, {}).get("n", 0))
        for label, s in labeled_summaries
    ]
    counts = " | ".join(f"**{label}**: {n} docs" for label, n in counts_by_label)
    lines = [counts, ""] + _unequal_docs_note(counts_by_label)
    lines += _unequal_gold_note(labeled_summaries, [corpus], field_names)
    lines.extend(_render_field_table(
        labeled_summaries, rows, field_scoring_types, primary
    ))
    produced = _render_produced_section(labeled_summaries, field_names, [corpus])
    if produced:
        lines += ["", *produced[:-1]]
    variants = _render_variant_match_section(labeled_summaries, field_names, [corpus])
    if variants:
        lines += ["", *variants[:-1]]
    usage_lines = _render_usage_section(
        labeled_summaries, [corpus],
        "**LLM usage**, over every document attempted in this corpus.",
    )
    if usage_lines:
        lines += ["", *usage_lines]
    return lines


def _overall_grid(
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
    common: List[str],
) -> List[GridRow]:
    return build_grid(
        labeled_summaries, field_names, field_measures,
        lambda s, f, m: _get_overall_f1(s, f, m, common),
        lambda s, f, m: _get_overall_gold_f1(s, f, m, common),
        _scope_getter(labeled_summaries, common),
    )


def _render_overall_section(  # pylint: disable=too-many-locals,too-many-arguments
    # pylint: disable=too-many-positional-arguments
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    rows: Sequence[GridRow],
    field_scoring_types: dict,
    corpora: List[str],
    common: List[str],
    primary: int = -1,
) -> List[str]:
    _, primary_summary = labeled_summaries[-1]
    omitted = [corpus for corpus in corpora if corpus not in common]
    n_total = sum(
        primary_summary.get("corpora", {}).get(c, {}).get("n", 0) for c in common
    )
    counts_by_label = [
        (label, sum(s.get("corpora", {}).get(c, {}).get("n", 0) for c in common))
        for label, s in labeled_summaries
    ]
    total_counts = " | ".join(f"**{label}**: {n} docs" for label, n in counts_by_label)
    lines = [
        f"### Overall ({n_total} docs across {len(common)} corpora)",
        "",
        total_counts,
        "",
    ]
    if omitted:
        lines += [
            f"> Excludes {', '.join(omitted)}, which not every run scored — see the "
            f"per-corpus sections below.",
            "",
        ]
    lines += _unequal_docs_note(counts_by_label)
    lines += _unequal_gold_note(labeled_summaries, common, field_names)
    lines.extend(_render_field_table(
        labeled_summaries, rows, field_scoring_types, primary
    ))
    produced = _render_produced_section(labeled_summaries, field_names, common)
    if produced:
        lines += ["", *produced[:-1]]
    variants = _render_variant_match_section(labeled_summaries, field_names, common)
    if variants:
        lines += ["", *variants[:-1]]
    return lines


def _field_definition(summary: dict, field: str) -> Tuple[str, List[str]]:
    return (
        summary.get("field_scoring_types", {}).get(field, "string"),
        summary.get("field_sources", {}).get(field, [field, field]),
    )


def _differently_scored_note(
    labeled_summaries: List[Tuple[str, dict]], field_names: List[str]
) -> List[str]:
    """Fields the runs did not measure the same way.

    A delta between two runs that scored a field differently is a difference between the
    measures as much as between the runs, and nothing else in the table says so: the Type
    column states the primary run's.
    """
    differing = []
    for field in field_names:
        scored_by = [
            summary for _, summary in labeled_summaries
            if field in summary.get("fields", [])
        ]
        definitions = {
            (scoring_type, tuple(sources))
            for scoring_type, sources in (
                _field_definition(summary, field) for summary in scored_by
            )
        }
        if len(definitions) > 1:
            differing.append(field)
    if not differing:
        return []
    return [
        "> ⚠️ **Scored differently between runs** (" + ", ".join(f"`{f}`" for f in differing)
        + "). Their deltas compare the measures as well as the runs.",
        "",
    ]


def _render_declared_chart(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        specs: Sequence = compute_chart_specs(declared, [
            (label, summary.get("cost") or {}) for label, summary in labeled_summaries
        ])
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


def _render_comparison_report(  # pylint: disable=too-many-locals
    labeled_summaries: List[Tuple[str, dict]],
    labeled_run_records: Optional[List[Tuple[str, Optional[dict]]]] = None,
    selection: Selection = Selection(),
    charts: ChartOutput = ChartOutput(),
    primary: int = -1,
) -> str:
    if not labeled_summaries:
        return ""

    _, primary_summary = labeled_summaries[-1]
    field_names = resolve_fields(labeled_summaries, selection)
    field_measures = resolve_measures(labeled_summaries, field_names, selection)
    field_scoring_types: dict = primary_summary.get("field_scoring_types", {})
    corpora = resolve_corpora(labeled_summaries, selection)
    # Only the corpora every run scored. An aggregate over a corpus one run lacks
    # would differ between columns for composition reasons, which is exactly what
    # an overall row is read as ruling out. The per-corpus sections below still
    # show everything, flagged where the columns are unequal.
    common = _common_corpora(labeled_summaries, corpora)
    corpus_grids = {
        corpus: _corpus_grid(corpus, labeled_summaries, field_names, field_measures)
        for corpus in corpora
    }
    overall_rows = _overall_grid(labeled_summaries, field_names, field_measures, common)
    check_expected_types(labeled_summaries, selection.expected_types)
    check_row_filter(overall_rows, selection.row_filter)
    overall_rows = filter_grid_rows(overall_rows, selection.row_filter)
    corpus_grids = {
        corpus: filter_grid_rows(rows, selection.row_filter)
        for corpus, rows in corpus_grids.items()
    }

    lines = ["## ScienceBeam Parser Evaluation", ""]
    lines += _coverage_lines(labeled_run_records or [])
    lines += _differently_scored_note(labeled_summaries, field_names)

    if len(corpora) > 1:
        lines.extend(_render_overall_section(
            labeled_summaries, field_names, overall_rows, field_scoring_types,
            corpora, common, primary,
        ))
        lines.append("")

    # A single corpus puts one group of bars on the axis, which says nothing the table
    # does not.
    chart_lines: List[str] = []
    labels = [label for label, _ in labeled_summaries]
    for declared in selection.declared_charts:
        chart_lines += _render_declared_chart(
            declared, labels, labeled_summaries, corpus_grids, overall_rows,
            common, charts,
        )
    # `--chart` names fields rather than charts, so it cannot interleave with the file's
    # order and is drawn after whatever that declared.
    if selection.charts and len(common) > 1:
        specs = chart_specs(
            selection, labels, corpus_grids, overall_rows, common,
        )
        if charts.out_dir is not None:
            render_charts(specs, charts.out_dir, charts.prefix)
        chart_lines += chart_markdown(
            specs, charts.rel_dir, charts.prefix, charts.base_url
        )[2:]
    if chart_lines:
        lines += ["### Charts", "", *chart_lines]

    usage_lines = _render_usage_section(
        labeled_summaries, corpora,
        "### LLM usage\n\nWhat producing these predictions spent, over every document"
        " attempted — deliberately a wider set than the scores above, since an errored"
        " document still spent its tokens. A variant short of usage for documents it"
        " attempted was served them from the predictions store, or produced them before"
        " this was recorded. Cost is absent where the backend states none.",
    )
    if usage_lines:
        lines += [*usage_lines, ""]

    cost_lines = render_cost_section(
        labeled_summaries,
        "What producing these predictions took, and on what, over every run that"
        " generated any of them rather than only the one that scored them. CPU is"
        " the whole machine's busy time, so it includes the benchmark client and"
        " anything else the host was doing, and it is absent where the parser ran"
        " on another host. Latency is over the documents that got a prediction;"
        " throughput and CPU per document are over every document those runs"
        " processed, retries included, since the machine paid for those too.",
    )
    if cost_lines:
        lines += [*cost_lines, ""]

    for corpus in corpora:
        n_primary = primary_summary.get("corpora", {}).get(corpus, {}).get("n", 0)
        corpus_lines = _render_corpus_section(
            corpus, labeled_summaries, field_names, corpus_grids[corpus],
            field_scoring_types, primary,
        )
        lines += [
            "<details>",
            f"<summary><b>{corpus}</b> ({n_primary} docs)</summary>",
            "",
            *corpus_lines,
            "",
            "</details>",
            "",
        ]

    return "\n".join(lines)


def _parse_labeled_summary(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected 'label=path', got {spec!r}")
    label, _, path_str = spec.partition("=")
    return label, Path(path_str)


def run_compare(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    labeled_summary_paths: List[Tuple[str, Path]],
    out_path: Optional[Path],
    selection: Selection = Selection(),
    chart_prefix: str = "",
    chart_base_url: str = "",
    primary: int = -1,
) -> None:
    labeled_summaries = [
        (label, json.loads(path.read_text()))
        for label, path in labeled_summary_paths
    ]
    # A run that only scored stored predictions has no run record, and so nothing
    # to say about its own coverage.
    labeled_run_records = [
        (
            label,
            json.loads((path.parent / "run.json").read_text())
            if (path.parent / "run.json").exists() else None,
        )
        for label, path in labeled_summary_paths
    ]
    wants_charts = bool(selection.charts or selection.declared_charts)
    charts = ChartOutput(
        out_dir=out_path.parent / "charts" if out_path and wants_charts else None,
        prefix=chart_prefix,
        base_url=chart_base_url,
    )
    report = _render_comparison_report(
        labeled_summaries, labeled_run_records, selection, charts, primary
    )
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        LOGGER.info("Comparison report written to %s", out_path)
    print(report)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare benchmark summaries from multiple runs"
    )
    parser.add_argument(
        "--summary",
        action="append",
        dest="summaries",
        metavar="LABEL:PATH",
        default=None,
        help=(
            "Summary to include as 'label=path/to/summary.json'. "
            "Repeat for each run. The last entry is the primary (reference for deltas)."
        ),
    )
    parser.add_argument("--out", default=None, help="Output path (default: stdout only)")
    parser.add_argument(
        "--comparison", default=None, metavar="NAME_OR_PATH",
        help=(
            "A comparison file naming the variants, rows and charts to render. A name"
            " resolves under benchmarks/comparisons/. Replaces --summary and the"
            " selection flags"
        ),
    )
    parser.add_argument(
        "--runs", default="benchmarks/runs",
        help="Where a comparison's named variants are resolved from",
    )
    parser.add_argument(
        "--split", default="train",
        help="The split a comparison's named variants were run against",
    )
    parser.add_argument(
        "--current-run", default=None, metavar="DIR",
        help="The run directory a comparison's `current: true` variant refers to",
    )
    parser.add_argument(
        "--field", action="append", default=None, dest="fields", metavar="FIELD",
        help=(
            "Show only this field, repeatable, in the order given. Checked against every"
            " summary, so a field only a baseline scored can be asked for"
        ),
    )
    parser.add_argument(
        "--method", action="append", default=None, dest="methods", metavar="METHOD",
        help="Show only this scoring method, repeatable (e.g. levenshtein)",
    )
    parser.add_argument(
        "--corpus", action="append", default=None, dest="corpora", metavar="CORPUS",
        help="Show only this corpus, repeatable, in the order given",
    )
    parser.add_argument(
        "--chart", action="append", default=None, dest="charts", metavar="FIELD",
        help=(
            "Also chart this field, repeatable: one image per method and scope, with the"
            " variants as series and the corpora along the axis. Needs --out"
        ),
    )
    parser.add_argument(
        "--chart-method", action="append", default=None, dest="chart_methods",
        metavar="METHOD",
        help=(
            "Chart only this method, repeatable. Unlike --method this narrows the"
            " images alone, leaving the tables as they are"
        ),
    )
    parser.add_argument(
        "--chart-prefix", default="",
        help=(
            "Prefix for chart filenames, so charts from different runs do not overwrite"
            " each other where they are published together"
        ),
    )
    parser.add_argument(
        "--chart-base-url", default="",
        help=(
            "Link charts under this URL instead of by relative path, for a surface that"
            " cannot render a local file. The files still have to be published there"
        ),
    )
    args = parser.parse_args(argv)

    if bool(args.comparison) == bool(args.summaries):
        parser.error("Give either --comparison or at least two --summary entries.")
    if args.summaries and len(args.summaries) < 2:
        parser.error("At least two --summary entries are required for a comparison.")
    if (args.charts or args.comparison) and not args.out:
        parser.error("Charts need --out, since the images are written beside the report.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.comparison:
            config = load_comparison(args.comparison)
            labeled_paths = resolve_variants(
                config, Path(args.runs), args.split,
                Path(args.current_run) if args.current_run else None,
            )
            selection = to_selection(config)
            primary = primary_index(config)
        else:
            labeled_paths = [_parse_labeled_summary(s) for s in args.summaries]
            primary = -1
            selection = Selection(
                fields=tuple(args.fields) if args.fields else None,
                methods=tuple(args.methods) if args.methods else None,
                corpora=tuple(args.corpora) if args.corpora else None,
                charts=tuple(args.charts or ()),
                chart_methods=tuple(args.chart_methods) if args.chart_methods else None,
            )
        run_compare(
            labeled_paths, Path(args.out) if args.out else None,
            selection, args.chart_prefix, args.chart_base_url, primary,
        )
    except SelectionError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
