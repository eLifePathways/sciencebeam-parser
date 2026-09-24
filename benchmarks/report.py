from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmarks.gold_presence import (
    GOLD_PRESENCE_KEY,
    GOLD_PRESENT_AGGREGATED_KEY,
    has_gold,
    is_split_worth_reporting,
    produced_bullet,
)
from benchmarks.llm_usage import usage_for_corpora

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


def _render_field_table(  # pylint: disable=too-many-locals
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
    field_scoring_types: dict,
    get_f1_fn: Callable[[dict, str, str], Optional[float]],
) -> List[str]:
    """Render comparison table. get_f1_fn(summary, field, method) -> Optional[float]."""
    primary_label, _ = labeled_summaries[-1]
    others = labeled_summaries[:-1]
    other_labels = [label for label, _ in others]

    col_labels = other_labels + [primary_label] + [f"Δ {lbl}" for lbl in other_labels]
    n_cols = 2 * len(others) + 3
    lines = [
        "| Field (method) | Type | " + " | ".join(col_labels) + " |",
        "|" + "|".join(["---"] * n_cols) + "|",
    ]

    for field in field_names:
        methods = field_measures.get(field, [])
        field_type = field_scoring_types.get(field, "string")
        for method in methods:
            primary_f1 = get_f1_fn(labeled_summaries[-1][1], field, method)
            other_f1s = [get_f1_fn(s, field, method) for _, s in others]
            deltas = [
                _fmt_delta(
                    primary_f1 - f1 if primary_f1 is not None and f1 is not None else None
                )
                for f1 in other_f1s
            ]
            cells = [_fmt_f1(f1) for f1 in other_f1s] + [_fmt_f1(primary_f1)] + deltas
            lines.append(f"| {field} ({method}) | {field_type} | " + " | ".join(cells) + " |")

    return lines


def _unequal_gold_note(
    labeled_summaries: List[Tuple[str, dict]], corpus: str, fields: List[str]
) -> List[str]:
    """Call out a split whose columns were scored over different documents.

    The `Gold` column states one denominator for the row, so where the runs covered
    different documents it belongs to one column only, and each variant's figure is over
    its own gold documents rather than over the stated ones.
    """
    for field in fields:
        counts = [
            (label, _gold_presence(summary, corpus, field))
            for label, summary in labeled_summaries
        ]
        known = [(label, p) for label, p in counts if p]
        if len({(p["n_gold"], p["n"]) for _, p in known}) <= 1:
            continue
        listed = ", ".join(f"{label} {p['n_gold']}/{p['n']}" for label, p in known)
        return [
            f"> ⚠️ **Unequal document sets** ({listed}). The `Gold` column states one of "
            "them; each figure is over that column's own gold documents.",
            "",
        ]
    return []


def _split_presence(
    labeled_summaries: List[Tuple[str, dict]], corpus: str, field: str
) -> Optional[dict]:
    """One presence per field, preferring the primary: whether the gold records a field is
    a property of the corpus rather than of the variant."""
    for _, summary in reversed(labeled_summaries):
        presence = _gold_presence(summary, corpus, field)
        if presence:
            return presence
    return None


def _split_fields(
    labeled_summaries: List[Tuple[str, dict]], corpus: str, field_names: List[str]
) -> List[str]:
    return [
        field for field in field_names
        if is_split_worth_reporting(
            [_gold_presence(s, corpus, field) for _, s in labeled_summaries]
        )
    ]


def _render_split_table(  # pylint: disable=too-many-locals
    labeled_summaries: List[Tuple[str, dict]],
    corpus: str,
    fields: List[str],
    field_measures: dict,
) -> List[str]:
    """The comparison table, restricted to the documents the gold can support."""
    scored = [
        (field, presence) for field, presence in (
            (field, _split_presence(labeled_summaries, corpus, field)) for field in fields
        ) if presence and presence["n_gold"]
    ]
    if not scored:
        return []
    primary_label, _ = labeled_summaries[-1]
    others = labeled_summaries[:-1]
    col_labels = (
        [label for label, _ in others] + [primary_label]
        + [f"Δ {label}" for label, _ in others]
    )
    lines = [
        "| Field (method) | Gold | " + " | ".join(col_labels) + " |",
        "|" + "|".join(["---"] * (2 + len(col_labels))) + "|",
    ]
    for field, presence in scored:
        gold = f"{presence['n_gold']}/{presence['n']}"
        for method in field_measures.get(field, []):
            values = [
                _get_f1(s, corpus, field, method, GOLD_PRESENT_AGGREGATED_KEY)
                for _, s in labeled_summaries
            ]
            primary_f1 = values[-1]
            deltas = [
                _fmt_delta(
                    primary_f1 - f1 if primary_f1 is not None and f1 is not None else None
                )
                for f1 in values[:-1]
            ]
            cells = [_fmt_f1(f1) for f1 in values] + deltas
            lines.append(f"| {field} ({method}) | {gold} | " + " | ".join(cells) + " |")
    return lines


def _render_gold_split_section(
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
    corpora: List[str],
) -> List[str]:
    """Each field scored over only the documents whose gold records it, and what each
    variant produced where it records nothing.

    Empty unless some field earns it, so a comparison over corpora that record everything
    is unchanged, as is one against a summary written before the split.
    """
    blocks: List[str] = []
    for corpus in corpora:
        fields = _split_fields(labeled_summaries, corpus, field_names)
        if not fields:
            continue
        n_docs = max(
            s.get("corpora", {}).get(corpus, {}).get("n", 0) for _, s in labeled_summaries
        )
        table = _render_split_table(labeled_summaries, corpus, fields, field_measures)
        bullets = [
            bullet for bullet in (
                produced_bullet(
                    field,
                    [(label, _gold_presence(s, corpus, field)) for label, s in labeled_summaries],
                )
                for field in fields
            ) if bullet
        ]
        blocks += [f"**{corpus}** ({n_docs} docs)", ""]
        blocks += _unequal_gold_note(labeled_summaries, corpus, fields)
        if table:
            blocks += [*table, ""]
        blocks += ["Produced where the gold records nothing:", "", *bullets, ""]
    if not blocks:
        return []
    return [
        "### Where the gold does not record the field",
        "",
        "Scored over only the documents whose gold records the field. A corpus recording"
        " none of it has nothing to score against, and shows the counts alone.",
        "",
        *blocks,
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


def _render_corpus_section(
    corpus: str,
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
    field_scoring_types: dict,
) -> List[str]:
    counts_by_label = [
        (label, s.get("corpora", {}).get(corpus, {}).get("n", 0))
        for label, s in labeled_summaries
    ]
    counts = " | ".join(f"**{label}**: {n} docs" for label, n in counts_by_label)
    lines = [counts, ""] + _unequal_docs_note(counts_by_label)
    lines.extend(_render_field_table(
        labeled_summaries, field_names, field_measures, field_scoring_types,
        _corpus_f1_getter(corpus),
    ))
    usage_lines = _render_usage_section(
        labeled_summaries, [corpus],
        "**LLM usage**, over every document attempted in this corpus.",
    )
    if usage_lines:
        lines += ["", *usage_lines]
    return lines


def _render_overall_section(  # pylint: disable=too-many-locals
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
    field_scoring_types: dict,
    corpora: List[str],
) -> List[str]:
    _, primary_summary = labeled_summaries[-1]
    # Only the corpora every run scored. An aggregate over a corpus one run lacks
    # would differ between columns for composition reasons, which is exactly what
    # an overall row is read as ruling out. The per-corpus sections below still
    # show everything, flagged where the columns are unequal.
    common = _common_corpora(labeled_summaries, corpora)
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
    lines.extend(_render_field_table(
        labeled_summaries, field_names, field_measures, field_scoring_types,
        lambda s, f, m: _get_overall_f1(s, f, m, common),
    ))
    return lines


def _render_comparison_report(
    labeled_summaries: List[Tuple[str, dict]],
    labeled_run_records: Optional[List[Tuple[str, Optional[dict]]]] = None,
) -> str:
    if not labeled_summaries:
        return ""

    _, primary_summary = labeled_summaries[-1]
    field_names: List[str] = primary_summary.get("fields", [])
    field_measures: dict = primary_summary.get("field_measures", {})
    field_scoring_types: dict = primary_summary.get("field_scoring_types", {})
    corpora = list(primary_summary.get("corpora", {}).keys())

    lines = ["## ScienceBeam Parser Evaluation", ""]
    lines += _coverage_lines(labeled_run_records or [])

    if len(corpora) > 1:
        lines.extend(_render_overall_section(
            labeled_summaries, field_names, field_measures, field_scoring_types, corpora,
        ))
        lines.append("")

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

    for corpus in corpora:
        n_primary = primary_summary.get("corpora", {}).get(corpus, {}).get("n", 0)
        corpus_lines = _render_corpus_section(
            corpus, labeled_summaries, field_names, field_measures, field_scoring_types,
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

    lines += _render_gold_split_section(
        labeled_summaries, field_names, field_measures, corpora,
    )

    return "\n".join(lines)


def _parse_labeled_summary(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected 'label=path', got {spec!r}")
    label, _, path_str = spec.partition("=")
    return label, Path(path_str)


def run_compare(
    labeled_summary_paths: List[Tuple[str, Path]],
    out_path: Optional[Path],
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
    report = _render_comparison_report(labeled_summaries, labeled_run_records)
    if out_path:
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
        required=True,
        help=(
            "Summary to include as 'label=path/to/summary.json'. "
            "Repeat for each run. The last entry is the primary (reference for deltas)."
        ),
    )
    parser.add_argument("--out", default=None, help="Output path (default: stdout only)")
    args = parser.parse_args(argv)

    if len(args.summaries) < 2:
        parser.error("At least two --summary entries are required for a comparison.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    labeled_paths = [_parse_labeled_summary(s) for s in args.summaries]
    run_compare(labeled_paths, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
