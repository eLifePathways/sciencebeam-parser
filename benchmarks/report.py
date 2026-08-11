from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)


def _get_f1(
    summary: dict,
    corpus: str,
    field: str,
    method: str,
) -> Optional[float]:
    aggregated = summary.get("corpora", {}).get(corpus, {}).get("aggregated", [])
    field_type = summary.get("field_scoring_types", {}).get(field, "string")
    for entry in aggregated:
        if entry.get("scoring_type") == field_type and entry.get("scoring_method") == method:
            f1 = entry.get("summary_scores", {}).get("by-field", {}).get(
                field, {}
            ).get("scores", {}).get("f1")
            if f1 is not None:
                return float(f1)
    return None


def _get_overall_f1(summary: dict, field: str, method: str) -> Optional[float]:
    """Doc-count-weighted mean F1 across all corpora."""
    total_n = 0
    weighted = 0.0
    for corpus, corpus_data in summary.get("corpora", {}).items():
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


def _corpus_f1_getter(corpus: str) -> Callable[[dict, str, str], Optional[float]]:
    return lambda s, f, m: _get_f1(s, corpus, f, m)


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
    return lines


def _render_overall_section(  # pylint: disable=too-many-locals
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
    field_scoring_types: dict,
    corpora: List[str],
) -> List[str]:
    _, primary_summary = labeled_summaries[-1]
    n_total = sum(
        primary_summary.get("corpora", {}).get(c, {}).get("n", 0) for c in corpora
    )
    counts_by_label = [
        (label, sum(s.get("corpora", {}).get(c, {}).get("n", 0) for c in corpora))
        for label, s in labeled_summaries
    ]
    total_counts = " | ".join(f"**{label}**: {n} docs" for label, n in counts_by_label)
    lines = [
        f"### Overall ({n_total} docs across {len(corpora)} corpora)",
        "",
        total_counts,
        "",
        *_unequal_docs_note(counts_by_label),
    ]
    lines.extend(_render_field_table(
        labeled_summaries, field_names, field_measures, field_scoring_types,
        _get_overall_f1,
    ))
    return lines


def _render_comparison_report(
    labeled_summaries: List[Tuple[str, dict]],
) -> str:
    if not labeled_summaries:
        return ""

    _, primary_summary = labeled_summaries[-1]
    field_names: List[str] = primary_summary.get("fields", [])
    field_measures: dict = primary_summary.get("field_measures", {})
    field_scoring_types: dict = primary_summary.get("field_scoring_types", {})
    corpora = list(primary_summary.get("corpora", {}).keys())

    lines = ["## ScienceBeam Parser Evaluation", ""]

    if len(corpora) > 1:
        lines.extend(_render_overall_section(
            labeled_summaries, field_names, field_measures, field_scoring_types, corpora,
        ))
        lines.append("")

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
    report = _render_comparison_report(labeled_summaries)
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
