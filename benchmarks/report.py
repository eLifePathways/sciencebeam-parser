from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

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


def _fmt_f1(f1: Optional[float]) -> str:
    return f"{f1:.3f}" if f1 is not None else "—"


def _fmt_delta(delta: Optional[float]) -> str:
    return f"{delta:+.3f}" if delta is not None else "—"


def _render_corpus_section(  # pylint: disable=too-many-locals
    corpus: str,
    labeled_summaries: List[Tuple[str, dict]],
    field_names: List[str],
    field_measures: dict,
    field_scoring_types: dict,
) -> List[str]:
    primary_label, primary_summary = labeled_summaries[-1]
    others = labeled_summaries[:-1]
    other_labels = [label for label, _ in others]

    lines: List[str] = []

    counts = " | ".join(
        f"**{label}**: {s.get('corpora', {}).get(corpus, {}).get('n', 0)} docs"
        for label, s in labeled_summaries
    )
    lines.append(counts)
    lines.append("")

    col_labels = (
        list(other_labels)
        + [primary_label]
        + [f"Δ {label}" for label in other_labels]
    )
    lines.append("| Field (method) | Type | " + " | ".join(col_labels) + " |")

    n_cols = 2 * len(others) + 3  # field, type, others, primary, delta-per-other
    lines.append("|" + "|".join(["---"] * n_cols) + "|")

    for field in field_names:
        methods = field_measures.get(field, [])
        field_type = field_scoring_types.get(field, "string")
        for method in methods:
            primary_f1 = _get_f1(primary_summary, corpus, field, method)
            other_f1s = [_get_f1(s, corpus, field, method) for _, s in others]
            deltas = [
                _fmt_delta(
                    primary_f1 - f1
                    if primary_f1 is not None and f1 is not None
                    else None
                )
                for f1 in other_f1s
            ]
            cells = (
                [_fmt_f1(f1) for f1 in other_f1s]
                + [_fmt_f1(primary_f1)]
                + deltas
            )
            lines.append(f"| {field} ({method}) | {field_type} | " + " | ".join(cells) + " |")

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
    for corpus in corpora:
        lines.append(f"### {corpus}")
        lines.append("")
        lines.extend(
            _render_corpus_section(
                corpus, labeled_summaries, field_names, field_measures, field_scoring_types
            )
        )
        lines.append("")

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
