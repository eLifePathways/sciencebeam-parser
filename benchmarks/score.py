from __future__ import annotations

import argparse
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

from sciencebeam_judge.evaluation.document_scoring import iter_score_document_fields
from sciencebeam_judge.evaluation.score_aggregation import (
    combine_and_compact_document_scores,
    summarise_combined_document_scores,
)
from sciencebeam_judge.evaluation.scoring_types.scoring_types import resolve_scoring_type
from sciencebeam_judge.parsing.xml import parse_xml

from benchmarks.affiliation_linking import (
    AFFILIATION_LINKED_FIELD,
    LINKED_SCORING_TYPE,
    score_linked_affiliations,
)
from benchmarks.fetch import included_corpora
from benchmarks.gold_presence import (
    GOLD_PRESENCE_KEY,
    GOLD_PRESENT_AGGREGATED_KEY,
    gold_records_field,
    has_gold,
    is_split_worth_reporting,
    produced_row,
    summarise_gold_presence,
)
from benchmarks.judge_setup import prepare_judge
from benchmarks.llm_usage import aggregate_llm_usage, read_manifest_entries
from benchmarks.prediction_files import iter_prediction_files, record_id_from_path
from benchmarks.variant_match import (
    VARIANT_MATCH_KEY,
    render_variant_match_table,
    summarise_variant_matches,
)

LOGGER = logging.getLogger(__name__)


def _score_pair(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    gold_xml: bytes,
    pred_xml: bytes,
    field_names: List[str],
    measures: List[str],
    xml_mapping: dict,
    scoring_types_by_field_map: Optional[Dict[str, List[str]]] = None,
    field_sources: Optional[Dict[str, Tuple[str, str]]] = None,
) -> List[dict]:
    # A link between two values is not something the judge's mapping can select, so the
    # field is scored apart from the ones it reads.
    judge_field_names = [f for f in field_names if f != AFFILIATION_LINKED_FIELD]
    sources = field_sources or {}
    source_names = sorted({
        name for field in judge_field_names for name in sources.get(field, (field, field))
    })
    gold_values = parse_xml(BytesIO(gold_xml), xml_mapping, fields=source_names)
    predicted_values = parse_xml(BytesIO(pred_xml), xml_mapping, fields=source_names)
    expected = {
        field: gold_values[sources.get(field, (field, field))[0]]
        for field in judge_field_names
        if sources.get(field, (field, field))[0] in gold_values
    }
    actual = {
        field: predicted_values[sources.get(field, (field, field))[1]]
        for field in judge_field_names
        if sources.get(field, (field, field))[1] in predicted_values
    }
    doc_scores = list(
        iter_score_document_fields(
            expected, actual,
            field_names=judge_field_names,
            measures=measures,
            scoring_types_by_field_map=scoring_types_by_field_map,
        )
    )
    if AFFILIATION_LINKED_FIELD in field_names:
        doc_scores += score_linked_affiliations(gold_xml, pred_xml, measures)
    return doc_scores


def _match_to_prf(ms: dict) -> dict:
    tp = ms.get("true_positive", 0)
    fp = ms.get("false_positive", 0)
    fn = ms.get("false_negative", 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _build_field_measures(
    field_names: List[str],
    default_methods: List[str],
    per_field: Dict[str, dict],
) -> Dict[str, List[str]]:
    return {
        f: per_field.get(f, {}).get("methods", default_methods)
        for f in field_names
    }


def _build_field_sources(
    field_names: List[str],
    per_field: Dict[str, dict],
) -> Dict[str, Tuple[str, str]]:
    """Which mapping entry each side of a field's comparison reads.

    A field whose gold and prediction come from different entries is how a comparison says
    "this gold against any of those predicted values" without a scoring type having to read
    a position and trust what put the value there.
    """
    return {
        field: (
            per_field.get(field, {}).get("expected", field),
            per_field.get(field, {}).get("actual", field),
        )
        for field in field_names
    }


def _build_field_scoring_types(
    field_names: List[str],
    default_type: str,
    per_field: Dict[str, dict],
) -> Dict[str, str]:
    scoring_types = {
        # Scored one way only, so its type is not the config's to set: the default
        # would otherwise label it `string` and the report would look it up as that.
        f: (
            LINKED_SCORING_TYPE if f == AFFILIATION_LINKED_FIELD
            else per_field.get(f, {}).get("type", default_type)
        )
        for f in field_names
    }
    # `linked` is not one of the judge's types, so there is nothing of its to resolve.
    for scoring_type in set(scoring_types.values()) - {LINKED_SCORING_TYPE}:
        # Resolved here rather than per document: scoring raises per document and the run
        # goes on, so an unknown type would warn once per document and report nothing.
        resolve_scoring_type(scoring_type)
    return scoring_types


def _doc_scores_to_dict(doc_scores: List[dict]) -> dict:
    """Reshape flat score list → {field: {scoring_type, method: {counts + precision/recall/f1}}}."""
    result: dict = {}
    for entry in doc_scores:
        field = entry["field_name"]
        ms = entry["match_score"]
        result.setdefault(field, {"scoring_type": entry["scoring_type"]})[
            entry["scoring_method"]
        ] = {**ms, **_match_to_prf(ms)}
    return result


def _doc_scores_from_dict(fields: Dict[str, dict]) -> Iterator[dict]:
    """Inverse of `_doc_scores_to_dict`, so a score file aggregates like a fresh score.

    The per-document precision/recall/f1 it also holds are ignored by the sums, which read
    only the counts.
    """
    for field_name, entry in fields.items():
        for method, match_score in entry.items():
            if method == "scoring_type":
                continue
            yield {
                "field_name": field_name,
                "scoring_type": entry.get("scoring_type", "string"),
                "scoring_method": method,
                "match_score": match_score,
            }


def _summarise_documents(
    documents: List[Dict[str, dict]],
    field_names: List[str],
    field_measures: Dict[str, List[str]],
) -> Dict[str, Any]:
    """One corpus, from the per-document scores, whether just computed or read back."""
    n = len(documents)
    all_doc_scores: List[dict] = []
    gold_present_doc_scores: List[dict] = []
    for fields in documents:
        doc_scores = [
            score for score in _doc_scores_from_dict(fields)
            if score["scoring_method"] in field_measures.get(score["field_name"], [])
        ]
        all_doc_scores += doc_scores
        gold_present_doc_scores += [
            score for score in doc_scores
            if gold_records_field(fields.get(score["field_name"]) or {})
        ]

    if not all_doc_scores:
        return {"n": n}

    result: Dict[str, Any] = {
        "n": n,
        "aggregated": summarise_combined_document_scores(
            combine_and_compact_document_scores(all_doc_scores), keys=field_names, count=n
        ),
        GOLD_PRESENCE_KEY: summarise_gold_presence(documents, field_names),
    }
    variant_match = summarise_variant_matches(documents, field_names)
    if variant_match:
        result[VARIANT_MATCH_KEY] = variant_match
    if gold_present_doc_scores:
        # No count: the documents behind it differ per field, so one number would be wrong
        # for all but the field it came from.
        result[GOLD_PRESENT_AGGREGATED_KEY] = summarise_combined_document_scores(
            combine_and_compact_document_scores(gold_present_doc_scores), keys=field_names
        )
    return result


def _read_scored_documents(scores_dir: Path) -> List[Dict[str, dict]]:
    """What a previous run scored, which is what it left behind rather than what it would
    score now: a document whose scoring failed keeps the file of the run before it."""
    if not scores_dir.exists():
        LOGGER.warning("No scores directory at %s", scores_dir)
        return []
    return [
        json.loads(path.read_text())["fields"]
        for path in sorted(scores_dir.glob("*.json"))
    ]


def _score_corpus(  # pylint: disable=too-many-locals
    corpus: str,
    data_dir: Path,
    run_dir: Path,
    field_names: List[str],
    all_measures: List[str],
    field_measures: Dict[str, List[str]],
    field_scoring_types: Dict[str, str],
    field_sources: Dict[str, Tuple[str, str]],
    xml_mapping: dict,
) -> Dict[str, Any]:
    pred_dir = run_dir / "predictions" / corpus
    scores_dir = run_dir / "scores" / corpus
    scores_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        LOGGER.warning("No predictions directory for corpus %r", corpus)
        return {"n": 0}

    scoring_types_by_field_map = {f: [t] for f, t in field_scoring_types.items()}
    documents: List[Dict[str, dict]] = []

    for pred_path in iter_prediction_files(pred_dir):
        record_id = record_id_from_path(pred_path)
        gold_path = data_dir / corpus / f"{record_id}.jats.xml"

        if not gold_path.exists():
            LOGGER.warning("Missing gold for %s/%s, skipping", corpus, record_id)
            continue

        try:
            doc_scores = _score_pair(
                gold_path.read_bytes(), pred_path.read_bytes(),
                field_names, all_measures, xml_mapping,
                scoring_types_by_field_map=scoring_types_by_field_map,
                field_sources=field_sources,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Scoring failed for %s/%s: %s", corpus, record_id, exc)
            continue

        doc_scores = [
            s for s in doc_scores
            if s["scoring_method"] in field_measures[s["field_name"]]
        ]

        fields = _doc_scores_to_dict(doc_scores)
        (scores_dir / f"{record_id}.json").write_text(json.dumps({
            "record_id": record_id,
            "corpus": corpus,
            "fields": fields,
        }, indent=2))

        documents.append(fields)

    return _summarise_documents(documents, field_names, field_measures)


def _coverage_note(run_record: dict) -> Optional[str]:
    """What the run had to retry, and what it never got.

    Neither is visible in the scores: a document without a prediction leaves the
    denominator rather than scoring zero, and a retried document scores like any
    other. Both say something about the run rather than about the models, so they
    are stated where the scores are.
    """
    recovered = run_record.get("n_recovered") or 0
    errors = run_record.get("n_errors") or 0
    if not recovered and not errors:
        return None
    parts = []
    if recovered:
        parts.append(f"{recovered} recovered on retry")
    if errors:
        parts.append(f"{errors} without a prediction, and so not scored")
    return "**Coverage:** " + ", ".join(parts)


def _unique_methods(aggregated: List[dict]) -> List[str]:
    """The same method appears once per scoring type, so column headers are deduplicated."""
    seen: set = set()
    methods: List[str] = []
    for entry in aggregated:
        method = entry["scoring_method"]
        if method not in seen:
            seen.add(method)
            methods.append(method)
    return methods


def _f1_from_aggregated(
    aggregated: List[dict], field: str, field_type: str, method: str
) -> Optional[float]:
    for entry in aggregated:
        if entry.get("scoring_type", "string") != field_type:
            continue
        if entry["scoring_method"] != method:
            continue
        return entry.get("summary_scores", {}).get("by-field", {}).get(
            field, {}
        ).get("scores", {}).get("f1")
    return None


def _fmt_f1(f1: Optional[float]) -> str:
    return f"{f1:.3f}" if f1 is not None else "—"


def _score_row(
    field: str,
    field_type: str,
    docs: str,
    aggregated: List[dict],
    methods: List[str],
) -> str:
    """One table row. An empty `aggregated` dashes every method, which is what a corpus
    recording nothing for the field gets: an f1 of 0.000 would read as a failure to
    extract something that is not there."""
    cells = "".join(
        " " + _fmt_f1(_f1_from_aggregated(aggregated, field, field_type, method)) + " |"
        for method in methods
    )
    return f"| {field} | {field_type} | {docs} |" + cells


def _render_produced_table(
    result: Dict[str, Any],
    field_names: List[str],
) -> List[str]:
    """The documents whose gold records no value for a field, and what was produced on
    them. Not a score: nothing in the PDF says whether a publisher recorded the field."""
    presence_by_field = result.get(GOLD_PRESENCE_KEY) or {}
    rows = [
        produced_row(field, [presence_by_field.get(field)])
        for field in field_names
        if is_split_worth_reporting([presence_by_field.get(field)])
    ]
    present = [row for row in rows if row]
    if not present:
        return []
    return [
        "Produced where the gold records nothing:",
        "",
        "| Field | No gold | Produced |",
        "|---|---|---|",
        *["| " + " | ".join(row) + " |" for row in present],
        "",
    ]


def _render_report(  # pylint: disable=too-many-locals
    corpus_results: Dict[str, Any],
    field_names: List[str],
    field_scoring_types: Dict[str, str],
    run_record: Optional[dict],
) -> str:
    lines = ["## ScienceBeam Parser Evaluation", ""]

    if run_record:
        image = run_record.get("parser_image") or "local"
        cfg = run_record.get("profile") or run_record.get("parser_config") or "default"
        mode = run_record.get("mode", "?")
        lines += [f"**Image:** `{image}`  **Config:** `{cfg}`  **Mode:** {mode}", ""]
        coverage = _coverage_note(run_record)
        if coverage:
            lines += [coverage, ""]

    for corpus, result in corpus_results.items():
        n = result.get("n", 0)
        aggregated = result.get("aggregated")
        lines.append(f"### {corpus} ({n} docs)")
        lines.append("")

        if not aggregated:
            lines += ["_No results._", ""]
            continue

        unique_methods = _unique_methods(aggregated)
        presence_by_field = result.get(GOLD_PRESENCE_KEY) or {}

        conditional = result.get(GOLD_PRESENT_AGGREGATED_KEY) or []
        lines.append(
            "| Field | Type | Docs |" + "".join(f" {m} F1 |" for m in unique_methods)
        )
        lines.append("|---|---|---|" + "---|" * len(unique_methods))

        for field in field_names:
            field_type = field_scoring_types.get(field, "string")
            presence = presence_by_field.get(field)
            split = bool(
                presence and presence["n_gold"] and is_split_worth_reporting([presence])
            )
            n_docs = presence["n"] if presence else n
            lines.append(_score_row(
                field, field_type, f"all {n_docs}" if split else str(n_docs),
                aggregated if has_gold(presence) else [], unique_methods,
            ))
            if split and presence:
                lines.append(_score_row(
                    field, field_type, f"gold {presence['n_gold']}",
                    conditional, unique_methods,
                ))

        lines.append("")
        lines += _render_produced_table(result, field_names)
        lines += render_variant_match_table(result.get(VARIANT_MATCH_KEY) or {}, field_names)

    return "\n".join(lines)


def run_score(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    config: dict,
    run_dir: Path,
    data_dir: Path,
    out_path: Optional[Path],
    split_override: Optional[str] = None,
    include: Optional[Iterable[str]] = None,
    from_scores: bool = False,
) -> None:
    xml_mapping = prepare_judge()
    field_names: List[str] = config["fields"]
    scoring_cfg = config.get("scoring", {})
    default_methods: List[str] = scoring_cfg.get("default_methods", ["levenshtein"])
    default_type: str = scoring_cfg.get("default_type", "string")
    per_field_cfg: Dict[str, dict] = scoring_cfg.get("per_field", {})
    field_measures = _build_field_measures(field_names, default_methods, per_field_cfg)
    field_scoring_types = _build_field_scoring_types(field_names, default_type, per_field_cfg)
    field_sources = _build_field_sources(field_names, per_field_cfg)
    all_measures = list(dict.fromkeys(m for methods in field_measures.values() for m in methods))

    run_record = None
    run_record_path = run_dir / "run.json"
    if run_record_path.exists():
        run_record = json.loads(run_record_path.read_text())
    else:
        LOGGER.warning("No run.json found in %s; split/corpus detection may be incomplete", run_dir)

    split = split_override or (run_record or {}).get("split", "train")

    # The corpora the run actually covered, which is not every corpus of the split
    # once one of them is opt-in: scoring an absent corpus only produces a warning
    # and an empty section in the report. The run record is the authority, but it
    # only exists where predictions were generated — a run whose predictions all
    # came from the store has none, so the caller's opt-in set has to be honoured
    # here too, or an opt-in corpus is fetched and predicted and then silently left
    # out of the summary and the comparison built from it.
    corpora = (run_record or {}).get("corpora") or included_corpora(
        config, split, include
    )

    corpus_results: Dict[str, Any] = {}
    for corpus in corpora:
        if from_scores:
            # Summarising what a run already scored, which is all this needs: the gold it
            # was scored against may no longer be cached, and re-scoring would not change
            # a figure.
            LOGGER.info("Summarising corpus %r from its score files...", corpus)
            corpus_results[corpus] = _summarise_documents(
                _read_scored_documents(run_dir / "scores" / corpus),
                field_names, field_measures,
            )
            continue
        LOGGER.info("Scoring corpus %r (split=%s)...", corpus, split)
        corpus_results[corpus] = _score_corpus(
            corpus, data_dir / split, run_dir, field_names, all_measures, field_measures,
            field_scoring_types, field_sources, xml_mapping
        )

    llm_usage = aggregate_llm_usage(read_manifest_entries(run_dir), corpora)

    (run_dir / "summary.json").write_text(json.dumps({
        "fields": field_names,
        "field_measures": field_measures,
        "field_scoring_types": field_scoring_types,
        "field_sources": {
            field: list(sources)
            for field, sources in field_sources.items()
            if sources != (field, field)
        },
        "corpora": corpus_results,
        **({"llm_usage": llm_usage} if llm_usage else {}),
    }, indent=2))

    report = _render_report(corpus_results, field_names, field_scoring_types, run_record)
    report_path = out_path or run_dir / "report.md"
    report_path.write_text(report)

    LOGGER.info("Report written to %s", report_path)
    print(report)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score: evaluate predictions with sciencebeam-judge"
    )
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument("--run", required=True, help="Run directory, e.g. benchmarks/runs/local")
    parser.add_argument("--data", default="benchmarks/data", help="Data cache directory")
    parser.add_argument(
        "--out", default=None, help="Output path for report.md (default: <run>/report.md)"
    )
    parser.add_argument(
        "--split", default=None, help="Dataset split override (default: read from run.json)"
    )
    parser.add_argument(
        "--from-scores", action="store_true",
        help="Summarise the run's existing score files rather than scoring again",
    )
    parser.add_argument(
        "--include-corpus", action="append", default=None, dest="include_corpus",
        metavar="CORPUS",
        help="Also score an opt-in corpus, repeatable. Ignored where run.json lists corpora",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_score(
        config=config,
        run_dir=Path(args.run),
        data_dir=Path(args.data),
        out_path=Path(args.out) if args.out else None,
        split_override=args.split,
        include=args.include_corpus,
        from_scores=args.from_scores,
    )


if __name__ == "__main__":
    main()
