from __future__ import annotations

import argparse
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from sciencebeam_judge.evaluation.document_scoring import iter_score_document_fields
from sciencebeam_judge.evaluation.score_aggregation import summarise_combined_document_scores
from sciencebeam_judge.parsing.xml import parse_xml, parse_xml_mapping
from sciencebeam_judge.parsing.xpath.xpath_functions import register_functions
from sciencebeam_judge.resources import DEFAULT_XML_MAPPING_PATH

LOGGER = logging.getLogger(__name__)


def _score_pair(
    gold_xml: bytes,
    pred_xml: bytes,
    field_names: List[str],
    measures: List[str],
    xml_mapping: dict,
    scoring_types_by_field_map: Optional[Dict[str, List[str]]] = None,
) -> List[dict]:
    expected = parse_xml(BytesIO(gold_xml), xml_mapping, fields=field_names)
    actual = parse_xml(BytesIO(pred_xml), xml_mapping, fields=field_names)
    return list(
        iter_score_document_fields(
            expected, actual,
            field_names=field_names,
            measures=measures,
            scoring_types_by_field_map=scoring_types_by_field_map,
        )
    )


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


def _build_field_scoring_types(
    field_names: List[str],
    default_type: str,
    per_field: Dict[str, dict],
) -> Dict[str, str]:
    return {
        f: per_field.get(f, {}).get("type", default_type)
        for f in field_names
    }


def _doc_scores_to_dict(doc_scores: List[dict]) -> dict:
    """Reshape flat score list → {field: {method: {precision, recall, f1}}}."""
    result: dict = {}
    for entry in doc_scores:
        result.setdefault(entry["field_name"], {})[entry["scoring_method"]] = (
            _match_to_prf(entry["match_score"])
        )
    return result


def _score_corpus(  # pylint: disable=too-many-locals
    corpus: str,
    data_dir: Path,
    run_dir: Path,
    field_names: List[str],
    all_measures: List[str],
    field_measures: Dict[str, List[str]],
    field_scoring_types: Dict[str, str],
    xml_mapping: dict,
) -> Dict[str, Any]:
    pred_dir = run_dir / "predictions" / corpus
    scores_dir = run_dir / "scores" / corpus
    scores_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        LOGGER.warning("No predictions directory for corpus %r", corpus)
        return {"n": 0}

    scoring_types_by_field_map = {f: [t] for f, t in field_scoring_types.items()}
    all_doc_scores: List[dict] = []
    n = 0

    for pred_path in sorted(pred_dir.glob("*.tei.xml")):
        record_id = pred_path.stem.replace(".tei", "")
        gold_path = data_dir / corpus / f"{record_id}.jats.xml"

        if not gold_path.exists():
            LOGGER.warning("Missing gold for %s/%s, skipping", corpus, record_id)
            continue

        try:
            doc_scores = _score_pair(
                gold_path.read_bytes(), pred_path.read_bytes(),
                field_names, all_measures, xml_mapping,
                scoring_types_by_field_map=scoring_types_by_field_map,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Scoring failed for %s/%s: %s", corpus, record_id, exc)
            continue

        doc_scores = [
            s for s in doc_scores
            if s["scoring_method"] in field_measures[s["field_name"]]
        ]

        (scores_dir / f"{record_id}.json").write_text(json.dumps({
            "record_id": record_id,
            "corpus": corpus,
            "fields": _doc_scores_to_dict(doc_scores),
        }, indent=2))

        all_doc_scores.extend(doc_scores)
        n += 1

    if not all_doc_scores:
        return {"n": n}

    aggregated = summarise_combined_document_scores(all_doc_scores, keys=field_names, count=n)
    return {"n": n, "aggregated": aggregated}


def _render_report(  # pylint: disable=too-many-locals
    corpus_results: Dict[str, Any],
    field_names: List[str],
    field_scoring_types: Dict[str, str],
    run_record: Optional[dict],
) -> str:
    lines = ["## ScienceBeam Parser Evaluation", ""]

    if run_record:
        image = run_record.get("parser_image") or "local"
        cfg = run_record.get("parser_config") or "default"
        mode = run_record.get("mode", "?")
        lines += [f"**Image:** `{image}`  **Config:** `{cfg}`  **Mode:** {mode}", ""]

    for corpus, result in corpus_results.items():
        n = result.get("n", 0)
        aggregated = result.get("aggregated")
        lines.append(f"### {corpus} ({n} docs)")
        lines.append("")

        if not aggregated:
            lines += ["_No results._", ""]
            continue

        # aggregated is a list of {scoring_type, scoring_method, summary_scores}.
        # The same method may appear multiple times (once per scoring type), so deduplicate
        # column headers and look up scores by (field_type, method).
        seen: set = set()
        unique_methods: List[str] = []
        for entry in aggregated:
            m = entry["scoring_method"]
            if m not in seen:
                seen.add(m)
                unique_methods.append(m)

        score_lookup = {
            (entry.get("scoring_type", "string"), entry["scoring_method"]): entry
            for entry in aggregated
        }

        lines.append("| Field | Type |" + "".join(f" {m} F1 |" for m in unique_methods))
        lines.append("|---|---|" + "---|" * len(unique_methods))

        for field in field_names:
            field_type = field_scoring_types.get(field, "string")
            row = f"| {field} | {field_type} |"
            for method in unique_methods:
                agg_entry = score_lookup.get((field_type, method))
                if agg_entry is None:
                    row += " — |"
                    continue
                by_field = agg_entry.get("summary_scores", {}).get("by-field", {})
                f1 = by_field.get(field, {}).get("scores", {}).get("f1", None)
                row += f" {f1:.3f} |" if f1 is not None else " — |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def run_score(  # pylint: disable=too-many-locals
    config: dict,
    run_dir: Path,
    data_dir: Path,
    out_path: Optional[Path],
) -> None:
    register_functions()
    xml_mapping = parse_xml_mapping(DEFAULT_XML_MAPPING_PATH)
    field_names: List[str] = config["fields"]
    scoring_cfg = config.get("scoring", {})
    default_methods: List[str] = scoring_cfg.get("default_methods", ["levenshtein"])
    default_type: str = scoring_cfg.get("default_type", "string")
    per_field_cfg: Dict[str, dict] = scoring_cfg.get("per_field", {})
    field_measures = _build_field_measures(field_names, default_methods, per_field_cfg)
    field_scoring_types = _build_field_scoring_types(field_names, default_type, per_field_cfg)
    all_measures = list(dict.fromkeys(m for methods in field_measures.values() for m in methods))

    run_record = None
    run_record_path = run_dir / "run.json"
    if run_record_path.exists():
        run_record = json.loads(run_record_path.read_text())
    else:
        LOGGER.warning("No run.json found in %s; split/corpus detection may be incomplete", run_dir)

    split = (run_record or {}).get("split", "train")

    corpora = list(config["dataset"]["splits"][split].keys())

    corpus_results: Dict[str, Any] = {}
    for corpus in corpora:
        LOGGER.info("Scoring corpus %r (split=%s)...", corpus, split)
        corpus_results[corpus] = _score_corpus(
            corpus, data_dir / split, run_dir, field_names, all_measures, field_measures,
            field_scoring_types, xml_mapping
        )

    (run_dir / "summary.json").write_text(json.dumps({
        "fields": field_names,
        "field_measures": field_measures,
        "field_scoring_types": field_scoring_types,
        "corpora": corpus_results,
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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_score(
        config=config,
        run_dir=Path(args.run),
        data_dir=Path(args.data),
        out_path=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
