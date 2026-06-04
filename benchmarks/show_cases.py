from __future__ import annotations

import argparse
import difflib
import json
import logging
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from lxml import etree as lxml_etree

from sciencebeam_judge.parsing.xml import parse_xml, parse_xml_mapping
from sciencebeam_judge.parsing.xpath.xpath_functions import register_functions
from sciencebeam_judge.resources import DEFAULT_XML_MAPPING_PATH

LOGGER = logging.getLogger(__name__)

_RESET = "\033[0m"
_RED = "\033[91m"
_GREEN = "\033[92m"


def word_diff(reference: Optional[str], candidate: Optional[str]) -> str:
    if reference is None and candidate is None:
        return "(both absent)"
    if reference is None:
        return f"(gold absent) {_GREEN}{candidate}{_RESET}"
    if candidate is None:
        return f"{_RED}(prediction absent){_RESET}"
    ref_words = reference.split()
    cand_words = candidate.split()
    matcher = difflib.SequenceMatcher(None, ref_words, cand_words, autojunk=False)
    parts = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            parts.append(" ".join(ref_words[i1:i2]))
        elif op == "replace":
            parts.append(f"{_RED}[-{' '.join(ref_words[i1:i2])}-]{_RESET}")
            parts.append(f"{_GREEN}{{+{' '.join(cand_words[j1:j2])}+}}{_RESET}")
        elif op == "delete":
            parts.append(f"{_RED}[-{' '.join(ref_words[i1:i2])}-]{_RESET}")
        elif op == "insert":
            parts.append(f"{_GREEN}{{+{' '.join(cand_words[j1:j2])}+}}{_RESET}")
    return " ".join(parts)


def _get_doc_score(field_scores: dict, method: str) -> Optional[float]:
    ms = field_scores.get(method)
    if ms is None:
        return None
    if "sim_sum" in ms:
        ec = ms.get("expected_count", 0)
        pc = ms.get("predicted_count", 0)
        precision = ms["sim_sum"] / pc if pc else 0.0
        recall = ms["sim_sum"] / ec if ec else 0.0
        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
    return ms.get("f1")


def _load_doc_score(score_path: Path, field: str, method: str) -> Optional[float]:
    if not score_path.exists():
        return None
    data = json.loads(score_path.read_text(encoding="utf-8"))
    return _get_doc_score(data.get("fields", {}).get(field, {}), method)


def _extract_field_text(xml_path: Path, field: str, xml_mapping: dict) -> Optional[str]:
    if not xml_path.exists():
        return None
    values = parse_xml(BytesIO(xml_path.read_bytes()), xml_mapping, fields=[field])
    items = values.get(field, [])
    return " | ".join(str(v) for v in items) if items else None


def _run_label(run: Path) -> str:
    parts = list(run.parts)
    try:
        idx = parts.index("runs")
        return "/".join(parts[idx + 1:])
    except ValueError:
        return run.name


def _comparison_label(run_b: Path) -> str:
    """Derive a short label from run_b path for use in export directory names.

    baselines/grobid/0.9.0-crf/train -> grobid-0.9.0-crf
    """
    label = _run_label(run_b)
    if label.startswith("baselines/"):
        label = label[len("baselines/"):]
    # Strip trailing split segment (last path component)
    parts = label.split("/")
    if len(parts) > 1:
        label = "/".join(parts[:-1])
    return label.replace("/", "-")


def find_cases(
    run_a: Path,
    run_b: Path,
    field: str,
    method: str,
    corpus_filter: Optional[str],
    mode: str,
) -> List[Tuple[float, str, str, float, float]]:
    """Return (delta, corpus, record_id, score_a, score_b) sorted by delta magnitude."""
    if not (run_a / "scores").exists():
        LOGGER.warning("No scores directory in %s", run_a)
        return []

    corpora = (
        [corpus_filter]
        if corpus_filter
        else sorted(d.name for d in (run_a / "scores").iterdir() if d.is_dir())
    )

    cases = []
    for corpus in corpora:
        score_dir_a = run_a / "scores" / corpus
        score_dir_b = run_b / "scores" / corpus
        if not score_dir_a.exists():
            continue
        for score_path_a in sorted(score_dir_a.glob("*.json")):
            score_a = _load_doc_score(score_path_a, field, method)
            score_b = _load_doc_score(score_dir_b / score_path_a.name, field, method)
            if score_a is None or score_b is None:
                continue
            delta = score_a - score_b
            if mode == "regression" and delta < 0:
                cases.append((delta, corpus, score_path_a.stem, score_a, score_b))
            elif mode == "improvement" and delta > 0:
                cases.append((delta, corpus, score_path_a.stem, score_a, score_b))

    cases.sort(key=lambda x: x[0], reverse=mode == "improvement")
    return cases


def _extract_texts(
    corpus: str,
    record_id: str,
    run_a: Path,
    run_b: Path,
    data_dir: Path,
    split: str,
    field: str,
    xml_mapping: dict,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    gold_path = data_dir / split / corpus / f"{record_id}.jats.xml"
    pred_path_a = run_a / "predictions" / corpus / f"{record_id}.tei.xml"
    pred_path_b = run_b / "predictions" / corpus / f"{record_id}.tei.xml"
    gold_text = _extract_field_text(gold_path, field, xml_mapping)
    text_a = _extract_field_text(pred_path_a, field, xml_mapping)
    text_b = _extract_field_text(pred_path_b, field, xml_mapping)
    return gold_text, text_a, text_b


def _print_case(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    delta: float,
    corpus: str,
    record_id: str,
    score_a: float,
    score_b: float,
    label_a: str,
    label_b: str,
    gold_text: Optional[str],
    text_a: Optional[str],
    text_b: Optional[str],
) -> None:
    width = max(len(label_a), len(label_b), 4)
    print(f"record : {corpus}/{record_id}  Δ={delta:+.3f}")
    if gold_text is not None:
        print(f"  {'gold':<{width}} : {gold_text}")
    print(f"  {label_a:<{width}} : {score_a:.3f} | {word_diff(gold_text, text_a)}")
    print(f"  {label_b:<{width}} : {score_b:.3f} | {word_diff(gold_text, text_b)}")
    print()


_PDFALTO_ENDPOINT = "/api/pdfalto"


def _pretty_print_xml(content: bytes) -> bytes:
    try:
        root = lxml_etree.fromstring(content)
        return lxml_etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    except lxml_etree.XMLSyntaxError:
        return content


def _fetch_pdfalto_xml(parser_url: str, pdf_path: Path) -> Optional[bytes]:
    """POST a PDF to the pdfalto endpoint and return the ALTO XML bytes, or None on failure."""
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{parser_url}{_PDFALTO_ENDPOINT}",
                files={"input": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
                timeout=60,
            )
            response.raise_for_status()
            return response.content
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.warning("Failed to fetch pdfalto XML for %s: %s", pdf_path.name, exc)
        return None


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy(src, dst)


def _export_case(
    out_dir: Path,
    record_id: str,
    corpus: str,
    field: str,
    gold_text: Optional[str],
    text_a: Optional[str],
    text_b: Optional[str],
    run_a: Path,
    run_b: Path,
    data_dir: Path,
    split: str,
    alto_xml: Optional[bytes] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix, text in [
        (f"{record_id}.gold.{field}.txt", gold_text),
        (f"{record_id}.run-a.{field}.txt", text_a),
        (f"{record_id}.run-b.{field}.txt", text_b),
    ]:
        (out_dir / suffix).write_text(text or "", encoding="utf-8")
    _copy_if_exists(
        data_dir / split / corpus / f"{record_id}.pdf", out_dir / f"{record_id}.pdf"
    )
    _copy_if_exists(
        data_dir / split / corpus / f"{record_id}.jats.xml",
        out_dir / f"{record_id}.jats.xml",
    )
    _copy_if_exists(
        run_a / "predictions" / corpus / f"{record_id}.tei.xml",
        out_dir / f"{record_id}.run-a.tei.xml",
    )
    _copy_if_exists(
        run_b / "predictions" / corpus / f"{record_id}.tei.xml",
        out_dir / f"{record_id}.run-b.tei.xml",
    )
    _copy_if_exists(
        run_a / "scores" / corpus / f"{record_id}.json",
        out_dir / f"{record_id}.run-a.scores.json",
    )
    _copy_if_exists(
        run_b / "scores" / corpus / f"{record_id}.json",
        out_dir / f"{record_id}.run-b.scores.json",
    )
    if alto_xml is not None:
        (out_dir / f"{record_id}.alto.xml").write_bytes(_pretty_print_xml(alto_xml))


def run_show_cases(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    run_a: Path,
    run_b: Path,
    field: str,
    method: str,
    corpus: Optional[str],
    mode: str,
    data_dir: Path,
    split: str,
    limit: Optional[int],
    parser_url: Optional[str] = None,
) -> None:
    register_functions()
    xml_mapping = parse_xml_mapping(DEFAULT_XML_MAPPING_PATH)

    label_a = _run_label(run_a)
    label_b = _run_label(run_b)
    comp_label = _comparison_label(run_b)

    cases = find_cases(run_a, run_b, field, method, corpus, mode)
    corpus_label = corpus or "all corpora"
    print(f"{len(cases)} {mode}s: {field}/{method} ({corpus_label})")
    print(f"  run-a ({label_a}): {run_a}")
    print(f"  run-b ({label_b}): {run_b}")
    print()

    to_show = cases[:limit] if limit is not None else cases
    examples_base = run_a / "examples" / f"vs-{comp_label}" / mode / field / method
    for delta, corp, record_id, score_a, score_b in to_show:
        gold_text, text_a, text_b = _extract_texts(
            corp, record_id, run_a, run_b, data_dir, split, field, xml_mapping,
        )
        alto_xml = None
        if parser_url:
            pdf_path = data_dir / split / corp / f"{record_id}.pdf"
            if pdf_path.exists():
                alto_xml = _fetch_pdfalto_xml(parser_url, pdf_path)
        _print_case(
            delta, corp, record_id, score_a, score_b,
            label_a, label_b, gold_text, text_a, text_b,
        )
        _export_case(
            examples_base / corp, record_id, corp, field,
            gold_text, text_a, text_b,
            run_a, run_b, data_dir, split,
            alto_xml=alto_xml,
        )

    if to_show:
        print(f"Exported {len(to_show)} examples to {examples_base}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Show per-document regressions or improvements between two runs"
    )
    parser.add_argument("--run-a", required=True, type=Path, help="Primary run directory")
    parser.add_argument("--run-b", required=True, type=Path, help="Baseline run directory")
    parser.add_argument("--field", required=True, help="Field name, e.g. title")
    parser.add_argument(
        "--method", default="edit_sim", help="Scoring method (default: edit_sim)"
    )
    parser.add_argument("--corpus", default=None, help="Corpus filter (default: all)")
    parser.add_argument("--mode", required=True, choices=["regression", "improvement"])
    parser.add_argument("--data", default="benchmarks/data", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--parser-url", default=None,
        help="Parser URL for on-demand pdfalto XML export (e.g. http://localhost:8080)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    run_show_cases(
        run_a=args.run_a,
        run_b=args.run_b,
        field=args.field,
        method=args.method,
        corpus=args.corpus,
        mode=args.mode,
        data_dir=args.data,
        split=args.split,
        limit=args.limit,
        parser_url=args.parser_url,
    )


if __name__ == "__main__":
    main()
