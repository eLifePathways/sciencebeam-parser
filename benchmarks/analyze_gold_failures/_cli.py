"""
Gold-label failure mode analysis for a specific benchmark field.

Given a field name (e.g. body_section_titles) and a ScienceBeam benchmark run,
this script:
  1. Pre-scans all documents to find those with gold JATS annotations for the field
  2. For each document, assigns a failure mode to every gold value:
       - Not found in raw text: absent from the ScienceBeam raw TEI output
       - Extraction failed: present in raw text but not predicted into the correct field
       - Partial/wrong match: extracted but similarity below threshold
       - Correct: extracted with sufficient similarity
  3. Optionally fetches ScienceBeam Parser model data (--parser-url) to attribute
     EXTRACTION_FAILED values to a specific model layer in the pipeline
  4. Aggregates results and writes a markdown report plus per-mode detail files

Usage:
    python -m benchmarks.analyze_gold_failures \\
        --field body_section_titles \\
        --run benchmarks/runs/train \\
        --out benchmarks/runs/train/gold-failure-analysis/body_section_titles/edit_sim
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from sciencebeam_judge.parsing.xml import parse_xml, parse_xml_mapping
from sciencebeam_judge.parsing.xpath.xpath_functions import register_functions
from sciencebeam_judge.resources import DEFAULT_XML_MAPPING_PATH

from benchmarks.analyze_field_regressions._models import (
    FIELD_MODEL,
    MODEL_RELEVANT_LABELS,
    _get_model_chain,
)

from ._aggregate import aggregate_results
from ._attribution import PARSER_DEFAULT_URL, attribute_failures, fetch_model_data
from ._modes import DEFAULT_SIMILARITY_THRESHOLD, assign_failure_modes
from ._report import render_mode_detail_reports, render_report
from ._screenshots import collect_tei_snippets, render_context_screenshots
from ._types import DocumentSummary, FailureMode

LOGGER = logging.getLogger(__name__)


def _format_eta(seconds: float) -> str:
    if seconds >= 3600:
        return f'{seconds / 3600:.1f}h'
    if seconds >= 60:
        return f'{int(seconds) // 60}m{int(seconds) % 60:02d}s'
    return f'{seconds:.0f}s'


def _extract_field_text(xml_path: Path, field: str, xml_mapping: dict) -> Optional[str]:
    if not xml_path.exists():
        return None
    try:
        values = parse_xml(BytesIO(xml_path.read_bytes()), xml_mapping, fields=[field])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.warning('Failed to parse %s: %s', xml_path.name, exc)
        return None
    items = values.get(field, [])
    return ' | '.join(str(v) for v in items) if items else None


def _parse_pipe_separated(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [v.strip() for v in text.split(' | ') if v.strip()]


def _load_sb_score(
    run: Path,
    corpus: str,
    record_id: str,
    field: str,
    method: str = 'edit_sim',
) -> Optional[float]:
    score_path = run / 'scores' / corpus / f'{record_id}.json'
    if not score_path.exists():
        return None
    data = json.loads(score_path.read_text(encoding='utf-8'))
    ms = data.get('fields', {}).get(field, {}).get(method, {})
    if not ms:
        return None
    if 'sim_sum' in ms:
        ec = ms.get('expected_count', 0)
        pc = ms.get('predicted_count', 0)
        precision = ms['sim_sum'] / pc if pc else 0.0
        recall = ms['sim_sum'] / ec if ec else 0.0
        return (2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return ms.get('f1')


def _find_all_documents(
    run: Path,
    data_dir: Path,
    split: str,
    corpus_filter: Optional[str],
) -> Iterator[Tuple[str, str]]:
    pred_root = run / 'predictions'
    if not pred_root.exists():
        LOGGER.warning('No predictions directory in run: %s', run)
        return

    corpora = (
        [corpus_filter]
        if corpus_filter
        else sorted(d.name for d in pred_root.iterdir() if d.is_dir())
    )
    for corpus in corpora:
        corpus_dir = pred_root / corpus
        gold_dir = data_dir / split / corpus
        for tei_file in sorted(corpus_dir.glob('*.tei.xml')):
            record_id = tei_file.stem.removesuffix('.tei')
            if (gold_dir / f'{record_id}.jats.xml').exists():
                yield corpus, record_id
            else:
                LOGGER.debug('No gold JATS for %s/%s, skipping', corpus, record_id)


def _prefilter_documents(
    doc_ids: List[Tuple[str, str]],
    data_dir: Path,
    split: str,
    field: str,
    xml_mapping: dict,
) -> Tuple[List[Tuple[str, str, List[str]]], int]:
    """Return (docs_with_gold_values, n_skipped).

    Parses each gold JATS upfront so we know the true analysis count and
    avoid re-parsing the same file during _analyze_document.
    """
    total = len(doc_ids)
    print(
        f'Pre-scanning {total} document(s) for gold {field!r} annotations...',
        file=sys.stderr,
    )
    docs_with_gold: List[Tuple[str, str, List[str]]] = []
    n_skipped = 0
    for corpus, record_id in doc_ids:
        gold_path = data_dir / split / corpus / f'{record_id}.jats.xml'
        gold_text = _extract_field_text(gold_path, field, xml_mapping)
        gold_values = _parse_pipe_separated(gold_text)
        if gold_values:
            docs_with_gold.append((corpus, record_id, gold_values))
        else:
            LOGGER.debug('No gold %r for %s/%s', field, corpus, record_id)
            n_skipped += 1
    return docs_with_gold, n_skipped


def _write_doc_artifacts(
    doc_dir: Path,
    record_id: str,
    field: str,
    gold_text: str,
    sb_field_text: Optional[str],
    gold_path: Optional[Path] = None,
    pred_path: Optional[Path] = None,
    pdf_path: Optional[Path] = None,
) -> None:
    doc_dir.mkdir(parents=True, exist_ok=True)
    sentinel = doc_dir / '.artifacts_exported'
    if sentinel.exists():
        return
    (doc_dir / f'{record_id}.gold.{field}.txt').write_text(gold_text, encoding='utf-8')
    (doc_dir / f'{record_id}.sb.{field}.txt').write_text(sb_field_text or '', encoding='utf-8')
    for src in (gold_path, pred_path):
        if src and src.exists():
            shutil.copy2(src, doc_dir / src.name)
    if pdf_path and pdf_path.exists():
        pdf_link = doc_dir / pdf_path.name
        if not pdf_link.exists():
            pdf_link.symlink_to(pdf_path.resolve())
    sentinel.touch()


def _analyze_document(  # pylint: disable=too-many-locals
    corpus: str,
    record_id: str,
    gold_values: List[str],
    run: Path,
    data_dir: Path,
    split: str,
    field: str,
    xml_mapping: dict,
    out_dir: Path,
    model_chain: List[str],
    relevant_labels: dict,
    parser_url: Optional[str],
    similarity_threshold: float,
    method: str,
) -> DocumentSummary:
    gold_path = data_dir / split / corpus / f'{record_id}.jats.xml'
    pred_path = run / 'predictions' / corpus / f'{record_id}.tei.xml'
    pdf_path = data_dir / split / corpus / f'{record_id}.pdf'

    raw_tei_text = (
        pred_path.read_text(encoding='utf-8', errors='replace')
        if pred_path.exists() else None
    )
    sb_field_text = _extract_field_text(pred_path, field, xml_mapping)

    _write_doc_artifacts(
        out_dir / 'by-doc' / corpus / record_id,
        record_id, field,
        gold_text=' | '.join(gold_values),
        sb_field_text=sb_field_text,
        gold_path=gold_path,
        pred_path=pred_path if pred_path.exists() else None,
        pdf_path=pdf_path if pdf_path.exists() else None,
    )

    results = assign_failure_modes(
        gold_values, raw_tei_text, sb_field_text, similarity_threshold
    )
    score_sb = _load_sb_score(run, corpus, record_id, field, method)

    doc = DocumentSummary(
        corpus=corpus,
        record_id=record_id,
        score_sb=score_sb,
        results=results,
    )

    if parser_url:
        extraction_failed = [r.value for r in results if r.mode == FailureMode.EXTRACTION_FAILED]
        if extraction_failed and model_chain:
            doc_dir = out_dir / 'by-doc' / corpus / record_id
            model_data = fetch_model_data(pdf_path, model_chain, parser_url, doc_dir)
            if model_data:
                doc.attributions = attribute_failures(
                    extraction_failed, model_chain, relevant_labels, model_data
                )

    return doc


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--field', required=True,
        help=f'Benchmark field to analyze. Known fields: {sorted(FIELD_MODEL)}',
    )
    parser.add_argument('--run', required=True, type=Path,
                        help='ScienceBeam benchmark run directory')
    parser.add_argument('--data', default=Path('benchmarks/data'), type=Path,
                        help='Benchmark data directory (default: benchmarks/data)')
    parser.add_argument('--split', default='train',
                        help='Dataset split (default: train)')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output directory for the report')
    parser.add_argument('--corpus', default=None,
                        help='Restrict analysis to a single corpus (default: all)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of documents to analyze')
    parser.add_argument(
        '--parser-url', default=None, dest='parser_url',
        help=(
            f'ScienceBeam Parser URL for pipeline attribution (default: off). '
            f'Example: {PARSER_DEFAULT_URL}'
        ),
    )
    parser.add_argument(
        '--threshold', type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
        help=(
            f'Similarity threshold for CORRECT classification '
            f'(default: {DEFAULT_SIMILARITY_THRESHOLD})'
        ),
    )
    parser.add_argument(
        '--method', default='edit_sim',
        help='Scoring method used to load benchmark scores (default: edit_sim)',
    )
    args = parser.parse_args()

    if args.field not in FIELD_MODEL:
        sys.exit(
            f'Unknown field: {args.field!r}. '
            f'Known fields: {sorted(FIELD_MODEL)}'
        )

    if not (args.run / 'predictions').exists():
        sys.exit(f'No predictions directory in run: {args.run}')

    register_functions()
    xml_mapping = parse_xml_mapping(DEFAULT_XML_MAPPING_PATH)

    model_chain = _get_model_chain(args.field)
    relevant_labels = MODEL_RELEVANT_LABELS.get(args.field, {})

    all_doc_ids = list(_find_all_documents(args.run, args.data, args.split, args.corpus))
    if args.limit:
        all_doc_ids = all_doc_ids[:args.limit]

    docs_to_analyze, n_skipped = _prefilter_documents(
        all_doc_ids, args.data, args.split, args.field, xml_mapping
    )
    skipped_note = (
        f' ({n_skipped} skipped — no gold annotations for {args.field!r})'
        if n_skipped else ''
    )
    print(
        f'Found {len(docs_to_analyze)}/{len(all_doc_ids)} document(s)'
        f' with gold {args.field!r} annotations{skipped_note}',
        file=sys.stderr,
    )

    if not docs_to_analyze:
        print('No documents with gold field data found.', file=sys.stderr)
        sys.exit(0)

    doc_summaries = []
    total = len(docs_to_analyze)
    t_start = time.monotonic()
    print(f'Analyzing {total} document(s)...', file=sys.stderr)
    for n, (corpus, record_id, gold_values) in enumerate(docs_to_analyze, 1):
        t0 = time.monotonic()
        doc = _analyze_document(
            corpus=corpus,
            record_id=record_id,
            gold_values=gold_values,
            run=args.run,
            data_dir=args.data,
            split=args.split,
            field=args.field,
            xml_mapping=xml_mapping,
            out_dir=args.out,
            model_chain=model_chain,
            relevant_labels=relevant_labels,
            parser_url=args.parser_url,
            similarity_threshold=args.threshold,
            method=args.method,
        )
        doc_summaries.append(doc)
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        elapsed = time.monotonic() - t_start
        rate = n / elapsed if elapsed > 0 else 0.0
        remaining = total - n
        eta = _format_eta(remaining / rate) if rate > 0 and remaining > 0 else '—'
        print(
            f'[{n}/{total}] {corpus}/{record_id}'
            f' {elapsed_ms}ms'
            f' | {rate:.1f}/s | ~{eta} left',
            file=sys.stderr,
        )

    summary = aggregate_results(
        doc_summaries=doc_summaries,
        model_chain=model_chain,
        field=args.field,
        run_sb=str(args.run),
        is_online=bool(args.parser_url),
        n_docs_skipped=n_skipped,
        method=args.method,
    )

    # Generate screenshots and collect TEI snippets before rendering context_window.md
    # so the markdown only includes image refs that actually exist.
    screenshots: Dict[str, bytes] = {}
    tei_snippets: Dict[str, str] = {}
    if summary.is_online:
        screenshot_items = [
            (doc.corpus, doc.record_id, value, attr.context_window, attr.predicted_label)
            for doc in doc_summaries
            for value, attr in doc.attributions.items()
            if attr.first_failed_model is not None
        ]
        screenshots = render_context_screenshots(args.out, screenshot_items)
        tei_snippets = collect_tei_snippets(args.out, screenshot_items)

    report = render_report(summary, doc_summaries)
    detail_reports = render_mode_detail_reports(
        summary, doc_summaries,
        available_screenshots=set(screenshots),
        tei_snippets=tei_snippets,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / 'report.md'
    report_path.write_text(report, encoding='utf-8')
    for filename, content in detail_reports.items():
        (args.out / filename).write_text(content, encoding='utf-8')
    for relpath, png_bytes in screenshots.items():
        out_path = args.out / relpath
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png_bytes)
    if screenshots:
        print(f'Wrote {len(screenshots)} screenshot(s)', file=sys.stderr)

    print(f'Report written to {report_path}', file=sys.stderr)
    print(report)
