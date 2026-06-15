from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from benchmarks.show_cases import export_case, extract_texts, find_cases

from ._types import FieldPresenceSummary, RegressionCase

LOGGER = logging.getLogger(__name__)


def _find_regression_cases(
    run_a: Path,
    run_b: Path,
    analysis_field: str,
    method: str,
    limit: Optional[int],
) -> Tuple[List[RegressionCase], int]:
    cases_raw = find_cases(
        run_a=run_a,
        run_b=run_b,
        field=analysis_field,
        method=method,
        corpus_filter=None,
        mode='regression',
    )
    all_cases = [
        RegressionCase(delta, corpus, record_id, score_a, score_b)
        for delta, corpus, record_id, score_a, score_b in cases_raw
    ]
    total = len(all_cases)
    return all_cases[:limit] if limit else all_cases, total


def _normalize_for_comparison(text: str) -> str:
    return re.sub(r'\s+', '', text)


def _get_field_presence(
    gold_text: str,
    raw_tei_path: Path,
) -> Optional[List[Tuple[str, bool]]]:
    if not raw_tei_path.exists():
        return None
    raw_normalized = _normalize_for_comparison(
        raw_tei_path.read_text(encoding='utf-8', errors='replace')
    )
    values = [v.strip() for v in gold_text.split(' | ') if v.strip()]
    return [(v, _normalize_for_comparison(v) in raw_normalized) for v in values]


def _write_field_presence(
    doc_dir: Path,
    record_id: str,
    analysis_field: str,
    presence: List[Tuple[str, bool, bool, bool]],
) -> None:
    rows = [
        '| Value | In GROBID | In raw text | In ScienceBeam |',
        '|-------|:---------:|:-----------:|:--------------:|',
    ]
    for value, in_grobid, in_raw, in_sb in presence:
        rows.append(
            f'| {value}'
            f' | {"yes" if in_grobid else "no"}'
            f' | {"yes" if in_raw else "no"}'
            f' | {"yes" if in_sb else "no"} |'
        )
    (doc_dir / f'{record_id}.{analysis_field}.presence.md').write_text(
        '\n'.join(rows) + '\n', encoding='utf-8'
    )


def _build_presence(
    gold_text: str,
    sb_tei: Path,
    grobid_tei: Path,
    sb_field_file: Path,
) -> Optional[List[Tuple[str, bool, bool, bool]]]:
    sb_presence = _get_field_presence(gold_text, sb_tei)
    grobid_presence = _get_field_presence(gold_text, grobid_tei)
    sb_field_text = sb_field_file.read_text(encoding='utf-8') if sb_field_file.exists() else ''
    if sb_presence is None and grobid_presence is None and not sb_field_text:
        return None
    sb_map = dict(sb_presence or [])
    grobid_map = dict(grobid_presence or [])
    sb_field_norm = _normalize_for_comparison(sb_field_text)
    gold_values = [v.strip() for v in gold_text.split(' | ') if v.strip()]
    return [
        (v, grobid_map.get(v, False), sb_map.get(v, False),
         bool(_normalize_for_comparison(v) in sb_field_norm))
        for v in gold_values
    ]


def _compute_presence_summary(
    case: RegressionCase,
    out_dir: Path,
    analysis_field: str,
    run_a: Path,
    run_b: Path,
) -> Optional[FieldPresenceSummary]:
    gold_file = (
        out_dir / 'by-doc' / case.corpus / case.record_id
        / f'{case.record_id}.gold.{analysis_field}.txt'
    )
    gold_text = gold_file.read_text(encoding='utf-8') if gold_file.exists() else ''
    if not gold_text:
        return None
    doc_dir = out_dir / 'by-doc' / case.corpus / case.record_id
    sb_tei = run_a / 'predictions' / case.corpus / f'{case.record_id}.tei.xml'
    grobid_tei = run_b / 'predictions' / case.corpus / f'{case.record_id}.tei.xml'
    sb_field_file = doc_dir / f'{case.record_id}.run-a.{analysis_field}.txt'
    three_way = _build_presence(gold_text, sb_tei, grobid_tei, sb_field_file)
    if three_way is None:
        return None
    _write_field_presence(doc_dir, case.record_id, analysis_field, three_way)
    return FieldPresenceSummary(
        in_grobid=sum(1 for _, g, _, _ in three_way if g),
        in_raw=sum(1 for _, _, r, _ in three_way if r),
        in_sb=sum(1 for _, _, _, s in three_way if s),
        total=len(three_way),
    )


def _export_doc_examples(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    case: RegressionCase,
    run_a: Path,
    run_b: Path,
    data_dir: Path,
    split: str,
    analysis_field: str,
    xml_mapping: dict,
    out_dir: Path,
) -> None:
    doc_dir = out_dir / 'by-doc' / case.corpus / case.record_id
    sentinel = doc_dir / '.examples_exported'
    if sentinel.exists():
        return
    try:
        gold_text, text_a, text_b = extract_texts(
            case.corpus, case.record_id, run_a, run_b,
            data_dir, split, analysis_field, xml_mapping,
        )
        export_case(
            doc_dir, case.record_id, case.corpus, analysis_field,
            gold_text, text_a, text_b,
            run_a, run_b, data_dir, split,
        )
        sentinel.touch()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.warning(
            'Failed to export examples for %s/%s: %s',
            case.corpus, case.record_id, exc,
        )
