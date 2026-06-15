#!/usr/bin/env python
"""
Automated GROBID feature parity analysis for a specific benchmark field.

Given a field name (e.g. reference_doi) and two benchmark runs (sbeam vs grobid baseline),
this script:
  1. Finds all regression documents for the field (sbeam F1 < grobid F1)
  2. Fetches model data (GROBID + parser) for each model in the field's hierarchy
  3. Runs find-important feature analysis per document per model
  4. Aggregates results and writes a markdown report

Requires GROBID and the ScienceBeam Parser to be running locally.

Usage:
    python -m benchmarks.analyze_field_regressions \\
        --field reference_doi \\
        --run-a benchmarks/runs/train \\
        --run-b benchmarks/runs/baselines/grobid/0.9.0-crf/default/train \\
        --out benchmarks/runs/train/field-analysis/reference_doi
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from sciencebeam_judge.parsing.xml import parse_xml_mapping
from sciencebeam_judge.parsing.xpath.xpath_functions import register_functions
from sciencebeam_judge.resources import DEFAULT_XML_MAPPING_PATH

from benchmarks.show_cases import export_case, extract_texts, find_cases
from sciencebeam_parser.app.parser import ScienceBeamParser
from sciencebeam_parser.service.server import get_app_config
from sciencebeam_parser.utils.feature_importance import find_important_data
from sciencebeam_parser.utils.model_data_diff import format_model_data_diff

LOGGER = logging.getLogger(__name__)

FIELD_RELEVANT_LABELS: Dict[str, frozenset] = {
    'reference_doi':        frozenset({'<pubnum>', '<web>'}),
    'reference_title':      frozenset({'<title>'}),
    'title':                frozenset({'<title>'}),
    'abstract':             frozenset({'<abstract>'}),
    'keywords':             frozenset({'<keyword>'}),
    'author_full_names':    frozenset({'<forenames>', '<surname>', '<author>'}),
    'affiliation_text':     frozenset({'<affiliation>'}),
    'body_section_titles':  frozenset({'<section>'}),
    'acknowledgement':      frozenset({'<acknowledgement>'}),
    'first_reference_text': frozenset({'<references>', '<reference>'}),
}

FIELD_MODEL: Dict[str, str] = {
    'title':                'header',
    'abstract':             'header',
    'author_full_names':    'name-header',
    'affiliation_text':     'affiliation-address',
    'keywords':             'header',
    'body_section_titles':  'fulltext',
    'acknowledgement':      'fulltext',
    'first_reference_text': 'reference-segmenter',
    'reference_title':      'citation',
    'reference_doi':        'citation',
}

MODEL_PARENT: Dict[str, str] = {
    'header':              'segmentation',
    'name-header':         'header',
    'affiliation-address': 'header',
    'fulltext':            'segmentation',
    'reference-segmenter': 'segmentation',
    'citation':            'reference-segmenter',
}


def _get_model_chain(analysis_field: str) -> List[str]:
    chain = []
    model: Optional[str] = FIELD_MODEL[analysis_field]
    while model is not None:
        chain.append(model)
        model = MODEL_PARENT.get(model)
    return list(reversed(chain))


GROBID_DEFAULT_URL = 'http://localhost:8070'
PARSER_DEFAULT_URL = 'http://localhost:8080'


@dataclass
class FieldPresenceSummary:
    in_grobid: int
    in_raw: int
    in_sb: int
    total: int


@dataclass
class RegressionCase:
    delta: float
    corpus: str
    record_id: str
    score_a: float
    score_b: float
    presence: Optional[FieldPresenceSummary] = None


@dataclass
class FeatureSummary:
    feature: str
    total_label_changes: int
    docs_affected: int
    transitions: Counter = field(default_factory=Counter)


@dataclass
class ModelSummary:
    model: str
    docs_analyzed: int
    docs_failed: int
    docs_with_feature_diffs: int
    features: List[FeatureSummary]


def _check_service(url: str, name: str) -> None:
    try:
        r = httpx.get(f'{url}/api/isalive', timeout=5)
        r.raise_for_status()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        sys.exit(f'Error: {name} not reachable at {url}: {exc}')


def _fetch_grobid_model_data(
    pdf_path: Path,
    model_name: str,
    grobid_url: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f'{grobid_url}/api/processFulltextDocument',
            data={'debugMode': 'true', 'models': model_name},
            files={'input': (pdf_path.name, pdf_path.read_bytes(), 'application/pdf')},
        )
        r.raise_for_status()
    lines = [
        line.replace('\t', ' ')
        for line in r.text.splitlines()
        if not line.startswith('=== model:')
    ]
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _fetch_parser_model_data(
    pdf_path: Path,
    model_name: str,
    parser_url: str,
    data_path: Path,
    feature_names_path: Path,
) -> None:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f'{parser_url}/api/models/{model_name}',
            params={'output_format': 'data'},
            headers={'accept': 'application/json'},
            files={'input': (pdf_path.name, pdf_path.read_bytes(), 'application/pdf')},
        )
        r.raise_for_status()
        data_path.write_bytes(r.content)

        r2 = client.get(f'{parser_url}/api/models/{model_name}/feature-names')
        r2.raise_for_status()
        feature_names_path.write_bytes(r2.content)


def _load_sbparser_models(model_chain: List[str]) -> Dict[str, object]:
    config = get_app_config()
    sb_parser = ScienceBeamParser.from_config(config)
    models: Dict[str, object] = {}
    for model_name in model_chain:
        sb_name = model_name.replace('-', '_')
        try:
            models[model_name] = sb_parser.fulltext_models.get_sequence_model_by_name(sb_name)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            sys.exit(f'Failed to load model {model_name!r}: {exc}')
    return models


def _write_diff(
    parser_data: Path,
    grobid_data: Path,
    feature_names_file: Path,
    diff_out: Path,
) -> None:
    feature_names = json.loads(
        feature_names_file.read_text(encoding='utf-8')
    )['feature_names']
    diff_text = format_model_data_diff(
        parser_data.read_text(encoding='utf-8'),
        grobid_data.read_text(encoding='utf-8'),
        feature_names,
    )
    diff_out.write_text(diff_text, encoding='utf-8')


def _analyze_doc_model(
    record_id: str,
    model_name: str,
    model,
    pdf_path: Path,
    grobid_url: str,
    parser_url: str,
    doc_dir: Path,
) -> Optional[dict]:
    """Fetch model data and run find-important for one doc/model pair. Returns JSON data or None."""
    grobid_data = doc_dir / 'grobid' / f'{model_name}.data'
    parser_data = doc_dir / 'sciencebeam-parser' / f'{model_name}.data'
    feature_names_file = doc_dir / 'sciencebeam-parser' / f'{model_name}.feature_names.json'
    json_out = doc_dir / f'{model_name}.find_important.json'
    diff_out = doc_dir / f'{model_name}.diff'

    if json_out.exists():
        LOGGER.info('Using cached result for %s / %s', record_id, model_name)
        if not diff_out.exists() and parser_data.exists() and grobid_data.exists():
            LOGGER.info('Generating missing diff for %s / %s', record_id, model_name)
            _write_diff(parser_data, grobid_data, feature_names_file, diff_out)
        return json.loads(json_out.read_text(encoding='utf-8'))

    if not pdf_path.exists():
        LOGGER.warning('PDF not found: %s', pdf_path)
        return None

    LOGGER.info('Fetching GROBID model data for %s / %s', record_id, model_name)
    _fetch_grobid_model_data(pdf_path, model_name, grobid_url, grobid_data)

    LOGGER.info('Fetching parser model data for %s / %s', record_id, model_name)
    _fetch_parser_model_data(pdf_path, model_name, parser_url, parser_data, feature_names_file)

    LOGGER.info('Running find-important for %s / %s', record_id, model_name)
    result = find_important_data(
        str(parser_data), str(grobid_data), str(feature_names_file), model
    )

    json_out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    _write_diff(parser_data, grobid_data, feature_names_file, diff_out)
    return result


_OTHER_LABELS = frozenset({'<other>', 'I-<other>'})


def _is_meaningful_label_change(sbeam_label: str, grobid_label: str) -> bool:
    """Return False for <other> ↔ I-<other> transitions, which are semantically equivalent."""
    return not (sbeam_label in _OTHER_LABELS and grobid_label in _OTHER_LABELS)


def _aggregate_model_results(
    model_name: str,
    doc_results: List[Tuple[str, Optional[dict]]],
) -> ModelSummary:
    feature_changes: Counter = Counter()
    feature_doc_count: Counter = Counter()
    feature_transitions: Dict[str, Counter] = {}

    docs_analyzed = 0
    docs_failed = 0
    docs_with_feature_diffs = 0

    for _record_id, result in doc_results:
        if result is None:
            docs_failed += 1
            continue
        docs_analyzed += 1
        if result.get('features_with_diffs'):
            docs_with_feature_diffs += 1
        for feat, fdata in result.get('per_feature', {}).items():
            meaningful = [
                tok for tok in fdata.get('changed_tokens', [])
                if _is_meaningful_label_change(tok['sbeam_label'], tok['grobid_label'])
            ]
            if meaningful:
                feature_changes[feat] += len(meaningful)
                feature_doc_count[feat] += 1
                if feat not in feature_transitions:
                    feature_transitions[feat] = Counter()
                for tok in meaningful:
                    feature_transitions[feat][(tok['sbeam_label'], tok['grobid_label'])] += 1

    summaries = [
        FeatureSummary(
            feature=feat,
            total_label_changes=feature_changes[feat],
            docs_affected=feature_doc_count[feat],
            transitions=feature_transitions.get(feat, Counter()),
        )
        for feat in sorted(feature_changes, key=lambda f: feature_changes[f], reverse=True)
    ]

    return ModelSummary(
        model=model_name,
        docs_analyzed=docs_analyzed,
        docs_failed=docs_failed,
        docs_with_feature_diffs=docs_with_feature_diffs,
        features=summaries,
    )


def _base_label(label: str) -> str:
    """Strip BIO prefix (B-/I-) from a label tag."""
    if label.startswith(('B-', 'I-')):
        return label[2:]
    return label


def _label_is_relevant(label: str, relevant_labels: frozenset) -> bool:
    return label in relevant_labels or _base_label(label) in relevant_labels


def _pair_is_relevant(s: str, g: str, relevant_labels: frozenset) -> bool:
    return _label_is_relevant(s, relevant_labels) or _label_is_relevant(g, relevant_labels)


def _has_relevant_transition(fs: FeatureSummary, relevant_labels: frozenset) -> bool:
    return any(_pair_is_relevant(s, g, relevant_labels) for s, g in fs.transitions)


def _render_feature_table(
    features: List[FeatureSummary],
    docs_analyzed: int,
    relevant_labels: Optional[frozenset] = None,
) -> List[str]:
    rows: List[str] = [
        '| Feature | Total Δlabels | Docs affected | Transition | Count |',
        '|---------|------:|------:|-----------|------:|',
    ]
    for fs in features:
        all_trans = fs.transitions.most_common()
        if relevant_labels:
            rel = [
                ((s, g), n) for (s, g), n in all_trans
                if _pair_is_relevant(s, g, relevant_labels)
            ]
            others = [
                ((s, g), n) for (s, g), n in all_trans
                if not _pair_is_relevant(s, g, relevant_labels)
            ]
            sorted_trans = rel + others
        else:
            sorted_trans = all_trans
        for i, ((s, g), n) in enumerate(sorted_trans):
            cell = f'`{s} → {g}`'
            if relevant_labels and _pair_is_relevant(s, g, relevant_labels):
                cell = f'**{cell}**'
            if i == 0:
                rows.append(
                    f'| {fs.feature} | {fs.total_label_changes}'
                    f' | {fs.docs_affected}/{docs_analyzed}'
                    f' | {cell} | {n} |'
                )
            else:
                rows.append(f'| | | | {cell} | {n} |')
        if not sorted_trans:
            rows.append(
                f'| {fs.feature} | {fs.total_label_changes}'
                f' | {fs.docs_affected}/{docs_analyzed} | — | |'
            )
    return rows


def _render_docs_table(
    cases: List[RegressionCase],
) -> List[str]:
    show_presence = any(c.presence is not None for c in cases)
    header = '| Doc | Corpus | GROBID | ScienceBeam | Δ |'
    sep = '|-----|--------|------:|------:|--:|'
    if show_presence:
        header += ' Gold | In GROBID | In raw text | In ScienceBeam |'
        sep += '---:|:---:|:---:|:---:|'
    header += ' Examples |'
    sep += '---------|'
    rows: List[str] = ['## Analyzed documents', '', header, sep]
    for case in cases:
        rel = Path('by-doc') / case.corpus / case.record_id
        cells = [
            case.record_id, case.corpus,
            f'{case.score_b:.2f}', f'{case.score_a:.2f}', f'{case.delta:+.2f}',
        ]
        if show_presence:
            p = case.presence
            cells += (
                [str(p.total), str(p.in_grobid), str(p.in_raw), str(p.in_sb)]
                if p is not None else ['-', '-', '-', '-']
            )
        cells.append(f'[{case.record_id}]({rel}/)')
        rows.append('| ' + ' | '.join(cells) + ' |')
    if show_presence:
        rows.append('')
        rows.append(
            '_Presence counts: how many gold values (out of total) appear in each prediction.'
            ' "In raw text" uses the raw ScienceBeam TEI (no external lookup).'
            ' Values absent from the raw text may have been retrieved'
            ' via external metadata lookup._'
        )
    return rows


def _generate_report(
    analysis_field: str,
    run_a: Path,
    run_b: Path,
    total_regressions: int,
    cases: List[RegressionCase],
    model_summaries: List[ModelSummary],
) -> str:
    relevant_labels = FIELD_RELEVANT_LABELS.get(analysis_field)
    regression_note = (
        f'{total_regressions} total, {len(cases)} analyzed'
        if len(cases) < total_regressions
        else str(total_regressions)
    )
    lines: List[str] = []
    lines += [
        f'# Field Analysis: {analysis_field}',
        '',
        f'- **Run A (ScienceBeam)**: `{run_a}`',
        f'- **Run B (GROBID)**: `{run_b}`',
        f'- **Regression documents**: {regression_note}',
        '',
    ]

    lines += _render_docs_table(cases)
    lines.append('')

    for ms in model_summaries:
        lines.append(
            f'## {ms.model} ({ms.docs_analyzed} docs analyzed'
            + (f', {ms.docs_failed} failed' if ms.docs_failed else '')
            + f', {ms.docs_with_feature_diffs} with feature diffs)'
        )
        lines.append('')
        if not ms.features:
            if ms.docs_with_feature_diffs:
                lines += ['_Features differ but caused no meaningful label changes._', '']
            else:
                lines += ['_No feature differences found._', '']
            continue

        if relevant_labels:
            labels_str = ', '.join(f'`{lbl}`' for lbl in sorted(relevant_labels))
            rel_feats = [
                fs for fs in ms.features if _has_relevant_transition(fs, relevant_labels)
            ]
            other_feats = [
                fs for fs in ms.features if not _has_relevant_transition(fs, relevant_labels)
            ]

            lines.append(f'### Features affecting {labels_str}')
            lines.append('')
            if rel_feats:
                lines += _render_feature_table(rel_feats, ms.docs_analyzed, relevant_labels)
            else:
                lines.append(f'_No features directly affect {labels_str} in these documents._')
            lines.append('')

            if other_feats:
                lines.append('### Other feature differences')
                lines.append('')
                lines += _render_feature_table(other_feats, ms.docs_analyzed)
                lines.append('')
        else:
            lines += _render_feature_table(ms.features, ms.docs_analyzed)
            lines.append('')

    return '\n'.join(lines)


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


DEFAULT_CONCURRENCY = 0  # 0 = auto: max(2, cpu_count)


def _resolve_concurrency(concurrency: int) -> int:
    if concurrency == 0:
        return max(2, os.cpu_count() or 2)
    return concurrency


def _format_eta(seconds: float) -> str:
    if seconds >= 3600:
        return f'{seconds / 3600:.1f}h'
    if seconds >= 60:
        return f'{int(seconds) // 60}m{int(seconds) % 60:02d}s'
    return f'{seconds:.0f}s'


class _AnalysisProgress:
    def __init__(self, total: int) -> None:
        self.total = total
        self._done = 0
        self._lock = threading.Lock()
        self._t_start = time.monotonic()

    def record(
        self, corpus: str, record_id: str, model_name: str, elapsed_ms: int, ok: bool
    ) -> None:
        with self._lock:
            self._done += 1
            done = self._done
        elapsed = time.monotonic() - self._t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - done
        eta = _format_eta(remaining / rate) if rate > 0 else '?'
        print(
            f'[{done}/{self.total}] {corpus}/{record_id} {model_name}'
            f' {"ok" if ok else "err"} {elapsed_ms}ms'
            f' | {rate:.1f}/s | ~{eta} left',
            file=sys.stderr,
        )


def _run_analysis_loop(  # pylint: disable=too-many-locals
    cases: List[RegressionCase],
    model_chain: List[str],
    sbparser_models: Dict[str, object],
    grobid_url: str,
    parser_url: str,
    data_dir: Path,
    split: str,
    out_dir: Path,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Dict[str, List[Tuple[str, Optional[dict]]]]:
    by_doc_dir = out_dir / 'by-doc'
    concurrency = _resolve_concurrency(concurrency)
    work = [
        (case, model_name)
        for case in cases
        for model_name in model_chain
    ]
    print(
        f'Analyzing {len(work)} tasks (concurrency={concurrency})',
        file=sys.stderr,
    )
    progress = _AnalysisProgress(total=len(work))

    def _run_one(
        case: RegressionCase, model_name: str
    ) -> Tuple[Optional[dict], int]:
        t0 = time.monotonic()
        result = None
        model = sbparser_models.get(model_name)
        if model is not None:
            pdf_path = data_dir / split / case.corpus / f'{case.record_id}.pdf'
            doc_dir = by_doc_dir / case.corpus / case.record_id
            try:
                result = _analyze_doc_model(
                    case.record_id, model_name, model,
                    pdf_path, grobid_url, parser_url,
                    doc_dir=doc_dir,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning(
                    'Failed %s/%s model=%s: %s',
                    case.corpus, case.record_id, model_name, exc,
                )
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        progress.record(case.corpus, case.record_id, model_name, elapsed_ms, result is not None)
        return result, elapsed_ms

    keyed: Dict[Tuple[str, str, str], Optional[dict]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(_run_one, case, model_name): (case, model_name)
            for case, model_name in work
        }
        for future in as_completed(future_map):
            case, model_name = future_map[future]
            result, _ = future.result()
            keyed[(model_name, case.corpus, case.record_id)] = result

    return {
        model_name: [
            (case.record_id, keyed.get((model_name, case.corpus, case.record_id)))
            for case in cases
        ]
        for model_name in model_chain
    }


def main() -> None:
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
    parser.add_argument('--run-a', required=True, type=Path, dest='run_a',
                        help='sbeam benchmark run directory')
    parser.add_argument('--run-b', required=True, type=Path, dest='run_b',
                        help='GROBID baseline run directory')
    parser.add_argument('--data', default=Path('benchmarks/data'), type=Path,
                        help='Benchmark data cache directory (default: benchmarks/data)')
    parser.add_argument('--split', default='train',
                        help='Dataset split (default: train)')
    parser.add_argument('--method', default='edit_sim',
                        help='Scoring method for regression detection (default: edit_sim)')
    parser.add_argument('--grobid-url', default=GROBID_DEFAULT_URL, dest='grobid_url',
                        help=f'GROBID URL (default: {GROBID_DEFAULT_URL})')
    parser.add_argument('--parser-url', default=PARSER_DEFAULT_URL, dest='parser_url',
                        help=f'Parser URL (default: {PARSER_DEFAULT_URL})')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output directory for the report')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of regression documents to analyze')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help='Number of concurrent analysis workers (0 = auto, default)')
    args = parser.parse_args()

    if args.field not in FIELD_MODEL:
        sys.exit(
            f'Unknown field: {args.field!r}. '
            f'Known fields: {sorted(FIELD_MODEL)}'
        )

    model_chain = _get_model_chain(args.field)

    if not (args.run_a / 'scores').exists():
        sys.exit(f'No scores directory in run-a: {args.run_a}')
    if not (args.run_b / 'scores').exists():
        sys.exit(f'No scores directory in run-b: {args.run_b}')

    _check_service(args.grobid_url, 'GROBID')
    _check_service(args.parser_url, 'ScienceBeam Parser')

    cases, total_regressions = _find_regression_cases(
        run_a=args.run_a,
        run_b=args.run_b,
        analysis_field=args.field,
        method=args.method,
        limit=args.limit,
    )

    print(
        f'Found {total_regressions} regression document(s) for field {args.field!r}'
        + (f', analyzing {len(cases)}.' if len(cases) < total_regressions else '.'),
        file=sys.stderr,
    )
    if not cases:
        print('Nothing to analyze.', file=sys.stderr)

    register_functions()
    xml_mapping = parse_xml_mapping(DEFAULT_XML_MAPPING_PATH)

    for case in cases:
        _export_doc_examples(
            case, args.run_a, args.run_b, args.data, args.split,
            args.field, xml_mapping, args.out,
        )

    for case in cases:
        case.presence = _compute_presence_summary(
            case, args.out, args.field, args.run_a, args.run_b,
        )

    LOGGER.info('Loading ScienceBeam Parser models: %s', model_chain)
    sbparser_models = _load_sbparser_models(model_chain)

    model_doc_results = _run_analysis_loop(
        cases=cases,
        model_chain=model_chain,
        sbparser_models=sbparser_models,
        grobid_url=args.grobid_url,
        parser_url=args.parser_url,
        data_dir=args.data,
        split=args.split,
        out_dir=args.out,
        concurrency=_resolve_concurrency(args.concurrency),
    )

    model_summaries = [
        _aggregate_model_results(model_name, model_doc_results[model_name])
        for model_name in model_chain
    ]

    report = _generate_report(
        analysis_field=args.field,
        run_a=args.run_a,
        run_b=args.run_b,
        total_regressions=total_regressions,
        cases=cases,
        model_summaries=model_summaries,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / 'report.md'
    report_path.write_text(report, encoding='utf-8')
    print(f'Report written to {report_path}', file=sys.stderr)
    print(report)


if __name__ == '__main__':
    main()
