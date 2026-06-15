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
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from benchmarks.show_cases import find_cases
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
class RegressionCase:
    delta: float
    corpus: str
    record_id: str
    score_a: float
    score_b: float


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
        f'- **Run A (sbeam)**: `{run_a}`',
        f'- **Run B (grobid)**: `{run_b}`',
        f'- **Regression documents**: {regression_note}',
        '',
    ]

    for ms in model_summaries:
        header = f'## {ms.model} ({ms.docs_analyzed} docs analyzed'
        if ms.docs_failed:
            header += f', {ms.docs_failed} failed'
        header += f', {ms.docs_with_feature_diffs} with feature diffs)'
        lines.append(header)
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


def _run_analysis_loop(  # pylint: disable=too-many-locals
    cases: List[RegressionCase],
    model_chain: List[str],
    sbparser_models: Dict[str, object],
    grobid_url: str,
    parser_url: str,
    data_dir: Path,
    split: str,
    out_dir: Path,
) -> Dict[str, List[Tuple[str, Optional[dict]]]]:
    model_doc_results: Dict[str, List[Tuple[str, Optional[dict]]]] = {
        m: [] for m in model_chain
    }
    total = len(cases) * len(model_chain)
    done = 0
    by_doc_dir = out_dir / 'by-doc'
    for case in cases:
        pdf_path = data_dir / split / case.corpus / f'{case.record_id}.pdf'
        doc_dir = by_doc_dir / case.record_id
        for model_name in model_chain:
            done += 1
            print(
                f'[{done}/{total}] {case.corpus}/{case.record_id} / {model_name}',
                file=sys.stderr,
            )
            model = sbparser_models.get(model_name)
            if model is None:
                model_doc_results[model_name].append((case.record_id, None))
                continue
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
                result = None
            model_doc_results[model_name].append((case.record_id, result))
    return model_doc_results


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
