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
import logging
import sys
from pathlib import Path

from sciencebeam_judge.parsing.xml import parse_xml_mapping
from sciencebeam_judge.parsing.xpath.xpath_functions import register_functions
from sciencebeam_judge.resources import DEFAULT_XML_MAPPING_PATH

from ._aggregate import _aggregate_model_results
from ._cases import (
    _compute_presence_summary,
    _export_doc_examples,
    _find_regression_cases,
)
from ._fetch import (
    GROBID_DEFAULT_URL,
    PARSER_DEFAULT_URL,
    _check_service,
    _load_sbparser_models,
)
from ._loop import DEFAULT_CONCURRENCY, _resolve_concurrency, _run_analysis_loop
from ._models import FIELD_MODEL, MODEL_RELEVANT_LABELS, _get_model_chain
from ._report import _generate_report

LOGGER = logging.getLogger(__name__)


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

    field_model_labels = MODEL_RELEVANT_LABELS.get(args.field, {})
    model_summaries = [
        _aggregate_model_results(
            model_name,
            model_doc_results[model_name],
            relevant_labels=field_model_labels.get(model_name),
        )
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
