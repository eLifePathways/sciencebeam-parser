"""
What the layout noise filter removes, as text.

Runs the layout pipeline over a set of PDFs, applies the noise filter, and reports every
block it would remove — with the page and position it sat at, and whether its text appears
in the publisher's JATS. Nothing downstream of segmentation runs, so no sequence model is
loaded and no gold is consulted.

Usage:
    python -m benchmarks.analyze_noise_filter \\
        --source-path 'benchmarks/data/train/*/*.pdf' \\
        --source-xml-path 'benchmarks/data/train/*/*.jats.xml' \\
        --out .temp/noise-filter/train
"""
import argparse
import logging
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

from sciencebeam_parser.app.parser import ScienceBeamParser
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.document.layout_noise_filter import LayoutNoiseFilterConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE

from ._collect import analyze_documents
from ._jats import index_jats_filenames_by_stem
from ._report import write_report

LOGGER = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 40


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--source-path', required=True,
                        help='Glob of PDFs; the parent directory names the corpus')
    parser.add_argument('--source-xml-path',
                        help='Glob of matching JATS XML, enabling the precision alarm')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output directory for report.md and the JSONL files')
    parser.add_argument('--limit', type=int, help='Process at most this many PDFs')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                        help=f'Removed blocks shown per corpus (default: {DEFAULT_SAMPLE_SIZE})')

    defaults = LayoutNoiseFilterConfig()
    group = parser.add_argument_group('filter thresholds')
    group.add_argument('--repetition-fraction', type=float,
                       default=defaults.repetition_fraction)
    group.add_argument('--position-consistency-fraction', type=float,
                       default=defaults.position_consistency_fraction)
    group.add_argument('--max-position-stddev', type=float,
                       default=defaults.max_position_stddev)
    group.add_argument('--max-height-ratio', type=float,
                       default=defaults.max_height_ratio)
    group.add_argument('--preserve-first-page-head', action='store_true',
                       default=defaults.preserve_first_page_head)
    group.add_argument('--preserve-first-page-foot', action='store_true',
                       default=defaults.preserve_first_page_foot)
    group.add_argument('--filter-outside-main-area', action='store_true',
                       default=defaults.filter_outside_main_area,
                       help='Also filter outside the page main area on a repeating pattern '
                            'or a line carrying no letters')
    group.add_argument('--min-repeating-pattern-length', type=int,
                       default=defaults.min_repeating_pattern_length)
    group.add_argument('--max-letterless-length', type=int,
                       default=defaults.max_letterless_length)
    return parser.parse_args(argv)


def get_noise_filter_config(args: argparse.Namespace) -> LayoutNoiseFilterConfig:
    return LayoutNoiseFilterConfig(
        enabled=True,
        repetition_fraction=args.repetition_fraction,
        position_consistency_fraction=args.position_consistency_fraction,
        max_position_stddev=args.max_position_stddev,
        max_height_ratio=args.max_height_ratio,
        preserve_first_page_head=args.preserve_first_page_head,
        preserve_first_page_foot=args.preserve_first_page_foot,
        filter_outside_main_area=args.filter_outside_main_area,
        min_repeating_pattern_length=args.min_repeating_pattern_length,
        max_letterless_length=args.max_letterless_length
    )


def run(args: argparse.Namespace) -> None:
    pdf_filenames = sorted(glob(args.source_path))
    if args.limit:
        pdf_filenames = pdf_filenames[:args.limit]
    if not pdf_filenames:
        raise SystemExit(f'no PDFs matched: {args.source_path!r}')
    LOGGER.info('PDFs: %d', len(pdf_filenames))

    jats_filename_by_stem = None
    if args.source_xml_path:
        jats_filename_by_stem = index_jats_filenames_by_stem(sorted(glob(args.source_xml_path)))
        LOGGER.info('JATS XML files: %d', len(jats_filename_by_stem))

    noise_filter_config = get_noise_filter_config(args)
    LOGGER.info('noise filter config: %r', noise_filter_config)

    sciencebeam_parser = ScienceBeamParser.from_config(AppConfig.load_yaml(DEFAULT_CONFIG_FILE))
    summaries, removed = analyze_documents(
        pdf_filenames,
        sciencebeam_parser=sciencebeam_parser,
        noise_filter_config=noise_filter_config,
        jats_filename_by_stem=jats_filename_by_stem
    )
    report_path = write_report(
        args.out,
        summaries=summaries,
        removed=removed,
        config=noise_filter_config,
        sample_size=args.sample_size
    )
    LOGGER.info('report written to %s', report_path)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stderr
    )
    run(parse_args(argv))
