"""Cross-check the recorded GROBID column layout against GROBID's own corpora.

Run by hand: it downloads the reference corpora, so it is deliberately not part
of the test suite. The offline per-model test asserts that the data generators
still match what is recorded here; this asserts that what is recorded still
matches GROBID.
"""

import argparse
import logging
from collections import Counter
from itertools import islice
from typing import Dict, List, Optional, Tuple

from sciencebeam_trainer_delft.utils.io import auto_download_input_file

from sciencebeam_parser.training.grobid_column_layout import (
    GrobidColumnLayout,
    LabelSlot,
    load_grobid_column_layout_by_name
)


LOGGER = logging.getLogger(__name__)


DEFAULT_MAX_LINES = 200000


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'ScienceBeam Parser: Check the recorded GROBID column layout'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        action='append',
        help='Model to check (repeatable, defaults to every recorded model)'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        action='append',
        help=(
            'Corpus to check against instead of the recorded'
            ' reference_training_corpus (repeatable)'
        )
    )
    parser.add_argument(
        '--max-lines',
        type=int,
        default=DEFAULT_MAX_LINES,
        help='Token lines to read per corpus'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser.parse_args(argv)


def get_corpus_stats(filename: str, max_lines: int) -> Tuple[Counter, Counter]:
    column_counts: Counter = Counter()
    trailing_values: Counter = Counter()
    with auto_download_input_file(filename, auto_decompress=True) as local_file:
        with open(local_file, 'r', encoding='utf-8') as fp:
            for line in islice((line for line in fp if line.strip()), max_lines):
                columns = line.split()
                column_counts[len(columns)] += 1
                if len(columns) >= 2:
                    trailing_values[columns[-2]] += 1
    return column_counts, trailing_values


def check_corpus(
    layout: GrobidColumnLayout,
    filename: str,
    max_lines: int
) -> List[str]:
    expected_column_count = len(layout.get_training_data_column_names()) + 1
    column_counts, trailing_values = get_corpus_stats(filename, max_lines)
    problems: List[str] = []
    if not column_counts:
        problems.append('no token lines found')
    if set(column_counts) - {expected_column_count}:
        problems.append(
            'expected %d columns, found %r' % (
                expected_column_count, dict(sorted(column_counts.items()))
            )
        )
    if layout.label_slot == LabelSlot.UNFILLED and set(trailing_values) - {'0'}:
        problems.append(
            'label_slot is %r, but the column before the label is not always `0`: %r' % (
                layout.label_slot, dict(trailing_values.most_common(5))
            )
        )
    LOGGER.info(
        '%s: %s\n  columns=%r\n  column before the label=%r',
        layout.name, filename,
        dict(sorted(column_counts.items())),
        dict(trailing_values.most_common(5))
    )
    return problems


def run(args: argparse.Namespace) -> int:
    layout_by_name = load_grobid_column_layout_by_name()
    model_names = args.model_name or sorted(layout_by_name)
    if args.corpus and len(model_names) != 1:
        raise ValueError('--corpus applies to a single --model-name')
    problem_count = 0
    checked: Dict[Tuple[str, str, int], str] = {}
    for model_name in model_names:
        layout = layout_by_name[model_name]
        corpus_list = args.corpus or list(layout.reference_training_corpus)
        if not corpus_list:
            LOGGER.warning('%s: no reference corpus recorded', model_name)
            continue
        for filename in corpus_list:
            # models sharing a layout share a corpus; downloading it once is enough
            key = (filename, layout.label_slot, len(layout.get_training_data_column_names()))
            already_checked_for = checked.get(key)
            if already_checked_for:
                LOGGER.info('%s: same check as %s', model_name, already_checked_for)
                continue
            checked[key] = model_name
            for problem in check_corpus(layout, filename, max_lines=args.max_lines):
                problem_count += 1
                LOGGER.error('%s: %s: %s', model_name, filename, problem)
    if problem_count:
        LOGGER.error('%d problem(s) found', problem_count)
    else:
        LOGGER.info('recorded layout agrees with every corpus checked')
    return 1 if problem_count else 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.debug:
        for name in [__name__, 'sciencebeam_parser', 'sciencebeam_trainer_delft']:
            logging.getLogger(name).setLevel('DEBUG')
    return run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    raise SystemExit(main())
