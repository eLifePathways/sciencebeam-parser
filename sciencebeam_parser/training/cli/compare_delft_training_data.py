"""Compare delft training data we generated with a published corpus.

Training data reconstructed from someone else's corpus is only trustworthy if it
can be held against what they published from the same source. The published
files carry no document ids, so sequences are joined on their first rows, which
is a heuristic -- and a safe one here, because a bad join shows up as a low match
rate rather than as data nobody checked.

Run by hand: it downloads the published corpus.
"""

import argparse
import gzip
import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from sciencebeam_trainer_delft.utils.io import auto_download_input_file

from sciencebeam_parser.training.grobid_column_layout import (
    GrobidColumnLayout,
    get_grobid_column_layout_for_model_name
)


LOGGER = logging.getLogger(__name__)

DEFAULT_JOIN_ROW_COUNT = 5
DEFAULT_MAX_REPORTED_ROWS = 50


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'ScienceBeam Parser: Compare generated delft data with a published corpus'
    )
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument(
        '--generated-path', type=str, required=True,
        help='The delft training data to check.'
    )
    parser.add_argument(
        '--published-path', type=str, required=False,
        help=(
            'What to check it against (default: the model\'s recorded'
            ' reference_training_corpus).'
        )
    )
    parser.add_argument(
        '--join-row-count', type=int, default=DEFAULT_JOIN_ROW_COUNT,
        help='Rows whose first two columns identify a sequence.'
    )
    parser.add_argument(
        '--max-reported-rows', type=int, default=DEFAULT_MAX_REPORTED_ROWS,
        help='Differing rows to print before saying only how many are left.'
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args(argv)


def read_sequences(filename: str) -> List[List[str]]:
    sequences: List[List[str]] = []
    current: List[str] = []
    with auto_download_input_file(filename, auto_decompress=False) as local_file:
        opener = gzip.open if local_file.endswith('.gz') else open
        with opener(local_file, 'rt', encoding='utf-8') as fp:  # type: ignore
            for line in fp:
                if not line.strip():
                    if current:
                        sequences.append(current)
                        current = []
                    continue
                current.append(line.rstrip('\n'))
    if current:
        sequences.append(current)
    return sequences


def get_join_key(sequence: Sequence[str], join_row_count: int) -> Tuple:
    return tuple(
        tuple(row.split()[:2]) for row in sequence[:join_row_count]
    )


def get_row_without_extra_columns(row: str, layout: GrobidColumnLayout) -> str:
    """The row as the published corpus would hold it.

    Our own output may carry the columns this project adds, which sit between
    GROBID's last feature column and the label.
    """
    extra_column_count = len(layout.extra_columns)
    if not extra_column_count:
        return row
    fields = row.split(' ')
    expected = len(layout.get_training_data_column_names()) + 1
    if len(fields) != expected + extra_column_count:
        return row
    label_index = len(fields) - 1
    keep = (
        fields[:label_index - extra_column_count] + [fields[label_index]]
    )
    return ' '.join(keep)


def get_label(row: str) -> str:
    return row.rsplit(' ', 1)[-1]


def compare(  # pylint: disable=too-many-locals, too-many-statements
    generated: List[List[str]],
    published: List[List[str]],
    layout: GrobidColumnLayout,
    join_row_count: int,
    max_reported_rows: int
) -> int:
    published_by_key: Dict[Tuple, List[List[str]]] = {}
    for sequence in published:
        published_by_key.setdefault(
            get_join_key(sequence, join_row_count), []
        ).append(sequence)

    totals: Counter = Counter()
    differing: List[Tuple[int, str, str]] = []
    generated_labels: Counter = Counter()
    published_labels: Counter = Counter()
    for sequence in published:
        for row in sequence:
            published_labels[get_label(row)] += 1

    for index, sequence in enumerate(generated):
        rows = [get_row_without_extra_columns(row, layout) for row in sequence]
        for row in rows:
            generated_labels[get_label(row)] += 1
        totals['generated_rows'] += len(rows)
        candidates = published_by_key.get(get_join_key(rows, join_row_count))
        if not candidates:
            totals['sequences_not_joined'] += 1
            continue
        if len(candidates) > 1:
            totals['sequences_joined_ambiguously'] += 1
            continue
        counterpart = candidates[0]
        totals['sequences_joined'] += 1
        totals['joined_generated_rows'] += len(rows)
        totals['joined_published_rows'] += len(counterpart)
        # Our rows should be the published rows, in order, minus any we chose
        # not to emit; anything else is a row that differs.
        cursor = 0
        in_order = 0
        for row in rows:
            found = None
            for look in range(cursor, len(counterpart)):
                if counterpart[look] == row:
                    found = look
                    break
            if found is None:
                totals['rows_not_in_published'] += 1
                if len(differing) < max_reported_rows:
                    expected = (
                        counterpart[cursor] if cursor < len(counterpart) else ''
                    )
                    differing.append((index, row, expected))
                continue
            in_order += 1
            cursor = found + 1
        totals['rows_identical_and_in_order'] += in_order
        totals['published_rows_not_emitted'] += len(counterpart) - in_order
        if in_order == len(counterpart) == len(rows):
            totals['sequences_identical'] += 1

    LOGGER.info('generated sequences: %d', len(generated))
    LOGGER.info('published sequences: %d', len(published))
    LOGGER.info('generated rows: %d', totals['generated_rows'])
    LOGGER.info('published rows: %d', sum(len(s) for s in published))
    LOGGER.info(
        'sequences joined: %d (ambiguous %d, not joined %d)',
        totals['sequences_joined'], totals['sequences_joined_ambiguously'],
        totals['sequences_not_joined']
    )
    LOGGER.info('sequences identical: %d', totals['sequences_identical'])
    if totals['joined_generated_rows']:
        LOGGER.info(
            'rows identical and in order: %d of %d joined (%.5f)',
            totals['rows_identical_and_in_order'], totals['joined_generated_rows'],
            totals['rows_identical_and_in_order'] / totals['joined_generated_rows']
        )
    LOGGER.info('rows not in the published corpus: %d', totals['rows_not_in_published'])
    LOGGER.info(
        'published rows we do not emit: %d', totals['published_rows_not_emitted']
    )
    for label in sorted(set(generated_labels) | set(published_labels)):
        generated_count = generated_labels[label]
        published_count = published_labels[label]
        LOGGER.info(
            '  label %-24s generated %-9d published %-9d %+d',
            label, generated_count, published_count,
            generated_count - published_count
        )
    for sequence_index, row, expected in differing:
        LOGGER.warning('sequence %d differs', sequence_index)
        LOGGER.warning('  ours     : %s', row)
        LOGGER.warning('  published: %s', expected)
    remaining = totals['rows_not_in_published'] - len(differing)
    if remaining > 0:
        LOGGER.warning('%d further differing rows not printed', remaining)
    return totals['rows_not_in_published']


def run(args: argparse.Namespace) -> int:
    layout = get_grobid_column_layout_for_model_name(args.model_name)
    published_path = args.published_path
    if not published_path:
        if not layout.reference_training_corpus:
            raise ValueError(
                'no reference_training_corpus recorded for %r; pass'
                ' --published-path' % args.model_name
            )
        published_path = layout.reference_training_corpus[0]
    LOGGER.info('generated: %r', args.generated_path)
    LOGGER.info('published: %r', published_path)
    return compare(
        generated=read_sequences(args.generated_path),
        published=read_sequences(published_path),
        layout=layout,
        join_row_count=args.join_row_count,
        max_reported_rows=args.max_reported_rows
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.debug:
        for name in [__name__, 'sciencebeam_parser', 'sciencebeam_trainer_delft']:
            logging.getLogger(name).setLevel('DEBUG')
    return 1 if run(args) else 0


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    raise SystemExit(main())
