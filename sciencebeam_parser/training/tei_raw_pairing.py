"""Pair the rows of a raw feature file with the labelled lines of its training TEI.

Training data we generate ourselves writes both files in one pass, so row `n` is
line `n` and a positional pairing is correct. GROBID's own corpora are not like
that: the TEI has been hand-corrected since the raw was written, so the two drift
and a pairing has to search.

`SegmentationTrainer.addFeaturesSegmentation` is what produced the published
training files, so this reproduces it rather than improving on it: the published
file is only an oracle for as long as the pairing does what GROBID's did.
"""

import logging
from typing import List, NamedTuple, Optional, Sequence


LOGGER = logging.getLogger(__name__)


# GROBID compares, then gives up once `pp - q > 5`, so the seventh candidate is
# the last one tried.
DEFAULT_MAX_LOOKAHEAD = 6

# GROBID abandons the rest of a document after this many consecutive rows it
# could not place, and discards a document whose run of failures at the end
# reaches DEFAULT_MAX_TRAILING_UNMATCHED.
DEFAULT_MAX_CONSECUTIVE_UNMATCHED = 20
DEFAULT_MAX_TRAILING_UNMATCHED = 10


class PairedRow(NamedTuple):
    raw_index: int
    labelled_index: int


class PairingResult(NamedTuple):
    paired_rows: Sequence[PairedRow]
    raw_row_count: int
    labelled_line_count: int
    unmatched_row_count: int
    """Rows inside the stretch the TEI covers that matched no line."""
    uncovered_row_count: int
    """Rows after the TEI ran out, which GROBID emits nothing for either."""
    is_document_discarded: bool

    @property
    def matched_row_count(self) -> int:
        return len(self.paired_rows)

    @property
    def covered_row_count(self) -> int:
        """Rows the TEI had lines to offer, matched or not."""
        return self.matched_row_count + self.unmatched_row_count

    @property
    def matched_ratio(self) -> float:
        """Of the rows the TEI covers, the share that found their line.

        Not over every raw row: a TEI the annotator stopped part way through
        leaves the rest of the file uncovered, and GROBID emits nothing for it
        either. `133108v1` is 356 rows of a 50,065-row raw file and every one of
        them matched, so the pairing is sound and the document is simply short.
        """
        if not self.covered_row_count:
            return 0.0
        return self.matched_row_count / self.covered_row_count

    @property
    def uncovered_ratio(self) -> float:
        """The share of the raw file the TEI stops short of."""
        if not self.raw_row_count:
            return 0.0
        return self.uncovered_row_count / self.raw_row_count


def get_match_key_for_labelled_line(
    token_text: str,
    line_text: Optional[str],
    line_as_token: bool
) -> str:
    """The first field of the raw row this labelled item should match.

    For a line-level model it is the line's first whitespace-delimited field,
    which is what GROBID matches on and what our own generator writes: over our
    86 segmentation documents it is the raw row's first column on every one of
    100,794 rows, where the tokenizer's first token is on 76.5% of them --
    `LIMIX: genetic` leads with `LIMIX:` in the raw and `LIMIX` after tokenizing.

    Taking both would match more lines than GROBID does, and on a page of
    repeated values -- a results table of `N/A` and `0.16` -- matching one line
    too many walks the cursor off the rows that follow.
    """
    if line_as_token and line_text:
        fields = line_text.split()
        if fields:
            return fields[0]
    return token_text


def get_paired_rows(  # pylint: disable=too-many-locals
    raw_row_tokens: Sequence[str],
    labelled_match_keys: Sequence[str],
    max_lookahead: int = DEFAULT_MAX_LOOKAHEAD,
    max_consecutive_unmatched: int = DEFAULT_MAX_CONSECUTIVE_UNMATCHED,
    max_trailing_unmatched: int = DEFAULT_MAX_TRAILING_UNMATCHED
) -> PairingResult:
    """One row per raw row, in order, matched by a forward-only cursor.

    The cursor advances past a line only when a row matched it, so a row that
    matches nothing leaves it where it was for the next row to try.
    """
    paired_rows: List[PairedRow] = []
    cursor = 0
    consecutive_unmatched = 0
    unmatched_row_count = 0
    uncovered_row_count = 0
    for raw_index, raw_token in enumerate(raw_row_tokens):
        if cursor >= len(labelled_match_keys):
            # the TEI ran out; GROBID's search body never runs again, so the rest
            # of the raw file produces nothing and nothing is counted against it
            uncovered_row_count = len(raw_row_tokens) - raw_index
            break
        matched = False
        for labelled_index in range(
            cursor, min(cursor + max_lookahead + 1, len(labelled_match_keys))
        ):
            if raw_token == labelled_match_keys[labelled_index]:
                paired_rows.append(PairedRow(raw_index, labelled_index))
                cursor = labelled_index + 1
                consecutive_unmatched = 0
                matched = True
                break
        if matched:
            continue
        unmatched_row_count += 1
        consecutive_unmatched += 1
        if consecutive_unmatched > max_consecutive_unmatched:
            LOGGER.debug(
                'abandoning the document after %d consecutive unmatched rows,'
                ' at raw row %d', consecutive_unmatched, raw_index
            )
            uncovered_row_count = len(raw_row_tokens) - raw_index - 1
            break
    return PairingResult(
        paired_rows=paired_rows,
        raw_row_count=len(raw_row_tokens),
        labelled_line_count=len(labelled_match_keys),
        unmatched_row_count=unmatched_row_count,
        uncovered_row_count=uncovered_row_count,
        is_document_discarded=consecutive_unmatched >= max_trailing_unmatched
    )
