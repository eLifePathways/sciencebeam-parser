"""The columns we add on top of GROBID's layout, derived from the training TEI.

A raw feature file written by our own generator already carries them. GROBID's
does not -- its segmentation rows stop at the column its own features end on --
so when its corpus is read they come off the TEI line the row was matched to,
which is where ours came from in the first place.
"""

import logging
from typing import Callable, Iterator, List, Mapping, Optional, Sequence, Tuple

from sciencebeam_parser.models.data import LabeledLayoutToken
from sciencebeam_parser.models.segmentation.data import format_feature_text
from sciencebeam_parser.training.grobid_column_layout import GrobidColumnLayout


LOGGER = logging.getLogger(__name__)

WHOLE_LINE_TEXT_COLUMN = 'whole_line_text'


def _get_whole_line_text(labeled_layout_token: LabeledLayoutToken) -> Optional[str]:
    line_text = labeled_layout_token.layout_token.line_meta.text
    if not line_text:
        return None
    return format_feature_text(line_text) or None


EXTRA_COLUMN_VALUE_GETTER_BY_NAME: Mapping[
    str, Callable[[LabeledLayoutToken], Optional[str]]
] = {
    WHOLE_LINE_TEXT_COLUMN: _get_whole_line_text
}


def is_missing_extra_columns(
    column_layout: GrobidColumnLayout,
    feature_column_count: int
) -> bool:
    """Whether a raw row stops where GROBID's own features do."""
    if not column_layout.extra_columns:
        return False
    expected = len(column_layout.get_data_generator_column_names()) - 1
    return feature_column_count == expected - len(column_layout.extra_columns)


def get_extra_column_values(
    column_layout: GrobidColumnLayout,
    labeled_layout_token: LabeledLayoutToken
) -> Optional[List[str]]:
    """The extra column values for one row, or None where one cannot be had.

    A row without them is not emitted: an empty field would be read as a column
    of its own, and a placeholder would be a value the model learns from.
    """
    values: List[str] = []
    for column_name in column_layout.extra_columns:
        getter = EXTRA_COLUMN_VALUE_GETTER_BY_NAME.get(column_name)
        if getter is None:
            raise ValueError(
                'no value can be derived for extra column %r of %r' % (
                    column_name, column_layout.name
                )
            )
        value = getter(labeled_layout_token)
        if not value:
            return None
        values.append(value)
    return values


def iter_rows_with_extra_columns(
    column_layout: GrobidColumnLayout,
    feature_rows: Sequence[Sequence[str]],
    row_tokens: Sequence[str],
    labeled_layout_tokens: Sequence[LabeledLayoutToken]
) -> Iterator[Tuple[List[str], str, LabeledLayoutToken]]:
    """Yield the rows that can carry the columns, whole.

    The token travels with its own row rather than beside the result: a row this
    drops would otherwise leave every row after it holding the token of the one
    before.
    """
    for feature_row, row_token, labeled_layout_token in zip(
        feature_rows, row_tokens, labeled_layout_tokens
    ):
        extra_values = get_extra_column_values(column_layout, labeled_layout_token)
        if extra_values is None:
            continue
        yield list(feature_row) + extra_values, row_token, labeled_layout_token
