from sciencebeam_parser.document.layout_document import LayoutLineMeta, LayoutToken
from sciencebeam_parser.models.data import LabeledLayoutToken
from sciencebeam_parser.training.extra_feature_columns import (
    get_extra_column_values,
    is_missing_extra_columns,
    iter_rows_with_extra_columns
)
from sciencebeam_parser.training.grobid_column_layout import (
    get_grobid_column_layout_for_model_name
)


SEGMENTATION_LAYOUT = get_grobid_column_layout_for_model_name('segmentation')


def _labeled_token(token_text: str, line_text) -> LabeledLayoutToken:
    return LabeledLayoutToken(
        label='B-<body>',
        layout_token=LayoutToken(
            text=token_text,
            line_meta=LayoutLineMeta(line_id=1, text=line_text)
        )
    )


class TestIsMissingExtraColumns:
    def test_should_recognise_a_row_that_stops_where_grobid_features_do(self):
        assert is_missing_extra_columns(SEGMENTATION_LAYOUT, 32) is True

    def test_should_not_claim_a_row_that_already_carries_them(self):
        assert is_missing_extra_columns(SEGMENTATION_LAYOUT, 33) is False


class TestGetExtraColumnValues:
    def test_should_format_the_line_so_it_stays_one_field(self):
        assert get_extra_column_values(
            SEGMENTATION_LAYOUT, _labeled_token('one', 'one two')
        ) == ['one two']

    def test_should_give_nothing_for_a_line_it_cannot_read(self):
        assert get_extra_column_values(
            SEGMENTATION_LAYOUT, _labeled_token('one', None)
        ) is None


class TestIterRowsWithExtraColumns:
    def test_should_keep_each_token_with_its_own_row(self):
        rows = list(iter_rows_with_extra_columns(
            SEGMENTATION_LAYOUT,
            [['a'], ['b'], ['c']],
            ['token_a', 'token_b', 'token_c'],
            [
                _labeled_token('token_a', 'token_a line'),
                _labeled_token('token_b', 'token_b line'),
                _labeled_token('token_c', 'token_c line'),
            ]
        ))
        assert [(row, token) for row, token, _ in rows] == [
            (['a', 'token_a line'], 'token_a'),
            (['b', 'token_b line'], 'token_b'),
            (['c', 'token_c line'], 'token_c'),
        ]

    def test_should_not_shift_a_token_onto_a_later_row_when_one_is_dropped(self):
        # the middle line cannot supply the column, so its row goes; the row
        # after it has to keep its own token rather than inherit the dropped
        # row's
        rows = list(iter_rows_with_extra_columns(
            SEGMENTATION_LAYOUT,
            [['a'], ['b'], ['c']],
            ['token_a', 'token_b', 'token_c'],
            [
                _labeled_token('token_a', 'token_a line'),
                _labeled_token('token_b', None),
                _labeled_token('token_c', 'token_c line'),
            ]
        ))
        assert [(row, token) for row, token, _ in rows] == [
            (['a', 'token_a line'], 'token_a'),
            (['c', 'token_c line'], 'token_c'),
        ]
