from __future__ import annotations

from benchmarks.report_grid import SCOPE_ALL, SCOPE_GOLD, GridRow
from benchmarks.report_movement import (
    default_baseline,
    moved_fields,
    row_moved,
    split_rows,
)


def _row(field: str, values, method: str = "levenshtein", scope: str = SCOPE_ALL):
    return GridRow(field, method, scope, scope == SCOPE_GOLD, 10, tuple(values))


class TestDefaultBaseline:
    def test_two_columns_leave_one_candidate(self):
        assert default_baseline(2, -1) == 0
        assert default_baseline(2, 0) == 1

    def test_more_columns_have_no_obvious_candidate(self):
        assert default_baseline(3, -1) is None

    def test_one_column_has_nothing_to_compare_with(self):
        assert default_baseline(1, -1) is None


class TestRowMoved:
    def test_equal_values_did_not_move(self):
        assert row_moved(_row("title", [0.5, 0.5]), baseline=0) is False

    def test_a_difference_below_the_threshold_did_not_move(self):
        assert row_moved(_row("title", [0.5005, 0.5]), baseline=0) is False

    def test_a_difference_at_the_threshold_moved(self):
        assert row_moved(_row("title", [0.501, 0.5]), baseline=0) is True

    def test_a_column_without_a_value_moved(self):
        assert row_moved(_row("title", [None, 0.5]), baseline=0) is True

    def test_a_row_neither_column_answers_did_not_move(self):
        assert row_moved(_row("title", [None, None]), baseline=0) is False

    def test_only_the_baseline_column_is_compared(self):
        # The first column is another tool's; the run matches its baseline exactly.
        row = _row("title", [0.1, 0.5, 0.5])
        assert row_moved(row, baseline=1) is False
        assert row_moved(row, baseline=0) is True

    def test_the_threshold_is_settable(self):
        assert row_moved(_row("title", [0.52, 0.5]), baseline=0, threshold=0.05) is False


class TestMovedFields:
    def test_a_field_moves_where_any_of_its_rows_moved(self):
        rows = [
            _row("title", [0.5, 0.5], method="exact"),
            _row("title", [0.9, 0.5], method="levenshtein"),
            _row("abstract", [0.5, 0.5]),
        ]
        assert moved_fields(rows, baseline=0) == {"title"}

    def test_a_field_moves_where_only_its_gold_row_moved(self):
        rows = [
            _row("keywords", [0.5, 0.5]),
            _row("keywords", [0.9, 0.5], scope=SCOPE_GOLD),
        ]
        assert moved_fields(rows, baseline=0) == {"keywords"}

    def test_nothing_moves_where_no_column_answers_for_it(self):
        assert moved_fields([_row("title", [0.1, 0.5, 0.9])], baseline=None) == set()


class TestSplitRows:
    def test_keeps_a_fields_rows_together(self):
        rows = [
            _row("title", [0.5, 0.5], method="exact"),
            _row("title", [0.9, 0.5], method="levenshtein"),
            _row("abstract", [0.5, 0.5]),
        ]
        moved, unmoved = split_rows(rows, baseline=0)
        assert [row.method for row in moved] == ["exact", "levenshtein"]
        assert [row.field for row in unmoved] == ["abstract"]

    def test_keeps_the_grids_order(self):
        rows = [_row("a", [0.9, 0.5]), _row("b", [0.5, 0.5]), _row("c", [0.9, 0.5])]
        moved, unmoved = split_rows(rows, baseline=0)
        assert [row.field for row in moved] == ["a", "c"]
        assert [row.field for row in unmoved] == ["b"]

    def test_without_a_baseline_everything_stays_together(self):
        rows = [_row("a", [0.9, 0.5]), _row("b", [0.5, 0.5])]
        moved, unmoved = split_rows(rows, baseline=None)
        assert moved == []
        assert [row.field for row in unmoved] == ["a", "b"]


class TestThresholdMatchesTheDisplay:
    """A row the split calls unmoved never shows a visible delta."""

    def test_a_difference_that_prints_as_a_delta_moved(self):
        assert format(0.0007, "+.3f") == "+0.001"
        assert row_moved(_row("title", [0.5007, 0.5]), baseline=0) is True

    def test_a_difference_that_prints_as_zero_did_not(self):
        assert format(0.0004, "+.3f") == "+0.000"
        assert row_moved(_row("title", [0.5004, 0.5]), baseline=0) is False
