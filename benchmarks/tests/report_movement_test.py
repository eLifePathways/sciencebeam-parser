from __future__ import annotations

from benchmarks.report_grid import SCOPE_ALL, SCOPE_GOLD, GridRow
from benchmarks.report_movement import moved_fields, row_moved, split_rows


def _row(field: str, values, method: str = "levenshtein", scope: str = SCOPE_ALL):
    return GridRow(field, method, scope, scope == SCOPE_GOLD, 10, tuple(values))


class TestRowMoved:
    def test_equal_values_did_not_move(self):
        assert row_moved(_row("title", [0.5, 0.5])) is False

    def test_a_difference_below_the_threshold_did_not_move(self):
        assert row_moved(_row("title", [0.5005, 0.5])) is False

    def test_a_difference_at_the_threshold_moved(self):
        assert row_moved(_row("title", [0.501, 0.5])) is True

    def test_a_column_without_a_value_moved(self):
        assert row_moved(_row("title", [None, 0.5])) is True

    def test_a_row_no_column_answers_did_not_move(self):
        assert row_moved(_row("title", [None, None])) is False

    def test_compares_against_the_stated_reference(self):
        row = _row("title", [0.5, 0.9, 0.5])
        assert row_moved(row, primary=0) is True
        assert row_moved(row, primary=-1) is True
        assert row_moved(_row("title", [0.5, 0.5, 0.9]), primary=0) is True

    def test_the_threshold_is_settable(self):
        assert row_moved(_row("title", [0.52, 0.5]), threshold=0.05) is False


class TestMovedFields:
    def test_a_field_moves_where_any_of_its_rows_moved(self):
        rows = [
            _row("title", [0.5, 0.5], method="exact"),
            _row("title", [0.9, 0.5], method="levenshtein"),
            _row("abstract", [0.5, 0.5]),
        ]
        assert moved_fields(rows) == {"title"}

    def test_a_field_moves_where_only_its_gold_row_moved(self):
        rows = [
            _row("keywords", [0.5, 0.5]),
            _row("keywords", [0.9, 0.5], scope=SCOPE_GOLD),
        ]
        assert moved_fields(rows) == {"keywords"}


class TestSplitRows:
    def test_keeps_a_fields_rows_together(self):
        rows = [
            _row("title", [0.5, 0.5], method="exact"),
            _row("title", [0.9, 0.5], method="levenshtein"),
            _row("abstract", [0.5, 0.5]),
        ]
        moved, unmoved = split_rows(rows)
        assert [row.method for row in moved] == ["exact", "levenshtein"]
        assert [row.field for row in unmoved] == ["abstract"]

    def test_keeps_the_grids_order(self):
        rows = [
            _row("a", [0.9, 0.5]), _row("b", [0.5, 0.5]), _row("c", [0.9, 0.5]),
        ]
        moved, unmoved = split_rows(rows)
        assert [row.field for row in moved] == ["a", "c"]
        assert [row.field for row in unmoved] == ["b"]

    def test_everything_moving_leaves_nothing_behind(self):
        moved, unmoved = split_rows([_row("a", [0.9, 0.5])])
        assert [row.field for row in moved] == ["a"]
        assert unmoved == []
