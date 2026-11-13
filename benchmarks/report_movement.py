"""Which fields a comparison moved."""
from __future__ import annotations

from typing import List, Sequence, Set, Tuple

from benchmarks.report_grid import GridRow

# A smaller difference prints as +0.000, so a row it separated would read as unchanged.
MOVED_THRESHOLD = 0.001


def row_moved(
    row: GridRow, primary: int = -1, threshold: float = MOVED_THRESHOLD
) -> bool:
    """Whether any column differs from the reference by at least `threshold`."""
    values = row.values
    reference = values[primary]
    for index, value in enumerate(values):
        if index == primary % len(values):
            continue
        if (reference is None) != (value is None):
            return True
        if reference is not None and value is not None:
            if abs(reference - value) >= threshold:
                return True
    return False


def moved_fields(
    rows: Sequence[GridRow], primary: int = -1, threshold: float = MOVED_THRESHOLD
) -> Set[str]:
    """The fields with at least one row that moved."""
    return {
        row.field for row in rows if row_moved(row, primary, threshold)
    }


def split_rows(
    rows: Sequence[GridRow], primary: int = -1, threshold: float = MOVED_THRESHOLD
) -> Tuple[List[GridRow], List[GridRow]]:
    """The rows of the fields that moved, then the rest, each in the grid's order."""
    moved = moved_fields(rows, primary, threshold)
    return (
        [row for row in rows if row.field in moved],
        [row for row in rows if row.field not in moved],
    )
