"""Which fields a comparison moved, against the one column that answers for it."""
from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from benchmarks.report_grid import GridRow

# A smaller difference prints as +0.000, so a row it separated would read as unchanged.
MOVED_THRESHOLD = 0.001


def default_baseline(n_columns: int, primary: int) -> Optional[int]:
    """The only column movement could mean, where there is one."""
    if n_columns != 2:
        return None
    return 1 - primary % n_columns


def row_moved(
    row: GridRow, baseline: int, primary: int = -1,
    threshold: float = MOVED_THRESHOLD,
) -> bool:
    """Whether the reference differs from the baseline by at least `threshold`."""
    reference = row.values[primary]
    value = row.values[baseline]
    if (reference is None) != (value is None):
        return True
    if reference is None or value is None:
        return False
    return abs(reference - value) >= threshold


def moved_fields(
    rows: Sequence[GridRow], baseline: Optional[int], primary: int = -1,
    threshold: float = MOVED_THRESHOLD,
) -> Set[str]:
    """The fields with at least one row that moved, empty where nothing answers for it."""
    if baseline is None:
        return set()
    return {
        row.field for row in rows if row_moved(row, baseline, primary, threshold)
    }


def split_rows(
    rows: Sequence[GridRow], baseline: Optional[int], primary: int = -1,
    threshold: float = MOVED_THRESHOLD,
) -> Tuple[List[GridRow], List[GridRow]]:
    """The rows of the fields that moved, then the rest, each in the grid's order."""
    moved = moved_fields(rows, baseline, primary, threshold)
    return (
        [row for row in rows if row.field in moved],
        [row for row in rows if row.field not in moved],
    )
