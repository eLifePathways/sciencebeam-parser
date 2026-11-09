"""The letters a comparison's tables head their columns with, and the legend naming them."""
from __future__ import annotations

import string
from typing import List, Sequence, Tuple


def _column_keys(n_columns: int) -> List[str]:
    """A letter per column, in the order the columns are declared."""
    return [
        string.ascii_uppercase[index] if index < len(string.ascii_uppercase)
        else f"V{index + 1}"
        for index in range(n_columns)
    ]


def keyed(labeled_summaries: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    """The summaries under their column letters, which is what the tables print.

    A label is long enough -- an image tag, a build timestamp and a profile -- that
    a table headed by them, twice over for the deltas, is mostly header.
    """
    keys = _column_keys(len(labeled_summaries))
    return [(key, summary) for key, (_, summary) in zip(keys, labeled_summaries)]


def render_legend(
    labels: Sequence[str], primary: int, described: Sequence[Tuple[str, str]] = ()
) -> List[str]:
    """Which variant each column letter stands for, and what it is where a comparison
    file says so. Not collapsed: the tables cannot be read without it.

    A label says what distinguishes a column, which is not the same as saying which
    profile produced it, and the file that knows is somewhere the reader is not.
    """
    if not labels:
        return []
    what_by_label = dict(described)
    with_what = any(what_by_label.get(label) for label in labels)
    reference = primary % len(labels)
    rows = []
    for index, (key, label) in enumerate(zip(_column_keys(len(labels)), labels)):
        cells = [key, label]
        if with_what:
            cells.append(what_by_label.get(label, ""))
        cells.append(
            "deltas are measured against this"
            if index == reference and len(labels) > 1 else ""
        )
        rows.append("| " + " | ".join(cells) + " |")
    header = ["", "Variant"] + (["What it is"] if with_what else []) + [""]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
        *rows,
        "",
    ]
