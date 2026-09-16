"""What a prediction file is called, and how to find one.

`.tei.xml` from the GROBID-compatible tools, `.jats.xml` from an annotation
model. Defined once, so the store and the scorer cannot disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Tuple

# Longest-first, so that stripping a suffix cannot leave part of another behind.
PREDICTION_SUFFIXES: Tuple[str, ...] = (".jats.xml", ".tei.xml")


def is_prediction_file(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in PREDICTION_SUFFIXES)


def record_id_from_name(name: str) -> str:
    """The record id a prediction file is named for."""
    for suffix in PREDICTION_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def record_id_from_path(path: Path) -> str:
    return record_id_from_name(path.name)


def iter_prediction_files(directory: Path) -> Iterator[Path]:
    """Every prediction in a directory, ordered by record id."""
    if not directory.is_dir():
        return iter(())
    found = [path for path in directory.iterdir() if is_prediction_file(path.name)]
    return iter(sorted(found, key=record_id_from_path))


def iter_prediction_names(names: Iterable[str]) -> Iterator[str]:
    """The prediction files among a listing of bare names."""
    return (name for name in names if is_prediction_file(name))
