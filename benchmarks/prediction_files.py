"""What a prediction file is called, and how to find one.

`.tei.xml` from the GROBID-compatible tools, `.jats.xml` from an annotation
model. Defined once, so the store and the scorer cannot disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

# Longest-first, so that stripping a suffix cannot leave part of another behind.
PREDICTION_SUFFIXES: Tuple[str, ...] = (".jats.xml", ".tei.xml")

SUFFIX_BY_FORMAT: Dict[str, str] = {"jats": ".jats.xml", "tei": ".tei.xml"}
PREDICTION_FORMATS: Tuple[str, ...] = tuple(SUFFIX_BY_FORMAT)


def is_prediction_file(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in PREDICTION_SUFFIXES)


def prediction_format_from_name(name: str) -> Optional[str]:
    for prediction_format, suffix in SUFFIX_BY_FORMAT.items():
        if name.endswith(suffix):
            return prediction_format
    return None


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


def iter_scored_prediction_files(
    directory: Path, prediction_format: Optional[str] = None
) -> Iterator[Path]:
    """One prediction per record: those in `prediction_format`, or the only one a record has."""
    if prediction_format is not None:
        suffix = SUFFIX_BY_FORMAT[prediction_format]
        return (path for path in iter_prediction_files(directory) if path.name.endswith(suffix))
    paths = list(iter_prediction_files(directory))
    record_ids = [record_id_from_path(path) for path in paths]
    duplicated = sorted({record_id for record_id in record_ids if record_ids.count(record_id) > 1})
    if duplicated:
        raise ValueError(
            f"{directory} holds more than one format for {duplicated[:3]}; name the format to score"
        )
    return iter(paths)
