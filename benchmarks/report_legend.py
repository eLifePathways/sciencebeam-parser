"""The short names a comparison's tables head their columns with, and the legend
naming them."""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


_MAX_INITIALS = 3
_TRAILING_PARENTHESIS = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
_WORD_SEPARATORS = re.compile(r"[-_:./+]+")


def _initials(text: str, limit: int = _MAX_INITIALS) -> str:
    """The first letter of each word, uppercased: `ScienceBeam refs retrained` is SRR.

    A word starting with a digit is a version, a hash or a timestamp, which say little
    and differ between otherwise identical labels.
    """
    words = [
        word
        for chunk in text.split() if not chunk[0].isdigit()
        for word in _WORD_SEPARATORS.split(chunk) if word and word[0].isalpha()
    ]
    return "".join(word[0] for word in words[:limit]).upper()


def _split_profile(label: str) -> Tuple[str, str]:
    """A label's name and its trailing parenthesis, which is the profile in the labels
    a run makes: `sciencebeam-parser:main-4b436dd1 (grobid_crf)`."""
    match = _TRAILING_PARENTHESIS.match(label)
    return (match.group(1), match.group(2)) if match else (label, "")


def _duplicated(keys: Sequence[str]) -> List[bool]:
    return [keys.count(key) > 1 for key in keys]


def column_keys(labels: Sequence[str]) -> List[str]:
    """A short name per column, from its label's initials.

    Columns sharing initials -- the same tool under two profiles -- add the profile's,
    as SPM+GC and SPM+DH; any still alike are numbered in the order they are declared.
    """
    names_and_profiles = [_split_profile(label) for label in labels]
    keys = [
        _initials(name) or f"V{index + 1}"
        for index, (name, _) in enumerate(names_and_profiles)
    ]
    keys = [
        f"{key}+{_initials(profile)}" if duplicated and _initials(profile) else key
        for key, duplicated, (_, profile) in zip(
            keys, _duplicated(keys), names_and_profiles
        )
    ]
    seen: Dict[str, int] = {}
    numbered = []
    for key, duplicated in zip(keys, _duplicated(keys)):
        seen[key] = seen.get(key, 0) + 1
        numbered.append(f"{key}{seen[key]}" if duplicated and seen[key] > 1 else key)
    return numbered


def keyed(labeled_summaries: List[Tuple[str, dict]]) -> List[Tuple[str, dict]]:
    """The summaries under their column keys, which is what the tables print.

    A label is long enough -- an image tag, a build timestamp and a profile -- that
    a table headed by them, twice over for the deltas, is mostly header.
    """
    keys = column_keys([label for label, _ in labeled_summaries])
    return [(key, summary) for key, (_, summary) in zip(keys, labeled_summaries)]


def render_legend(
    labels: Sequence[str], primary: int, described: Sequence[Tuple[str, str]] = ()
) -> List[str]:
    """Which variant each column key stands for, and what it is where a comparison
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
    for index, (key, label) in enumerate(zip(column_keys(labels), labels)):
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
