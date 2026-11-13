"""Fitting a report into what a GitHub comment can carry."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

# GitHub refuses an issue comment body over 65,536 characters. The margin covers the
# marker line the workflow prepends, the `<details>` wrapper a later run adds when it
# collapses the comment, any note appended after rendering, and GitHub counting UTF-16
# code units where Python counts code points.
COMMENT_LIMIT = 65536
COMMENT_BUDGET = 60000

Section = Tuple[str, List[str]]
Note = Callable[[Sequence[str]], List[str]]


def dropped_corpora_note(dropped: Sequence[str], location: str) -> List[str]:
    """The line a shortened comment carries in place of the sections it left out."""
    where = (
        f"The full report is {location}."
        if location else "The full report is in this run's artifact."
    )
    plural = "section is" if len(dropped) == 1 else "sections are"
    return [
        f"> \u26a0\ufe0f **{len(dropped)} corpus {plural} not in this comment**"
        f" ({', '.join(dropped)}). {where}",
        "",
    ]


def fit(
    head: Sequence[str], sections: Sequence[Section], budget: int, note: Note
) -> str:
    """`head` plus as many whole sections as fit, naming the ones left out."""
    for kept in range(len(sections), -1, -1):
        dropped = [name for name, _ in sections[kept:]]
        lines = list(head)
        for _, block in sections[:kept]:
            lines += block
        if dropped:
            lines += note(dropped)
        text = "\n".join(lines)
        if len(text) <= budget:
            return text
    return "\n".join(head)
