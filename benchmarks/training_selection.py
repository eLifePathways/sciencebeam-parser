"""Which documents a corpus holds, as an ordered list it carries rather than a draw.

`sample_indices` shuffles over the row count and takes the first *n*, so the
nesting it offers holds only while the corpus is the same size. It is not: `ore`
went from 40 rows to 37, and the ten documents `smoke` named became a different
ten. A corpus whose documents are reviewed cannot have its membership recomputed
underneath it.

So the selection is recorded, ordered, and a mode is a prefix of it. Nesting,
growth and "raising a mode adds rather than swaps" are then one property instead
of three that have to agree. Topping up appends by a hash of the document id,
which does not move when the corpus around it does.

A document whose source row has gone **keeps its place**. Removing it would shift
every later document forward into the prefix and evict the one at the end, which
is the eviction this exists to prevent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

LOGGER = logging.getLogger(__name__)

__all__ = [
    "SELECTION_DIRECTORY_NAME",
    "ModeSelection",
    "get_selection_file_path",
    "read_selection",
    "select_for_mode",
    "write_selection",
]

SELECTION_DIRECTORY_NAME = "selection"


def _rank(seed: int, document_id: str) -> str:
    return hashlib.sha256(f"{seed}:{document_id}".encode("utf-8")).hexdigest()


def get_selection_file_path(selection_dir: Path, split: str, corpus: str) -> Path:
    return selection_dir / split / f"{corpus}.txt"


def read_selection(selection_dir: Path, split: str, corpus: str) -> List[str]:
    """The corpus's documents in order, or empty when it has none recorded yet."""
    file_path = get_selection_file_path(selection_dir, split, corpus)
    if not file_path.is_file():
        return []
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def write_selection(
    selection_dir: Path, split: str, corpus: str, document_ids: Sequence[str]
) -> Path:
    file_path = get_selection_file_path(selection_dir, split, corpus)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(document_ids) + "\n", encoding="utf-8")
    return file_path


@dataclasses.dataclass(frozen=True)
class ModeSelection:
    """What one mode names, and what became of it.

    `present` is what can be fetched and generated; `missing` is in the prefix and
    no longer upstream, which is reported rather than quietly dropped. `appended`
    is what topping up added, and is what makes the recorded list grow.
    """

    selection: Sequence[str]
    present: Sequence[str]
    missing: Sequence[str]
    appended: Sequence[str]

    @property
    def prefix(self) -> List[str]:
        return list(self.present) + list(self.missing)


def select_for_mode(
    selection: Sequence[str],
    upstream_ids: Iterable[str],
    raw_n: Optional[int],
    seed: int,
) -> ModeSelection:
    """The documents this mode names, topping the recorded list up if it is short.

    What never shrinks is the recorded list: a mode smaller than it takes a shorter
    prefix and leaves every id in place, so no document leaves the corpus and a
    mode raised again later names the same ones it did before.
    """
    upstream = list(upstream_ids)
    upstream_set: Set[str] = set(upstream)
    recorded = list(selection)
    recorded_set = set(recorded)

    candidates = sorted(
        (document_id for document_id in upstream if document_id not in recorded_set),
        key=lambda document_id: _rank(seed, document_id),
    )
    wanted = len(recorded) + len(candidates) if raw_n is None else raw_n
    appended = candidates[: max(0, wanted - len(recorded))]
    grown = recorded + appended

    prefix = grown if raw_n is None else grown[:raw_n]
    missing = [document_id for document_id in prefix if document_id not in upstream_set]
    if missing:
        LOGGER.warning(
            "%d document(s) in the selection are no longer in the dataset: %s",
            len(missing),
            ", ".join(missing),
        )
    return ModeSelection(
        selection=grown,
        present=[
            document_id for document_id in prefix if document_id in upstream_set
        ],
        missing=missing,
        appended=appended,
    )
