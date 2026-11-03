"""Reading a document from the corrected deposit its corpus declares.

A deposit is sometimes thinner than the article it describes, and the corpus
that holds the documents declares what it is missing beside its training-source
config, as an entry and the corrected deposit that entry comes to. Interpreting
the entry belongs to that repo, where it is written and reviewed; what happens
here is only to read the corrected deposit in place of the published one, so
that generation and the gold it is scored against are the same document.

The published deposit is kept beside it, so this says the same thing however
many times it runs, and a document whose corrected deposit has gone -- a change
a release has caught up with -- is given its published one back.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)

DEPOSIT_SUFFIX = ".jats.xml"
AS_DEPOSITED_SUFFIX = ".as-deposited.xml"


def _restore_withdrawn(corrections_dir: Path, corpus: str, deposits_dir: Path) -> None:
    for as_deposited in sorted(deposits_dir.glob(f"*{AS_DEPOSITED_SUFFIX}")):
        document_id = as_deposited.name[: -len(AS_DEPOSITED_SUFFIX)]
        corrected = corrections_dir / corpus / f"{document_id}{DEPOSIT_SUFFIX}"
        if corrected.exists():
            continue
        deposit_path = deposits_dir / f"{document_id}{DEPOSIT_SUFFIX}"
        deposit_path.write_text(
            as_deposited.read_text(encoding="utf-8"), encoding="utf-8"
        )
        as_deposited.unlink()


def substitute_corrected_deposits(
    corrections_dir: Path, corpus: str, deposits_dir: Path
) -> List[str]:
    """Put each corrected deposit in place of the one the dataset published."""
    if not deposits_dir.is_dir():
        return []
    _restore_withdrawn(corrections_dir, corpus, deposits_dir)

    substituted: List[str] = []
    for corrected in sorted((corrections_dir / corpus).glob(f"*{DEPOSIT_SUFFIX}")):
        document_id = corrected.name[: -len(DEPOSIT_SUFFIX)]
        deposit_path = deposits_dir / f"{document_id}{DEPOSIT_SUFFIX}"
        if not deposit_path.exists():
            continue
        substituted.append(document_id)
        corrected_text = corrected.read_text(encoding="utf-8")
        published = deposit_path.read_text(encoding="utf-8")
        if published == corrected_text:
            continue
        as_deposited = deposit_path.with_name(
            document_id + AS_DEPOSITED_SUFFIX
        )
        if not as_deposited.exists():
            as_deposited.write_text(published, encoding="utf-8")
        deposit_path.write_text(corrected_text, encoding="utf-8")
    return substituted
