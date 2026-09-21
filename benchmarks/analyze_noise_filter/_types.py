from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RemovedBlock:
    """One layout block the noise filter removes, with enough context to judge it."""
    document_id: str
    corpus: str
    page_number: int
    page_count: int
    note_type: str
    y_relative: Optional[float]
    height_relative: Optional[float]
    line_count: int
    text: str
    # True: the text is in the publisher's JATS, so the removal takes content.
    # False: it is not, which on its own proves nothing.
    # None: the text is too short to look up, or no JATS was given.
    in_jats: Optional[bool]


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    corpus: str
    page_count: int
    block_count: int
    line_count: int
    removed_block_count: int
    removed_line_count: int
    in_jats_count: int
    error: Optional[str] = None
