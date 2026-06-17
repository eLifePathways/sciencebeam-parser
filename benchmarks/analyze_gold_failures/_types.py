from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class FailureMode(IntEnum):
    NOT_IN_RAW_TEXT = 0
    EXTRACTION_FAILED = 1
    PARTIAL_WRONG = 2
    CORRECT = 3


FAILURE_MODE_LABEL = {
    FailureMode.NOT_IN_RAW_TEXT: 'Not found in raw text',
    FailureMode.EXTRACTION_FAILED: 'Extraction failed',
    FailureMode.PARTIAL_WRONG: 'Partial/wrong match',
    FailureMode.CORRECT: 'Correct',
}

FAILURE_MODE_SHORT = {
    FailureMode.NOT_IN_RAW_TEXT: 'NR',
    FailureMode.EXTRACTION_FAILED: 'Failed',
    FailureMode.PARTIAL_WRONG: 'Partial',
    FailureMode.CORRECT: 'Correct',
}


@dataclass
class GoldValueResult:
    value: str
    mode: FailureMode
    in_raw: bool
    in_sb_field: bool
    best_sb_match: Optional[str] = None
    best_sb_similarity: Optional[float] = None
    # Similarity of the best-matching fixed-length window in the raw TEI text.
    # Only populated for NOT_IN_RAW_TEXT results; implicitly ~1.0 for all other modes
    # (since in_raw=True guarantees the gold appears in the raw text).
    best_raw_similarity: Optional[float] = None


@dataclass
class PipelineAttribution:
    correct_models: List[str]
    failed_models: List[str]
    recommended_action: str
    # Set to the model name when attribution is definite (text found with wrong label).
    # None when no model could locate the text or all models correctly classify it.
    first_failed_model: Optional[str] = None
    # What the first_failed_model actually predicted on the matching token span.
    predicted_label: Optional[str] = None
    expected_label: Optional[str] = None
    # Raw text from the model data at the matching position.  Set for any definite
    # attribution (sliding window, prefix block, fuzzy block).  Allows Sim to be
    # computed for all attributed rows and shows encoding differences in "Data text".
    candidate_text: Optional[str] = None
    # Explanation shown when first_failed_model is None (e.g. "Text not found in any
    # model data" or "All models correctly classify this text").
    attribution_note: Optional[str] = None
    # Surrounding model-data lines at the point of mislabelling.
    # Each entry is (display_text, label, is_matched_span).
    context_window: Optional[List[Tuple[str, str, bool]]] = None
    # True when context_window entries are block-level (segmentation); False for token-level.
    context_is_block_level: bool = False


@dataclass
class DocumentSummary:
    corpus: str
    record_id: str
    score_sb: Optional[float]
    results: List[GoldValueResult]
    # value text -> attribution, populated only in online mode for EXTRACTION_FAILED values
    attributions: Dict[str, PipelineAttribution] = field(default_factory=dict)

    @property
    def mode_counts(self) -> List[int]:
        counts = [0] * len(FailureMode)
        for r in self.results:
            counts[int(r.mode)] += 1
        return counts

    @property
    def total_gold(self) -> int:
        return len(self.results)


@dataclass
class FailureModeAggregate:
    mode: FailureMode
    total_values: int
    docs_affected: int
    examples: List[Tuple[str, str, GoldValueResult]]  # (corpus, record_id, result)


@dataclass
class AttributionAggregate:
    model: str
    failure_count: int
    doc_count: int = 0


@dataclass
class FieldFailureSummary:
    field: str
    run_sb: str
    total_docs: int
    total_gold: int
    mode_aggregates: List[FailureModeAggregate]
    attribution_aggregates: List[AttributionAggregate]
    recommended_action: str
    is_online: bool
    n_docs_skipped: int = 0
    method: str = 'edit_sim'
