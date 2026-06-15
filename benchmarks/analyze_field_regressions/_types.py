from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FieldPresenceSummary:
    in_grobid: int
    in_raw: int
    in_sb: int
    total: int


@dataclass
class RegressionCase:
    delta: float
    corpus: str
    record_id: str
    score_a: float
    score_b: float
    presence: Optional[FieldPresenceSummary] = None


@dataclass
class FeatureSummary:
    feature: str
    total_label_changes: int
    docs_affected: int
    transitions: Counter = field(default_factory=Counter)
    relevant_docs_affected: Optional[int] = None
    relevant_label_changes: Optional[int] = None
    other_docs_affected: Optional[int] = None
    other_label_changes: Optional[int] = None


@dataclass
class ModelSummary:
    model: str
    docs_analyzed: int
    docs_failed: int
    docs_with_feature_diffs: int
    features: List[FeatureSummary]
