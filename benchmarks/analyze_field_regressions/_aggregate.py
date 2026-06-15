from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ._types import FeatureSummary, ModelSummary

_OTHER_LABELS = frozenset({'<other>', 'I-<other>'})


def _is_meaningful_label_change(sbeam_label: str, grobid_label: str) -> bool:
    """Return False for <other> ↔ I-<other> transitions, which are semantically equivalent."""
    return not (sbeam_label in _OTHER_LABELS and grobid_label in _OTHER_LABELS)


def _base_label(label: str) -> str:
    """Strip BIO prefix (B-/I-) from a label tag."""
    if label.startswith(('B-', 'I-')):
        return label[2:]
    return label


def _label_is_relevant(label: str, relevant_labels: frozenset) -> bool:
    return label in relevant_labels or _base_label(label) in relevant_labels


def _pair_is_relevant(s: str, g: str, relevant_labels: frozenset) -> bool:
    return _label_is_relevant(s, relevant_labels) or _label_is_relevant(g, relevant_labels)


@dataclass
class _FeatureCounters:
    changes: Counter = field(default_factory=Counter)
    doc_count: Counter = field(default_factory=Counter)
    transitions: Dict[str, Counter] = field(default_factory=dict)
    relevant_changes: Counter = field(default_factory=Counter)
    relevant_doc_count: Counter = field(default_factory=Counter)
    other_changes: Counter = field(default_factory=Counter)
    other_doc_count: Counter = field(default_factory=Counter)


def _accumulate_feature(
    fc: '_FeatureCounters',
    feat: str,
    meaningful: list,
    relevant_labels: Optional[frozenset],
) -> None:
    fc.changes[feat] += len(meaningful)
    fc.doc_count[feat] += 1
    if feat not in fc.transitions:
        fc.transitions[feat] = Counter()
    for tok in meaningful:
        fc.transitions[feat][(tok['sbeam_label'], tok['grobid_label'])] += 1
    if not relevant_labels:
        return
    rel_toks = [
        tok for tok in meaningful
        if _pair_is_relevant(tok['sbeam_label'], tok['grobid_label'], relevant_labels)
    ]
    other_toks = [tok for tok in meaningful if tok not in rel_toks]
    if rel_toks:
        fc.relevant_changes[feat] += len(rel_toks)
        fc.relevant_doc_count[feat] += 1
    if other_toks:
        fc.other_changes[feat] += len(other_toks)
        fc.other_doc_count[feat] += 1


def _aggregate_model_results(
    model_name: str,
    doc_results: List[Tuple[str, Optional[dict]]],
    relevant_labels: Optional[frozenset] = None,
) -> ModelSummary:
    fc = _FeatureCounters()
    docs_analyzed = 0
    docs_failed = 0
    docs_with_feature_diffs = 0

    for _record_id, result in doc_results:
        if result is None:
            docs_failed += 1
            continue
        docs_analyzed += 1
        if result.get('features_with_diffs'):
            docs_with_feature_diffs += 1
        for feat, fdata in result.get('per_feature', {}).items():
            meaningful = [
                tok for tok in fdata.get('changed_tokens', [])
                if _is_meaningful_label_change(tok['sbeam_label'], tok['grobid_label'])
            ]
            if meaningful:
                _accumulate_feature(fc, feat, meaningful, relevant_labels)

    summaries = [
        FeatureSummary(
            feature=feat,
            total_label_changes=fc.changes[feat],
            docs_affected=fc.doc_count[feat],
            transitions=fc.transitions.get(feat, Counter()),
            relevant_docs_affected=(
                fc.relevant_doc_count.get(feat) if relevant_labels else None
            ),
            relevant_label_changes=(
                fc.relevant_changes.get(feat) if relevant_labels else None
            ),
            other_docs_affected=(
                fc.other_doc_count.get(feat) if relevant_labels else None
            ),
            other_label_changes=(
                fc.other_changes.get(feat) if relevant_labels else None
            ),
        )
        for feat in sorted(fc.changes, key=lambda f: fc.changes[f], reverse=True)
    ]

    return ModelSummary(
        model=model_name,
        docs_analyzed=docs_analyzed,
        docs_failed=docs_failed,
        docs_with_feature_diffs=docs_with_feature_diffs,
        features=summaries,
    )
