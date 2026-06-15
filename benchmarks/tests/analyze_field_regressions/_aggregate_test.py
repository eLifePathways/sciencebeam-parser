from __future__ import annotations

from typing import List, Optional

from benchmarks.analyze_field_regressions._aggregate import (
    _aggregate_model_results,
    _is_meaningful_label_change,
    _label_is_relevant,
    _pair_is_relevant,
)
from benchmarks.analyze_field_regressions._types import FeatureSummary

_REF_DOI_LABELS = frozenset({'<pubnum>', '<web>'})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_result(features: dict) -> dict:
    """Build a minimal find_important result dict."""
    return {
        'features_with_diffs': list(features.keys()),
        'per_feature': {
            feat: {'changed_tokens': tokens}
            for feat, tokens in features.items()
        },
    }


def _tok(sbeam: str, grobid: str) -> dict:
    return {'sbeam_label': sbeam, 'grobid_label': grobid}


def _make_summary(
    feature: str,
    transitions: List,
    relevant_labels: Optional[frozenset] = None,
) -> FeatureSummary:
    """Build FeatureSummary via _aggregate_model_results for one doc."""
    result = _make_result({feature: [_tok(s, g) for s, g in transitions]})
    ms = _aggregate_model_results('citation', [('doc1', result)], relevant_labels)
    feats = [f for f in ms.features if f.feature == feature]
    assert feats, f'feature {feature!r} not in results'
    return feats[0]


# ---------------------------------------------------------------------------
# _is_meaningful_label_change
# ---------------------------------------------------------------------------

class TestIsMeaningfulLabelChange:
    def test_other_to_i_other_is_not_meaningful(self):
        assert not _is_meaningful_label_change('<other>', 'I-<other>')

    def test_i_other_to_other_is_not_meaningful(self):
        assert not _is_meaningful_label_change('I-<other>', '<other>')

    def test_both_other_is_not_meaningful(self):
        assert not _is_meaningful_label_change('<other>', '<other>')

    def test_real_transition_is_meaningful(self):
        assert _is_meaningful_label_change('B-<pubnum>', 'B-<other>')

    def test_other_to_label_is_meaningful(self):
        assert _is_meaningful_label_change('<other>', 'B-<pubnum>')


# ---------------------------------------------------------------------------
# _label_is_relevant / _pair_is_relevant
# ---------------------------------------------------------------------------

class TestRelevanceChecks:
    def test_exact_match(self):
        assert _label_is_relevant('<pubnum>', _REF_DOI_LABELS)

    def test_bio_prefix_stripped(self):
        assert _label_is_relevant('B-<pubnum>', _REF_DOI_LABELS)
        assert _label_is_relevant('I-<pubnum>', _REF_DOI_LABELS)

    def test_irrelevant_label(self):
        assert not _label_is_relevant('<title>', _REF_DOI_LABELS)

    def test_pair_relevant_when_sbeam_matches(self):
        assert _pair_is_relevant('B-<pubnum>', '<other>', _REF_DOI_LABELS)

    def test_pair_relevant_when_grobid_matches(self):
        assert _pair_is_relevant('<other>', 'B-<pubnum>', _REF_DOI_LABELS)

    def test_pair_irrelevant_when_neither_matches(self):
        assert not _pair_is_relevant('<other>', '<title>', _REF_DOI_LABELS)


# ---------------------------------------------------------------------------
# _aggregate_model_results
# ---------------------------------------------------------------------------

class TestAggregateModelResults:
    def test_counts_label_changes_per_feature(self):
        result = _make_result({
            'is_http': [_tok('B-<pubnum>', '<other>'), _tok('I-<pubnum>', '<other>')],
        })
        ms = _aggregate_model_results('citation', [('doc1', result)])
        assert ms.features[0].feature == 'is_http'
        assert ms.features[0].total_label_changes == 2
        assert ms.features[0].docs_affected == 1

    def test_skips_none_results(self):
        ms = _aggregate_model_results('citation', [('doc1', None)])
        assert ms.docs_failed == 1
        assert ms.docs_analyzed == 0
        assert not ms.features

    def test_excludes_meaningless_transitions(self):
        result = _make_result({
            'feat': [_tok('<other>', 'I-<other>'), _tok('B-<pubnum>', '<other>')],
        })
        ms = _aggregate_model_results('citation', [('doc1', result)])
        assert ms.features[0].total_label_changes == 1

    def test_relevant_counts_when_labels_given(self):
        result = _make_result({
            'feat': [
                _tok('B-<pubnum>', '<other>'),
                _tok('B-<title>', '<other>'),
            ],
        })
        ms = _aggregate_model_results('citation', [('doc1', result)], _REF_DOI_LABELS)
        fs = ms.features[0]
        assert fs.relevant_docs_affected == 1
        assert fs.relevant_label_changes == 1
        assert fs.other_docs_affected == 1
        assert fs.other_label_changes == 1

    def test_relevant_docs_none_without_labels(self):
        result = _make_result({'feat': [_tok('B-<pubnum>', '<other>')]})
        ms = _aggregate_model_results('citation', [('doc1', result)])
        assert ms.features[0].relevant_docs_affected is None
        assert ms.features[0].other_docs_affected is None

    def test_relevant_doc_not_counted_when_only_other_transitions(self):
        result = _make_result({
            'feat': [_tok('B-<title>', '<other>')],
        })
        ms = _aggregate_model_results('citation', [('doc1', result)], _REF_DOI_LABELS)
        fs = ms.features[0]
        assert fs.relevant_docs_affected is None or fs.relevant_docs_affected == 0
        assert fs.other_docs_affected == 1

    def test_counts_across_multiple_docs(self):
        result1 = _make_result({'feat': [_tok('B-<pubnum>', '<other>')]})
        result2 = _make_result({'feat': [_tok('B-<pubnum>', '<other>')]})
        ms = _aggregate_model_results(
            'citation', [('doc1', result1), ('doc2', result2)], _REF_DOI_LABELS
        )
        fs = ms.features[0]
        assert fs.docs_affected == 2
        assert fs.relevant_docs_affected == 2
        assert fs.total_label_changes == 2

    def test_features_sorted_by_total_label_changes_descending(self):
        result = _make_result({
            'feat_a': [_tok('B-<pubnum>', '<other>')],
            'feat_b': [_tok('B-<pubnum>', '<other>'), _tok('B-<pubnum>', '<other>')],
        })
        ms = _aggregate_model_results('citation', [('doc1', result)])
        assert ms.features[0].feature == 'feat_b'
        assert ms.features[1].feature == 'feat_a'
