from __future__ import annotations

from collections import Counter
from typing import List, Optional

from benchmarks.analyze_field_regressions import (
    MODEL_RELEVANT_LABELS,
    FeatureSummary,
    _aggregate_model_results,
    _feature_section_label,
    _get_model_chain,
    _is_meaningful_label_change,
    _label_is_relevant,
    _normalize_for_comparison,
    _pair_is_relevant,
    _render_feature_summary_table,
    _render_feature_table,
    _resolve_concurrency,
)

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
# _get_model_chain
# ---------------------------------------------------------------------------

class TestGetModelChain:
    def test_citation_chain(self):
        assert _get_model_chain('reference_doi') == [
            'segmentation', 'reference-segmenter', 'citation'
        ]

    def test_header_chain(self):
        assert _get_model_chain('title') == ['segmentation', 'header']

    def test_name_header_chain(self):
        chain = _get_model_chain('author_full_names')
        assert chain == ['segmentation', 'header', 'name-header']

    def test_leaf_only_when_no_parent(self):
        # fulltext has no parent listed in MODEL_PARENT
        chain = _get_model_chain('body_section_titles')
        assert chain[-1] == 'fulltext'
        assert 'segmentation' in chain


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
# _normalize_for_comparison
# ---------------------------------------------------------------------------

class TestNormalizeForComparison:
    def test_removes_spaces(self):
        assert _normalize_for_comparison('10.7554 / eLife.12345') == '10.7554/eLife.12345'

    def test_removes_newlines(self):
        assert _normalize_for_comparison('10.7554\n/eLife.12345') == '10.7554/eLife.12345'

    def test_no_whitespace_unchanged(self):
        assert _normalize_for_comparison('10.7554/eLife.12345') == '10.7554/eLife.12345'


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


# ---------------------------------------------------------------------------
# _feature_section_label
# ---------------------------------------------------------------------------

class TestFeatureSectionLabel:
    def _fs(self, total_label_changes: int) -> FeatureSummary:
        return FeatureSummary(
            feature='f', total_label_changes=total_label_changes, docs_affected=1
        )

    def test_plural(self):
        label = _feature_section_label([self._fs(3), self._fs(5)])
        assert '2 features' in label
        assert '8 label changes' in label

    def test_singular(self):
        label = _feature_section_label([self._fs(1)])
        assert '1 feature,' in label
        assert '1 label change' in label
        assert 'changes' not in label.split('1 label change')[1][:2]


# ---------------------------------------------------------------------------
# _render_feature_summary_table
# ---------------------------------------------------------------------------

class TestRenderFeatureSummaryTable:
    def _make_fs(self, rel_docs=None, rel_labels=None, other_docs=None, other_labels=None):
        return FeatureSummary(
            feature='is_http',
            total_label_changes=10,
            docs_affected=5,
            transitions=Counter(),
            relevant_docs_affected=rel_docs,
            relevant_label_changes=rel_labels,
            other_docs_affected=other_docs,
            other_label_changes=other_labels,
        )

    def test_no_focus_shows_total_columns(self):
        rows = _render_feature_summary_table([self._make_fs()])
        header = rows[0]
        assert 'Total docs' in header
        assert 'Total Δlabels' in header
        assert 'Relevant' not in header
        assert 'Other' not in header
        assert 'is_http' in rows[2]

    def test_relevant_focus_shows_relevant_columns(self):
        rows = _render_feature_summary_table([self._make_fs(rel_docs=3, rel_labels=7)],
                                             focus='relevant')
        header = rows[0]
        assert 'Relevant docs' in header
        assert 'Relevant Δlabels' in header
        assert 'Other' not in header
        data_row = rows[2]
        assert '3' in data_row
        assert '7' in data_row

    def test_other_focus_shows_other_columns(self):
        rows = _render_feature_summary_table([self._make_fs(other_docs=2, other_labels=4)],
                                             focus='other')
        header = rows[0]
        assert 'Other docs' in header
        assert 'Other Δlabels' in header
        assert 'Relevant' not in header
        data_row = rows[2]
        assert '2' in data_row
        assert '4' in data_row


# ---------------------------------------------------------------------------
# _render_feature_table
# ---------------------------------------------------------------------------

class TestRenderFeatureTable:
    def _make_fs_with_transitions(
        self, transitions, rel_docs=None, rel_labels=None,
        other_docs=None, other_labels=None
    ):
        return FeatureSummary(
            feature='is_http',
            total_label_changes=sum(transitions.values()),
            docs_affected=1,
            transitions=Counter(transitions),
            relevant_docs_affected=rel_docs,
            relevant_label_changes=rel_labels,
            other_docs_affected=other_docs,
            other_label_changes=other_labels,
        )

    def test_no_focus_shows_all_transitions(self):
        fs = self._make_fs_with_transitions({
            ('B-<pubnum>', '<other>'): 5,
            ('B-<title>', '<other>'): 3,
        })
        rows = _render_feature_table([fs])
        content = '\n'.join(rows)
        assert '<pubnum>' in content
        assert '<title>' in content

    def test_relevant_focus_filters_to_relevant_transitions(self):
        fs = self._make_fs_with_transitions(
            {('B-<pubnum>', '<other>'): 5, ('B-<title>', '<other>'): 3},
            rel_docs=1, rel_labels=5,
        )
        rows = _render_feature_table([fs], relevant_labels=_REF_DOI_LABELS, focus='relevant')
        content = '\n'.join(rows)
        assert '<pubnum>' in content
        assert '<title>' not in content
        assert 'Relevant docs' in rows[0]

    def test_other_focus_filters_to_non_relevant_transitions(self):
        fs = self._make_fs_with_transitions(
            {('B-<pubnum>', '<other>'): 5, ('B-<title>', '<other>'): 3},
            other_docs=1, other_labels=3,
        )
        rows = _render_feature_table([fs], relevant_labels=_REF_DOI_LABELS, focus='other')
        content = '\n'.join(rows)
        assert '<title>' in content
        assert '<pubnum>' not in content
        assert 'Other docs' in rows[0]

    def test_dash_row_when_no_transitions_match(self):
        fs = self._make_fs_with_transitions(
            {('B-<title>', '<other>'): 5},
            rel_docs=0, rel_labels=0,
        )
        rows = _render_feature_table([fs], relevant_labels=_REF_DOI_LABELS, focus='relevant')
        content = '\n'.join(rows)
        assert '— ' in content or '—' in content


# ---------------------------------------------------------------------------
# MODEL_RELEVANT_LABELS
# ---------------------------------------------------------------------------

class TestModelRelevantLabels:
    def test_reference_doi_segmentation(self):
        labels = MODEL_RELEVANT_LABELS['reference_doi']['segmentation']
        assert '<references>' in labels

    def test_reference_doi_reference_segmenter(self):
        labels = MODEL_RELEVANT_LABELS['reference_doi']['reference-segmenter']
        assert '<reference>' in labels

    def test_reference_doi_citation(self):
        labels = MODEL_RELEVANT_LABELS['reference_doi']['citation']
        assert '<pubnum>' in labels
        assert '<web>' in labels

    def test_reference_title_shares_hierarchy_labels(self):
        assert MODEL_RELEVANT_LABELS['reference_title']['segmentation'] == \
               MODEL_RELEVANT_LABELS['reference_doi']['segmentation']

    def test_title_header_model(self):
        assert '<title>' in MODEL_RELEVANT_LABELS['title']['header']

    def test_title_segmentation_model(self):
        assert '<header>' in MODEL_RELEVANT_LABELS['title']['segmentation']


# ---------------------------------------------------------------------------
# _resolve_concurrency
# ---------------------------------------------------------------------------

class TestResolveConcurrency:
    def test_nonzero_returned_as_is(self):
        assert _resolve_concurrency(8) == 8

    def test_zero_returns_at_least_two(self):
        result = _resolve_concurrency(0)
        assert result >= 2
