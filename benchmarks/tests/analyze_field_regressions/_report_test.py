from __future__ import annotations

from collections import Counter

from benchmarks.analyze_field_regressions._report import (
    _feature_section_label,
    _render_feature_summary_table,
    _render_feature_table,
)
from benchmarks.analyze_field_regressions._types import FeatureSummary

_REF_DOI_LABELS = frozenset({'<pubnum>', '<web>'})


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
