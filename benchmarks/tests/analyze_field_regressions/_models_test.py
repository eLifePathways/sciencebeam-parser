from __future__ import annotations

from benchmarks.analyze_field_regressions._models import (
    MODEL_RELEVANT_LABELS,
    _get_model_chain,
)


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
