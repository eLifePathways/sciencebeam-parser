from __future__ import annotations

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.models.citation.extract import (
    CitationSemanticExtractor,
    VALID_REFERENCE_TYPES,
)
from sciencebeam_parser.models.citation.labels import CITATION_LABELS

from benchmarks.analyze_field_regressions._models import (
    MODEL_RELEVANT_LABELS,
    REFERENCE_PRESENCE_LABELS,
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

    def test_first_reference_text_chain_includes_citation(self):
        # the citation model decides which references survive, so which one is first
        assert _get_model_chain('first_reference_text') == [
            'segmentation', 'reference-segmenter', 'citation'
        ]

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


class TestReferencePresenceLabels:
    def test_matches_the_labels_that_keep_a_reference_in_the_output(self):
        extractor = CitationSemanticExtractor()
        block = LayoutBlock.for_text('Example')
        keeps_reference = frozenset(
            label
            for label in CITATION_LABELS
            if type(  # pylint: disable=unidiomatic-typecheck
                extractor.get_semantic_content_for_entity_name(label, block)
            ) in VALID_REFERENCE_TYPES
        )
        assert keeps_reference == REFERENCE_PRESENCE_LABELS

    def test_first_reference_text_reads_them_from_the_citation_model(self):
        assert MODEL_RELEVANT_LABELS['first_reference_text']['citation'] == \
               REFERENCE_PRESENCE_LABELS
