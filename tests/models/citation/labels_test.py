import logging

import pytest

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import SemanticNote
from sciencebeam_parser.models.citation.extract import CitationSemanticExtractor
from sciencebeam_parser.models.citation.labels import (
    CITATION_LABELS,
    IDENTIFIER_LABEL,
    NOTE_CITATION_LABELS,
    OTHER_LABEL
)
from sciencebeam_parser.models.citation.training_data import (
    TRAINING_XML_ELEMENT_PATH_BY_LABEL
)
from sciencebeam_parser.training.jats.field_vocab import CITATION_LABEL_BY_SUB_FIELD


LOGGER = logging.getLogger(__name__)


TEXT_1 = 'text 1'


class TestCitationLabels:
    def test_should_declare_note_labels_within_the_label_set(self):
        assert NOTE_CITATION_LABELS <= CITATION_LABELS

    def test_should_declare_identifier_label_within_the_label_set(self):
        assert IDENTIFIER_LABEL in CITATION_LABELS

    def test_should_generate_training_tei_for_every_label_except_other(self):
        assert set(TRAINING_XML_ELEMENT_PATH_BY_LABEL.keys()) == CITATION_LABELS - {OTHER_LABEL}

    def test_should_only_use_known_labels_for_jats_sub_fields(self):
        assert set(CITATION_LABEL_BY_SUB_FIELD.values()) <= CITATION_LABELS

    def test_should_use_the_identifier_label_for_every_jats_identifier_sub_field(self):
        identifier_labels = {
            label
            for sub_field, label in CITATION_LABEL_BY_SUB_FIELD.items()
            if sub_field.endswith(('-doi', '-pmid', '-pmcid'))
        }
        assert identifier_labels == {IDENTIFIER_LABEL}

    @pytest.mark.parametrize("label", sorted(CITATION_LABELS - {OTHER_LABEL}))
    def test_should_extract_semantic_content_for_every_label_that_is_not_a_declared_note(
        self,
        label: str
    ):
        semantic_content = CitationSemanticExtractor().get_semantic_content_for_entity_name(
            label, layout_block=LayoutBlock.for_text(TEXT_1)
        )
        LOGGER.debug('semantic_content: %r', semantic_content)
        if label in NOTE_CITATION_LABELS:
            assert isinstance(semantic_content, SemanticNote)
        else:
            assert not isinstance(semantic_content, SemanticNote)
