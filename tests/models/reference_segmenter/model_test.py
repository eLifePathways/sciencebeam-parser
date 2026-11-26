from typing import List, Tuple

from sciencebeam_parser.document.layout_document import LayoutToken
from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT,
    FEATURE_FLAVOUR_GROBID,
    FEATURE_FLAVOUR_SCIENCEBEAM,
    LabeledLayoutToken
)
from sciencebeam_parser.models.reference_segmenter.model import (
    ReferenceSegmenterModel,
    iter_reference_entity_blocks
)


def _get_entities(
    labeled_texts: List[Tuple[str, str]],
    **kwargs
) -> List[Tuple[str, str]]:
    return [
        (tag, layout_block.text)
        for tag, layout_block in iter_reference_entity_blocks([
            LabeledLayoutToken(label, LayoutToken(text))
            for label, text in labeled_texts
        ], **kwargs)
    ]


class TestIterReferenceEntityBlocks:
    def test_should_continue_a_reference_past_other_text_inside_it(self):
        assert _get_entities([
            ('B-<reference>', 'Fernley'),
            ('I-<reference>', 'A.'),
            ('O', 'Mossby'),
            ('I-<reference>', 'https://ferns.test'),
            ('B-<reference>', 'Bracken'),
            ('I-<reference>', 'B.'),
        ]) == [
            ('<reference>', 'Fernley A. Mossby https://ferns.test'),
            ('<reference>', 'Bracken B.'),
        ]

    def test_should_leave_other_text_inside_a_reference_out_where_asked(self):
        assert _get_entities([
            ('B-<reference>', 'Fernley'),
            ('I-<reference>', 'A.'),
            ('O', 'Mossby'),
            ('I-<reference>', 'https://ferns.test'),
            ('B-<reference>', 'Bracken'),
        ], keep_other_text_in_references=False) == [
            ('<reference>', 'Fernley A. https://ferns.test'),
            ('O', 'Mossby'),
            ('<reference>', 'Bracken'),
        ]

    def test_should_start_a_reference_on_its_beginning_after_other_text(self):
        assert _get_entities([
            ('B-<reference>', 'Fernley'),
            ('O', 'Mossby'),
            ('B-<reference>', 'Bracken'),
        ]) == [
            ('<reference>', 'Fernley'),
            ('O', 'Mossby'),
            ('<reference>', 'Bracken'),
        ]

    def test_should_keep_labels_and_text_before_the_first_reference(self):
        assert _get_entities([
            ('O', 'References'),
            ('B-<label>', '1'),
            ('B-<reference>', 'Fernley'),
            ('B-<label>', '2'),
            ('B-<reference>', 'Bracken'),
        ]) == [
            ('O', 'References'),
            ('<label>', '1'),
            ('<reference>', 'Fernley'),
            ('<label>', '2'),
            ('<reference>', 'Bracken'),
        ]


class TestReferenceSegmenterModelFeatureFlavour:
    def test_should_give_grobid_features_unless_configured_otherwise(self):
        model = ReferenceSegmenterModel(None)
        assert model.get_data_generator(
            DEFAULT_DOCUMENT_FEATURES_CONTEXT
        ).feature_flavour == FEATURE_FLAVOUR_GROBID

    def test_should_give_the_features_the_config_names(self):
        model = ReferenceSegmenterModel(
            None,
            model_config={'feature_flavour': FEATURE_FLAVOUR_SCIENCEBEAM}
        )
        assert model.get_data_generator(
            DEFAULT_DOCUMENT_FEATURES_CONTEXT
        ).feature_flavour == FEATURE_FLAVOUR_SCIENCEBEAM
