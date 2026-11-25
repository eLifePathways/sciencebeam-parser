from typing import List, Tuple

from sciencebeam_parser.document.layout_document import LayoutToken
from sciencebeam_parser.models.data import LabeledLayoutToken
from sciencebeam_parser.models.reference_segmenter.model import iter_reference_entity_blocks


def _get_entities(labeled_texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [
        (tag, layout_block.text)
        for tag, layout_block in iter_reference_entity_blocks([
            LabeledLayoutToken(label, LayoutToken(text))
            for label, text in labeled_texts
        ])
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
