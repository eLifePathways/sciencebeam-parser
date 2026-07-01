import logging
from typing import Sequence, Union

import pytest
from lxml import etree
from lxml.builder import E

from sciencebeam_parser.document.tei.common import TEI_E
from sciencebeam_parser.models.data import LabeledLayoutToken
from sciencebeam_parser.utils.xml_writer import XmlTreeWriter

from sciencebeam_parser.models.training_data import (
    AbstractTrainingTeiParser,
    is_line_break_element,
    is_same_or_parent_path_of
)


LOGGER = logging.getLogger(__name__)


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'
TOKEN_4 = 'token4'


class TestIsSameOrParentPathOf:
    def test_should_return_true_for_same_path(self):
        assert is_same_or_parent_path_of(['parent', 'child1'], ['parent', 'child1']) is True

    def test_should_return_true_for_parent_path_of_child(self):
        assert is_same_or_parent_path_of(['parent'], ['parent', 'child1']) is True

    def test_should_return_false_for_child_path_of_parent(self):
        assert is_same_or_parent_path_of(['parent', 'child1'], ['parent']) is False

    def test_should_return_false_for_siblings(self):
        assert is_same_or_parent_path_of(['parent', 'child1'], ['parent', 'child2']) is False

    def test_should_return_false_for_different_parent(self):
        assert is_same_or_parent_path_of(['parent1', 'child1'], ['parent2', 'child1']) is False


ROOT_TRAINING_XML_ELEMENT_PATH = ['text']

TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<head>': ROOT_TRAINING_XML_ELEMENT_PATH + ['head'],
    '<paragraph>': ROOT_TRAINING_XML_ELEMENT_PATH + ['p'],
    '<page>': ROOT_TRAINING_XML_ELEMENT_PATH + ['biblScope[@unit="page"]']
}


class SampleTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
            use_tei_namespace=False
        )


# pylint: disable=not-callable

def _get_training_tei_with_text(
    text_items: Sequence[Union[etree.ElementBase, str]]
) -> etree.ElementBase:
    xml_writer = XmlTreeWriter(E('tei'), element_maker=E)
    xml_writer.require_path(
        ROOT_TRAINING_XML_ELEMENT_PATH
    )
    xml_writer.append_all(*text_items)
    LOGGER.debug('training tei: %r', etree.tostring(xml_writer.root))
    return xml_writer.root


class TestIsLineBreakElement:
    def test_should_return_true_for_lb_without_namespace(self):
        assert is_line_break_element(E('lb')) is True

    def test_should_return_true_for_lb_with_namespace(self):
        assert is_line_break_element(TEI_E('lb')) is True


class TestTrainingTeiParser:
    def test_should_parse_labelled_tokens_as_tag_result(self):
        tei_root = _get_training_tei_with_text([
            E('head', TOKEN_1, E('lb')),
            '\n',
            E('p', TOKEN_2, E('lb')),
            '\n'
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<head>'),
            (TOKEN_2, 'B-<paragraph>')
        ]]

    def test_should_parse_same_line_labelled_token_with_same_line_meta(self):
        tei_root = _get_training_tei_with_text([
            E('head', TOKEN_1),
            ' ',
            E('p', TOKEN_2, E('lb')),
            '\n'
        ])
        labeled_layout_tokens_list = (
            SampleTrainingTeiParser().parse_training_tei_to_labeled_layout_tokens_list(
                tei_root
            )
        )
        assert len(labeled_layout_tokens_list) == 1
        labeled_layout_tokens = labeled_layout_tokens_list[0]
        assert len(labeled_layout_tokens) == 2
        assert isinstance(labeled_layout_tokens[0], LabeledLayoutToken)
        assert labeled_layout_tokens[0].layout_token.text == TOKEN_1
        assert labeled_layout_tokens[1].layout_token.text == TOKEN_2
        assert (
            labeled_layout_tokens[0].layout_token.line_meta
            == labeled_layout_tokens[1].layout_token.line_meta
        )

    def test_should_parse_multi_line_labelled_token_with_diff_line_meta(self):
        tei_root = _get_training_tei_with_text([
            E('head', TOKEN_1, E('lb')),
            '\n',
            E('p', TOKEN_2, E('lb')),
            '\n'
        ])
        labeled_layout_tokens_list = (
            SampleTrainingTeiParser().parse_training_tei_to_labeled_layout_tokens_list(
                tei_root
            )
        )
        assert len(labeled_layout_tokens_list) == 1
        labeled_layout_tokens = labeled_layout_tokens_list[0]
        assert len(labeled_layout_tokens) == 2
        assert isinstance(labeled_layout_tokens[0], LabeledLayoutToken)
        assert labeled_layout_tokens[0].layout_token.text == TOKEN_1
        assert labeled_layout_tokens[1].layout_token.text == TOKEN_2
        assert (
            labeled_layout_tokens[0].layout_token.line_meta
            != labeled_layout_tokens[1].layout_token.line_meta
        )

    def test_should_ignore_from_and_to_attributes_when_matching_element_path(self):
        # e.g. <biblScope unit="page" from="1619" to="1621"> added by
        # CitationTeiTrainingDataGenerator.get_post_processed_xml_root for sciencebeam-judge;
        # from/to must not affect which label the element resolves to
        tei_root = _get_training_tei_with_text([
            E('biblScope', {'unit': 'page', 'from': '1619', 'to': '1621'}, TOKEN_1, E('lb')),
            '\n'
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<page>')
        ]]

    def test_should_raise_error_for_element_with_more_than_one_matching_attribute(self):
        tei_root = _get_training_tei_with_text([
            E('biblScope', {'unit': 'page', 'other': 'value'}, TOKEN_1, E('lb')),
            '\n'
        ])
        with pytest.raises(ValueError):
            SampleTrainingTeiParser().parse_training_tei_to_tag_result(
                tei_root
            )

    def test_unannotated_tokens_after_annotated_span_use_O(self):
        # training_data.py emits plain 'O' for all unannotated tokens regardless of
        # position. The I-<other> GROBID convention is applied later by
        # translate_tag_result_tags_IOB_to_grobid in generate_delft_data.py.
        tei_root = _get_training_tei_with_text([
            E('head', TOKEN_1, E('lb')),
            '\n',
            TOKEN_2, E('lb'),
            '\n',
            TOKEN_3, E('lb'),
            '\n',
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<head>'),
            (TOKEN_2, 'O'),
            (TOKEN_3, 'O'),
        ]]
