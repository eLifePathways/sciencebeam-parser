import logging
from collections import Counter
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
        )


TRAINING_XML_ELEMENT_PATH_BY_LABEL_WITH_ROOT = {
    **TRAINING_XML_ELEMENT_PATH_BY_LABEL,
    # segmentation and header both have one: a label whose path is the root
    '<other>': ROOT_TRAINING_XML_ELEMENT_PATH,
}


class SampleTrainingTeiParserWithRootLabel(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL_WITH_ROOT
            ),
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

    def test_should_match_an_element_path_whatever_namespace_it_is_in(self):
        # GROBID's own segmentation corpus ships both dialects: 230 of its 925 TEI
        # files are in the TEI namespace and the rest are not
        xml_writer = XmlTreeWriter(TEI_E('tei'), element_maker=TEI_E)
        xml_writer.require_path(ROOT_TRAINING_XML_ELEMENT_PATH)
        xml_writer.append_all(TEI_E('head', TOKEN_1, TEI_E('lb')), '\n')
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            xml_writer.root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<head>')
        ]]

    def test_should_take_the_label_of_the_innermost_recognised_element(self):
        tei_root = _get_training_tei_with_text([
            E('back', E('head', TOKEN_1, E('lb'))),
            '\n'
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<head>')
        ]]

    def test_should_fall_back_to_the_enclosing_label_for_an_unrecognised_element(self):
        tei_root = _get_training_tei_with_text([
            E('head', E('unknown', TOKEN_1, E('lb'))),
            '\n'
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<head>')
        ]]

    def test_should_not_take_the_root_label_over_the_enclosing_one(self):
        # a label whose path is the root is a key of its own, and reaching it
        # before the enclosing element would answer `<other>` for everything
        # unrecognised rather than the region the line is printed in
        tei_root = _get_training_tei_with_text([
            E('head', E('unknown', TOKEN_1, E('lb'))),
            '\n'
        ])
        parser = SampleTrainingTeiParserWithRootLabel()
        assert parser.parse_training_tei_to_tag_result(tei_root) == [[
            (TOKEN_1, 'B-<head>')
        ]]

    def test_should_fall_back_to_the_root_label_with_no_enclosing_element(self):
        tei_root = _get_training_tei_with_text([
            E('unknown', TOKEN_1, E('lb')),
            '\n'
        ])
        parser = SampleTrainingTeiParserWithRootLabel()
        assert parser.parse_training_tei_to_tag_result(tei_root) == [[
            (TOKEN_1, '<other>')
        ]]

    def test_should_count_an_element_path_it_did_not_recognise(self):
        tei_root = _get_training_tei_with_text([
            E('back', E('head', TOKEN_1, E('lb'))),
            '\n',
            E('head', TOKEN_2, E('lb')),
            '\n'
        ])
        unrecognised_path_counts: Counter = Counter()
        SampleTrainingTeiParser().parse_training_tei_to_labeled_layout_tokens_list(
            tei_root,
            unrecognised_path_counts=unrecognised_path_counts
        )
        assert dict(unrecognised_path_counts) == {'back/head': 1}

    def test_should_ignore_a_comment_between_lines(self):
        tei_root = _get_training_tei_with_text([
            E('head', TOKEN_1, E('lb')),
            etree.Comment('a note the annotator left'),
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

    def test_should_start_a_new_entity_for_each_sibling_element_with_the_same_label(self):
        tei_root = _get_training_tei_with_text([
            E('p', TOKEN_1, E('lb')),
            '\n',
            E('p', TOKEN_2, E('lb')),
            '\n'
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<paragraph>'),
            (TOKEN_2, 'B-<paragraph>')
        ]]

    @pytest.mark.parametrize('separator', ['\n', ' ', '\n  ', ', '])
    def test_should_start_a_new_entity_for_a_sibling_element_whatever_separates_them(
        self, separator: str
    ):
        tei_root = _get_training_tei_with_text([
            E('p', TOKEN_1, E('lb')),
            separator,
            E('p', TOKEN_2, E('lb')),
            '\n'
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        label_by_token = dict(tag_result[0])
        assert label_by_token[TOKEN_1] == 'B-<paragraph>'
        assert label_by_token[TOKEN_2] == 'B-<paragraph>'

    def test_should_keep_an_element_spanning_multiple_lines_as_one_entity(self):
        tei_root = _get_training_tei_with_text([
            E('p', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb')),
            '\n'
        ])
        tag_result = SampleTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<paragraph>'),
            (TOKEN_2, 'I-<paragraph>')
        ]]

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
