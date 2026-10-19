# pylint: disable=not-callable
import logging
from typing import Sequence

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine
)
from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT
)
from sciencebeam_parser.models.training_data import (
    OTHER_LABELS
)
from sciencebeam_parser.models.segmentation.data import SegmentationDataGenerator
from sciencebeam_parser.models.segmentation.training_data import (
    TRAINING_XML_ELEMENT_PATH_BY_LABEL,
    SegmentationTeiTrainingDataGenerator,
    SegmentationTrainingTeiParser
)
from sciencebeam_parser.utils.xml_writer import XmlTreeWriter
from sciencebeam_parser.utils.xml import get_text_content, get_text_content_list

from tests.models.training_data_test_utils import (
    get_labeled_model_data_list,
    get_model_data_list_for_layout_document,
    get_next_layout_line_for_text
)


LOGGER = logging.getLogger(__name__)


TEXT_1 = 'this is text 1'
TEXT_2 = 'this is text 2'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'
TOKEN_4 = 'token4'


def get_data_generator() -> SegmentationDataGenerator:
    return SegmentationDataGenerator(
        DEFAULT_DOCUMENT_FEATURES_CONTEXT,
        use_first_token_of_block=True
    )


class TestSegmentationTeiTrainingDataGenerator:
    def test_should_include_layout_document_text_in_tei_output(self):
        training_data_generator = SegmentationTeiTrainingDataGenerator()
        layout_document = LayoutDocument.for_blocks([LayoutBlock.for_text(TEXT_1)])
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            get_model_data_list_for_layout_document(
                layout_document,
                data_generator=get_data_generator()
            )
        )
        text_nodes = xml_root.xpath('./text')
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == TEXT_1

    def test_should_keep_original_whitespace(self):
        training_data_generator = SegmentationTeiTrainingDataGenerator()
        text = 'Token1, Token2  ,Token3'
        layout_document = LayoutDocument.for_blocks([LayoutBlock(lines=[
            LayoutLine.for_text(text, tail_whitespace='\n')
        ])])
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            get_model_data_list_for_layout_document(
                layout_document,
                data_generator=get_data_generator()
            )
        )
        text_nodes = xml_root.xpath('./text')
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == text

    def test_should_add_line_feeds(self):
        training_data_generator = SegmentationTeiTrainingDataGenerator()
        layout_document = LayoutDocument.for_blocks([LayoutBlock(lines=[
            LayoutLine.for_text(TEXT_1, tail_whitespace='\n'),
            LayoutLine.for_text(TEXT_2, tail_whitespace='\n')
        ])])
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            get_model_data_list_for_layout_document(
                layout_document,
                data_generator=get_data_generator()
            )
        )
        text_nodes = xml_root.xpath('./text')
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == '\n'.join([TEXT_1, TEXT_2])

    def test_should_lb_elements_before_line_feeds(self):
        training_data_generator = SegmentationTeiTrainingDataGenerator()
        layout_document = LayoutDocument.for_blocks([LayoutBlock(lines=[
            LayoutLine.for_text(TEXT_1, tail_whitespace='\n'),
            LayoutLine.for_text(TEXT_2, tail_whitespace='\n')
        ])])
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            get_model_data_list_for_layout_document(
                layout_document,
                data_generator=get_data_generator()
            )
        )
        text_nodes = xml_root.xpath('./text')
        assert len(text_nodes) == 1
        lb_nodes = text_nodes[0].xpath('lb')
        assert len(lb_nodes) == 2
        assert lb_nodes[0].getparent().text == TEXT_1
        assert lb_nodes[0].tail == '\n' + TEXT_2

    def test_should_generate_tei_from_model_data(self):
        layout_document = LayoutDocument.for_blocks([LayoutBlock(lines=[
            get_next_layout_line_for_text(TEXT_1),
            get_next_layout_line_for_text(TEXT_2)
        ])])
        data_generator = SegmentationDataGenerator(
            DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            use_first_token_of_block=True
        )
        model_data_iterable = data_generator.iter_model_data_for_layout_document(
            layout_document
        )
        training_data_generator = SegmentationTeiTrainingDataGenerator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            model_data_iterable
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        text_nodes = xml_root.xpath('./text')
        assert len(text_nodes) == 1
        lb_nodes = text_nodes[0].xpath('lb')
        assert len(lb_nodes) == 2
        assert lb_nodes[0].getparent().text == TEXT_1
        assert lb_nodes[0].tail == '\n' + TEXT_2

    def test_should_generate_tei_from_model_data_using_model_labels(self):
        label_and_layout_line_list = [
            ('<header>', get_next_layout_line_for_text(TEXT_1)),
            ('<body>', get_next_layout_line_for_text(TEXT_2))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = SegmentationTeiTrainingDataGenerator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_text_content_list(xml_root.xpath('./text/front')) == [TEXT_1]
        assert get_text_content_list(xml_root.xpath('./text/body')) == [TEXT_2]
        assert get_text_content_list(xml_root.xpath('./text')) == [f'{TEXT_1}\n{TEXT_2}\n']

    def test_should_map_unknown_label_to_note(self):
        label_and_layout_line_list = [
            ('<unknown>', get_next_layout_line_for_text(TEXT_1))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = SegmentationTeiTrainingDataGenerator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_text_content_list(xml_root.xpath('./text/note[@type="unknown"]')) == [TEXT_1]
        assert get_text_content_list(xml_root.xpath('./text')) == [f'{TEXT_1}\n']


class TestSegmentationTrainingTeiParser:
    def test_should_parse_single_token_labelled_training_tei_lines(self):
        tei_root = E('tei', E('text', *[
            E('front', TOKEN_1, E('lb')),
            '\n',
            E('body', TOKEN_2, E('lb')),
            '\n'
        ]))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<header>'),
            (TOKEN_2, 'B-<body>')
        ]]

    def test_should_parse_review_as_its_own_label(self):
        # <review> is ours, for the peer-review sub-articles ORE prints; GROBID has no
        # counterpart, which is why it has to survive our own reader unchanged
        tei_root = E('tei', E('text', *[
            E('div', {'type': 'review'}, TOKEN_1, E('lb')),
            '\n'
        ]))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<review>')
        ]]

    def test_should_parse_review_nested_in_a_wrapper_as_its_own_label(self):
        tei_root = E('tei', E('text', *[
            E('back', E('div', {'type': 'review'}, TOKEN_1, E('lb'))),
            '\n'
        ]))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<review>')
        ]]

    def test_should_restart_a_region_after_a_nested_one(self):
        # GROBID nests a page number inside the body, and its writer starts a new
        # chunk after it; ours writes the two as siblings, so nothing here moves
        # for a corpus we generated
        tei_root = E('tei', E('text', *[
            E('body', TOKEN_1, E('lb'), '\n',
              E('page', TOKEN_2, E('lb')), '\n',
              TOKEN_3, E('lb')),
            '\n'
        ]))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<body>'),
            (TOKEN_2, 'B-<page>'),
            (TOKEN_3, 'B-<body>'),
        ]]

    def test_should_end_a_line_at_an_element_without_a_line_break(self):
        # GROBID's corpus prints a page number between two body lines as
        # `<page>12 </page>` with no <lb/> of its own; accumulating across the
        # boundary would hand the body line that follows a line text of both
        tei_root = E('tei', E('text', *[
            E('body', TOKEN_1, E('lb'), '\n',
              E('page', TOKEN_2),
              TOKEN_3, E('lb')),
            '\n'
        ]))
        parser = SegmentationTrainingTeiParser()
        tag_result = parser.parse_training_tei_to_tag_result(tei_root)
        assert tag_result == [[
            (TOKEN_1, 'B-<body>'),
            (TOKEN_2, 'B-<page>'),
            (TOKEN_3, 'B-<body>'),
        ]]
        line_texts = [
            labeled_layout_token.layout_token.line_meta.text
            for labeled_layout_token
            in parser.parse_training_tei_to_labeled_layout_tokens_list(tei_root)[0]
        ]
        assert line_texts == [TOKEN_1, TOKEN_2, TOKEN_3]

    def test_should_parse_single_label_with_multiple_lines(self):
        tei_root = E('tei', E('text', *[
            E('front', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb')),
            '\n'
        ]))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<header>'),
            (TOKEN_2, 'I-<header>')
        ]]

    def test_should_only_output_first_token_of_unlabelled_lines(self):
        tei_root = E('tei', E('text', *[
            TOKEN_1,
            ' ',
            TOKEN_2,
            E('lb'),
            '\n',
            TOKEN_3,
            ' ',
            TOKEN_4,
            E('lb'),
            '\n'
        ]))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, 'O'),
            (TOKEN_3, 'O')
        ]]

    def test_should_parse_single_label_with_multiple_tokens_on_multiple_lines(self):
        tei_root = E('tei', E('text', *[
            E(
                'front',
                TOKEN_1,
                ' ',
                TOKEN_2,
                E('lb'),
                '\n',
                TOKEN_3,
                ' ',
                TOKEN_4,
                E('lb')
            ),
            '\n'
        ]))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, 'B-<header>'),
            (TOKEN_3, 'I-<header>')
        ]]

    @pytest.mark.parametrize(
        "tei_label,element_path", list(TRAINING_XML_ELEMENT_PATH_BY_LABEL.items())
    )
    def test_should_parse_all_supported_labels(
        self,
        tei_label: str,
        element_path: Sequence[str]
    ):
        xml_writer = XmlTreeWriter(E('tei'), element_maker=E)
        xml_writer.require_path(element_path)
        xml_writer.append_all(
            TOKEN_1,
            ' ',
            TOKEN_2,
            E('lb'),
            '\n',
            TOKEN_3,
            ' ',
            TOKEN_4,
            E('lb')
        )
        tei_root = xml_writer.root
        LOGGER.debug('tei_root: %r', etree.tostring(tei_root))
        tag_result = SegmentationTrainingTeiParser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        if tei_label in OTHER_LABELS:
            assert tag_result == [[
                (TOKEN_1, 'O'),
                (TOKEN_3, 'O')
            ]]
        else:
            assert tag_result == [[
                (TOKEN_1, f'B-{tei_label}'),
                (TOKEN_3, f'I-{tei_label}')
            ]]
