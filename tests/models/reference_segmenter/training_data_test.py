import logging
from typing import Sequence, Union

from lxml import etree
from lxml.builder import E

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine
)
from sciencebeam_parser.document.tei.common import tei_xpath
from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT
)
from sciencebeam_parser.models.reference_segmenter.data import ReferenceSegmenterDataGenerator
from sciencebeam_parser.models.reference_segmenter.training_data import (
    ROOT_TRAINING_XML_ELEMENT_PATH,
    ReferenceSegmenterTeiTrainingDataGenerator,
    ReferenceSegmenterTrainingTeiParser
)
from sciencebeam_parser.utils.xml import get_text_content, get_text_content_list
from sciencebeam_parser.utils.xml_writer import XmlTreeWriter

from tests.test_utils import log_on_exception
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


def get_data_generator() -> ReferenceSegmenterDataGenerator:
    return ReferenceSegmenterDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)


def get_tei_training_data_generator() -> ReferenceSegmenterTeiTrainingDataGenerator:
    return ReferenceSegmenterTeiTrainingDataGenerator()


def get_training_tei_parser() -> ReferenceSegmenterTrainingTeiParser:
    return ReferenceSegmenterTrainingTeiParser()


@log_on_exception
class TestReferenceSegmenterTeiTrainingDataGenerator:
    def test_should_include_layout_document_text_in_tei_output(self):
        training_data_generator = get_tei_training_data_generator()
        layout_document = LayoutDocument.for_blocks([LayoutBlock.for_text(TEXT_1)])
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            get_model_data_list_for_layout_document(
                layout_document,
                data_generator=get_data_generator()
            )
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        text_nodes = tei_xpath(xml_root, './text/listBibl')
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == TEXT_1

    def test_should_keep_original_whitespace(self):
        training_data_generator = get_tei_training_data_generator()
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
        text_nodes = xml_root.xpath('./text/listBibl')
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == text

    def test_should_add_line_feeds(self):
        training_data_generator = get_tei_training_data_generator()
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
        text_nodes = xml_root.xpath('./text/listBibl')
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == '\n'.join([TEXT_1, TEXT_2])

    def test_should_lb_elements_before_line_feeds(self):
        training_data_generator = get_tei_training_data_generator()
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
        text_nodes = xml_root.xpath('./text/listBibl')
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
        data_generator = get_data_generator()
        model_data_iterable = data_generator.iter_model_data_for_layout_document(
            layout_document
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            model_data_iterable
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        text_nodes = xml_root.xpath('./text/listBibl')
        assert len(text_nodes) == 1
        lb_nodes = text_nodes[0].xpath('lb')
        assert len(lb_nodes) == 2
        assert lb_nodes[0].getparent().text == TEXT_1
        assert lb_nodes[0].tail == '\n' + TEXT_2

    def test_should_generate_tei_single_bibl_element_without_label(self):
        label_and_layout_line_list = [
            ('<reference>', get_next_layout_line_for_text('Ref 1'))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl/bibl')
        ) == ['Ref 1']

    def test_should_generate_single_tei_bibl_element_with_label_at_the_beginning(self):
        label_and_layout_line_list = [
            ('<label>', get_next_layout_line_for_text('Label 1')),
            ('<reference>', get_next_layout_line_for_text('Ref 1'))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl/bibl/label')
        ) == ['Label 1']
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl/bibl')
        ) == ['Label 1\nRef 1']

    def test_should_generate_multiple_tei_bibl_elements_with_label(self):
        label_and_layout_line_list = [
            ('<label>', get_next_layout_line_for_text('Label 1')),
            ('<reference>', get_next_layout_line_for_text('Ref 1')),
            ('<label>', get_next_layout_line_for_text('Label 2')),
            ('<reference>', get_next_layout_line_for_text('Ref 2'))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl/bibl/label')
        ) == ['Label 1', 'Label 2']
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl/bibl')
        ) == ['Label 1\nRef 1', 'Label 2\nRef 2']

    def test_should_generate_multiple_tei_bibl_elements_without_label(self):
        label_and_layout_line_list = [
            ('<reference>', get_next_layout_line_for_text(TEXT_1)),
            ('<reference>', get_next_layout_line_for_text(TEXT_2))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl/bibl')
        ) == [TEXT_1, TEXT_2]
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl')
        ) == [f'{TEXT_1}\n{TEXT_2}\n']

    def test_should_map_other_label_as_text_without_note(self):
        label_and_layout_line_list = [
            ('<other>', get_next_layout_line_for_text(TEXT_1))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert not xml_root.xpath('./text/listBibl//note')
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl')
        ) == [f'{TEXT_1}\n']

    def test_should_map_unknown_label_to_note(self):
        label_and_layout_line_list = [
            ('<unknown>', get_next_layout_line_for_text(TEXT_1))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl/note[@type="unknown"]')
        ) == [TEXT_1]
        assert get_text_content_list(
            xml_root.xpath('./text/listBibl')
        ) == [f'{TEXT_1}\n']


# pylint: disable=not-callable

def _get_training_tei_with_references(
    references: Sequence[Union[etree.ElementBase, str]]
) -> etree.ElementBase:
    xml_writer = XmlTreeWriter(E('tei'), element_maker=E)
    xml_writer.require_path(
        ROOT_TRAINING_XML_ELEMENT_PATH
    )
    xml_writer.append_all(*references)
    LOGGER.debug('training tei: %r', etree.tostring(xml_writer.root))
    return xml_writer.root


class TestTableTrainingTeiParser:
    def test_should_parse_single_token_labelled_training_tei_lines(self):
        tei_root = _get_training_tei_with_references([E(
            'bibl',
            E('label', TOKEN_1, E('lb')),
            '\n',
            TOKEN_2, E('lb'),
            '\n'
        )])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<label>'),
            (TOKEN_2, 'B-<reference>')
        ]]

    def test_should_start_a_new_reference_for_each_bibl_without_a_label(self):
        tei_root = _get_training_tei_with_references([
            E('bibl', TOKEN_1, E('lb')),
            '\n',
            E('bibl', TOKEN_2, E('lb')),
            '\n'
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<reference>'),
            (TOKEN_2, 'B-<reference>')
        ]]

    def test_should_start_the_reference_after_whitespace_preceding_the_label(self):
        tei_root = _get_training_tei_with_references([E(
            'bibl',
            ' ',
            E('label', TOKEN_1),
            ' ',
            TOKEN_2,
            E('lb'),
            '\n'
        )])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<label>'),
            (TOKEN_2, 'B-<reference>')
        ]]

    def test_should_keep_a_reference_spanning_multiple_lines_as_one_entity(self):
        tei_root = _get_training_tei_with_references([E(
            'bibl',
            TOKEN_1,
            E('lb'),
            '\n',
            TOKEN_2,
            E('lb'),
            '\n'
        )])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<reference>'),
            (TOKEN_2, 'I-<reference>')
        ]]

    def test_should_parse_single_label_with_multiple_lines(self):
        tei_root = _get_training_tei_with_references([E(
            'bibl',
            E('label', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb')),
            '\n'
        )])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<label>'),
            (TOKEN_2, 'I-<label>')
        ]]

    def test_should_output_multiple_tokens_of_each_unlabelled_lines(self):
        tei_root = _get_training_tei_with_references([
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
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, 'O'),
            (TOKEN_2, 'O'),
            (TOKEN_3, 'O'),
            (TOKEN_4, 'O')
        ]]

    def test_should_parse_single_label_with_multiple_tokens_on_multiple_lines(self):
        tei_root = _get_training_tei_with_references([E(
            'bibl',
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
        )])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, 'B-<reference>'),
            (TOKEN_2, 'I-<reference>'),
            (TOKEN_3, 'I-<reference>'),
            (TOKEN_4, 'I-<reference>')
        ]]
