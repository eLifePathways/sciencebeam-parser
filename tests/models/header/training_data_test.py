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
from sciencebeam_parser.models.header.data import HeaderDataGenerator
from sciencebeam_parser.models.header.training_data import (
    ROOT_TRAINING_XML_ELEMENT_PATH,
    TRAINING_XML_ELEMENT_PATH_BY_LABEL_WITHOUT_ALIAS,
    HeaderTeiTrainingDataGenerator,
    HeaderTrainingTeiParser
)
from sciencebeam_parser.models.training_data import OTHER_LABELS
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
TEXT_3 = 'this is text 3'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'
TOKEN_4 = 'token4'


def get_data_generator() -> HeaderDataGenerator:
    return HeaderDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)


def get_tei_training_data_generator() -> HeaderTeiTrainingDataGenerator:
    return HeaderTeiTrainingDataGenerator()


def get_training_tei_parser() -> HeaderTrainingTeiParser:
    return HeaderTrainingTeiParser()


class TestHeaderTeiTrainingDataGenerator:
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
        text_nodes = xml_root.xpath('./text/front')
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
        text_nodes = xml_root.xpath('./text/front')
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
        text_nodes = xml_root.xpath('./text/front')
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
        text_nodes = xml_root.xpath('./text/front')
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
        text_nodes = xml_root.xpath('./text/front')
        assert len(text_nodes) == 1
        lb_nodes = text_nodes[0].xpath('lb')
        assert len(lb_nodes) == 2
        assert lb_nodes[0].getparent().text == TEXT_1
        assert lb_nodes[0].tail == '\n' + TEXT_2

    def test_should_generate_tei_from_model_data_using_model_labels(self):
        label_and_layout_line_list = [
            ('<title>', get_next_layout_line_for_text(TEXT_1)),
            ('<abstract>', get_next_layout_line_for_text(TEXT_2))
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
            xml_root.xpath('./text/front/docTitle/titlePart')
        ) == [TEXT_1]
        assert get_text_content_list(
            xml_root.xpath('./text/front/div[@type="abstract"]')
        ) == [TEXT_2]
        assert get_text_content_list(
            xml_root.xpath('./text/front')
        ) == [f'{TEXT_1}\n{TEXT_2}\n']

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
            xml_root.xpath('./text/front/note[@type="unknown"]')
        ) == [TEXT_1]
        assert get_text_content_list(
            xml_root.xpath('./text/front')
        ) == [f'{TEXT_1}\n']

    def test_should_not_open_a_second_front_for_a_label_mapped_to_the_root(self):
        label_and_layout_line_list = [
            ('O', get_next_layout_line_for_text(TEXT_1)),
            ('<note>', get_next_layout_line_for_text(TEXT_2))
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
        assert len(xml_root.xpath('./text/front')) == 1
        assert get_text_content_list(
            xml_root.xpath('./text/front')
        ) == [f'{TEXT_1}\n{TEXT_2}\n']

    def test_should_put_other_text_between_labels_directly_under_front(self):
        label_and_layout_line_list = [
            ('<title>', get_next_layout_line_for_text(TEXT_1)),
            ('O', get_next_layout_line_for_text(TEXT_2)),
            ('<author>', get_next_layout_line_for_text(TEXT_3))
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
        assert [
            get_text_content(node).strip()
            for node in xml_root.xpath('./text/front/docTitle')
        ] == [TEXT_1]
        assert [
            get_text_content(node).strip()
            for node in xml_root.xpath('./text/front/byline')
        ] == [TEXT_3]
        assert get_text_content_list(
            xml_root.xpath('./text/front')
        ) == [f'{TEXT_1}\n{TEXT_2}\n{TEXT_3}\n']

    def test_should_not_join_separate_labels(self):
        label_and_layout_line_list = [
            ('<title>', get_next_layout_line_for_text(TEXT_1)),
            ('<title>', get_next_layout_line_for_text(TEXT_2))
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
            xml_root.xpath('./text/front/docTitle/titlePart')
        ) == [TEXT_1, TEXT_2]
        assert get_text_content_list(
            xml_root.xpath('./text/front')
        ) == [f'{TEXT_1}\n{TEXT_2}\n']


class TestHeaderTrainingTeiParser:
    def test_should_parse_single_token_labelled_training_tei_lines(self):
        tei_root = E('tei', E('text', E('front', *[
            E('docTitle', E('titlePart', TOKEN_1, E('lb'))),
            '\n',
            E('byline', E('docAuthor', TOKEN_2, E('lb'))),
            '\n'
        ])))
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<title>'),
            (TOKEN_2, 'B-<author>')
        ]]

    def test_should_parse_single_label_with_multiple_lines(self):
        tei_root = E('tei', E('text', E('front', *[
            E('docTitle', E('titlePart', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb'))),
            '\n'
        ])))
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<title>'),
            (TOKEN_2, 'I-<title>')
        ]]

    def test_should_not_fail_with_line_feed_inside_doctitle_element(self):
        tei_root = E('tei', E('text', E('front', *[
            E('docTitle', '\n', E('titlePart', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb')), '\n'),
        ])))
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<title>'),
            (TOKEN_2, 'I-<title>')
        ]]

    def test_should_interpret_text_in_byline_as_unlabelled(self):
        tei_root = E('tei', E('text', E('front', *[
            E('byline', TOKEN_1, E('lb')),
            '\n'
        ])))
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'O')
        ]]

    def test_should_output_multiple_tokens_of_each_unlabelled_lines(self):
        tei_root = E('tei', E('text', E('front', *[
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
        ])))
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
        tei_root = E('tei', E('text', E('front', *[
            E('docTitle', E(
                'titlePart',
                TOKEN_1,
                ' ',
                TOKEN_2,
                E('lb'),
                '\n',
                TOKEN_3,
                ' ',
                TOKEN_4,
                E('lb')
            )),
            '\n'
        ])))
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, 'B-<title>'),
            (TOKEN_2, 'I-<title>'),
            (TOKEN_3, 'I-<title>'),
            (TOKEN_4, 'I-<title>')
        ]]

    @pytest.mark.parametrize(
        "tei_label,element_path",
        list(TRAINING_XML_ELEMENT_PATH_BY_LABEL_WITHOUT_ALIAS.items())
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
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        if tei_label in OTHER_LABELS or element_path == ROOT_TRAINING_XML_ELEMENT_PATH:
            assert tag_result == [[
                (TOKEN_1, 'O'),
                (TOKEN_2, 'O'),
                (TOKEN_3, 'O'),
                (TOKEN_4, 'O')
            ]]
        else:
            assert tag_result == [[
                (TOKEN_1, f'B-{tei_label}'),
                (TOKEN_2, f'I-{tei_label}'),
                (TOKEN_3, f'I-{tei_label}'),
                (TOKEN_4, f'I-{tei_label}')
            ]]

    def test_should_parse_generated_other_text_back_to_the_same_labels(self):
        label_and_layout_line_list = [
            ('<title>', get_next_layout_line_for_text(TOKEN_1)),
            ('O', get_next_layout_line_for_text(TOKEN_2)),
            ('<author>', get_next_layout_line_for_text(TOKEN_3))
        ]
        xml_root = get_tei_training_data_generator(
        ).get_training_tei_xml_for_model_data_iterable(
            get_labeled_model_data_list(
                label_and_layout_line_list,
                data_generator=get_data_generator()
            )
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        assert get_training_tei_parser().parse_training_tei_to_tag_result(xml_root) == [[
            (TOKEN_1, 'B-<title>'),
            (TOKEN_2, 'O'),
            (TOKEN_3, 'B-<author>')
        ]]
