import logging
from typing import Sequence, Union

import pytest

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
from sciencebeam_parser.models.fulltext.data import FullTextDataGenerator
from sciencebeam_parser.models.fulltext.training_data import (
    ROOT_TRAINING_XML_ELEMENT_PATH,
    TRAINING_XML_ELEMENT_PATH_BY_LABEL,
    FullTextTeiTrainingDataGenerator,
    FullTextTrainingTeiParser
)
from sciencebeam_parser.models.training_data import OTHER_LABELS
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


def get_data_generator() -> FullTextDataGenerator:
    return FullTextDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)


def get_tei_training_data_generator() -> FullTextTeiTrainingDataGenerator:
    return FullTextTeiTrainingDataGenerator()


def get_training_tei_parser() -> FullTextTrainingTeiParser:
    return FullTextTrainingTeiParser()


@log_on_exception
class TestFullTextTeiTrainingDataGenerator:
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
        text_nodes = tei_xpath(xml_root, './text')
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
        text_nodes = xml_root.xpath('./text')
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
        text_nodes = xml_root.xpath('./text')
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
        data_generator = get_data_generator()
        model_data_iterable = data_generator.iter_model_data_for_layout_document(
            layout_document
        )
        training_data_generator = get_tei_training_data_generator()
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

    def test_should_generate_tei_for_most_labels(self):
        label_and_layout_line_list = [
            ('<section>', get_next_layout_line_for_text('Section Title 1')),
            ('<paragraph>', get_next_layout_line_for_text('Paragraph 1')),
            ('<citation_marker>', get_next_layout_line_for_text('See Citation 1')),
            ('<figure_marker>', get_next_layout_line_for_text('See Figure 1')),
            ('<table_marker>', get_next_layout_line_for_text('See Table 1')),
            ('<equation_marker>', get_next_layout_line_for_text('See Eq 1')),
            ('<section_marker>', get_next_layout_line_for_text('See Section 1')),
            ('<figure>', get_next_layout_line_for_text('Figure 1')),
            ('<table>', get_next_layout_line_for_text('Table 1')),
            ('<equation>', get_next_layout_line_for_text('Eq 1')),
            ('<item>', get_next_layout_line_for_text('Item 1')),
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
            xml_root.xpath('./text/head')
        ) == ['Section Title 1']
        assert get_text_content_list(
            xml_root.xpath('./text/p/ref[@type="biblio"]')
        ) == ['See Citation 1']
        assert get_text_content_list(
            xml_root.xpath('./text/p/ref[@type="table"]')
        ) == ['See Table 1']
        assert get_text_content_list(
            xml_root.xpath('./text/p/ref[@type="figure"]')
        ) == ['See Figure 1']
        assert get_text_content_list(
            xml_root.xpath('./text/p/ref[@type="formula"]')
        ) == ['See Eq 1']
        assert get_text_content_list(
            xml_root.xpath('./text/p/ref[@type="section"]')
        ) == ['See Section 1']
        assert get_text_content_list(
            xml_root.xpath('./text/figure[not(@type="table")]')
        ) == ['Figure 1']
        assert get_text_content_list(
            xml_root.xpath('./text/figure[@type="table"]')
        ) == ['Table 1']
        assert get_text_content_list(
            xml_root.xpath('./text/formula')
        ) == ['Eq 1']
        assert get_text_content_list(
            xml_root.xpath('./text/item')
        ) == ['Item 1']

    def test_should_add_equation_label_at_the_end_inside_equation_element(self):
        label_and_layout_line_list = [
            ('<equation>', get_next_layout_line_for_text('Eq 1')),
            ('<equation_label>', get_next_layout_line_for_text('Eq Label 1')),
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
            xml_root.xpath('./text/formula/label')
        ) == ['Eq Label 1']
        assert get_text_content_list(
            xml_root.xpath('./text/formula')
        ) == ['Eq 1\nEq Label 1']

    def test_should_add_equation_label_to_the_middle_of_formula_element(self):
        label_and_layout_line_list = [
            ('<equation>', get_next_layout_line_for_text('Eq 1')),
            ('<equation_label>', get_next_layout_line_for_text('Eq Label 1')),
            ('I-<equation>', get_next_layout_line_for_text('Continued Eq 1'))
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
            xml_root.xpath('./text/formula/label')
        ) == ['Eq Label 1']
        assert get_text_content_list(
            xml_root.xpath('./text/formula')
        ) == ['\n'.join(['Eq 1', 'Eq Label 1', 'Continued Eq 1'])]

    def test_should_generate_tei_from_model_data_using_model_labels(self):
        label_and_layout_line_list = [
            ('<section>', get_next_layout_line_for_text(TEXT_1)),
            ('<paragraph>', get_next_layout_line_for_text(TEXT_2))
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
            xml_root.xpath('./text/head')
        ) == [TEXT_1]
        assert get_text_content_list(
            xml_root.xpath('./text/p')
        ) == [TEXT_2]
        assert get_text_content_list(
            xml_root.xpath('./text')
        ) == [f'{TEXT_1}\n{TEXT_2}\n']

    def test_should_map_other_label_to_note(self):
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
        assert get_text_content_list(
            xml_root.xpath('./text/note[@type="other"]')
        ) == [TEXT_1]
        assert get_text_content_list(
            xml_root.xpath('./text')
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
            xml_root.xpath('./text/note[@type="unknown"]')
        ) == [TEXT_1]
        assert get_text_content_list(
            xml_root.xpath('./text')
        ) == [f'{TEXT_1}\n']

    def test_should_not_join_separate_labels(self):
        label_and_layout_line_list = [
            ('<section>', get_next_layout_line_for_text(TEXT_1)),
            ('<section>', get_next_layout_line_for_text(TEXT_2))
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
            xml_root.xpath('./text/head')
        ) == [TEXT_1, TEXT_2]
        assert get_text_content_list(
            xml_root.xpath('./text')
        ) == [f'{TEXT_1}\n{TEXT_2}\n']


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


class TestTableTrainingTeiParser:
    def test_should_parse_single_token_labelled_training_tei_lines(self):
        tei_root = _get_training_tei_with_text([
            E('head', TOKEN_1, E('lb')),
            '\n',
            E('p', TOKEN_2, E('lb')),
            '\n'
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<section>'),
            (TOKEN_2, 'B-<paragraph>')
        ]]

    def test_should_continue_the_paragraph_after_a_nested_citation_marker(self):
        tei_root = _get_training_tei_with_text([
            E('p', TOKEN_1, ' ', E('ref', {'type': 'biblio'}, TOKEN_2), ' ', TOKEN_3, E('lb')),
            '\n',
            E('p', TOKEN_4, E('lb')),
            '\n'
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<paragraph>'),
            (TOKEN_2, 'B-<citation_marker>'),
            (TOKEN_3, 'I-<paragraph>'),
            (TOKEN_4, 'B-<paragraph>')
        ]]

    def test_should_parse_single_label_with_multiple_lines(self):
        tei_root = _get_training_tei_with_text([
            E('p', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb')),
            '\n'
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<paragraph>'),
            (TOKEN_2, 'I-<paragraph>')
        ]]

    def test_should_output_multiple_tokens_of_each_unlabelled_lines(self):
        tei_root = _get_training_tei_with_text([
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
        tei_root = _get_training_tei_with_text([E(
            'p',
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
            (TOKEN_1, 'B-<paragraph>'),
            (TOKEN_2, 'I-<paragraph>'),
            (TOKEN_3, 'I-<paragraph>'),
            (TOKEN_4, 'I-<paragraph>')
        ]]

    def test_should_continue_paragraph_after_child_element(self):
        tei_root = _get_training_tei_with_text([E(
            'p',
            TOKEN_1,
            E('lb'),
            '\n',
            E('ref', {'type': 'biblio'}, TOKEN_2, E('lb')),
            '\n',
            TOKEN_3,
            E('lb'),
            '\n'
        )])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, 'B-<paragraph>'),
            (TOKEN_2, 'B-<citation_marker>'),
            (TOKEN_3, 'I-<paragraph>')
        ]]

    @pytest.mark.parametrize(
        "tei_label,element_path",
        list(TRAINING_XML_ELEMENT_PATH_BY_LABEL.items())
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
                (TOKEN_1, '<other>'),
                (TOKEN_2, '<other>'),
                (TOKEN_3, '<other>'),
                (TOKEN_4, '<other>')
            ]]
        else:
            assert tag_result == [[
                (TOKEN_1, f'B-{tei_label}'),
                (TOKEN_2, f'I-{tei_label}'),
                (TOKEN_3, f'I-{tei_label}'),
                (TOKEN_4, f'I-{tei_label}')
            ]]
