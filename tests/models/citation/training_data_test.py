import logging
from typing import Iterable, Optional, Sequence, Tuple

import pytest
from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine
)
from sciencebeam_parser.document.semantic_document import (
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticReference
)
from sciencebeam_parser.document.tei.common import (
    TEI_E,
    get_tei_xpath_text_content_list,
    tei_xpath,
    TEI_NS_PREFIX
)
from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT,
    LayoutModelData
)
from sciencebeam_parser.models.model import (
    iter_entity_layout_blocks_for_labeled_layout_tokens
)
from sciencebeam_parser.models.citation.data import CitationDataGenerator
from sciencebeam_parser.models.citation.extract import CitationSemanticExtractor
from sciencebeam_parser.models.citation.labels import IDENTIFIER_LABEL
from sciencebeam_parser.models.citation.training_data import (
    ROOT_TRAINING_XML_ELEMENT_PATH,
    TRAINING_XML_ELEMENT_PATH_BY_LABEL,
    CitationTeiTrainingDataGenerator,
    CitationTrainingTeiParser
)
from sciencebeam_parser.utils.xml import get_text_content
from sciencebeam_parser.utils.xml_writer import XmlTreeWriter

from tests.test_utils import log_on_exception
from tests.models.training_data_test_utils import (
    get_labeled_model_data_list,
    get_labeled_model_data_list_list,
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


BIBL_XPATH = './tei:text/tei:back/tei:listBibl/tei:bibl'


def get_data_generator() -> CitationDataGenerator:
    return CitationDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)


def get_tei_training_data_generator() -> CitationTeiTrainingDataGenerator:
    return CitationTeiTrainingDataGenerator()


def get_training_tei_parser() -> CitationTrainingTeiParser:
    return CitationTrainingTeiParser()


def get_training_tei_xml_for_model_data_iterable(
    model_data_iterable: Iterable[LayoutModelData]
) -> etree.ElementBase:
    training_data_generator = get_tei_training_data_generator()
    xml_root = training_data_generator.get_training_tei_xml_for_model_data_iterable(
        model_data_iterable
    )
    LOGGER.debug('xml: %r', etree.tostring(xml_root))
    return xml_root


def get_semantic_references_for_training_tei_xml(
    xml_root: etree.ElementBase
) -> Sequence[SemanticReference]:
    extractor = CitationSemanticExtractor()
    labeled_layout_tokens_list = (
        get_training_tei_parser().parse_training_tei_to_labeled_layout_tokens_list(xml_root)
    )
    return [
        semantic_content
        for labeled_layout_tokens in labeled_layout_tokens_list
        for semantic_content in extractor.iter_semantic_content_for_entity_blocks(
            iter_entity_layout_blocks_for_labeled_layout_tokens(labeled_layout_tokens)
        )
        if isinstance(semantic_content, SemanticReference)
    ]


def get_external_identifier_types_and_values(
    semantic_reference: SemanticReference
) -> Sequence[Tuple[Optional[str], Optional[str]]]:
    return [
        (external_identifier.external_identifier_type, external_identifier.value)
        for external_identifier in semantic_reference.iter_by_type(SemanticExternalIdentifier)
    ]


def get_training_tei_xml_for_layout_document(
    layout_document: LayoutDocument
) -> etree.ElementBase:
    return get_training_tei_xml_for_model_data_iterable(
        get_model_data_list_for_layout_document(
            layout_document,
            data_generator=get_data_generator()
        )
    )


@log_on_exception
class TestCitationTeiTrainingDataGenerator:
    def test_should_include_layout_document_text_in_tei_output(self):
        layout_document = LayoutDocument.for_blocks([LayoutBlock.for_text(TEXT_1)])
        xml_root = get_training_tei_xml_for_layout_document(layout_document)
        assert xml_root.tag == TEI_NS_PREFIX + 'TEI'
        text_nodes = tei_xpath(xml_root, BIBL_XPATH)
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == TEXT_1

    def test_should_keep_original_whitespace(self):
        text = 'Token1, Token2  ,Token3'
        layout_document = LayoutDocument.for_blocks([LayoutBlock(lines=[
            LayoutLine.for_text(text, tail_whitespace='\n')
        ])])
        xml_root = get_training_tei_xml_for_layout_document(layout_document)
        text_nodes = tei_xpath(xml_root, BIBL_XPATH)
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == text

    def test_should_add_line_feeds(self):
        layout_document = LayoutDocument.for_blocks([LayoutBlock(lines=[
            LayoutLine.for_text(TEXT_1, tail_whitespace='\n'),
            LayoutLine.for_text(TEXT_2, tail_whitespace='\n')
        ])])
        xml_root = get_training_tei_xml_for_layout_document(layout_document)
        text_nodes = tei_xpath(xml_root, BIBL_XPATH)
        assert len(text_nodes) == 1
        assert get_text_content(text_nodes[0]).rstrip() == '\n'.join([TEXT_1, TEXT_2])

    def test_should_lb_elements_before_line_feeds(self):
        layout_document = LayoutDocument.for_blocks([LayoutBlock(lines=[
            LayoutLine.for_text(TEXT_1, tail_whitespace='\n'),
            LayoutLine.for_text(TEXT_2, tail_whitespace='\n')
        ])])
        xml_root = get_training_tei_xml_for_layout_document(layout_document)
        text_nodes = tei_xpath(xml_root, BIBL_XPATH)
        assert len(text_nodes) == 1
        lb_nodes = tei_xpath(text_nodes[0], 'tei:lb')
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
        xml_root = get_training_tei_xml_for_model_data_iterable(model_data_iterable)
        text_nodes = tei_xpath(xml_root, BIBL_XPATH)
        assert len(text_nodes) == 1
        lb_nodes = tei_xpath(text_nodes[0], 'tei:lb')
        assert len(lb_nodes) == 2
        assert lb_nodes[0].getparent().text == TEXT_1
        assert lb_nodes[0].tail == '\n' + TEXT_2

    def test_should_generate_tei_for_most_labels(self):
        label_and_layout_line_list = [
            ('<title>', get_next_layout_line_for_text('Title 1')),
            ('<author>', get_next_layout_line_for_text('Author 1')),
            ('<editor>', get_next_layout_line_for_text('Editor 1')),
            ('<institution>', get_next_layout_line_for_text('Institution 1')),
            ('<collaboration>', get_next_layout_line_for_text('Collaboration 1')),
            ('<journal>', get_next_layout_line_for_text('Journal 1')),
            ('<series>', get_next_layout_line_for_text('Series 1')),
            ('<booktitle>', get_next_layout_line_for_text('Book Title 1')),
            ('<date>', get_next_layout_line_for_text('Date 1')),
            ('<volume>', get_next_layout_line_for_text('Volume 1')),
            ('<issue>', get_next_layout_line_for_text('Issue 1')),
            ('<pages>', get_next_layout_line_for_text('Pages 1')),
            ('<publisher>', get_next_layout_line_for_text('Publisher 1')),
            ('<location>', get_next_layout_line_for_text('Location 1')),
            ('<tech>', get_next_layout_line_for_text('Tech 1')),
            ('<pubnum>', get_next_layout_line_for_text('Pubnum 1')),
            ('<web>', get_next_layout_line_for_text('Web 1')),
            ('<note>', get_next_layout_line_for_text('Note 1'))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        xml_root = get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:title[@level="a"]'
        ) == ['Title 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:author'
        ) == ['Author 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:editor'
        ) == ['Editor 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:orgName[not(@type)]'
        ) == ['Institution 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:orgName[@type="collaboration"]'
        ) == ['Collaboration 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:title[@level="j"]'
        ) == ['Journal 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:title[@level="s"]'
        ) == ['Series 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:title[@level="m"]'
        ) == ['Book Title 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:date'
        ) == ['Date 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:biblScope[@unit="volume"]'
        ) == ['Volume 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:biblScope[@unit="issue"]'
        ) == ['Issue 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:biblScope[@unit="page"]'
        ) == ['Pages 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:publisher'
        ) == ['Publisher 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:pubPlace'
        ) == ['Location 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:note[@type="report"]'
        ) == ['Tech 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:idno'
        ) == ['Pubnum 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:ptr[@type="web"]'
        ) == ['Web 1']
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:note[not(@type)]'
        ) == ['Note 1']

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
        assert not tei_xpath(xml_root, f'{BIBL_XPATH}//tei:note')
        assert get_tei_xpath_text_content_list(
            xml_root, BIBL_XPATH
        ) == [f'{TEXT_1}\n']

    def test_should_map_unknown_label_to_note(self):
        label_and_layout_line_list = [
            ('<unknown>', get_next_layout_line_for_text(TEXT_1))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        xml_root = get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:note[@type="unknown"]'
        ) == [TEXT_1]

    @pytest.mark.parametrize(
        "test_input,expected_type",
        [
            ('10.1234/test', SemanticExternalIdentifierTypes.DOI),
            ('PMID: 1234567', SemanticExternalIdentifierTypes.PMID),
            ('arXiv: 0706.0001', SemanticExternalIdentifierTypes.ARXIV),
            ('xyz', None)
        ]
    )
    def test_should_map_pubnum_to_idno_with_type_if_detected(
        self,
        test_input: str,
        expected_type: Optional[str]
    ):
        label_and_layout_line_list = [
            ('<pubnum>', get_next_layout_line_for_text(test_input))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        xml_root = get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        if expected_type:
            assert get_tei_xpath_text_content_list(
                xml_root, f'{BIBL_XPATH}/tei:idno[@type="{expected_type}"]'
            ) == [test_input]
        else:
            assert get_tei_xpath_text_content_list(
                xml_root, f'{BIBL_XPATH}/tei:idno[not(@type)]'
            ) == [test_input]

    def test_should_generate_tei_from_multiple_model_data_lists_using_model_labels(self):
        label_and_layout_line_list_list = [
            [
                ('<title>', get_next_layout_line_for_text(TEXT_1))
            ], [
                ('<title>', get_next_layout_line_for_text(TEXT_2))
            ]
        ]
        labeled_model_data_list_list = get_labeled_model_data_list_list(
            label_and_layout_line_list_list,
            data_generator=get_data_generator()
        )
        training_data_generator = get_tei_training_data_generator()
        xml_root = training_data_generator.get_training_tei_xml_for_multiple_model_data_iterables(
            labeled_model_data_list_list
        )
        LOGGER.debug('xml: %r', etree.tostring(xml_root))
        aff_nodes = tei_xpath(xml_root, BIBL_XPATH)
        assert len(aff_nodes) == 2
        assert get_tei_xpath_text_content_list(
            aff_nodes[0], './tei:title[@level="a"]'
        ) == [TEXT_1]
        assert get_tei_xpath_text_content_list(
            aff_nodes[1], './tei:title[@level="a"]'
        ) == [TEXT_2]


def _get_training_tei_with_references(
    references: Sequence[etree.ElementBase]
) -> etree.ElementBase:
    xml_writer = XmlTreeWriter(TEI_E('tei'), element_maker=TEI_E)
    xml_writer.require_path(
        ROOT_TRAINING_XML_ELEMENT_PATH[:-1]
    )
    xml_writer.append_all(*references)
    LOGGER.debug('training tei: %r', etree.tostring(xml_writer.root))
    return xml_writer.root


class TestCitationTrainingTeiParser:
    def test_should_parse_single_token_labelled_training_tei_lines(self):
        tei_root = _get_training_tei_with_references([
            TEI_E('bibl', *[
                TEI_E('author', TOKEN_1, TEI_E('lb')),
                '\n',
                TEI_E('title', {'level': 'a'}, TOKEN_2, TEI_E('lb')),
                '\n'
            ])
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<author>'),
            (TOKEN_2, 'B-<title>')
        ]]

    def test_should_start_a_new_entity_for_sibling_identifiers_separated_by_whitespace(self):
        tei_root = _get_training_tei_with_references([
            TEI_E('bibl', *[
                TEI_E('idno', TOKEN_1),
                ' ',
                TEI_E('idno', TOKEN_2, TEI_E('lb')),
                '\n'
            ])
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<pubnum>'),
            (TOKEN_2, 'B-<pubnum>')
        ]]

    def test_should_parse_single_label_with_multiple_lines(self):
        tei_root = _get_training_tei_with_references([
            TEI_E('bibl', *[
                TEI_E(
                    'author',
                    TOKEN_1,
                    TEI_E('lb'),
                    '\n',
                    TOKEN_2,
                    TEI_E('lb')
                ),
                '\n',
            ])
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'B-<author>'),
            (TOKEN_2, 'I-<author>')
        ]]

    def test_should_interpret_text_in_bibl_as_unlabelled(self):
        tei_root = _get_training_tei_with_references([
            TEI_E('bibl', *[
                TOKEN_1, TEI_E('lb'),
                '\n'
            ])
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, 'O')
        ]]

    def test_should_output_multiple_tokens_of_each_unlabelled_lines(self):
        tei_root = _get_training_tei_with_references([
            TEI_E('bibl', *[
                TOKEN_1,
                ' ',
                TOKEN_2,
                TEI_E('lb'),
                '\n',
                TOKEN_3,
                ' ',
                TOKEN_4,
                TEI_E('lb'),
                '\n'
            ])
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
        tei_root = _get_training_tei_with_references([
            TEI_E('bibl', *[
                TEI_E(
                    'author',
                    TOKEN_1,
                    ' ',
                    TOKEN_2,
                    TEI_E('lb'),
                    '\n',
                    TOKEN_3,
                    ' ',
                    TOKEN_4,
                    TEI_E('lb')
                ),
                '\n',
            ])
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, 'B-<author>'),
            (TOKEN_2, 'I-<author>'),
            (TOKEN_3, 'I-<author>'),
            (TOKEN_4, 'I-<author>')
        ]]

    @pytest.mark.parametrize(
        "external_identifier_type",
        [
            SemanticExternalIdentifierTypes.ARXIV,
            SemanticExternalIdentifierTypes.DOI,
            SemanticExternalIdentifierTypes.PII,
            SemanticExternalIdentifierTypes.PMCID,
            SemanticExternalIdentifierTypes.PMID,
            'other'
        ]
    )
    def test_should_parse_any_idno_type_as_identifier_label(
        self,
        external_identifier_type: str
    ):
        tei_root = _get_training_tei_with_references([
            TEI_E('bibl', *[
                TEI_E('idno', {'type': external_identifier_type}, TOKEN_1, TEI_E('lb')),
                '\n'
            ])
        ])
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        assert tag_result == [[
            (TOKEN_1, f'B-{IDENTIFIER_LABEL}')
        ]]

    @pytest.mark.parametrize(
        "text,expected_type,expected_value",
        [
            ('10.1234/test', SemanticExternalIdentifierTypes.DOI, '10.1234/test'),
            ('arXiv: 0706.0001', SemanticExternalIdentifierTypes.ARXIV, '0706.0001'),
            ('PMID: 1234567', SemanticExternalIdentifierTypes.PMID, '1234567'),
            ('PMC1234567', SemanticExternalIdentifierTypes.PMCID, 'PMC1234567')
        ]
    )
    def test_should_round_trip_identifier_to_typed_semantic_external_identifier(
        self,
        text: str,
        expected_type: str,
        expected_value: str
    ):
        label_and_layout_line_list = [
            (IDENTIFIER_LABEL, get_next_layout_line_for_text(text))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        xml_root = get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        assert get_tei_xpath_text_content_list(
            xml_root, f'{BIBL_XPATH}/tei:idno[@type="{expected_type}"]'
        ) == [text]
        references = get_semantic_references_for_training_tei_xml(xml_root)
        assert len(references) == 1
        assert get_external_identifier_types_and_values(references[0]) == [
            (expected_type, expected_value)
        ]

    def test_should_round_trip_multiple_identifiers_of_one_reference(self):
        label_and_layout_line_list = [
            ('<title>', get_next_layout_line_for_text('Title 1')),
            (IDENTIFIER_LABEL, get_next_layout_line_for_text('10.1234/test')),
            ('O', get_next_layout_line_for_text('and')),
            (IDENTIFIER_LABEL, get_next_layout_line_for_text('arXiv: 0706.0001')),
            ('O', get_next_layout_line_for_text('and')),
            (IDENTIFIER_LABEL, get_next_layout_line_for_text('PMID: 1234567'))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        xml_root = get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        references = get_semantic_references_for_training_tei_xml(xml_root)
        assert len(references) == 1
        assert get_external_identifier_types_and_values(references[0]) == [
            (SemanticExternalIdentifierTypes.DOI, '10.1234/test'),
            (SemanticExternalIdentifierTypes.ARXIV, '0706.0001'),
            (SemanticExternalIdentifierTypes.PMID, '1234567')
        ]

    def test_should_round_trip_directly_adjacent_identifiers_of_one_reference(self):
        label_and_layout_line_list = [
            (IDENTIFIER_LABEL, get_next_layout_line_for_text('10.1234/test')),
            (IDENTIFIER_LABEL, get_next_layout_line_for_text('PMID: 1234567'))
        ]
        labeled_model_data_list = get_labeled_model_data_list(
            label_and_layout_line_list,
            data_generator=get_data_generator()
        )
        xml_root = get_training_tei_xml_for_model_data_iterable(
            labeled_model_data_list
        )
        assert len(tei_xpath(xml_root, f'{BIBL_XPATH}/tei:idno')) == 2
        references = get_semantic_references_for_training_tei_xml(xml_root)
        assert len(references) == 1
        assert get_external_identifier_types_and_values(references[0]) == [
            (SemanticExternalIdentifierTypes.DOI, '10.1234/test'),
            (SemanticExternalIdentifierTypes.PMID, '1234567')
        ]

    @pytest.mark.parametrize(
        "tei_label,element_path",
        list(TRAINING_XML_ELEMENT_PATH_BY_LABEL.items())
    )
    def test_should_parse_all_supported_labels(
        self,
        tei_label: str,
        element_path: Sequence[str]
    ):
        xml_writer = XmlTreeWriter(TEI_E('tei'), element_maker=TEI_E)
        xml_writer.require_path(element_path)
        xml_writer.append_all(
            TOKEN_1,
            ' ',
            TOKEN_2,
            TEI_E('lb'),
            '\n',
            TOKEN_3,
            ' ',
            TOKEN_4,
            TEI_E('lb')
        )
        tei_root = xml_writer.root
        LOGGER.debug('tei_root: %r', etree.tostring(tei_root))
        tag_result = get_training_tei_parser().parse_training_tei_to_tag_result(
            tei_root
        )
        LOGGER.debug('tag_result: %r', tag_result)
        assert tag_result == [[
            (TOKEN_1, f'B-{tei_label}'),
            (TOKEN_2, f'I-{tei_label}'),
            (TOKEN_3, f'I-{tei_label}'),
            (TOKEN_4, f'I-{tei_label}')
        ]]
