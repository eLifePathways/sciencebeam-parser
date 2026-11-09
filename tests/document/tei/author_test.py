import logging

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutBlock
)
from sciencebeam_parser.document.semantic_document import (
    SemanticAddressLine,
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticCountry,
    SemanticDepartment,
    SemanticGivenName,
    SemanticInstitution,
    SemanticLaboratory,
    SemanticMarker,
    SemanticPostBox,
    SemanticPostCode,
    SemanticRegion,
    SemanticSettlement,
    SemanticSurname,
)
from sciencebeam_parser.document.tei.common import TeiElementWrapper
from sciencebeam_parser.document.tei.factories import (
    DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
)
from sciencebeam_parser.document.tei.author import (
    get_tei_affiliation_for_semantic_affiliation_address_element,
    get_tei_author_for_semantic_author_element
)
from tests.document.tei.common_test import (
    ITALICS_FONT_1,
    BOLD_FONT_1
)


LOGGER = logging.getLogger(__name__)


class TestGetTeiAffiliationForSemanticAffiliationAddress:
    def test_should_add_all_fields(self):
        semantic_affiliation_address = SemanticAffiliationAddress([
            SemanticMarker(layout_block=LayoutBlock.for_text('1')),
            SemanticInstitution(layout_block=LayoutBlock.for_text('Institution1')),
            SemanticDepartment(layout_block=LayoutBlock.for_text('Department1')),
            SemanticLaboratory(layout_block=LayoutBlock.for_text('Lab1')),
            SemanticAddressLine(layout_block=LayoutBlock.for_text('AddressLine1')),
            SemanticPostCode(layout_block=LayoutBlock.for_text('PostCode1')),
            SemanticPostBox(layout_block=LayoutBlock.for_text('PostBox1')),
            SemanticRegion(layout_block=LayoutBlock.for_text('Region1')),
            SemanticSettlement(layout_block=LayoutBlock.for_text('Settlement1')),
            SemanticCountry(layout_block=LayoutBlock.for_text('Country1')),
        ])
        tei_aff = TeiElementWrapper(
            get_tei_affiliation_for_semantic_affiliation_address_element(
                semantic_affiliation_address,
                context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
            )
        )
        LOGGER.debug('tei_aff: %r', etree.tostring(tei_aff.element))
        assert tei_aff.get_xpath_text_content_list(
            'tei:note[@type="raw_affiliation"]'
        ) == [semantic_affiliation_address.get_text()]
        assert tei_aff.get_xpath_text_content_list(
            'tei:note[@type="raw_affiliation"]/tei:label'
        ) == ['1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:orgName[@type="institution"]'
        ) == ['Institution1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:orgName[@type="department"]'
        ) == ['Department1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:orgName[@type="laboratory"]'
        ) == ['Lab1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:address/tei:addrLine'
        ) == ['AddressLine1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:address/tei:postCode'
        ) == ['PostCode1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:address/tei:postBox'
        ) == ['PostBox1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:address/tei:region'
        ) == ['Region1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:address/tei:settlement'
        ) == ['Settlement1']
        assert tei_aff.get_xpath_text_content_list(
            'tei:address/tei:country'
        ) == ['Country1']

    def test_should_add_raw_affiliation_without_marker(self):
        semantic_affiliation_address = SemanticAffiliationAddress([
            SemanticInstitution(layout_block=LayoutBlock.for_text('Institution1'))
        ])
        tei_aff = TeiElementWrapper(
            get_tei_affiliation_for_semantic_affiliation_address_element(
                semantic_affiliation_address,
                context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
            )
        )
        LOGGER.debug('tei_aff: %r', etree.tostring(tei_aff.element))
        assert tei_aff.get_xpath_text_content_list(
            'tei:note[@type="raw_affiliation"]'
        ) == [semantic_affiliation_address.get_text()]
        assert tei_aff.get_xpath_text_content_list(
            'tei:note[@type="raw_affiliation"]/tei:label'
        ) == []

    def test_should_add_raw_affiliation_with_formatting(self):
        semantic_affiliation_address = SemanticAffiliationAddress([
            SemanticMarker(layout_block=LayoutBlock.for_text('1')),
            SemanticInstitution(layout_block=LayoutBlock.merge_blocks([
                LayoutBlock.for_text('bold', font=BOLD_FONT_1),
                LayoutBlock.for_text('italic', font=ITALICS_FONT_1)
            ]))
        ])
        tei_aff = TeiElementWrapper(
            get_tei_affiliation_for_semantic_affiliation_address_element(
                semantic_affiliation_address,
                context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
            )
        )
        LOGGER.debug('tei_aff: %r', etree.tostring(tei_aff.element))
        assert tei_aff.get_xpath_text_content_list(
            'tei:note[@type="raw_affiliation"]'
        ) == [semantic_affiliation_address.get_text()]
        assert tei_aff.get_xpath_text_content_list(
            'tei:note[@type="raw_affiliation"]/tei:label'
        ) == ['1']


class TestGetTeiAuthorForSemanticAffiliationAddress:
    def test_should_add_all_fields(self):
        affiliations_by_marker = {
            '1': [SemanticAffiliationAddress([
                SemanticInstitution(layout_block=LayoutBlock.for_text('Institution1'))
            ])],
            '2': [SemanticAffiliationAddress([
                SemanticInstitution(layout_block=LayoutBlock.for_text('Institution2'))
            ])],
            '3': [SemanticAffiliationAddress([
                SemanticInstitution(layout_block=LayoutBlock.for_text('Other'))
            ])]
        }
        semantic_author = SemanticAuthor([
            SemanticMarker(layout_block=LayoutBlock.for_text('1')),
            SemanticMarker(layout_block=LayoutBlock.for_text('2')),
            SemanticGivenName(layout_block=LayoutBlock.for_text('GivenName1')),
            SemanticSurname(layout_block=LayoutBlock.for_text('Surname1'))
        ])
        tei_author = TeiElementWrapper(
            get_tei_author_for_semantic_author_element(
                semantic_author,
                context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT,
                affiliations_by_marker=affiliations_by_marker
            )
        )
        LOGGER.debug('tei_author: %r', etree.tostring(tei_author.element))
        assert tei_author.get_xpath_text_content_list(
            '//tei:note[@type="marker"]'
        ) == ['1', '2']
        assert tei_author.get_xpath_text_content_list(
            'tei:persName/tei:forename'
        ) == ['GivenName1']
        assert tei_author.get_xpath_text_content_list(
            'tei:persName/tei:surname'
        ) == ['Surname1']
        assert tei_author.get_xpath_text_content_list(
            'tei:affiliation/tei:orgName[@type="institution"]'
        ) == ['Institution1', 'Institution2']
