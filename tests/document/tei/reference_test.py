import logging

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutBlock
)
from sciencebeam_parser.document.semantic_document import (
    SemanticAuthor,
    SemanticDate,
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticExternalUrl,
    SemanticGivenName,
    SemanticIssue,
    SemanticBookTitle,
    SemanticJournal,
    SemanticLabel,
    SemanticLocation,
    SemanticPageRange,
    SemanticPublisher,
    SemanticRawEditors,
    SemanticReference,
    SemanticSurname,
    SemanticTitle,
    SemanticVolume
)
from sciencebeam_parser.document.tei.common import (
    TeiElementWrapper
)
from sciencebeam_parser.document.tei.factories import (
    DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
)
from sciencebeam_parser.document.tei.references import (
    get_tei_reference_element
)


LOGGER = logging.getLogger(__name__)


WEB_URL_1 = 'http://host/path'
DOI_1 = '10.1234/test'


class TestGetTeiReference:
    def test_should_add_all_fields(self):
        semantic_ref = SemanticReference([
            SemanticLabel(layout_block=LayoutBlock.for_text('1.')),
            SemanticTitle(layout_block=LayoutBlock.for_text('Title 1')),
            SemanticAuthor([
                SemanticGivenName(layout_block=LayoutBlock.for_text('Given Name 1')),
                SemanticSurname(layout_block=LayoutBlock.for_text('Surname 1'))
            ]),
            SemanticRawEditors(layout_block=LayoutBlock.for_text('Editor 1')),
            SemanticJournal(layout_block=LayoutBlock.for_text('Journal 1')),
            SemanticBookTitle(layout_block=LayoutBlock.for_text('Book Title 1')),
            SemanticVolume(layout_block=LayoutBlock.for_text('Volume 1')),
            SemanticIssue(layout_block=LayoutBlock.for_text('Issue 1')),
            SemanticPageRange(
                layout_block=LayoutBlock.for_text('12-15'),
                from_page='12',
                to_page='15'
            ),
            SemanticPublisher(layout_block=LayoutBlock.for_text('Publisher 1')),
            SemanticLocation(layout_block=LayoutBlock.for_text('Location 1')),
            SemanticExternalUrl(
                layout_block=LayoutBlock.for_text(WEB_URL_1),
                value=WEB_URL_1
            ),
            SemanticExternalIdentifier(
                layout_block=LayoutBlock.for_text(DOI_1),
                value=DOI_1,
                external_identifier_type=SemanticExternalIdentifierTypes.DOI
            ),
            SemanticDate(
                layout_block=LayoutBlock.for_text('1991'),
                year=1991
            ),
            SemanticDate(
                layout_block=LayoutBlock.for_text('1992'),
                year=1992
            ),
        ])
        tei_ref = TeiElementWrapper(get_tei_reference_element(
            semantic_ref,
            context=DEFAULT_TEI_ELEMENT_FACTORY_CONTEXT
        ))
        LOGGER.debug('tei_ref: %r', etree.tostring(tei_ref.element))
        assert tei_ref.get_xpath_text_content_list(
            'tei:note[@type="label"]'
        ) == ['1.']
        assert tei_ref.get_xpath_text_content_list(
            'tei:analytic/tei:title[@type="main"]'
        ) == ['Title 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:analytic/tei:author/tei:persName/tei:forename'
        ) == ['Given Name 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:analytic/tei:author/tei:persName/tei:surname'
        ) == ['Surname 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:editor'
        ) == ['Editor 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:title[@level="j"]'
        ) == ['Journal 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:title[@level="m"]'
        ) == ['Book Title 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:imprint/tei:biblScope[@unit="volume"]'
        ) == ['Volume 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:imprint/tei:biblScope[@unit="issue"]'
        ) == ['Issue 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:imprint/tei:biblScope[@unit="page"]/@from'
        ) == ['12']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:imprint/tei:biblScope[@unit="page"]/@to'
        ) == ['15']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:imprint/tei:publisher'
        ) == ['Publisher 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:meeting/tei:address/tei:addrLine'
        ) == ['Location 1']
        assert tei_ref.get_xpath_text_content_list(
            'tei:analytic/tei:idno[@type="DOI"]'
        ) == [DOI_1]
        assert tei_ref.get_xpath_text_content_list(
            'tei:ref[@type="url"]'
        ) == [WEB_URL_1]
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:imprint/tei:date[@type="published"]/@when'
        ) == ['1991']
        assert tei_ref.get_xpath_text_content_list(
            'tei:monogr/tei:imprint/tei:date[not(@type)]/@when'
        ) == ['1992']
