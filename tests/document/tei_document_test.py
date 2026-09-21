import logging

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutGraphic,
    LayoutPageCoordinates
)
from sciencebeam_parser.document.semantic_document import (
    SemanticAbstract,
    SemanticAffiliationAddress,
    SemanticCaption,
    SemanticDocument,
    SemanticAuthor,
    SemanticFigure,
    SemanticFigureCitation,
    SemanticGivenName,
    SemanticGraphic,
    SemanticHeading,
    SemanticInstitution,
    SemanticLabel,
    SemanticMarker,
    SemanticMiddleName,
    SemanticMixedNote,
    SemanticNameSuffix,
    SemanticNameTitle,
    SemanticParagraph,
    SemanticRawEquation,
    SemanticRawEquationContent,
    SemanticRawReference,
    SemanticRawReferenceText,
    SemanticReference,
    SemanticReferenceCitation,
    SemanticReferenceList,
    SemanticSection,
    SemanticSectionTypes,
    SemanticSurname,
    SemanticTable,
    SemanticTableCitation,
    SemanticTextContentWrapper,
    SemanticTitle
)
from sciencebeam_parser.document.tei.attribution import DocumentAttribution
from sciencebeam_parser.document.tei_document import (
    get_tei_for_semantic_document
)
from tests.document.tei.common_test import (
    TOKEN_1,
    TOKEN_2
)


LOGGER = logging.getLogger(__name__)


WEB_URL_1 = 'http://host/path'
DOI_1 = '10.1234/test'

COORDINATES_1 = LayoutPageCoordinates(10, 20, 110, 120)


class TestGetTeiForSemanticDocument:  # pylint: disable=too-many-public-methods
    def test_should_return_empty_document(self):
        semantic_document = SemanticDocument()
        tei_document = get_tei_for_semantic_document(semantic_document)
        assert not tei_document.xpath('//tei:div')

    def test_should_set_manuscript_title(self):
        semantic_document = SemanticDocument()
        semantic_document.front.add_content(
            SemanticTitle(layout_block=LayoutBlock.for_text(TOKEN_1))
        )
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]'
        ) == [TOKEN_1]

    def test_should_set_abstract(self):
        semantic_document = SemanticDocument()
        semantic_document.front.add_content(
            SemanticAbstract(LayoutBlock.for_text(TOKEN_1))
        )
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:abstract/tei:p'
        ) == [TOKEN_1]

    def test_should_add_single_author(self):
        semantic_document = SemanticDocument()
        title = SemanticNameTitle(layout_block=LayoutBlock.for_text('Title1'))
        given_name = SemanticGivenName(layout_block=LayoutBlock.for_text('Given1'))
        middle_name = SemanticMiddleName(layout_block=LayoutBlock.for_text('Middle1'))
        surname = SemanticSurname(layout_block=LayoutBlock.for_text('Surname1'))
        suffix = SemanticNameSuffix(layout_block=LayoutBlock.for_text('Suffix1'))
        author = SemanticAuthor([title, given_name, middle_name, surname, suffix])
        semantic_document.front.add_content(author)
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:roleName'
        ) == ['Title1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:forename[@type="first"]'
        ) == ['Given1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:forename[@type="middle"]'
        ) == ['Middle1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:surname'
        ) == ['Surname1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:genName'
        ) == ['Suffix1']

    def test_should_use_author_name_part_values(self):
        semantic_document = SemanticDocument()
        given_name = SemanticGivenName(layout_block=LayoutBlock.for_text('GIVEN1'))
        given_name.value = 'Given1'
        middle_name = SemanticMiddleName(layout_block=LayoutBlock.for_text('MIDDLE1'))
        middle_name.value = 'Middle1'
        surname = SemanticSurname(layout_block=LayoutBlock.for_text('SURNAME1'))
        surname.value = 'Surname1'
        author = SemanticAuthor([given_name, middle_name, surname])
        semantic_document.front.add_content(author)
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:forename[@type="first"]'
        ) == ['Given1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:forename[@type="middle"]'
        ) == ['Middle1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:surname'
        ) == ['Surname1']

    def test_should_add_single_author_with_affiliation(self):
        semantic_document = SemanticDocument()
        title = SemanticNameTitle(layout_block=LayoutBlock.for_text('Title1'))
        given_name = SemanticGivenName(layout_block=LayoutBlock.for_text('Given1'))
        middle_name = SemanticMiddleName(layout_block=LayoutBlock.for_text('Middle1'))
        surname = SemanticSurname(layout_block=LayoutBlock.for_text('Surname1'))
        suffix = SemanticNameSuffix(layout_block=LayoutBlock.for_text('Suffix1'))
        author_marker = SemanticMarker(layout_block=LayoutBlock.for_text('1'))
        author = SemanticAuthor([title, given_name, middle_name, surname, suffix, author_marker])
        aff_marker = SemanticMarker(layout_block=LayoutBlock.for_text('1'))
        institution = SemanticInstitution(layout_block=LayoutBlock.for_text('Institution1'))
        aff = SemanticAffiliationAddress([aff_marker, institution], content_id='aff0')
        semantic_document.front.add_content(author)
        semantic_document.front.add_content(aff)
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:roleName'
        ) == ['Title1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:forename[@type="first"]'
        ) == ['Given1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:forename[@type="middle"]'
        ) == ['Middle1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:surname'
        ) == ['Surname1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author//tei:genName'
        ) == ['Suffix1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author/tei:affiliation/tei:note[@type="raw_affiliation"]'
        ) == [aff.get_text()]
        assert tei_document.get_xpath_text_content_list(
            '//tei:author/tei:affiliation/tei:note[@type="raw_affiliation"]/tei:label'
        ) == [aff_marker.get_text()]
        assert tei_document.get_xpath_text_content_list(
            '//tei:author/tei:affiliation/@key'
        ) == ['aff0']
        assert tei_document.get_xpath_text_content_list(
            '//tei:author/tei:affiliation/tei:orgName[@type="institution"]'
        ) == [institution.get_text()]

    def test_should_add_orphan_affiliation(self):
        semantic_document = SemanticDocument()
        aff_marker = SemanticMarker(layout_block=LayoutBlock.for_text('1'))
        institution = SemanticInstitution(layout_block=LayoutBlock.for_text('Institution1'))
        aff = SemanticAffiliationAddress([aff_marker, institution], content_id='aff0')
        semantic_document.front.add_content(aff)
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:author/tei:affiliation/tei:note[@type="raw_affiliation"]'
        ) == [aff.get_text()]
        assert tei_document.get_xpath_text_content_list(
            '//tei:author/tei:affiliation/tei:note[@type="raw_affiliation"]/tei:label'
        ) == [aff_marker.get_text()]
        assert tei_document.get_xpath_text_content_list(
            '//tei:author/tei:affiliation/@key'
        ) == ['aff0']

    def test_should_create_body_section(self):
        semantic_document = SemanticDocument()
        section = semantic_document.body_section.add_new_section()
        section.add_heading_block(LayoutBlock.for_text(TOKEN_1))
        paragraph = section.add_new_paragraph()
        paragraph.add_block_content(LayoutBlock.for_text(TOKEN_2))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:head'
        ) == [TOKEN_1]
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p'
        ) == [TOKEN_2]

    def test_should_create_back_section(self):
        semantic_document = SemanticDocument()
        section = semantic_document.back_section.add_new_section()
        section.add_heading_block(LayoutBlock.for_text(TOKEN_1))
        paragraph = section.add_new_paragraph()
        paragraph.add_block_content(LayoutBlock.for_text(TOKEN_2))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="annex"]/tei:div/tei:head'
        ) == [TOKEN_1]
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="annex"]/tei:div/tei:p'
        ) == [TOKEN_2]

    def test_should_create_acknowledgment_section(self):
        semantic_document = SemanticDocument()
        section = semantic_document.back_section.add_new_section(
            SemanticSectionTypes.ACKNOWLEDGEMENT
        )
        section.add_heading_block(LayoutBlock.for_text(TOKEN_1))
        paragraph = section.add_new_paragraph()
        paragraph.add_block_content(LayoutBlock.for_text(TOKEN_2))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="acknowledgement"]/tei:head'
        ) == [TOKEN_1]
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="acknowledgement"]/tei:p'
        ) == [TOKEN_2]

    def test_should_add_notes_to_section(self):
        semantic_document = SemanticDocument()
        section = semantic_document.body_section.add_new_section()
        section.add_note(LayoutBlock.for_text(TOKEN_1), 'other')
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:note[@type="other"]'
        ) == [TOKEN_1]

    def test_should_add_notes_to_body(self):
        semantic_document = SemanticDocument()
        semantic_document.body_section.add_note(LayoutBlock.for_text(TOKEN_1), 'other')
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:note[@type="other"]'
        ) == [TOKEN_1]

    def test_should_add_notes_to_back(self):
        semantic_document = SemanticDocument()
        semantic_document.back_section.add_note(LayoutBlock.for_text(TOKEN_1), 'other')
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="annex"]/tei:note[@type="other"]'
        ) == [TOKEN_1]

    def test_should_add_section_figures_to_body(self):
        semantic_document = SemanticDocument()
        semantic_document.body_section.add_content(SemanticSection([
            SemanticFigure([
                SemanticLabel(layout_block=LayoutBlock.for_text('Label 1')),
                SemanticCaption(layout_block=LayoutBlock.for_text('Caption 1'))
            ], content_id='fig_0')
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        figure_xpath = (
            '//tei:body/tei:figure[not(contains(@type, "table"))]'
        )
        assert tei_document.get_xpath_text_content_list(
            f'{figure_xpath}/tei:head'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{figure_xpath}/tei:label'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{figure_xpath}/tei:figDesc'
        ) == ['Caption 1']
        assert tei_document.get_xpath_text_content_list(f'{figure_xpath}/@xml:id') == ['fig_0']
        assert not tei_document.xpath(
            '//tei:back/tei:div[@type="annex"]/tei:div'
        )

    def test_should_add_section_figures_to_back(self):
        semantic_document = SemanticDocument()
        semantic_document.back_section.add_content(SemanticSection([
            SemanticFigure([
                SemanticLabel(layout_block=LayoutBlock.for_text('Label 1')),
                SemanticCaption(layout_block=LayoutBlock.for_text('Caption 1'))
            ], content_id='fig_0')
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        figure_xpath = (
            '//tei:back/tei:div[@type="annex"]/tei:figure[not(contains(@type, "table"))]'
        )
        assert tei_document.get_xpath_text_content_list(
            f'{figure_xpath}/tei:head'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{figure_xpath}/tei:label'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{figure_xpath}/tei:figDesc'
        ) == ['Caption 1']
        assert tei_document.get_xpath_text_content_list(f'{figure_xpath}/@xml:id') == ['fig_0']
        assert not tei_document.xpath(
            '//tei:back/tei:div[@type="annex"]/tei:div'
        )

    def test_should_add_section_tables_to_body(self):
        semantic_document = SemanticDocument()
        semantic_document.body_section.add_content(SemanticSection([
            SemanticTable([
                SemanticLabel(layout_block=LayoutBlock.for_text('Label 1')),
                SemanticCaption(layout_block=LayoutBlock.for_text('Caption 1'))
            ], content_id='tab_0')
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        table_xpath = '//tei:body/tei:figure[@type="table"]'
        assert tei_document.get_xpath_text_content_list(
            f'{table_xpath}/tei:head'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{table_xpath}/tei:label'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{table_xpath}/tei:figDesc'
        ) == ['Caption 1']
        assert tei_document.get_xpath_text_content_list(f'{table_xpath}/@xml:id') == ['tab_0']
        assert not tei_document.xpath(
            '//tei:body/tei:div'
        )

    def test_should_add_section_tables_to_back(self):
        semantic_document = SemanticDocument()
        semantic_document.back_section.add_content(SemanticSection([
            SemanticTable([
                SemanticLabel(layout_block=LayoutBlock.for_text('Label 1')),
                SemanticCaption(layout_block=LayoutBlock.for_text('Caption 1'))
            ], content_id='tab_0')
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        table_xpath = '//tei:back/tei:div[@type="annex"]/tei:figure[@type="table"]'
        assert tei_document.get_xpath_text_content_list(
            f'{table_xpath}/tei:head'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{table_xpath}/tei:label'
        ) == ['Label 1']
        assert tei_document.get_xpath_text_content_list(
            f'{table_xpath}/tei:figDesc'
        ) == ['Caption 1']
        assert tei_document.get_xpath_text_content_list(f'{table_xpath}/@xml:id') == ['tab_0']
        assert not tei_document.xpath(
            '//tei:body/tei:div'
        )

    def test_should_add_asset_citation_for_resolved_figure(self):
        semantic_document = SemanticDocument()
        semantic_document.body_section.add_content(SemanticSection([
            SemanticParagraph([
                SemanticTextContentWrapper(layout_block=LayoutBlock.for_text('See')),
                SemanticFigureCitation(
                    layout_block=LayoutBlock.for_text('Fig 1'),
                    target_content_id='fig_0'
                )
            ]),
            SemanticFigure([
                SemanticLabel(layout_block=LayoutBlock.for_text('Figure 1'))
            ], content_id='fig_0')
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p'
        ) == ['See Fig 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p/tei:ref[@type="figure"]'
        ) == ['Fig 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p/tei:ref[@type="figure"]/@target'
        ) == ['#fig_0']

    def test_should_add_asset_citation_for_resolved_table(self):
        semantic_document = SemanticDocument()
        semantic_document.body_section.add_content(SemanticSection([
            SemanticParagraph([
                SemanticTextContentWrapper(layout_block=LayoutBlock.for_text('See')),
                SemanticTableCitation(
                    layout_block=LayoutBlock.for_text('Tab 1'),
                    target_content_id='tab_0'
                )
            ]),
            SemanticTable([
                SemanticLabel(layout_block=LayoutBlock.for_text('Table 1'))
            ], content_id='tab_0')
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p'
        ) == ['See Tab 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p/tei:ref[@type="table"]'
        ) == ['Tab 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p/tei:ref[@type="table"]/@target'
        ) == ['#tab_0']

    def test_should_add_asset_citation_for_resolved_reference(self):
        semantic_document = SemanticDocument()
        semantic_document.body_section.add_content(SemanticSection([
            SemanticParagraph([
                SemanticTextContentWrapper(layout_block=LayoutBlock.for_text('See')),
                SemanticReferenceCitation(
                    layout_block=LayoutBlock.for_text('Ref 1'),
                    target_content_id='b0'
                )
            ]),
            SemanticReferenceList([
                SemanticReference([
                    SemanticLabel(layout_block=LayoutBlock.for_text('1'))
                ], content_id='b0')
            ])
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p'
        ) == ['See Ref 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p/tei:ref[@type="bibr"]'
        ) == ['Ref 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p/tei:ref[@type="bibr"]/@target'
        ) == ['#b0']

    def test_should_add_raw_equation_with_label_to_paragraph(self):
        # to be consistent with Java GROBID
        semantic_document = SemanticDocument()
        semantic_document.body_section.add_content(SemanticSection([
            SemanticParagraph([
                SemanticTextContentWrapper(layout_block=LayoutBlock.for_text('Next')),
                SemanticRawEquation([
                    SemanticRawEquationContent(layout_block=LayoutBlock.for_text('Equation 1')),
                    SemanticLabel(layout_block=LayoutBlock.for_text('(1)'))
                ])
            ]),
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:formula'
        ) == ['Equation 1 (1)']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:formula/tei:label'
        ) == ['(1)']
        assert tei_document.get_xpath_text_content_list(
            '//tei:body/tei:div/tei:p'
        ) == ['Next']

    def test_should_add_raw_references(self):
        semantic_document = SemanticDocument()
        semantic_raw_ref = SemanticRawReference([
            SemanticRawReferenceText(layout_block=LayoutBlock.for_text('Reference 1'))
        ], content_id='b0')
        semantic_document.back_section.add_content(SemanticReferenceList([
            semantic_raw_ref
        ]))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="references"]/tei:listBibl'
            '/tei:biblStruct/tei:note[@type="raw_reference"]'
        ) == ['Reference 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="references"]/tei:listBibl'
            '/tei:biblStruct/@xml:id'
        ) == ['b0']

    def test_should_add_parsed_references(self):
        semantic_document = SemanticDocument()
        semantic_ref = SemanticReference([
            SemanticTitle(layout_block=LayoutBlock.for_text('Reference Title 1')),
            SemanticRawReferenceText(layout_block=LayoutBlock.for_text('Reference 1'))
        ])
        semantic_ref.content_id = 'b0'
        semantic_document.back_section.add_content(
            SemanticReferenceList([
                SemanticHeading(layout_block=LayoutBlock.for_text('References')),
                semantic_ref
            ])
        )
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="references"]/tei:listBibl/tei:head'
        ) == ['References']
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="references"]/tei:listBibl'
            '/tei:biblStruct/tei:analytic/tei:title[@type="main"]'
        ) == ['Reference Title 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="references"]/tei:listBibl'
            '/tei:biblStruct/tei:note[@type="raw_reference"]'
        ) == ['Reference 1']
        assert tei_document.get_xpath_text_content_list(
            '//tei:back/tei:div[@type="references"]/tei:listBibl'
            '/tei:biblStruct/@xml:id'
        ) == ['b0']

    def test_should_unmatched_graphics_to_back(self):
        semantic_document = SemanticDocument()
        semantic_document.back_section.add_content(SemanticMixedNote([
            SemanticGraphic(
                layout_graphic=LayoutGraphic(
                    coordinates=COORDINATES_1
                ),
                relative_path='image1.svg'
            )
        ], note_type='unmatched_graphics'))
        tei_document = get_tei_for_semantic_document(semantic_document)
        LOGGER.debug('tei xml: %r', etree.tostring(tei_document.root))
        graphics_xpath = '//tei:note[@type="unmatched_graphics"]//tei:graphic'
        assert tei_document.xpath_nodes(graphics_xpath)
        assert tei_document.get_xpath_text_content_list(
            f'{graphics_xpath}/@url'
        ) == ['image1.svg']


ATTRIBUTION_1 = DocumentAttribution(
    version='1.2.3',
    profile_digest='digest1',
    profile_name='profile1'
)


class TestGetTeiForSemanticDocumentAttribution:
    def _get_semantic_document(self) -> SemanticDocument:
        semantic_document = SemanticDocument()
        semantic_document.front.add_content(
            SemanticTitle(layout_block=LayoutBlock.for_text(TOKEN_1))
        )
        semantic_document.front.add_content(
            SemanticAbstract(layout_block=LayoutBlock.for_text(TOKEN_2))
        )
        return semantic_document

    def test_should_not_add_attribution_when_not_given(self):
        tei_document = get_tei_for_semantic_document(self._get_semantic_document())
        assert not tei_document.xpath('//tei:encodingDesc')

    def test_should_name_the_profile_that_served_it(self):
        tei_document = get_tei_for_semantic_document(
            self._get_semantic_document(), attribution=ATTRIBUTION_1
        )
        assert tei_document.get_xpath_text_content_list(
            '//tei:encodingDesc/tei:appInfo/tei:application/tei:label[@type="profile"]'
        ) == [ATTRIBUTION_1.profile_name]
        assert tei_document.get_xpath_text_content_list(
            '//tei:encodingDesc/tei:appInfo/tei:application'
            '/tei:label[@type="profile-digest"]'
        ) == [ATTRIBUTION_1.profile_digest]

    def test_should_produce_byte_identical_output_for_two_runs(self):
        serialized = [
            etree.tostring(
                get_tei_for_semantic_document(
                    self._get_semantic_document(), attribution=ATTRIBUTION_1
                ).root
            )
            for _ in range(2)
        ]
        assert serialized[0] == serialized[1]
