import logging

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import (
    SemanticAddressLine,
    SemanticAffiliationAddress,
    SemanticCountry,
    SemanticDepartment,
    SemanticInstitution,
    SemanticLaboratory,
    SemanticMarker,
    SemanticNote,
    SemanticPostBox,
    SemanticPostCode,
    SemanticRegion,
    SemanticSettlement
)
from sciencebeam_parser.models.affiliation_address.extract import (
    AffiliationAddressSemanticExtractor
)


LOGGER = logging.getLogger(__name__)


class TestNameSemanticExtractor:
    def test_should_extract_single_affiliation_address(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<department>', LayoutBlock.for_text('Department 1')),
                ('<laboratory>', LayoutBlock.for_text('Laboratory 1')),
                ('<addrLine>', LayoutBlock.for_text('Address Line 1')),
                ('<postCode>', LayoutBlock.for_text('Post Code 1')),
                ('<postBox>', LayoutBlock.for_text('Post Box 1')),
                ('<region>', LayoutBlock.for_text('Region 1')),
                ('<settlement>', LayoutBlock.for_text('Settlement 1')),
                ('<country>', LayoutBlock.for_text('Country 1'))
            ])
        )
        assert len(semantic_content_list) == 1
        author = semantic_content_list[0]
        assert isinstance(author, SemanticAffiliationAddress)
        assert author.view_by_type(SemanticMarker).get_text() == '1'
        assert author.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert author.view_by_type(SemanticDepartment).get_text() == 'Department 1'
        assert author.view_by_type(SemanticLaboratory).get_text() == 'Laboratory 1'
        assert author.view_by_type(SemanticAddressLine).get_text() == 'Address Line 1'
        assert author.view_by_type(SemanticPostCode).get_text() == 'Post Code 1'
        assert author.view_by_type(SemanticPostBox).get_text() == 'Post Box 1'
        assert author.view_by_type(SemanticRegion).get_text() == 'Region 1'
        assert author.view_by_type(SemanticSettlement).get_text() == 'Settlement 1'
        assert author.view_by_type(SemanticCountry).get_text() == 'Country 1'

    def test_should_extract_preceeding_other_text(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('O', LayoutBlock.for_text('Other 1')),
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 1')),
            ])
        )
        assert len(semantic_content_list) == 2
        note = semantic_content_list[0]
        assert isinstance(note, SemanticNote)
        author = semantic_content_list[1]
        assert isinstance(author, SemanticAffiliationAddress)
        assert author.view_by_type(SemanticMarker).get_text() == '1'
        assert author.view_by_type(SemanticInstitution).get_text() == 'Institution 1'

    def test_should_split_affiliation_on_second_institution(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<institution>', LayoutBlock.for_text('Institution 2')),
            ])
        )
        assert len(semantic_content_list) == 2
        aff1 = semantic_content_list[0]
        assert isinstance(aff1, SemanticAffiliationAddress)
        assert aff1.view_by_type(SemanticMarker).get_text() == '1'
        assert aff1.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        aff2 = semantic_content_list[1]
        assert isinstance(aff2, SemanticAffiliationAddress)
        assert aff2.view_by_type(SemanticInstitution).get_text() == 'Institution 2'

    def test_should_remove_trailing_dot_from_country(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<country>', LayoutBlock.for_text('Country1.'))
            ])
        )
        assert len(semantic_content_list) == 1
        aff1 = semantic_content_list[0]
        assert isinstance(aff1, SemanticAffiliationAddress)
        assert aff1.view_by_type(SemanticMarker).get_text() == '1'
        assert aff1.view_by_type(SemanticCountry).get_text() == 'Country1'
