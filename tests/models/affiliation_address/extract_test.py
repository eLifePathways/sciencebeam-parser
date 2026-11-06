import logging
from typing import List, Sequence, Type

from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutLineMeta
from sciencebeam_parser.document.semantic_document import (
    SemanticAddressLine,
    SemanticAffiliationAddress,
    SemanticContentWrapper,
    SemanticCountry,
    SemanticDepartment,
    SemanticInstitution,
    SemanticLaboratory,
    SemanticMarker,
    SemanticMixedContentWrapper,
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


def _block_on_line(text: str, line_id: int) -> LayoutBlock:
    return LayoutBlock.for_text(text, line_meta=LayoutLineMeta(line_id=line_id))


def _get_texts_by_type(
    semantic_content_list: Sequence[SemanticContentWrapper],
    type_: Type[SemanticMixedContentWrapper]
) -> List[List[str]]:
    result: List[List[str]] = []
    for semantic_content in semantic_content_list:
        assert isinstance(semantic_content, SemanticAffiliationAddress)
        result.append(semantic_content.view_by_type(type_).get_text_list())
    return result


class TestAffiliationAddressSemanticExtractor:
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
        aff = semantic_content_list[0]
        assert isinstance(aff, SemanticAffiliationAddress)
        assert aff.view_by_type(SemanticMarker).get_text() == '1'
        assert aff.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert aff.view_by_type(SemanticDepartment).get_text() == 'Department 1'
        assert aff.view_by_type(SemanticLaboratory).get_text() == 'Laboratory 1'
        assert aff.view_by_type(SemanticAddressLine).get_text() == 'Address Line 1'
        assert aff.view_by_type(SemanticPostCode).get_text() == 'Post Code 1'
        assert aff.view_by_type(SemanticPostBox).get_text() == 'Post Box 1'
        assert aff.view_by_type(SemanticRegion).get_text() == 'Region 1'
        assert aff.view_by_type(SemanticSettlement).get_text() == 'Settlement 1'
        assert aff.view_by_type(SemanticCountry).get_text() == 'Country 1'

    def test_should_keep_preceding_other_text_in_the_affiliation_as_grobid_does(self):
        # GROBID keeps text labelled as other in the raw affiliation string, and a
        # marker on the same line, with no marker on the affiliation yet, joins it
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('O', LayoutBlock.for_text('Other 1')),
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 1')),
            ])
        )
        assert len(semantic_content_list) == 1
        aff = semantic_content_list[0]
        assert isinstance(aff, SemanticAffiliationAddress)
        assert aff.view_by_type(SemanticNote).get_text() == 'Other 1'
        assert aff.view_by_type(SemanticMarker).get_text() == '1'
        assert aff.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert aff.get_text() == 'Other 1 1 Institution 1'

    def test_should_start_new_affiliation_on_a_marker_where_it_has_one_already(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<marker>', LayoutBlock.for_text('2')),
                ('<institution>', LayoutBlock.for_text('Institution 2')),
            ])
        )
        assert len(semantic_content_list) == 2
        aff1, aff2 = semantic_content_list
        assert isinstance(aff1, SemanticAffiliationAddress)
        assert aff1.view_by_type(SemanticMarker).get_text() == '1'
        assert aff1.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert isinstance(aff2, SemanticAffiliationAddress)
        assert aff2.view_by_type(SemanticMarker).get_text() == '2'
        assert aff2.view_by_type(SemanticInstitution).get_text() == 'Institution 2'
        assert [aff1.content_id, aff2.content_id] == ['aff0', 'aff1']

    def test_should_start_new_affiliation_on_a_marker_starting_a_line(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<institution>', _block_on_line('Institution 1', 1)),
                ('<marker>', _block_on_line('2', 2)),
                ('<institution>', _block_on_line('Institution 2', 2)),
            ])
        )
        assert len(semantic_content_list) == 2
        aff1, aff2 = semantic_content_list
        assert isinstance(aff1, SemanticAffiliationAddress)
        assert not aff1.has_type(SemanticMarker)
        assert aff1.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert isinstance(aff2, SemanticAffiliationAddress)
        assert aff2.view_by_type(SemanticMarker).get_text() == '2'
        assert aff2.view_by_type(SemanticInstitution).get_text() == 'Institution 2'

    def test_should_give_a_trailing_marker_on_the_same_line_to_the_affiliation(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<institution>', _block_on_line('Institution 1', 1)),
                ('<marker>', _block_on_line('1', 1)),
            ])
        )
        assert len(semantic_content_list) == 1
        aff = semantic_content_list[0]
        assert isinstance(aff, SemanticAffiliationAddress)
        assert aff.view_by_type(SemanticMarker).get_text() == '1'
        assert aff.view_by_type(SemanticInstitution).get_text() == 'Institution 1'

    def test_should_take_the_content_over_where_a_second_marker_follows_an_institution(self):
        # GROBID 0.9.1: "Institution 1 1 2" is one institution with two markers
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<marker>', LayoutBlock.for_text('1')),
                ('<marker>', LayoutBlock.for_text('2')),
            ])
        )
        assert len(semantic_content_list) == 2
        aff1, aff2 = semantic_content_list
        assert isinstance(aff1, SemanticAffiliationAddress)
        assert aff1.view_by_type(SemanticMarker).get_text() == '1'
        assert aff1.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert isinstance(aff2, SemanticAffiliationAddress)
        assert aff2.view_by_type(SemanticMarker).get_text() == '2'
        assert aff2.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert [aff1.content_id, aff2.content_id] == ['aff0', 'aff1']

    def test_should_split_on_second_institution_after_a_marker(self):
        # GROBID 0.9.1: "Institution 1 1 Institution 2 2" on one line
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 2')),
                ('<marker>', LayoutBlock.for_text('2')),
            ])
        )
        assert len(semantic_content_list) == 2
        aff1, aff2 = semantic_content_list
        assert isinstance(aff1, SemanticAffiliationAddress)
        assert aff1.view_by_type(SemanticMarker).get_text() == '1'
        assert aff1.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert isinstance(aff2, SemanticAffiliationAddress)
        assert aff2.view_by_type(SemanticMarker).get_text() == '2'
        assert aff2.view_by_type(SemanticInstitution).get_text() == 'Institution 2'

    def test_should_split_on_second_institution_starting_a_line(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<institution>', _block_on_line('Institution 1', 1)),
                ('<institution>', _block_on_line('Institution 2', 2)),
            ])
        )
        assert _get_texts_by_type(semantic_content_list, SemanticInstitution) == [
            ['Institution 1'], ['Institution 2']
        ]

    def test_should_split_on_second_department_starting_a_line_but_not_on_the_same_line(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<department>', _block_on_line('Department 1', 1)),
                ('<department>', _block_on_line('Department 2', 1)),
                ('<department>', _block_on_line('Department 3', 2)),
            ])
        )
        assert _get_texts_by_type(semantic_content_list, SemanticDepartment) == [
            ['Department 1', 'Department 2'], ['Department 3']
        ]

    def test_should_close_the_affiliation_at_a_gap_in_the_sequence(self):
        # the tokens of blocks another header block sat before, as GROBID sees a
        # jump in the token offsets
        institution_2_block = LayoutBlock.for_text('Institution 2')
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks(
                [
                    ('<institution>', LayoutBlock.for_text('Institution 1')),
                    ('<institution>', institution_2_block),
                ],
                segment_start_token_ids={id(next(iter(institution_2_block.iter_all_tokens())))}
            )
        )
        assert _get_texts_by_type(semantic_content_list, SemanticInstitution) == [
            ['Institution 1'], ['Institution 2']
        ]

    def test_should_not_split_affiliation_on_second_institution_without_address(self):
        # GROBID's AffiliationAddressParser only starts a new affiliation on a
        # second institution on the same line once the current one has an address
        # or a marker came between
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<institution>', LayoutBlock.for_text('Institution 2')),
            ])
        )
        assert len(semantic_content_list) == 1
        aff = semantic_content_list[0]
        assert isinstance(aff, SemanticAffiliationAddress)
        assert aff.view_by_type(SemanticMarker).get_text() == '1'
        assert aff.view_by_type(SemanticInstitution).get_text_list() == [
            'Institution 1', 'Institution 2'
        ]

    def test_should_split_affiliation_on_second_institution_after_an_address(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<settlement>', LayoutBlock.for_text('Settlement 1')),
                ('<institution>', LayoutBlock.for_text('Institution 2')),
                ('<department>', LayoutBlock.for_text('Department 2')),
                ('<country>', LayoutBlock.for_text('Country 2')),
                ('<department>', LayoutBlock.for_text('Department 3')),
            ])
        )
        assert len(semantic_content_list) == 3
        aff1, aff2, aff3 = semantic_content_list
        assert isinstance(aff1, SemanticAffiliationAddress)
        assert aff1.view_by_type(SemanticMarker).get_text() == '1'
        assert aff1.view_by_type(SemanticInstitution).get_text() == 'Institution 1'
        assert isinstance(aff2, SemanticAffiliationAddress)
        assert not aff2.has_type(SemanticMarker)
        assert aff2.view_by_type(SemanticInstitution).get_text() == 'Institution 2'
        assert aff2.view_by_type(SemanticDepartment).get_text() == 'Department 2'
        assert isinstance(aff3, SemanticAffiliationAddress)
        assert aff3.view_by_type(SemanticDepartment).get_text() == 'Department 3'

    def test_should_keep_a_lone_marker_as_a_note(self):
        # GROBID drops an affiliation holding nothing but a marker
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', _block_on_line('1', 1)),
                ('<marker>', _block_on_line('2', 2)),
                ('<institution>', _block_on_line('Institution 2', 2)),
            ])
        )
        assert len(semantic_content_list) == 2
        note, aff = semantic_content_list
        assert isinstance(note, SemanticNote)
        assert note.note_type == 'affiliation_marker'
        assert note.get_text() == '1'
        assert isinstance(aff, SemanticAffiliationAddress)
        assert aff.view_by_type(SemanticMarker).get_text() == '2'
        assert aff.view_by_type(SemanticInstitution).get_text() == 'Institution 2'

    def test_should_not_repeat_a_country_as_grobid_does_not(self):
        semantic_content_list = list(
            AffiliationAddressSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<institution>', LayoutBlock.for_text('Institution 1')),
                ('<country>', LayoutBlock.for_text('Country 1')),
                ('O', LayoutBlock.for_text('and')),
                ('<country>', LayoutBlock.for_text('Country 1')),
                ('<country>', LayoutBlock.for_text('Country 2')),
            ])
        )
        assert len(semantic_content_list) == 1
        aff = semantic_content_list[0]
        assert isinstance(aff, SemanticAffiliationAddress)
        assert aff.view_by_type(SemanticCountry).get_text_list() == ['Country 1', 'Country 2']
        assert aff.get_text() == 'Institution 1 Country 1 and Country 1 Country 2'

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
