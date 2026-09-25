import logging
from typing import List

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import (
    SemanticAuthor,
    SemanticEditor,
    SemanticGivenName,
    SemanticMarker,
    SemanticMiddleName,
    SemanticMixedContentWrapper,
    SemanticNameSuffix,
    SemanticNameTitle,
    SemanticNote,
    SemanticSurname
)
from sciencebeam_parser.models.name.extract import (
    NameSemanticExtractor,
    iter_semantic_markers_for_layout_block
)


LOGGER = logging.getLogger(__name__)


class TestIterSemanticMarkersForLayoutBlock:
    def test_should_extract_single_marker(self):
        semantic_markers = list(iter_semantic_markers_for_layout_block(
            LayoutBlock.for_text('1')
        ))
        assert len(semantic_markers) == 1
        assert isinstance(semantic_markers[0], SemanticMarker)
        assert semantic_markers[0].get_text() == '1'

    def test_should_split_markers_on_comma(self):
        semantic_markers = list(iter_semantic_markers_for_layout_block(
            LayoutBlock.for_text('1,2')
        ))
        semantic_content_wrapper = SemanticMixedContentWrapper(semantic_markers)
        assert semantic_content_wrapper.view_by_type(SemanticMarker).get_text_list() == ['1', '2']
        assert semantic_content_wrapper.merged_block.text == '1,2'

    def test_should_split_markers_on_space(self):
        semantic_markers = list(iter_semantic_markers_for_layout_block(
            LayoutBlock.for_text('1 2')
        ))
        LOGGER.debug('semantic_markers: %r', semantic_markers)
        semantic_content_wrapper = SemanticMixedContentWrapper(semantic_markers)
        assert semantic_content_wrapper.view_by_type(SemanticMarker).get_text_list() == ['1', '2']
        assert semantic_content_wrapper.merged_block.text == '1 2'

    def test_should_not_split_markers_on_digit(self):
        semantic_markers = list(iter_semantic_markers_for_layout_block(
            LayoutBlock.for_text('11,12')
        ))
        semantic_content_wrapper = SemanticMixedContentWrapper(semantic_markers)
        assert semantic_content_wrapper.view_by_type(SemanticMarker).get_text_list() == ['11', '12']
        assert semantic_content_wrapper.merged_block.text == '11,12'

    def test_should_split_marker_on_non_numeric_characters(self):
        semantic_markers = list(iter_semantic_markers_for_layout_block(
            LayoutBlock.for_text('+*!')
        ))
        semantic_content_wrapper = SemanticMixedContentWrapper(semantic_markers)
        assert semantic_content_wrapper.view_by_type(SemanticMarker).get_text_list() == [
            '+', '*', '!'
        ]
        assert semantic_content_wrapper.merged_block.text == '+*!'


class TestNameSemanticExtractor:
    def test_should_extract_single_author(self):
        semantic_content_list = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text('Mr')),
                ('<forename>', LayoutBlock.for_text('John')),
                ('<middlename>', LayoutBlock.for_text('M')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<suffix>', LayoutBlock.for_text('Suffix1')),
                ('<marker>', LayoutBlock.for_text('1'))
            ], name_type=SemanticAuthor)
        )
        assert len(semantic_content_list) == 1
        author = semantic_content_list[0]
        assert isinstance(author, SemanticAuthor)
        assert author.view_by_type(SemanticNameTitle).get_text() == 'Mr'
        assert author.given_name_text == 'John'
        assert author.view_by_type(SemanticMiddleName).get_text() == 'M'
        assert author.surname_text == 'Smith'
        assert author.view_by_type(SemanticNameSuffix).get_text() == 'Suffix1'
        assert author.view_by_type(SemanticMarker).get_text() == '1'

    def test_should_be_able_to_extract_single_editor(self):
        semantic_content_list = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith'))
            ], name_type=SemanticEditor)
        )
        assert len(semantic_content_list) == 1
        author = semantic_content_list[0]
        assert isinstance(author, SemanticEditor)
        assert author.given_name_text == 'John'
        assert author.surname_text == 'Smith'

    def test_should_extract_multiple_authors(self):
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<marker>', LayoutBlock.for_text('1')),
                ('O', LayoutBlock.for_text(',')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
                ('<marker>', LayoutBlock.for_text('2'))
            ])
        )
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        assert author_1.view_by_type(SemanticMarker).get_text() == '1'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'
        assert author_2.view_by_type(SemanticMarker).get_text() == '2'

    def test_should_split_on_comma_before_marker(self):
        # The model currently does not provide segmentation as such
        # That is why the segmentation of authors is currently rule-based
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('O', LayoutBlock.for_text(',')),
                ('<marker>', LayoutBlock.for_text('2')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        assert author_1.view_by_type(SemanticMarker).get_text() == '1'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'
        assert author_2.view_by_type(SemanticMarker).get_text() == '2'

    def test_should_split_on_double_comma_before_marker(self):
        # The model currently does not provide segmentation as such
        # That is why the segmentation of authors is currently rule-based
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<marker>', LayoutBlock.for_text('1')),
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('O', LayoutBlock.for_text(', ,')),
                ('<marker>', LayoutBlock.for_text('2')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        assert author_1.view_by_type(SemanticMarker).get_text() == '1'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'
        assert author_2.view_by_type(SemanticMarker).get_text() == '2'

    def test_should_split_on_second_title(self):
        # This matches the current rules in GROBID
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text('Mr')),
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<title>', LayoutBlock.for_text('Ms')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.get_text_by_type(SemanticNameTitle) == 'Mr'
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.get_text_by_type(SemanticNameTitle) == 'Ms'
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'

    def test_should_split_on_second_firstname(self):
        # This matches the current rules in GROBID
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'

    def test_should_split_on_second_surname(self):
        # This matches the current rules in GROBID
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Madison')),
                ('<forename>', LayoutBlock.for_text('Maria')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'

    def test_should_split_not_split_on_second_middlename(self):
        # This matches the current rules in GROBID
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('John')),
                ('<middlename>', LayoutBlock.for_text('M')),
                ('<middlename>', LayoutBlock.for_text('J')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        assert author_1.get_text_by_type(SemanticMiddleName) == 'M J'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'

    def test_should_split_not_split_on_second_suffix(self):
        # This matches the current rules in GROBID
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<suffix>', LayoutBlock.for_text('X')),
                ('<suffix>', LayoutBlock.for_text('Y')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        assert author_1.get_text_by_type(SemanticNameSuffix) == 'X Y'
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'

    def test_should_parse_multiple_markers(self):
        semantic_content_list: List[SemanticAuthor] = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('John')),
                ('<surname>', LayoutBlock.for_text('Smith')),
                ('<marker>', LayoutBlock.for_text('1')),
                ('O', LayoutBlock.for_text(',')),
                ('<marker>', LayoutBlock.for_text('2')),
                ('O', LayoutBlock.for_text(',')),
                ('<forename>', LayoutBlock.for_text('Maria')),
                ('<surname>', LayoutBlock.for_text('Madison')),
                ('<marker>', LayoutBlock.for_text('2')),
                ('O', LayoutBlock.for_text(',')),
                ('<marker>', LayoutBlock.for_text('3')),
            ])
        )
        LOGGER.debug('semantic_content_list: %s', semantic_content_list)
        assert len(semantic_content_list) == 2
        author_1 = semantic_content_list[0]
        assert isinstance(author_1, SemanticAuthor)
        assert author_1.given_name_text == 'John'
        assert author_1.surname_text == 'Smith'
        assert author_1.view_by_type(SemanticMarker).get_text_list() == ['1', '2']
        author_2 = semantic_content_list[1]
        assert isinstance(author_2, SemanticAuthor)
        assert author_2.given_name_text == 'Maria'
        assert author_2.surname_text == 'Madison'
        assert author_2.view_by_type(SemanticMarker).get_text_list() == ['2', '3']

    def test_should_convert_name_parts_to_title_case(self):
        semantic_content_list = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('JOHN')),
                ('<middlename>', LayoutBlock.for_text('MIDDLE')),
                ('<surname>', LayoutBlock.for_text('SMITH')),
            ], name_type=SemanticAuthor)
        )
        assert len(semantic_content_list) == 1
        author = semantic_content_list[0]
        assert isinstance(author, SemanticAuthor)
        given_name = list(author.iter_by_type(SemanticGivenName))[0]
        assert given_name.get_text() == 'JOHN'
        assert given_name.value == 'John'
        middle_name = list(author.iter_by_type(SemanticMiddleName))[0]
        assert middle_name.get_text() == 'MIDDLE'
        assert middle_name.value == 'Middle'
        surname = list(author.iter_by_type(SemanticSurname))[0]
        assert surname.get_text() == 'SMITH'
        assert surname.value == 'Smith'

    def test_should_treat_two_letter_uppercase_given_name_as_given_midde_name(self):
        semantic_content_list = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('JM')),
                ('<surname>', LayoutBlock.for_text('SMITH')),
            ], name_type=SemanticAuthor)
        )
        assert len(semantic_content_list) == 1
        author = semantic_content_list[0]
        assert isinstance(author, SemanticAuthor)
        given_name = list(author.iter_by_type(SemanticGivenName))[0]
        assert given_name.get_text() == 'J'
        assert given_name.value == 'J'
        middle_name = list(author.iter_by_type(SemanticMiddleName))[0]
        assert middle_name.get_text() == 'M'
        assert middle_name.value == 'M'
        surname = list(author.iter_by_type(SemanticSurname))[0]
        assert surname.get_text() == 'SMITH'
        assert surname.value == 'Smith'

    def test_should_consider_names_without_last_name_as_invalid(self):
        semantic_content_list = list(
            NameSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<forename>', LayoutBlock.for_text('J'))
            ], name_type=SemanticAuthor)
        )
        assert len(semantic_content_list) == 1
        author = semantic_content_list[0]
        assert isinstance(author, SemanticNote)
        assert author.get_text() == 'J'
