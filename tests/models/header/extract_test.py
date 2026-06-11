from sciencebeam_parser.document.semantic_document import (
    SemanticAbstract,
    SemanticFront,
    SemanticRawAddress,
    SemanticRawAffiliation,
    SemanticRawAffiliationAddress,
    SemanticTitle
)
from sciencebeam_parser.document.layout_document import (
    LOGGER,
    LayoutBlock,
    join_layout_tokens
)

from sciencebeam_parser.models.header.extract import (
    HeaderSemanticExtractor,
    _split_trailing_title_punct,
    get_cleaned_abstract_text,
    get_cleaned_abstract_layout_block
)


TITLE_1 = 'the title 1'
ABSTRACT_1 = 'the abstract 1'
AUTHOR_1 = 'Author 1'
AUTHOR_2 = 'Author 2'
AFFILIATION_1 = 'Affiliation 1'
AFFILIATION_2 = 'Affiliation 2'
ADDRESS_1 = 'Address 1'
ADDRESS_2 = 'Address 2'

OTHER_1 = 'Other 1'


class TestGetCleanedAbstractText:
    def test_should_return_none_if_passed_in_text_was_none(self):
        assert get_cleaned_abstract_text(None) is None

    def test_should_return_empty_str_if_passed_in_text_was_empty(self):
        assert get_cleaned_abstract_text('') == ''

    def test_should_return_abstract_if_it_doesnt_contain_prefix(self):
        assert get_cleaned_abstract_text(ABSTRACT_1) == ABSTRACT_1

    def test_should_return_remove_abstract_prefix(self):
        assert get_cleaned_abstract_text(
            'Abstract ' + ABSTRACT_1
        ) == ABSTRACT_1

    def test_should_return_remove_abstract_dot_prefix(self):
        assert get_cleaned_abstract_text(
            'Abstract. ' + ABSTRACT_1
        ) == ABSTRACT_1

    def test_should_return_remove_abstract_colon_prefix(self):
        assert get_cleaned_abstract_text(
            'Abstract: ' + ABSTRACT_1
        ) == ABSTRACT_1


class TestGetCleanedAbstractLayoutBlock:
    def test_should_return_none_if_passed_in_text_was_none(self):
        assert get_cleaned_abstract_layout_block(None) is None

    def test_should_return_empty_str_if_passed_in_text_was_empty(self):
        layout_block = LayoutBlock(lines=[])
        assert get_cleaned_abstract_layout_block(layout_block) == layout_block

    def test_should_return_abstract_if_it_doesnt_contain_prefix(self):
        layout_block = LayoutBlock.for_text(ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1

    def test_should_return_remove_abstract_prefix(self):
        layout_block = LayoutBlock.for_text('Abstract ' + ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1

    def test_should_return_remove_abstract_dot_prefix(self):
        layout_block = LayoutBlock.for_text('Abstract. ' + ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1

    def test_should_return_remove_abstract_colon_prefix(self):
        layout_block = LayoutBlock.for_text('Abstract: ' + ABSTRACT_1)
        cleaned_layout_block = get_cleaned_abstract_layout_block(layout_block)
        assert join_layout_tokens(cleaned_layout_block.lines[0].tokens) == ABSTRACT_1


class TestSplitTrailingTitlePunct:
    def test_no_trailing_punct_returns_unchanged(self):
        block = LayoutBlock.for_text(TITLE_1)
        clean, trailing = _split_trailing_title_punct(block)
        assert join_layout_tokens(clean.lines[0].tokens) == TITLE_1
        assert trailing == ''

    def test_trailing_dot_is_stripped_and_returned(self):
        block = LayoutBlock.for_text(TITLE_1 + ' .')
        clean, trailing = _split_trailing_title_punct(block)
        assert join_layout_tokens(clean.lines[0].tokens) == TITLE_1
        assert trailing == '.'

    def test_empty_block_returns_unchanged(self):
        block = LayoutBlock(lines=[])
        clean, trailing = _split_trailing_title_punct(block)
        assert clean == block
        assert trailing == ''


class TestHeaderSemanticExtractor:
    def test_should_set_title_and_abstract(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(TITLE_1)),
                ('<abstract>', LayoutBlock.for_text(ABSTRACT_1))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        assert front.get_text_by_type(SemanticTitle) == TITLE_1
        assert front.get_text_by_type(SemanticAbstract) == ABSTRACT_1

    def test_should_strip_trailing_dot_from_title_into_trailing_text(self):
        title_with_dot = TITLE_1 + ' .'
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(title_with_dot))
            ])
        )
        front = SemanticFront(semantic_content_list)
        semantic_title = next(front.iter_by_type(SemanticTitle))
        assert front.get_text_by_type(SemanticTitle) == TITLE_1
        assert semantic_title.trailing_text == '.'

    def test_should_ignore_additional_title_and_abstract(self):
        # Note: this behaviour should be reviewed
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<title>', LayoutBlock.for_text(TITLE_1)),
                ('<abstract>', LayoutBlock.for_text(ABSTRACT_1)),
                ('<title>', LayoutBlock.for_text('other')),
                ('<abstract>', LayoutBlock.for_text('other'))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        assert front.get_text_by_type(SemanticTitle) == TITLE_1
        assert front.get_text_by_type(SemanticAbstract) == ABSTRACT_1

    def test_should_add_raw_authors(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<author>', LayoutBlock.for_text(AUTHOR_1))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        assert front.get_raw_authors_text() == AUTHOR_1

    def test_should_add_raw_affiliation_address(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<address>', LayoutBlock.for_text(ADDRESS_1))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert len(aff_address_list) == 1
        aff_address = aff_address_list[0]
        assert aff_address.get_text_by_type(SemanticRawAffiliation) == AFFILIATION_1
        assert aff_address.get_text_by_type(SemanticRawAddress) == ADDRESS_1

    def test_should_split_raw_affiliation_on_new_aff_without_address(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]

    def test_should_split_raw_affiliation_on_new_aff_with_address(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<address>', LayoutBlock.for_text(ADDRESS_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2)),
                ('<address>', LayoutBlock.for_text(ADDRESS_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]
        assert [
            aff_address.get_text_by_type(SemanticRawAddress)
            for aff_address in aff_address_list
        ] == [ADDRESS_1, ADDRESS_2]

    def test_should_split_raw_affiliation_separated_by_other(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('O', LayoutBlock.for_text(OTHER_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]

    def test_should_split_raw_affiliation_separated_by_known_label(self):
        semantic_content_list = list(
            HeaderSemanticExtractor().iter_semantic_content_for_entity_blocks([
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_1)),
                ('<author>', LayoutBlock.for_text(AUTHOR_1)),
                ('<affiliation>', LayoutBlock.for_text(AFFILIATION_2))
            ])
        )
        front = SemanticFront(semantic_content_list)
        LOGGER.debug('front: %s', front)
        aff_address_list = list(front.iter_by_type(SemanticRawAffiliationAddress))
        assert [
            aff_address.get_text_by_type(SemanticRawAffiliation)
            for aff_address in aff_address_list
        ] == [AFFILIATION_1, AFFILIATION_2]
