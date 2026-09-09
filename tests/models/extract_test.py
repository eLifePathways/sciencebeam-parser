import logging

import pytest

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import SemanticNote, SemanticTitle
from sciencebeam_parser.models.extract import (
    SimpleModelSemanticExtractor,
    get_regex_cleaned_layout_block_with_prefix_suffix
)


class TestGetRegexCleanedLayoutBlockWithPrefixSuffix:
    def test_should_return_original_block_for_non_matching_regex(self):
        layout_block = LayoutBlock.for_text('test')
        prefix_block, cleaned_block, suffix_block = (
            get_regex_cleaned_layout_block_with_prefix_suffix(
                layout_block,
                r'other'
            )
        )
        assert not prefix_block.lines
        assert cleaned_block == layout_block
        assert not suffix_block.lines

    def test_should_return_original_block_for_empty_block(self):
        layout_block = LayoutBlock(lines=[])
        prefix_block, cleaned_block, suffix_block = (
            get_regex_cleaned_layout_block_with_prefix_suffix(
                layout_block,
                r'other'
            )
        )
        assert not prefix_block.lines
        assert cleaned_block == layout_block
        assert not suffix_block.lines

    def test_should_return_prefix_for_prefix_match(self):
        layout_block = LayoutBlock.for_text('a b c d')
        prefix_block, cleaned_block, suffix_block = (
            get_regex_cleaned_layout_block_with_prefix_suffix(
                layout_block,
                r'.*?(b.*)'
            )
        )
        assert prefix_block.text == 'a'
        assert cleaned_block.text == 'b c d'
        assert not suffix_block.lines

    def test_should_return_suffix_for_suffix_match(self):
        layout_block = LayoutBlock.for_text('a b c d')
        prefix_block, cleaned_block, suffix_block = (
            get_regex_cleaned_layout_block_with_prefix_suffix(
                layout_block,
                r'(.*)d'
            )
        )
        assert not prefix_block.lines
        assert cleaned_block.text == 'a b c'
        assert suffix_block.text == 'd'

    def test_should_return_prefix_suffix_for_prefix_suffix_match(self):
        layout_block = LayoutBlock.for_text('a b c d')
        prefix_block, cleaned_block, suffix_block = (
            get_regex_cleaned_layout_block_with_prefix_suffix(
                layout_block,
                r'a(.*)d'
            )
        )
        assert prefix_block.text == 'a'
        assert cleaned_block.text == 'b c'
        assert suffix_block.text == 'd'


class SimpleExtractor(SimpleModelSemanticExtractor):
    def iter_semantic_content_for_entity_blocks(self, entity_tokens, **kwargs):
        return [
            self.get_semantic_content_for_entity_name(name, layout_block=layout_block)
            for name, layout_block in entity_tokens
        ]


def _get_semantic_content_for_entity_name(
    extractor: SimpleModelSemanticExtractor,
    name: str
):
    return extractor.get_semantic_content_for_entity_name(
        name, layout_block=LayoutBlock.for_text('text 1')
    )


class TestSimpleModelSemanticExtractor:
    def test_should_use_mapped_semantic_content_class(self):
        extractor = SimpleExtractor({'<title>': SemanticTitle})
        assert isinstance(
            _get_semantic_content_for_entity_name(extractor, '<title>'),
            SemanticTitle
        )

    def test_should_keep_unmapped_label_as_note(self):
        extractor = SimpleExtractor({'<title>': SemanticTitle})
        semantic_content = _get_semantic_content_for_entity_name(extractor, '<other-label>')
        assert isinstance(semantic_content, SemanticNote)
        assert semantic_content.note_type == '<other-label>'

    def test_should_not_warn_about_unmapped_label_without_expected_note_tags(
        self,
        caplog: pytest.LogCaptureFixture
    ):
        extractor = SimpleExtractor({'<title>': SemanticTitle})
        with caplog.at_level(logging.WARNING):
            _get_semantic_content_for_entity_name(extractor, '<other-label>')
        assert not caplog.records

    @pytest.mark.parametrize("name", ['<note>', 'O', '<other>'])
    def test_should_not_warn_about_expected_note_or_other_tags(
        self,
        name: str,
        caplog: pytest.LogCaptureFixture
    ):
        extractor = SimpleExtractor(
            {'<title>': SemanticTitle},
            expected_note_tags={'<note>'}
        )
        with caplog.at_level(logging.WARNING):
            _get_semantic_content_for_entity_name(extractor, name)
        assert not caplog.records

    def test_should_warn_once_about_an_unexpected_note_tag(
        self,
        caplog: pytest.LogCaptureFixture
    ):
        extractor = SimpleExtractor(
            {'<title>': SemanticTitle},
            expected_note_tags={'<note>'}
        )
        with caplog.at_level(logging.WARNING):
            _get_semantic_content_for_entity_name(extractor, '<idno>')
            _get_semantic_content_for_entity_name(extractor, '<idno>')
        assert len(caplog.records) == 1
        assert '<idno>' in caplog.records[0].getMessage()
