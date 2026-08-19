from abc import ABC, abstractmethod
import logging
import re
from typing import AbstractSet, Iterable, Mapping, Optional, Set, Tuple

from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticNote,
    T_SemanticContentFactory
)
from sciencebeam_parser.document.layout_document import EMPTY_BLOCK, LayoutBlock, LayoutTokensText
from sciencebeam_parser.utils.labels import OTHER_LABELS


LOGGER = logging.getLogger(__name__)


class ModelSemanticExtractor(ABC):
    @abstractmethod
    def iter_semantic_content_for_entity_blocks(
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        pass


def get_regex_cleaned_layout_block_with_prefix_suffix(
    layout_block: LayoutBlock,
    regex_pattern: Optional[str]
) -> Tuple[LayoutBlock, LayoutBlock, LayoutBlock]:
    if not layout_block or not layout_block.lines or not regex_pattern:
        return EMPTY_BLOCK, layout_block, EMPTY_BLOCK
    layout_tokens_text = LayoutTokensText(layout_block)
    text = str(layout_tokens_text)
    m = re.match(regex_pattern, text, re.IGNORECASE)
    if not m:
        LOGGER.debug('text does not match regex: %r', text)
        return EMPTY_BLOCK, layout_block, EMPTY_BLOCK
    start = m.start(1)
    end = m.end(1)
    LOGGER.debug('start: %d, end: %d, len: %d (text: %r)', start, end, len(text), text)
    return (
        LayoutBlock.for_tokens(list(
            layout_tokens_text.iter_layout_tokens_between(0, start)
        )),
        LayoutBlock.for_tokens(list(
            layout_tokens_text.iter_layout_tokens_between(start, end)
        )),
        LayoutBlock.for_tokens(list(
            layout_tokens_text.iter_layout_tokens_between(end, len(text))
        ))
    )


class SimpleModelSemanticExtractor(ModelSemanticExtractor):
    def __init__(
        self,
        semantic_content_class_by_tag: Optional[
            Mapping[str, T_SemanticContentFactory]
        ] = None,
        *,
        expected_note_tags: Optional[AbstractSet[str]] = None
    ):
        super().__init__()
        self.semantic_content_class_by_tag = semantic_content_class_by_tag or {}
        self.expected_note_tags = expected_note_tags
        self._reported_unexpected_note_tags: Set[str] = set()

    def _report_unexpected_note_tag(self, name: str) -> None:
        if self.expected_note_tags is None:
            return
        if name in self.expected_note_tags or name in OTHER_LABELS:
            return
        if name in self._reported_unexpected_note_tags:
            return
        self._reported_unexpected_note_tags.add(name)
        LOGGER.warning(
            'no semantic content for label %r, keeping it as a note'
            ' (it will not appear in any extracted field)',
            name
        )

    def get_semantic_content_for_entity_name(
        self,
        name: str,
        layout_block: LayoutBlock
    ) -> SemanticContentWrapper:
        semantic_content_class = self.semantic_content_class_by_tag.get(name)
        if semantic_content_class:
            return semantic_content_class(layout_block=layout_block)
        self._report_unexpected_note_tag(name)
        return SemanticNote(
            layout_block=layout_block,
            note_type=name
        )
