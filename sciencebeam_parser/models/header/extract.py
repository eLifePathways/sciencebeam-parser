import logging
import re
from collections import OrderedDict
from typing import AbstractSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticRawAddress,
    SemanticRawAffiliation,
    SemanticRawAffiliationAddress,
    SemanticTitle,
    SemanticAbstract,
    SemanticRawAuthors,
    T_SemanticContentFactory
)
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutLine,
    LayoutTokensText
)
from sciencebeam_parser.models.extract import SimpleModelSemanticExtractor
from sciencebeam_parser.utils.language import detect_language


LOGGER = logging.getLogger(__name__)


TITLE_TRAILING_PUNCT_CHARS: AbstractSet[str] = {'.'}


def _split_trailing_title_punct(
    layout_block: LayoutBlock,
) -> Tuple[LayoutBlock, str]:
    """Strip a trailing punctuation token from the title layout block.

    Returns the cleaned block and the stripped character (empty string if nothing stripped).
    GROBID moves such punctuation outside the <title> element via cleanField.
    """
    lines = layout_block.lines
    if not lines:
        return layout_block, ''
    last_line = lines[-1]
    tokens = last_line.tokens
    if not tokens or tokens[-1].text not in TITLE_TRAILING_PUNCT_CHARS:
        return layout_block, ''
    trailing = tokens[-1].text
    new_lines = lines[:-1] + [LayoutLine(tokens=tokens[:-1])]
    return LayoutBlock(lines=new_lines), trailing


# based on:
#   grobid-core/src/main/java/org/grobid/core/data/BiblioItem.java
ABSTRACT_REGEX = r'^(?:(?:abstract|summary|résumé|abrégé|a b s t r a c t)(?:[.:])?)?\s*(.*)'


SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG: Mapping[str, T_SemanticContentFactory] = {
    '<author>': SemanticRawAuthors,
    '<affiliation>': SemanticRawAffiliation,
    '<address>': SemanticRawAddress
}


# A further block labelled `<abstract>` is often not one. A block far shorter than the
# first is a fragment or a caption rather than another abstract: over a benchmark run no
# block matching a gold abstract falls below this, and 58 that match none do.
MIN_ABSTRACT_VARIANT_LENGTH_RATIO = 0.15


class AbstractsMode:
    """Which abstracts reach the output where the model labels more than one."""

    FIRST = 'first'
    VARIANTS = 'variants'
    MERGED_BY_LANGUAGE = 'merged_by_language'


ABSTRACTS_MODES = (AbstractsMode.FIRST, AbstractsMode.VARIANTS, AbstractsMode.MERGED_BY_LANGUAGE)

DEFAULT_ABSTRACTS_MODE = AbstractsMode.FIRST


def _get_token_count(layout_block: LayoutBlock) -> int:
    return sum(1 for _ in layout_block.iter_all_tokens())


def get_semantic_abstract_for_layout_block(
    layout_block: LayoutBlock,
    with_language: bool = True
) -> SemanticAbstract:
    return SemanticAbstract(
        layout_block=layout_block,
        language=detect_language(str(LayoutTokensText(layout_block))) if with_language else None
    )


def merge_abstracts_by_language(
    semantic_abstracts: Sequence[SemanticAbstract]
) -> List[SemanticAbstract]:
    """One abstract per language, in the order each language first appeared.

    An abstract whose language could not be told is left on its own: there is nothing to
    group it by, and guessing which group it belongs to is what detection declined to do.
    """
    blocks_by_language: "OrderedDict[str, List[LayoutBlock]]" = OrderedDict()
    order: List[Tuple[Optional[str], Optional[SemanticAbstract]]] = []
    for semantic_abstract in semantic_abstracts:
        language = semantic_abstract.language
        if not language:
            order.append((None, semantic_abstract))
            continue
        if language not in blocks_by_language:
            blocks_by_language[language] = []
            order.append((language, None))
        blocks_by_language[language].append(semantic_abstract.merged_block)
    merged: List[SemanticAbstract] = []
    for language, existing_abstract in order:
        if language is None:
            assert existing_abstract is not None
            merged.append(existing_abstract)
            continue
        merged.append(SemanticAbstract(
            layout_block=LayoutBlock.merge_blocks(blocks_by_language[language]),
            language=language
        ))
    return merged


def is_abstract_variant(
    layout_block: LayoutBlock,
    primary_layout_block: LayoutBlock
) -> bool:
    primary_token_count = _get_token_count(primary_layout_block)
    if not primary_token_count:
        return False
    ratio = _get_token_count(layout_block) / primary_token_count
    return ratio >= MIN_ABSTRACT_VARIANT_LENGTH_RATIO


def get_cleaned_abstract_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    m = re.match(ABSTRACT_REGEX, text, re.IGNORECASE)
    if not m:
        LOGGER.debug('text does not match regex: %r', text)
        return text
    return m.group(1)


def get_cleaned_abstract_layout_block(
    layout_block: Optional[LayoutBlock]
) -> Optional[LayoutBlock]:
    if not layout_block or not layout_block.lines:
        return layout_block
    layout_tokens_text = LayoutTokensText(layout_block)
    text = str(layout_tokens_text)
    m = re.match(ABSTRACT_REGEX, text, re.IGNORECASE)
    if not m:
        LOGGER.debug('text does not match regex: %r', text)
        return layout_block
    start = m.start(1)
    LOGGER.debug('start: %d (text: %r)', start, text)
    return LayoutBlock.for_tokens(list(
        layout_tokens_text.iter_layout_tokens_between(start, len(text))
    ))


class HeaderSemanticExtractor(SimpleModelSemanticExtractor):
    def __init__(self):
        super().__init__(semantic_content_class_by_tag=SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG)

    def iter_semantic_content_for_entity_blocks(  # noqa pylint: disable=too-many-branches,too-many-locals
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        abstracts_mode: str = DEFAULT_ABSTRACTS_MODE,
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        entity_tokens = list(entity_tokens)
        LOGGER.debug('entity_tokens: %s', entity_tokens)
        carry_variants = abstracts_mode != AbstractsMode.FIRST
        merge_by_language = abstracts_mode == AbstractsMode.MERGED_BY_LANGUAGE
        has_title: bool = False
        primary_abstract_block: Optional[LayoutBlock] = None
        semantic_abstracts: List[SemanticAbstract] = []
        aff_address: Optional[SemanticRawAffiliationAddress] = None
        next_previous_label: str = ''
        for name, layout_block in entity_tokens:
            previous_label = next_previous_label
            next_previous_label = name
            if name == '<title>' and not has_title:
                clean_block, trailing = _split_trailing_title_punct(layout_block)
                yield SemanticTitle(layout_block=clean_block, trailing_text=trailing)
                has_title = True
                continue
            if name == '<abstract>':
                abstract_layout_block = get_cleaned_abstract_layout_block(
                    layout_block
                )
                assert abstract_layout_block is not None
                if primary_abstract_block is None or (
                    carry_variants
                    and is_abstract_variant(abstract_layout_block, primary_abstract_block)
                ):
                    semantic_abstract = get_semantic_abstract_for_layout_block(
                        abstract_layout_block, with_language=carry_variants
                    )
                    if primary_abstract_block is None:
                        primary_abstract_block = abstract_layout_block
                    if merge_by_language:
                        semantic_abstracts.append(semantic_abstract)
                    else:
                        yield semantic_abstract
                    continue
            if name in {'<affiliation>', '<address>'}:
                if (
                    aff_address is not None
                    and name == '<affiliation>'
                    and previous_label in {'<affiliation>', '<address>'}
                ):
                    yield aff_address
                    aff_address = None
                if aff_address is None:
                    aff_address = SemanticRawAffiliationAddress()
                aff_address.add_content(self.get_semantic_content_for_entity_name(
                    name, layout_block
                ))
                continue
            if aff_address is not None:
                yield aff_address
                aff_address = None
            yield self.get_semantic_content_for_entity_name(
                name, layout_block
            )
        if aff_address is not None:
            yield aff_address
        yield from merge_abstracts_by_language(semantic_abstracts)
