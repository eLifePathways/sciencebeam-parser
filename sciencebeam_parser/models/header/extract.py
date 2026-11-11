import logging
import re
from collections import OrderedDict
from itertools import groupby
from typing import (
    AbstractSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
)

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
from sciencebeam_parser.utils.language import (
    MIN_TITLE_LANGUAGE_DETECTION_WORD_COUNT,
    WORD_PATTERN,
    detect_language,
    detect_language_among,
    detect_title_language
)


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


class TitlesMode:
    """What becomes of a title the model labelled as one."""

    AS_LABELLED = 'as_labelled'
    SPLIT_BY_LANGUAGE = 'split_by_language'


TITLES_MODES = (TitlesMode.AS_LABELLED, TitlesMode.SPLIT_BY_LANGUAGE)

DEFAULT_TITLES_MODE = TitlesMode.SPLIT_BY_LANGUAGE


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


TITLE_SAME_TEXT_PATTERN = re.compile(r'[^0-9a-z\u00c0-\u024f]+')


def _get_language(
    lines: Sequence[LayoutLine],
    document_languages: AbstractSet[str] = frozenset()
) -> Optional[str]:
    """Which language the lines are written in, read as a title.

    A title too evenly balanced to name on its own is named among the languages the
    document is written in elsewhere, where that settles it. One too short to be a title
    is left unnamed either way, so a fragment cannot be cut off as a title of its own.
    """
    text = str(LayoutTokensText(LayoutBlock(lines=list(lines))))
    language = detect_title_language(text)
    if language is not None:
        return language
    if len(WORD_PATTERN.findall(text)) < MIN_TITLE_LANGUAGE_DETECTION_WORD_COUNT:
        return None
    return detect_language_among(text, document_languages)


def _iter_line_runs_by_block(layout_block: LayoutBlock) -> Iterator[List[LayoutLine]]:
    """The title's lines, grouped by the block each was set in."""
    for _, block_lines in groupby(
        layout_block.lines, key=lambda line: line.tokens[0].line_meta.block_id
    ):
        yield list(block_lines)


def _iter_candidate_runs(layout_block: LayoutBlock) -> Iterator[List[LayoutLine]]:
    """Where the title may be cut: between blocks, or between lines of a single block.

    A title set across two blocks is cut between them; one block holding both languages
    leaves only its lines to cut at.
    """
    runs = list(_iter_line_runs_by_block(layout_block))
    if len(runs) > 1:
        yield from runs
        return
    yield from ([line] for line in layout_block.lines)


def iter_titles_split_by_language(
    layout_block: LayoutBlock,
    document_languages: AbstractSet[str] = frozenset()
) -> Iterator[Tuple[LayoutBlock, Optional[str]]]:
    """One title per language, where the model labelled several as one.

    A title given in two languages is labelled as a single title where the model has not
    learnt to separate them. A run that reads as the language before it, or whose language
    cannot be told, stays with it, so a title is only ever cut where both sides name a
    language and those languages differ, and the block or line boundary says where.
    """
    parts: List[Tuple[List[LayoutLine], Optional[str]]] = []
    for run in _iter_candidate_runs(layout_block):
        language = _get_language(run, document_languages)
        if parts and (language is None or language == parts[-1][1]):
            parts[-1][0].extend(run)
            continue
        parts.append((list(run), language))
    if len({language for _, language in parts if language}) < 2:
        yield layout_block, _get_language(layout_block.lines, document_languages)
        return
    for lines, language in parts:
        yield LayoutBlock(lines=lines), language


def _title_key(semantic_title: SemanticTitle) -> str:
    return TITLE_SAME_TEXT_PATTERN.sub('', semantic_title.get_text().lower())


def iter_titles_printed_once(
    semantic_titles: Sequence[SemanticTitle]
) -> Iterator[SemanticTitle]:
    """Each title a page prints, in the order it prints them, once.

    A page sets its title again under a heading or at the head of a second page, and
    every printing is labelled, so the same title arrives more than once.  The letters
    and digits are what say two printings are one title: the repeat is set in capitals
    where the first is not, or punctuated differently.
    """
    seen: Set[str] = set()
    for semantic_title in semantic_titles:
        key = _title_key(semantic_title)
        if key in seen:
            continue
        seen.add(key)
        yield semantic_title


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


def _get_semantic_titles(
    title_blocks: Sequence[Tuple[LayoutBlock, str]],
    titles_mode: str,
    document_languages: AbstractSet[str]
) -> List[SemanticTitle]:
    """The titles to yield, named among the languages the document carries elsewhere.

    Built once the whole header has been read, because the abstract says which languages
    the document is written in and a title is too short to say so on its own.
    """
    semantic_titles: List[SemanticTitle] = []
    for layout_block, trailing in title_blocks:
        if titles_mode == TitlesMode.AS_LABELLED:
            parts = [(
                layout_block,
                _get_language(layout_block.lines, document_languages)
            )]
        else:
            parts = list(iter_titles_split_by_language(layout_block, document_languages))
        for index, (title_block, language) in enumerate(parts):
            semantic_titles.append(SemanticTitle(
                layout_block=title_block,
                trailing_text=trailing if index == len(parts) - 1 else '',
                language=language
            ))
    return semantic_titles


class HeaderSemanticExtractor(SimpleModelSemanticExtractor):
    def __init__(self):
        super().__init__(semantic_content_class_by_tag=SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG)

    def iter_semantic_content_for_entity_blocks(  # noqa pylint: disable=too-many-branches,too-many-locals
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        abstracts_mode: str = DEFAULT_ABSTRACTS_MODE,
        titles_mode: str = DEFAULT_TITLES_MODE,
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        entity_tokens = list(entity_tokens)
        LOGGER.debug('entity_tokens: %s', entity_tokens)
        carry_variants = abstracts_mode != AbstractsMode.FIRST
        merge_by_language = abstracts_mode == AbstractsMode.MERGED_BY_LANGUAGE
        primary_abstract_block: Optional[LayoutBlock] = None
        title_blocks: List[Tuple[LayoutBlock, str]] = []
        document_languages: Set[str] = set()
        semantic_abstracts: List[SemanticAbstract] = []
        aff_address: Optional[SemanticRawAffiliationAddress] = None
        next_previous_label: str = ''
        for name, layout_block in entity_tokens:
            previous_label = next_previous_label
            next_previous_label = name
            if name == '<title>':
                title_blocks.append(_split_trailing_title_punct(layout_block))
                continue
            if name == '<abstract>':
                abstract_layout_block = get_cleaned_abstract_layout_block(
                    layout_block
                )
                assert abstract_layout_block is not None
                abstract_language = detect_language(
                    str(LayoutTokensText(abstract_layout_block))
                )
                if abstract_language:
                    document_languages.add(abstract_language)
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
        yield from iter_titles_printed_once(
            _get_semantic_titles(title_blocks, titles_mode, document_languages)
        )
        yield from merge_abstracts_by_language(semantic_abstracts)
