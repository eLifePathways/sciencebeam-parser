import functools
import itertools
import unicodedata
from typing import AbstractSet, Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)


class DehyphenateMode:
    """How a word the page broke at a line end with a hyphen is written."""

    OFF = 'off'
    GROBID = 'grobid'
    BY_DOCUMENT = 'by_document'


DEHYPHENATE_MODES = (DehyphenateMode.OFF, DehyphenateMode.GROBID, DehyphenateMode.BY_DOCUMENT)

DEFAULT_DEHYPHENATE_MODE = DehyphenateMode.BY_DOCUMENT

HYPHEN = '-'
SOFT_HYPHEN = '\u00ad'


class DocumentWords(NamedTuple):
    """Lower-case words, and hyphenated words, the document sets inside a line."""

    words: AbstractSet[str]
    hyphenated_words: AbstractSet[str]


EMPTY_DOCUMENT_WORDS = DocumentWords(words=frozenset(), hyphenated_words=frozenset())


def _is_word(text: str) -> bool:
    return unicodedata.normalize('NFC', text).isalpha()


def _is_word_part(text: str) -> bool:
    return unicodedata.normalize('NFC', text).isalnum()


def _get_word_key(text: str) -> str:
    return unicodedata.normalize('NFC', text).lower()


def _is_hyphen_on_the_line(token: LayoutToken) -> bool:
    return (
        token.text == HYPHEN
        and not token.font.is_subscript
        and not token.font.is_superscript
    )


def _get_broken_word_prefix(tokens: Sequence[LayoutToken]) -> Optional[str]:
    """The word part before a hyphen ending the line, if the line ends in one."""
    if not tokens:
        return None
    last = tokens[-1]
    if last.text.endswith(SOFT_HYPHEN) and _is_word_part(last.text[:-1]):
        return last.text[:-1]
    if (
        _is_hyphen_on_the_line(last)
        and len(tokens) >= 2
        and not tokens[-2].whitespace
        and _is_word_part(tokens[-2].text)
    ):
        return tokens[-2].text
    return None


def _get_hyphenated_words_in_line(tokens: Sequence[LayoutToken]) -> List[str]:
    return [
        _get_word_key(f'{before.text}-{after.text}')
        for before, hyphen, after in zip(tokens, tokens[1:], tokens[2:])
        if hyphen.text == HYPHEN
        and not before.whitespace
        and not hyphen.whitespace
        and _is_word_part(before.text)
        and _is_word_part(after.text)
    ]


def _iter_unbroken_words_in_block(layout_block: LayoutBlock) -> Iterable[str]:
    """The words of a block, leaving out the parts of a word broken at a line end."""
    lines = [line.tokens for line in layout_block.lines if line.tokens]
    broken_word_token_count_by_line = [
        (1 if tokens[-1].text.endswith(SOFT_HYPHEN) else 2)
        if _get_broken_word_prefix(tokens) is not None else 0
        for tokens in lines
    ]
    for index, tokens in enumerate(lines):
        starts_with_continuation = (
            index > 0
            and broken_word_token_count_by_line[index - 1] > 0
            and tokens[0].text.islower()
        )
        unbroken_start = 1 if starts_with_continuation else 0
        unbroken_end = len(tokens) - broken_word_token_count_by_line[index]
        for token in tokens[unbroken_start:unbroken_end]:
            if _is_word(token.text):
                yield _get_word_key(token.text)


def get_document_words(layout_document: LayoutDocument) -> DocumentWords:
    blocks = list(layout_document.iter_all_blocks())
    return DocumentWords(
        words=frozenset(
            word for block in blocks for word in _iter_unbroken_words_in_block(block)
        ),
        hyphenated_words=frozenset(
            word
            for block in blocks
            for line in block.lines
            for word in _get_hyphenated_words_in_line(line.tokens)
        )
    )


def _with_line_end_joined(tokens: List[LayoutToken], remove_hyphen: bool) -> List[LayoutToken]:
    last = tokens[-1]
    if last.text == HYPHEN and remove_hyphen:
        return tokens[:-2] + [tokens[-2]._replace(whitespace='')]
    if last.text.endswith(SOFT_HYPHEN):
        prefix = last.text[:-1]
        return tokens[:-1] + [last._replace(
            text=prefix if remove_hyphen else prefix + HYPHEN, whitespace=''
        )]
    return tokens[:-1] + [last._replace(whitespace='')]


def _should_remove_hyphen_by_document(
    prefix: str,
    continuation: str,
    document_words: DocumentWords
) -> Optional[bool]:
    """True or False where the document sets the joined or the hyphenated word, else None."""
    if _get_word_key(f'{prefix}-{continuation}') in document_words.hyphenated_words:
        return False
    is_in_the_case_of_a_word = continuation.islower() or (
        prefix.isupper() and continuation.isupper()
    )
    if (
        _is_word(continuation)
        and is_in_the_case_of_a_word
        and _get_word_key(prefix + continuation) in document_words.words
    ):
        return True
    return None


def _dehyphenate_by_document(
    layout_block: LayoutBlock,
    document_words: DocumentWords
) -> LayoutBlock:
    lines = [list(line.tokens) for line in layout_block.lines if line.tokens]
    for index, (tokens, next_tokens) in enumerate(zip(lines, lines[1:])):
        prefix = _get_broken_word_prefix(tokens)
        if prefix is None:
            continue
        remove_hyphen = _should_remove_hyphen_by_document(
            prefix, next_tokens[0].text, document_words
        )
        if remove_hyphen is not None:
            lines[index] = _with_line_end_joined(tokens, remove_hyphen=remove_hyphen)
    return LayoutBlock(lines=[LayoutLine(tokens=tokens) for tokens in lines])


def _get_y(token: LayoutToken) -> float:
    return token.coordinates.y if token.coordinates else -1


def _is_all_lower_case(text: str) -> bool:
    return bool(text) and all(c.islower() for c in text)


def _requires_grobid_dehyphenation(
    indexed_tokens: Sequence[Tuple[int, LayoutToken]],
    index: int
) -> bool:
    """GROBID 0.9.0's `LayoutTokensUtil.doesRequireDehypenisation`."""
    if index + 1 >= len(indexed_tokens):
        return False
    line_index, hyphen = indexed_tokens[index]
    next_line_index, next_token = indexed_tokens[index + 1]
    y = _get_y(hyphen)
    is_line_break = next_line_index != line_index
    if not is_line_break and _get_y(next_token) == y:
        return False
    if not _is_all_lower_case(next_token.text):
        return False
    if index < 1 or _get_y(next_token) > y:
        return True
    previous = indexed_tokens[index - 1][1]
    return previous.text.isalpha() and (_get_y(previous) < y or (y == -1 and is_line_break))


def _dehyphenate_as_grobid(layout_block: LayoutBlock) -> LayoutBlock:
    indexed_tokens = [
        (line_index, token)
        for line_index, line in enumerate(layout_block.lines)
        for token in line.tokens
    ]
    output: List[Tuple[int, LayoutToken]] = []
    for index, (line_index, token) in enumerate(indexed_tokens):
        if not _is_hyphen_on_the_line(token):
            output.append((line_index, token))
            continue
        if _requires_grobid_dehyphenation(indexed_tokens, index):
            if output:
                output[-1] = (output[-1][0], output[-1][1]._replace(whitespace=''))
            continue
        is_line_break = (
            index + 1 < len(indexed_tokens) and indexed_tokens[index + 1][0] != line_index
        )
        output.append((line_index, token._replace(whitespace='') if is_line_break else token))
    return LayoutBlock(lines=[
        LayoutLine(tokens=[token for _, token in line_tokens])
        for _, line_tokens in itertools.groupby(output, key=lambda item: item[0])
    ])


def dehyphenate_layout_block(
    layout_block: LayoutBlock,
    mode: str,
    document_words: DocumentWords = EMPTY_DOCUMENT_WORDS
) -> LayoutBlock:
    if mode == DehyphenateMode.GROBID:
        return _dehyphenate_as_grobid(layout_block)
    if mode == DehyphenateMode.BY_DOCUMENT:
        return _dehyphenate_by_document(layout_block, document_words)
    return layout_block


T_DehyphenateBlock = Callable[[LayoutBlock], LayoutBlock]


def keep_layout_block(layout_block: LayoutBlock) -> LayoutBlock:
    return layout_block


def get_dehyphenate_block(
    layout_document: LayoutDocument,
    mode: str
) -> T_DehyphenateBlock:
    """Dehyphenates a block of `layout_document` by `mode`."""
    if mode == DehyphenateMode.OFF:
        return keep_layout_block
    return functools.partial(
        dehyphenate_layout_block,
        mode=mode,
        document_words=(
            get_document_words(layout_document) if mode == DehyphenateMode.BY_DOCUMENT
            else EMPTY_DOCUMENT_WORDS
        )
    )
