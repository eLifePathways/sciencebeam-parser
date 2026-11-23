import logging
import re
from functools import cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


LOGGER = logging.getLogger(__name__)

# Single-character tokens matching these are skipped during both phrase loading and matching.
# Mirrors GROBID's TextUtilities.delimiters / FastMatcher delimiter handling.
_DELIMITER_CHARS: frozenset = frozenset(
    '\n\r\t\f '
    + '‌'          # zero-width non-joiner
    + ' '          # non-breaking space
    + '([*,:;?.!/)]\\'
    + '-−–—‐'  # hyphen / dash variants
    + '«»'               # angle quotation marks
    + '„“”'         # double quotation variants
    + '‘’\''             # single quotation
    + '`$#@•'                 # misc / bullet
    + '♦♥♣♠'   # card suits
    + '。、，・'   # CJK punctuation
    + '†‡§¶⁋ǂ'  # footnote markers
)

# Matches a "word" token: one or more characters that are NOT delimiters or whitespace.
_WORD_RE = re.compile(
    '[^\\s\\(\\)\\[\\]\\*,:;?\\.!/\\\\\\-'
    + '−–—‐'
    + '«»„“”‘’'
    + '`\'$#@•'
    + '♦♥♣♠'
    + '。、，・'
    + '†‡§¶⁋ǂ'
    + '‌ '
    + ']+'
)


def _phrase_to_word_tokens(phrase: str) -> List[str]:
    """Return the lowercased non-delimiter word tokens of a phrase."""
    return _WORD_RE.findall(phrase.lower())


_Candidate = Tuple[Tuple[str, ...], int, int]


class SequencePhraseMatch:
    """
    Case-insensitive multi-token phrase matcher.

    Replicates GROBID's FastMatcher.matchLayoutToken(tokens, ignoreDelimiters=True,
    caseSensitive=False).  Delimiter single-character tokens are skipped (but still
    counted in the position index) both during phrase loading and during matching,
    so that spans are reported as raw token-list indices inclusive of any embedded
    punctuation (e.g. tokens ["Model", ",", "Colorado"] -> match span {0, 1, 2}).

    A phrase is held as its tuple of tokens, beside the set of token tuples that
    some phrase continues. Those two sets are what a trie of one dict per node
    would hold, built in half the time and a third less memory -- 620,000 place
    names are a second and 140MB rather than two and 220.
    """

    def __init__(self, phrases: Iterable[str]) -> None:
        self._complete: Set[Tuple[str, ...]] = set()
        self._continued: Set[Tuple[str, ...]] = set()
        count = 0
        for phrase in phrases:
            self._add_phrase(phrase)
            count += 1
        LOGGER.debug('SequencePhraseMatch: loaded %d phrases', count)

    def _add_phrase(self, phrase: str) -> None:
        tokens = tuple(_phrase_to_word_tokens(phrase))
        if not tokens:
            return
        self._complete.add(tokens)
        continued = self._continued
        for end in range(1, len(tokens)):
            continued.add(tokens[:end])

    def match_spans(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Return the inclusive (start, end) token-list spans of every match, in the
        order FastMatcher.matchLayoutToken reports them.

        Delimiter tokens (single-char in _DELIMITER_CHARS) are skipped for
        matching purposes; a span still covers any delimiters inside it.
        """
        complete = self._complete
        continued = self._continued
        candidates: List[_Candidate] = []
        spans: List[Tuple[int, int]] = []

        for current_pos, token_text in enumerate(tokens):
            if token_text in (' ', '\n'):
                continue
            if len(token_text) == 1 and token_text in _DELIMITER_CHARS:
                continue
            token_lower = token_text.lower()
            next_candidates: List[_Candidate] = []
            for phrase, start_pos, last_non_sep_pos in candidates:
                if phrase in complete:
                    spans.append((start_pos, last_non_sep_pos))
                extended = phrase + (token_lower,)
                if extended in continued or extended in complete:
                    next_candidates.append((extended, start_pos, current_pos))
            started = (token_lower,)
            if started in continued or started in complete:
                next_candidates.append((started, current_pos, current_pos))
            candidates = next_candidates

        for phrase, start_pos, last_non_sep_pos in candidates:
            if phrase in complete:
                spans.append((start_pos, last_non_sep_pos))

        return spans

    def match_token_indices(self, tokens: List[str]) -> Set[int]:
        """Return the token-list indices that fall within at least one matched span."""
        return {
            index
            for start, end in self.match_spans(tokens)
            for index in range(start, end + 1)
        }

    def match_token_indices_as_grobid_citation_features(
        self,
        tokens: List[str]
    ) -> Set[int]:
        """
        Return the token-list indices GROBID's citation features flag for these matches.

        FeaturesVectorCitation walks the spans with a cursor that only moves forward,
        in the order they were reported rather than by start, so a span reported after
        an overlapping one that starts later can leave some of its tokens unflagged.
        """
        spans = self.match_spans(tokens)
        flagged: Set[int] = set()
        cursor = 0
        for index in range(len(tokens)):
            if not spans:
                break
            if cursor == len(spans) - 1 and spans[cursor][1] < index:
                continue
            for span_index in range(cursor, len(spans)):
                start, end = spans[span_index]
                if start <= index <= end:
                    flagged.add(index)
                    cursor = span_index
                    break
                if start > index:
                    cursor = span_index
                    break
        return flagged


@cache
def _load_phrase_match_from_text_files(paths: Tuple[str, ...]) -> SequencePhraseMatch:
    LOGGER.info('loading phrase match from: %r', paths)
    return SequencePhraseMatch(
        line
        for path in paths
        for line in Path(path).read_text(encoding='utf-8').splitlines()
        if line
    )


def load_phrase_match_from_paths(paths: Sequence[str]) -> Optional[SequencePhraseMatch]:
    """Return one matcher over the phrases of every file, or None without files."""
    if not paths:
        return None
    return _load_phrase_match_from_text_files(tuple(paths))
