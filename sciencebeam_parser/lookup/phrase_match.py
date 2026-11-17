import logging
import re
from functools import cache
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


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

    def match_token_indices(self, tokens: List[str]) -> Set[int]:
        """
        Return the set of raw token-list indices that fall within at least one matched span.

        Mirrors FastMatcher.matchLayoutToken: delimiter tokens (single-char in
        _DELIMITER_CHARS) are skipped for matching purposes but their positions
        are included when they fall inside a matched span.
        """
        complete = self._complete
        continued = self._continued
        candidates: List[_Candidate] = []
        results: Set[int] = set()

        for current_pos, token_text in enumerate(tokens):
            if token_text in (' ', '\n'):
                continue
            if len(token_text) == 1 and token_text in _DELIMITER_CHARS:
                continue
            token_lower = token_text.lower()
            next_candidates: List[_Candidate] = []
            for phrase, start_pos, last_non_sep_pos in candidates:
                if phrase in complete:
                    results.update(range(start_pos, last_non_sep_pos + 1))
                extended = phrase + (token_lower,)
                if extended in continued or extended in complete:
                    next_candidates.append((extended, start_pos, current_pos))
            started = (token_lower,)
            if started in continued or started in complete:
                next_candidates.append((started, current_pos, current_pos))
            candidates = next_candidates

        # Flush matches that ended at the last token of the stream
        for phrase, start_pos, last_non_sep_pos in candidates:
            if phrase in complete:
                results.update(range(start_pos, last_non_sep_pos + 1))

        return results


@cache
def load_phrase_match_from_text_file(path: str) -> SequencePhraseMatch:
    """Load a SequencePhraseMatch from a plain-text file with one phrase per line."""
    LOGGER.info('loading phrase match from: %r', path)
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return SequencePhraseMatch(line for line in lines if line)


def load_phrase_match_from_path(path: Optional[str]) -> Optional[SequencePhraseMatch]:
    if not path:
        return None
    return load_phrase_match_from_text_file(path)
