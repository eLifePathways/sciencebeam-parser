import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


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


_MatchState = Tuple[List[Dict], List[int], List[int]]


class SequencePhraseMatch:
    """
    Case-insensitive multi-token phrase matcher.

    Replicates GROBID's FastMatcher.matchLayoutToken(tokens, ignoreDelimiters=True,
    caseSensitive=False).  Delimiter single-character tokens are skipped (but still
    counted in the position index) both during phrase loading and during matching,
    so that spans are reported as raw token-list indices inclusive of any embedded
    punctuation (e.g. tokens ["Model", ",", "Colorado"] -> match span {0, 1, 2}).
    """

    def __init__(self, phrases: Iterable[str]) -> None:
        self._trie: Dict = {}
        count = 0
        for phrase in phrases:
            self._add_phrase(phrase)
            count += 1
        LOGGER.debug('SequencePhraseMatch: loaded %d phrases', count)

    def _add_phrase(self, phrase: str) -> None:
        tokens = _phrase_to_word_tokens(phrase)
        if not tokens:
            return
        node = self._trie
        for token in tokens:
            if token not in node:
                node[token] = {}
            node = node[token]
        node['#'] = {}

    def _advance(
        self,
        state: _MatchState,
        token_lower: str,
        current_pos: int,
        results: Set[int]
    ) -> _MatchState:
        """
        Advance all open matches by one non-delimiter token.
        Completed matches (nodes containing '#') are flushed into *results*.
        Returns the updated match state.
        """
        current_matches, start_positions, last_non_sep_positions = state
        new_matches: List[Dict] = []
        new_starts: List[int] = []
        new_last_non_sep: List[int] = []
        for i, current_match in enumerate(current_matches):
            child = current_match.get(token_lower)
            if child is not None:
                new_matches.append(child)
                new_starts.append(start_positions[i])
                new_last_non_sep.append(current_pos)
            if '#' in current_match:
                results.update(range(start_positions[i], last_non_sep_positions[i] + 1))
        root_child = self._trie.get(token_lower)
        if root_child is not None:
            new_matches.append(root_child)
            new_starts.append(current_pos)
            new_last_non_sep.append(current_pos)
        return new_matches, new_starts, new_last_non_sep

    def match_token_indices(self, tokens: List[str]) -> Set[int]:
        """
        Return the set of raw token-list indices that fall within at least one matched span.

        Mirrors FastMatcher.matchLayoutToken: delimiter tokens (single-char in
        _DELIMITER_CHARS) are skipped for matching purposes but their positions
        are included when they fall inside a matched span.
        """
        state: _MatchState = ([], [], [])
        results: Set[int] = set()

        for current_pos, token_text in enumerate(tokens):
            if token_text in (' ', '\n'):
                continue
            if len(token_text) == 1 and token_text in _DELIMITER_CHARS:
                continue
            state = self._advance(state, token_text.lower(), current_pos, results)

        # Flush matches that ended at the last token of the stream
        current_matches, start_positions, last_non_sep_positions = state
        for i, current_match in enumerate(current_matches):
            if '#' in current_match:
                results.update(range(start_positions[i], last_non_sep_positions[i] + 1))

        return results


def load_phrase_match_from_text_file(path: str) -> SequencePhraseMatch:
    """Load a SequencePhraseMatch from a plain-text file with one phrase per line."""
    LOGGER.info('loading phrase match from: %r', path)
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return SequencePhraseMatch(line for line in lines if line)


def load_phrase_match_from_path(path: Optional[str]) -> Optional[SequencePhraseMatch]:
    if not path:
        return None
    return load_phrase_match_from_text_file(path)
