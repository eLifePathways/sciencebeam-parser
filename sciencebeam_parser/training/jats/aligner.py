# pylint: disable=too-many-lines
import logging
import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from sciencebeam_alignment.align import LocalSequenceMatcher, SimpleScoring

from sciencebeam_parser.document.layout_document import LayoutDocument, LayoutToken
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.field_extractor import JatsFieldValue
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames, JatsSubFieldNames
from sciencebeam_parser.training.jats.text_normalizer import normalize_for_alignment


LOGGER = logging.getLogger(__name__)

_NO_TOKEN_INDEX = -1

# Fields whose matches establish the document region boundary (end of front matter).
# Body content is searched from the end of the last anchor match, so that front-matter
# fields which false-match citations late in the document do not push the search start
# past the actual body position.
_ANCHOR_FIELDS: FrozenSet[str] = frozenset({
    JatsFieldNames.TITLE,
    JatsFieldNames.ABSTRACT,
})

# The abstract match establishes the end of the front-matter region.  All non-anchor,
# non-body fields (authors, affiliations, keywords) are then confined to
# [0, abstract_end + _FRONT_MATTER_BUFFER] so they cannot false-match citations that
# appear later in the document.  Authors physically precede the abstract in most PDFs
# but follow it in JATS ordering, so without this constraint they would be searched
# from last_match_end (≈abstract end) and miss their true page-1 position.
_FRONT_MATTER_END_FIELDS: FrozenSet[str] = frozenset({JatsFieldNames.ABSTRACT})
_FRONT_MATTER_BUFFER = 2000

# When the "Keywords" section header is matched, individual keyword values are searched
# from that position forward rather than from position 0.  Without this, short common
# keywords ("confidence", "Bayesian") false-match in the title or abstract.
_KEYWORDS_SECTION_ANCHOR_FIELDS: FrozenSet[str] = frozenset({JatsFieldNames.KEYWORDS_TITLE})
_KEYWORDS_FIELDS: FrozenSet[str] = frozenset({JatsFieldNames.KEYWORDS})

# Fields that appear after the front matter region. They search from the body floor
# (end of last anchor match) rather than from the global last_match_end.
_BODY_CONTENT_FIELDS: FrozenSet[str] = frozenset({
    JatsFieldNames.BODY_SECTION_TITLE,
    JatsFieldNames.BODY_SECTION_PARAGRAPH,
    JatsFieldNames.BODY_FIGURE,
    JatsFieldNames.BODY_TABLE,
    JatsFieldNames.ACK_SECTION_TITLE,
    JatsFieldNames.ACK_SECTION_PARAGRAPH,
    JatsFieldNames.APPENDIX_GROUP_TITLE,
    JatsFieldNames.APPENDIX,
    JatsFieldNames.BACK_SECTION_TITLE,
    JatsFieldNames.BACK_SECTION_PARAGRAPH,
})

# Reference fields use a dedicated floor so that appendix/body content matched
# after the reference section cannot push body_content_end past the references.
# reference_list_title anchors the search; references then advance reference_floor
# incrementally.  Without this separation, body content from mathematical appendices
# (which may physically appear after references in the PDF) would advance
# body_content_end past all reference positions.
_REFERENCE_ANCHOR_FIELDS: FrozenSet[str] = frozenset({
    JatsFieldNames.REFERENCE_LIST_TITLE,
})
_REFERENCE_FIELDS: FrozenSet[str] = frozenset({
    JatsFieldNames.REFERENCE,
})

# Fields that appear entirely after the main body and reference sections
# (e.g. ORE peer-review sub-articles).  Searched from last_match_end rather than
# from the front-matter window, preventing author-response text (which often quotes
# the paper verbatim) from overwriting body / figure tokens via the global fallback.
_POST_BODY_FIELDS: FrozenSet[str] = frozenset({
    JatsFieldNames.SUB_ARTICLE,
})

# Smith-Waterman scoring: match=2, mismatch=-1, gap=-1
_SCORING = SimpleScoring(match_score=2, mismatch_score=-1, gap_score=-1)

# Window size limits to keep LocalSequenceMatcher fast (O(window * needle))
_DEFAULT_MIN_WINDOW = 2000
_WINDOW_NEEDLE_MULTIPLIER = 6

# Sub-field containment buffer: search sub-fields this many chars beyond the parent's
# matched range.  Keeps short sub-field values (e.g. "USA", "2020") from matching
# identical text elsewhere in the document.
_SUB_FIELD_PARENT_BUFFER = 200
# Pure-number labels precede the JATS parent text; this extends the search backward.
_SUB_FIELD_PARENT_PRE_BUFFER = 20
# Digit-prefix labels (e.g. "1-") may also precede the parent match start because SW
# aligns the parent by skipping the label digit, leaving p_start at the suffix char.
# A small buffer is enough — single-digit + space = 2 chars.
_SUB_FIELD_LABEL_DIGIT_PRE_BUFFER = 3
# Author names appear before the title/journal anchor text that SW latches onto.
# When the JATS given-names are initials ("RC") but the PDF has "R. C." (each initial
# as a separate token), the run of gaps makes SW skip the author prefix entirely and
# start the parent match at the title.  200 chars covers long multi-author lists too.
_SUB_FIELD_REFERENCE_AUTHOR_PRE_BUFFER = 200
# Identifier sub-fields (DOI, PMID, PMCID) are the only ones that appear in a URL tail
# AFTER the parent match text.  Their match ends advance the backward-search floor so
# the next reference's author search cannot reach back into the URL.
_REFERENCE_IDENTIFIER_SUB_FIELDS = frozenset({
    JatsSubFieldNames.REFERENCE_DOI,
    JatsSubFieldNames.REFERENCE_PMID,
    JatsSubFieldNames.REFERENCE_PMCID,
})

# Anchor+chain labelling strategy:
# Smith-Waterman produces many tiny (1–4 char) matching blocks while traversing
# interleaved sidebar content.  Those blocks must not cause sidebar tokens to be
# labelled.  We use two constants:
#
#   _MIN_ANCHOR_BLOCK_SIZE: a block is an "anchor" only if it is at least this
#     many characters long.  Sidebar words are almost never exact multi-word
#     substrings of the abstract/keywords needle, so their SW blocks stay small.
#
#   _MAX_HAYSTACK_GAP_TO_FILL: a small (non-anchor) block is included only if it
#     starts within this many characters of the previous *included* block end.
#     This fills legitimate intra-field gaps (e.g. the 2-char comma gap between
#     "confidence, bayesian," and "ddm") without re-entering a sidebar whose
#     last anchor lies hundreds of chars earlier.
#
# Fallback: when a field value produces NO anchor blocks at all (the entire text
# is shorter than _MIN_ANCHOR_BLOCK_SIZE), every block is labelled so short
# fields are never silently dropped.
_MIN_ANCHOR_BLOCK_SIZE = 5
_MAX_HAYSTACK_GAP_TO_FILL = 3
# Maximum number of non-author tokens permitted between two author spans before
# the gap fill gives up.  Covers up to ~5 initials with periods and separators.
_MAX_AUTHOR_GAP_TOKENS = 10

# Type alias for the return value of _fuzzy_match_field_value:
#   (abs_start, abs_end, [(block_start, block_end), ...])
_MatchResult = Tuple[int, int, List[Tuple[int, int]]]

# When a parent REFERENCE match fails at the primary threshold (0.8), retry at this
# lower value.  JATS author initials may be concatenated ("CA") while the PDF expands
# them ("C. A."), and institutional refs can omit boilerplate text that pads the needle
# without appearing in the PDF reference list.  0.55 is sufficient to capture truncated
# PDF references (e.g. a ref whose year/volume/URL are absent, giving quality ~0.60)
# while rejecting genuinely absent ones.
_REFERENCE_PARENT_MIN_THRESHOLD = 0.55


@dataclass
class AlignmentConfig:
    threshold: float = 0.8
    max_window: int = 8000


class _TokenIndex:
    """Flat character-level haystack built from all layout tokens.

    Tracks which character offset belongs to which token so that a match
    range [a, b) in the haystack can be mapped back to a set of tokens.

    Line-break hyphens are removed and the two word halves concatenated so
    that tokens ["hyphen", "-"] at end of line followed by ["ation"] at the
    start of the next line appear as "hyphenation" in the haystack, matching
    the unhyphenated form found in the JATS source text.
    """

    def __init__(
        self,
        tokens: List[LayoutToken],
        skip_tokens: Optional[Set[int]] = None,
        no_space_after: Optional[Set[int]] = None,
    ) -> None:
        self.tokens = tokens
        if skip_tokens is None:
            skip_tokens = set()
        if no_space_after is None:
            no_space_after = set()
        parts: List[str] = []
        token_index_at: List[int] = []

        for tok_idx, token in enumerate(tokens):
            if tok_idx in skip_tokens:
                continue
            norm = normalize_for_alignment(token.text)
            if not norm:
                continue
            for _ in norm:
                token_index_at.append(tok_idx)
            parts.append(norm)
            if tok_idx not in no_space_after:
                parts.append(' ')
                token_index_at.append(_NO_TOKEN_INDEX)

        self.haystack = ''.join(parts)
        self._token_index_at = token_index_at
        self._skip_tokens = skip_tokens

    def tokens_in_range(self, start: int, end: int) -> List[LayoutToken]:
        seen: Set[int] = set()
        result_indices: List[int] = []
        for i in range(start, min(end, len(self._token_index_at))):
            tok_idx = self._token_index_at[i]
            if tok_idx != _NO_TOKEN_INDEX and tok_idx not in seen:
                seen.add(tok_idx)
                result_indices.append(tok_idx)
        # Include bare end-of-line hyphen tokens (skip_tokens) whose preceding
        # word token was collected.  These hyphens are invisible in the haystack
        # but are physically part of the hyphenated word and should carry the
        # same label as the surrounding tokens.
        if self._skip_tokens:
            filled: List[int] = []
            added_skips: Set[int] = set()
            for tok_idx in result_indices:
                filled.append(tok_idx)
                next_idx = tok_idx + 1
                if next_idx in self._skip_tokens and next_idx not in added_skips:
                    filled.append(next_idx)
                    added_skips.add(next_idx)
            result_indices = filled
        return [self.tokens[i] for i in result_indices]

    def is_token_start(self, pos: int) -> bool:
        """Return True if pos is the first character of a layout token (not mid-token)."""
        if pos <= 0:
            return True
        if pos >= len(self._token_index_at):
            return False
        return self._token_index_at[pos - 1] != self._token_index_at[pos]

    def is_in_token(self, pos: int) -> bool:
        """Return True if pos is within a token (not a space between tokens)."""
        if pos < 0 or pos >= len(self._token_index_at):
            return False
        return self._token_index_at[pos] != _NO_TOKEN_INDEX

    def is_token_boundary_after(self, pos: int) -> bool:
        """Return True if the character at pos is the last in its token (or pos is past end)."""
        if pos >= len(self._token_index_at):
            return True
        if not self.is_in_token(pos):
            return True
        next_pos = pos + 1
        if next_pos >= len(self._token_index_at):
            return True
        return self._token_index_at[pos] != self._token_index_at[next_pos]


def _build_token_index(layout_document: LayoutDocument) -> _TokenIndex:
    all_tokens: List[LayoutToken] = []
    skip_tokens: Set[int] = set()
    no_space_after: Set[int] = set()
    for line in layout_document.iter_all_lines():
        line_tokens: List[LayoutToken] = line.tokens or []
        for i, token in enumerate(line_tokens):
            tok_global_idx = len(all_tokens)
            all_tokens.append(token)
            if i == len(line_tokens) - 1:
                norm = normalize_for_alignment(token.text)
                if norm == '-':
                    # Bare end-of-line hyphen produced by the PDF tokenizer when
                    # a word is split across lines (e.g. ["hyphen", "-"] then
                    # ["ation"]). Skip the "-" entirely and suppress the trailing
                    # space on the preceding word so the two halves join without a
                    # gap: "hyphen" + "ation" → "hyphenation".
                    skip_tokens.add(tok_global_idx)
                    if i > 0:
                        no_space_after.add(tok_global_idx - 1)
    return _TokenIndex(all_tokens, skip_tokens=skip_tokens, no_space_after=no_space_after)


def _match_quality(
    matching_blocks: List[Tuple[int, int, int]],
    needle_len: int,
) -> float:
    """Fraction of needle characters matched (0..1)."""
    if needle_len == 0:
        return 1.0
    matched = sum(size for _, _, size in matching_blocks if size)
    return matched / needle_len


def _scan_tail_chars_at_token_starts(
    tail: str,
    scan_start: int,
    token_index: '_TokenIndex',
) -> int:
    """Scan for each char in tail at token-start positions with gaps ≤ _MAX_HAYSTACK_GAP_TO_FILL.

    Returns the absolute position just after the last successfully matched char.
    If no chars can be matched, returns scan_start.  Used to bridge gaps such as
    ". " between dotted initials ("L. Y.") when the needle has "ly".
    """
    pos = scan_start
    end = scan_start
    for char in tail:
        limit = pos + _MAX_HAYSTACK_GAP_TO_FILL + 1
        found = False
        while pos < limit and pos < len(token_index.haystack):
            if token_index.is_token_start(pos) and token_index.haystack[pos] == char:
                end = pos + 1
                pos = end
                found = True
                break
            pos += 1
        if not found:
            break
    return end


def _extend_match_for_needle_tail(
    window: str,
    needle: str,
    window_start: int,
    abs_a_end: int,
    matched_blocks: List[Tuple[int, int, int]],
    abs_block_ranges: List[Tuple[int, int]],
    token_index: Optional['_TokenIndex'] = None,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Greedily extend the SW match to cover any unmatched needle suffix.

    Smith-Waterman terminates when extending the match would not increase the
    score.  A single-character suffix separated from the last block by one or
    two gap characters produces a net-zero extension (e.g. two gap penalties
    cancel one match bonus), so SW stops early.  This function scans forward
    within _MAX_HAYSTACK_GAP_TO_FILL characters of the current match end for
    the first unmatched needle character and appends consecutive matches as an
    extra block, mirroring the pre-anchor fill logic on the other end.

    Example: needle "brockmann d", match ends at "brockmann "; " , d" in the
    haystack has "d" at gap 2, which is within the fill threshold.

    When token_index is provided, a second pass uses _scan_tail_chars_at_token_starts
    to bridge larger gaps between remaining suffix chars (e.g. "l . y ." where
    the needle suffix "ly" needs to hop over ". " to reach "y").
    """
    needle_tail = needle[max(bi + size for _, bi, size in matched_blocks):]
    if not needle_tail:
        return abs_a_end, abs_block_ranges

    win_pos = abs_a_end - window_start
    first_pos = window[win_pos: win_pos + _MAX_HAYSTACK_GAP_TO_FILL + 1].find(needle_tail[0])
    if first_pos == -1:
        return abs_a_end, abs_block_ranges

    tail_start = win_pos + first_pos
    match_count = 0
    for tc in needle_tail:
        if tail_start + match_count >= len(window) or window[tail_start + match_count] != tc:
            break
        match_count += 1

    if not match_count:
        return abs_a_end, abs_block_ranges

    ext_start = window_start + tail_start
    # If chars remain unmatched and a token_index is available, try to bridge gaps
    # to find them at token boundaries (handles ". " gaps between dotted initials).
    if token_index is not None and match_count < len(needle_tail):
        bridge_end = _scan_tail_chars_at_token_starts(
            needle_tail[match_count:], ext_start + match_count, token_index
        )
        if bridge_end > ext_start + match_count:
            return bridge_end, abs_block_ranges + [(ext_start, bridge_end)]

    return ext_start + match_count, abs_block_ranges + [(ext_start, ext_start + match_count)]


def _fuzzy_search_in_window(
    haystack: str,
    needle: str,
    window_start: int,
    window_end: int,
    threshold: float,
    token_index: Optional['_TokenIndex'] = None,
) -> Optional[_MatchResult]:
    """Try to find `needle` in haystack[window_start:window_end].

    Returns (abs_start, abs_end, [(block_start, block_end), ...]) if quality >=
    threshold, else None.  Block ranges are in absolute haystack coordinates.
    After the SW match, a greedy tail extension fills any unmatched needle
    suffix whose first character falls within _MAX_HAYSTACK_GAP_TO_FILL chars
    of the current match end (mirrors the pre-anchor fill on the trailing side).
    """
    window = haystack[window_start:window_end]
    sm = LocalSequenceMatcher(a=window, b=needle, scoring=_SCORING)
    blocks = sm.get_matching_blocks()
    quality = _match_quality(blocks, len(needle))
    if quality < threshold:
        return None
    matched_blocks = [(ai, bi, size) for ai, bi, size in blocks if size]
    if not matched_blocks:
        return None
    a_start = matched_blocks[0][0] + window_start
    last = matched_blocks[-1]
    a_end = last[0] + last[2] + window_start
    abs_block_ranges: List[Tuple[int, int]] = [
        (ai + window_start, ai + size + window_start)
        for ai, _bi, size in matched_blocks
    ]
    a_end, abs_block_ranges = _extend_match_for_needle_tail(
        window, needle, window_start, a_end, matched_blocks, abs_block_ranges, token_index
    )
    return a_start, a_end, abs_block_ranges


def _get_unmasked_segments(
    search_start: int,
    search_end: int,
    masked_ranges: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Split [search_start, search_end) into segments that don't overlap masked_ranges.

    Masked boundaries are hard: SW runs on each segment independently and
    cannot produce a match that spans across a masked span.
    """
    clipped = sorted(
        (max(m0, search_start), min(m1, search_end))
        for m0, m1 in masked_ranges
        if m0 < search_end and m1 > search_start
    )
    segments: List[Tuple[int, int]] = []
    cur = search_start
    for m_start, m_end in clipped:
        if cur < m_start:
            segments.append((cur, m_start))
        cur = max(cur, m_end)
    if cur < search_end:
        segments.append((cur, search_end))
    return segments


def _is_pure_number(text: str) -> bool:
    return bool(re.fullmatch(r'\d+', text))


_BRACKET_LABEL_RE = re.compile(r'^(\[)(.+)(\])$|^(\()(.+)(\))$')
_LABEL_DIGIT_PREFIX_RE = re.compile(r'^(\d+)(.+)$')

# How far before the first segment start to search for the bracket label inner content.
# Needed because the parent SW match may start at "]" (skipping the preceding "[" and
# the inner number), so the sub-field search range starts after the label tokens.
_BRACKET_LABEL_BACK_BUFFER = 10


def _find_inner_token(
    token_index: _TokenIndex,
    inner: str,
    segments: List[Tuple[int, int]],
) -> Optional[_MatchResult]:
    """Locate `inner` as a standalone token within segments; pure-number fast path."""
    if _is_pure_number(inner):
        return _exact_number_match(token_index, inner, segments)
    haystack = token_index.haystack
    inner_len = len(inner)
    for seg_start, seg_end in segments:
        pos = seg_start
        while pos <= seg_end - inner_len:
            idx = haystack.find(inner, pos, seg_end)
            if idx == -1:
                break
            end = idx + inner_len
            if (token_index.is_in_token(idx)
                    and token_index.is_token_start(idx)
                    and token_index.is_token_boundary_after(end - 1)):
                return (idx, end, [(idx, end)])
            pos = idx + 1
    return None


def _try_bracket_label_match(  # pylint: disable=too-many-locals
    token_index: _TokenIndex,
    needle: str,
    segments: List[Tuple[int, int]],
) -> Optional[_MatchResult]:
    """Match bracket-style labels like [1] or (2) whose tokens are split by the PDF tokeniser.

    When the PDF tokeniser produces three tokens "[", "1", "]" the haystack has
    "[ 1 ]".  Smith-Waterman cannot match "[1]" across those spaces because the
    scoring library's traceback terminates early at gap moves, yielding quality < threshold.

    This function strips the outer brackets, matches the inner content, then extends
    the match range to cover the bracket tokens that immediately surround the hit.
    The search extends _BRACKET_LABEL_BACK_BUFFER chars before the first segment start
    because the parent SW match often begins at "]", leaving "[" and the number before
    its first matched block (and thus outside the nominal search range).
    """
    m = _BRACKET_LABEL_RE.fullmatch(needle)
    if not m:
        return None
    if m.group(1):
        open_b, inner, close_b = m.group(1), m.group(2), m.group(3)
    else:
        open_b, inner, close_b = m.group(4), m.group(5), m.group(6)

    extended = (
        [(max(0, segments[0][0] - _BRACKET_LABEL_BACK_BUFFER), segments[0][1])]
        + list(segments[1:])
    ) if segments else segments

    inner_match = _find_inner_token(token_index, inner, extended)
    if inner_match is None:
        return None

    m_start, m_end, blocks = inner_match
    haystack = token_index.haystack

    # Extend to include adjacent opening bracket token (up to 2 chars before m_start).
    new_start = m_start
    new_blocks: List[Tuple[int, int]] = list(blocks)
    for pos in range(max(0, m_start - 2), m_start):
        if (haystack[pos] == open_b
                and token_index.is_in_token(pos)
                and token_index.is_token_start(pos)):
            new_start = pos
            new_blocks = [(pos, pos + 1)] + new_blocks
            break

    # Extend to include adjacent closing bracket token (up to 2 chars after m_end).
    new_end = m_end
    for pos in range(m_end, min(m_end + 2, len(haystack))):
        if (haystack[pos] == close_b
                and token_index.is_in_token(pos)
                and token_index.is_token_start(pos)):
            new_end = pos + 1
            new_blocks = new_blocks + [(pos, pos + 1)]
            break

    return new_start, new_end, new_blocks


def _try_numeric_prefix_label_match(
    token_index: _TokenIndex,
    label_needle: str,
    segments: List[Tuple[int, int]],
) -> Optional[_MatchResult]:
    """Match labels like "1-" when SW fails because the PDF has "1 -" (space between
    digit and suffix).  Finds the digit prefix as an exact token via _exact_number_match
    then extends the match to cover the immediately adjacent suffix characters."""
    m = _LABEL_DIGIT_PREFIX_RE.match(label_needle)
    if not m:
        return None
    numeric_part, suffix = m.group(1), m.group(2)
    result = _exact_number_match(token_index, numeric_part, segments)
    if result is None:
        return None
    num_start, num_end, num_blocks = result
    haystack = token_index.haystack
    pos = num_end
    while pos < len(haystack) and haystack[pos] == ' ':
        pos += 1
    if haystack[pos:pos + len(suffix)] == suffix:
        return num_start, pos + len(suffix), num_blocks + [(pos, pos + len(suffix))]
    return result


def _is_exact_sw_match(result: _MatchResult, needle_len: int) -> bool:
    """True when SW found the needle as one contiguous block (no gaps)."""
    _, _, blocks = result
    return len(blocks) == 1 and (blocks[0][1] - blocks[0][0]) == needle_len


def _is_punct_suffix_token(
    token_index: _TokenIndex,
    haystack: str,
    end: int,
) -> bool:
    """Return True when the characters after `end` (still in the same token) are all
    punctuation.  This covers "1." or "1," where the PDF tokeniser attaches the
    delimiter to the digit, so the exact number "1" cannot be matched with a clean
    token boundary but is still the correct label to extract."""
    pos = end
    while pos < len(haystack) and token_index.is_in_token(pos):
        if not haystack[pos].isspace() and haystack[pos] not in '.,;:)]}':
            return False
        if not token_index.is_token_boundary_after(pos):
            pos += 1
            continue
        break
    return True


def _exact_number_match(
    token_index: _TokenIndex,
    needle: str,
    segments: List[Tuple[int, int]],
) -> Optional[_MatchResult]:
    haystack = token_index.haystack
    needle_len = len(needle)
    for seg_start, seg_end in segments:
        pos = seg_start
        while pos <= seg_end - needle_len:
            idx = haystack.find(needle, pos, seg_end)
            if idx == -1:
                break
            end = idx + needle_len
            if (token_index.is_in_token(idx)
                    and token_index.is_token_start(idx)
                    and (token_index.is_token_boundary_after(end - 1)
                         or _is_punct_suffix_token(token_index, haystack, end))):
                return idx, end, [(idx, end)]
            pos = idx + 1
    return None


def _fuzzy_match_field_value(  # pylint: disable=too-many-locals
    token_index: _TokenIndex,
    field_value: JatsFieldValue,
    config: AlignmentConfig,
    search_start: int,
    search_end: Optional[int] = None,
    masked_ranges: Optional[List[Tuple[int, int]]] = None,
) -> Optional[_MatchResult]:
    needle = normalize_for_alignment(field_value.text)
    if not needle:
        return None

    haystack = token_index.haystack
    hay_end = len(haystack) if search_end is None else min(search_end, len(haystack))

    segments: List[Tuple[int, int]] = (
        _get_unmasked_segments(search_start, hay_end, masked_ranges)
        if masked_ranges
        else [(search_start, hay_end)]
    )

    if _is_pure_number(needle):
        return _exact_number_match(token_index, needle, segments)

    need_len = len(needle)
    window_size = max(
        _DEFAULT_MIN_WINDOW,
        min(config.max_window, need_len * _WINDOW_NEEDLE_MULTIPLIER),
    )
    stride = max(1, window_size - need_len - 20)

    gap_match: Optional[_MatchResult] = None
    for seg_start, seg_end in segments:
        start = seg_start
        while start < seg_end:
            end = min(start + window_size, seg_end)
            result = _fuzzy_search_in_window(
                haystack, needle, start, end, config.threshold, token_index
            )
            if result is not None:
                if _is_exact_sw_match(result, need_len):
                    return result
                if gap_match is None:
                    gap_match = result
            if end >= seg_end:
                break
            start += stride

    if gap_match is None:
        bracket_match = _try_bracket_label_match(token_index, needle, segments)
        if bracket_match is not None:
            return bracket_match

    if gap_match is None and field_value.sub_field_name == JatsSubFieldNames.REFERENCE_LABEL:
        prefix_match = _try_numeric_prefix_label_match(token_index, needle, segments)
        if prefix_match is not None:
            return prefix_match

    return gap_match


def _search_range(
    fv: JatsFieldValue,
    last_match_end: int,
    body_floor: int,
    body_content_end: int,
    front_matter_end: int,
    keywords_floor: int,
    reference_floor: int,
    parent_match_by_field: Dict[str, Tuple[int, int, int]],
) -> Tuple[int, Optional[int]]:
    """Return (search_start, search_end) for fv given current position state."""
    if fv.sub_field_name is not None and fv.field_name in parent_match_by_field:
        p_start, p_end, pre_parent_ref_floor = parent_match_by_field[fv.field_name]
        if fv.sub_field_name == JatsSubFieldNames.REFERENCE_LABEL and _is_pure_number(fv.text):
            pre = _SUB_FIELD_PARENT_PRE_BUFFER
        elif (
            fv.sub_field_name == JatsSubFieldNames.REFERENCE_LABEL
            and bool(_LABEL_DIGIT_PREFIX_RE.match(fv.text))
        ):
            pre = _SUB_FIELD_LABEL_DIGIT_PRE_BUFFER
        elif fv.sub_field_name in {
            JatsSubFieldNames.REFERENCE_AUTHOR,
            JatsSubFieldNames.REFERENCE_SOURCE,
        }:
            # Source can precede the parent's SW match start when the PDF orders
            # source before article-title but the JATS parent text has them reversed.
            # SW then latches onto the article-title anchor and sets p_start after
            # the source, so we need the same backward buffer as for authors.
            pre = _SUB_FIELD_REFERENCE_AUTHOR_PRE_BUFFER
        else:
            pre = 0
        # For authors and source: never extend before the end of the previous
        # reference (prevents sub-field matches from bleeding into earlier bibls).
        sub_start = max(p_start - pre, pre_parent_ref_floor) \
            if fv.sub_field_name in {
                JatsSubFieldNames.REFERENCE_AUTHOR,
                JatsSubFieldNames.REFERENCE_SOURCE,
            } \
            else p_start - pre
        return max(0, sub_start), p_end + _SUB_FIELD_PARENT_BUFFER
    if fv.field_name in _BODY_CONTENT_FIELDS:
        return max(0, max(body_floor, body_content_end) - 200), None
    if fv.field_name in _REFERENCE_ANCHOR_FIELDS or fv.field_name in _REFERENCE_FIELDS:
        # References use a dedicated floor that is independent of body_content_end.
        # This prevents appendix or late body content from advancing body_content_end
        # past the reference section, which would make the reference search start
        # skip over all reference positions.
        return (max(0, reference_floor - 200) if reference_floor > 0 else max(0, body_floor)), None
    if fv.field_name in _ANCHOR_FIELDS or fv.field_name in _POST_BODY_FIELDS:
        # Anchor fields (abstract, title) and post-body fields (sub-articles) both
        # search from last_match_end so they follow reading order and cannot fall
        # back to the front-matter window.
        return max(0, last_match_end - 200), None
    if front_matter_end > 0:
        # Front-matter constrained fields (authors, affs, keywords).
        # Keywords are anchored to just after the keywords header/abstract so
        # short common words ("confidence", "Bayesian") don't false-match in the
        # abstract.  Other front-matter fields start from position 0.
        is_keywords = fv.field_name in _KEYWORDS_FIELDS
        start = max(keywords_floor, front_matter_end) if is_keywords else 0
        return start, front_matter_end + _FRONT_MATTER_BUFFER
    return max(0, last_match_end - 200), None


def _pre_anchor_indices(
    block_ranges: List[Tuple[int, int]]
) -> Optional[Set[int]]:
    """Return indices of non-anchor blocks tightly preceding the first anchor.

    Returns None when there are no anchor blocks at all (caller should label
    every block unconditionally).  Otherwise walks backward from the block
    just before the first anchor; stops when the gap exceeds
    _MAX_HAYSTACK_GAP_TO_FILL.  The result covers DOI-prefix segments
    ("10", ".", "1128", "/") that precede a longer segment but should still
    be labeled.
    """
    first_anchor_idx = next(
        (i for i, (bs, be) in enumerate(block_ranges) if be - bs >= _MIN_ANCHOR_BLOCK_SIZE),
        None,
    )
    if first_anchor_idx is None:
        return None
    result: Set[int] = set()
    prev_start = block_ranges[first_anchor_idx][0]
    for i in range(first_anchor_idx - 1, -1, -1):
        _, be = block_ranges[i]
        if prev_start - be <= _MAX_HAYSTACK_GAP_TO_FILL:
            result.add(i)
            prev_start = block_ranges[i][0]
        else:
            break
    return result


def _is_haystack_token_start(token_index: _TokenIndex, pos: int) -> bool:
    """Return True if pos is the first character of a layout token in the haystack.

    A SW matching block may start mid-token when a single character in the tail of
    a longer token happens to match the first character of the needle (e.g. the
    final 'o' of "introdução" matching needle "o crescimento...").  Including such
    a block in the pre-anchor fill would cause tokens_in_range to return the preceding
    token and overwrite its label — typically a heading label with a paragraph label.

    DOI sub-tokens ("10", ".", "3233", "/") always occupy their own token and always
    start at a token boundary, so they are unaffected by this guard.
    """
    return token_index.is_token_start(pos)


def _extend_match_with_given_names_tail(
    match_range: '_MatchResult',
    original_text: str,
    fallback_text: str,
    token_index: '_TokenIndex',
) -> '_MatchResult':
    """After a surname-fallback match, try to extend it with the given-names portion.

    When the mid-token fallback preference selects the fallback (surname-only) match,
    the given-names initial (e.g. "T" from "Guardian T") is missing.  This function
    searches within _MAX_HAYSTACK_GAP_TO_FILL chars of the fallback match end for the
    first character of the given-names and extends the block list if found.
    """
    given_tail = normalize_for_alignment(original_text)[
        len(normalize_for_alignment(fallback_text)):
    ].lstrip()
    if not given_tail:
        return match_range
    fb_end = match_range[1]
    win = token_index.haystack[fb_end: fb_end + _MAX_HAYSTACK_GAP_TO_FILL + 1 + len(given_tail)]
    idx = win.find(given_tail[0])
    if idx == -1 or idx > _MAX_HAYSTACK_GAP_TO_FILL:
        return match_range
    abs_tail_start = fb_end + idx
    if not token_index.is_token_start(abs_tail_start):
        return match_range
    tail_end = _scan_tail_chars_at_token_starts(given_tail, abs_tail_start, token_index)
    if tail_end == abs_tail_start:
        return match_range
    return (
        match_range[0],
        max(match_range[1], tail_end),
        match_range[2] + [(abs_tail_start, tail_end)],
    )


def _has_mid_token_within_gap_blocks(
    block_ranges: List[Tuple[int, int]],
    token_index: '_TokenIndex',
) -> bool:
    """Return True if block_ranges contains a non-anchor within-gap block that starts mid-token.

    This identifies SW matches that achieved quality only by reaching into the middle of a
    word — e.g. 't' inside 'staff' when searching for author initial 'T'.  When a fallback
    needle (surname only) exists, the aligner can instead try the fallback to find the
    correct earlier occurrence.
    """
    prev_end: Optional[int] = None
    for block_start, block_end in block_ranges:
        is_anchor = (block_end - block_start) >= _MIN_ANCHOR_BLOCK_SIZE
        within_gap = (
            prev_end is not None
            and block_start - prev_end <= _MAX_HAYSTACK_GAP_TO_FILL
        )
        if within_gap and not is_anchor and not token_index.is_token_start(block_start):
            return True
        prev_end = block_end
    return False


def _label_tokens_for_blocks(
    annotated: JatsAnnotatedLayoutDocument,
    token_index: _TokenIndex,
    block_ranges: List[Tuple[int, int]],
    field_name: str,
    sub_field_name: Optional[str],
    instance_id: int,
) -> None:
    """Label tokens using anchor+chain strategy (see module constants for rationale).

    The forward anchor+chain is extended by a backward pre-anchor pass: non-anchor
    blocks that immediately precede the first anchor (tight gap ≤ _MAX_HAYSTACK_GAP_TO_FILL)
    are included, so dense sub-tokens like DOI segments ("10", ".", "1128", "/")
    that appear before the first long segment are not silently dropped.  Blocks with
    a larger gap before the first anchor (sidebar text, page headers) remain excluded.

    Pre-anchor blocks that start mid-token (tail characters of a longer token
    incidentally matching the needle start) are skipped to prevent overwriting labels
    already set by earlier field values on that token.
    """
    pre_anchor = _pre_anchor_indices(block_ranges)
    if pre_anchor is None:
        # No anchor blocks at all: label everything so short field values are preserved.
        for block_start, block_end in block_ranges:
            for token in token_index.tokens_in_range(block_start, block_end):
                annotated.set_token_label(token, field_name, sub_field_name, instance_id)
        return
    prev_included_end: Optional[int] = None
    for i, (block_start, block_end) in enumerate(block_ranges):
        is_anchor = (block_end - block_start) >= _MIN_ANCHOR_BLOCK_SIZE
        within_gap = (
            prev_included_end is not None
            and block_start - prev_included_end <= _MAX_HAYSTACK_GAP_TO_FILL
        )
        if not is_anchor and not within_gap and i not in pre_anchor:
            continue
        if i in pre_anchor and not within_gap and not _is_haystack_token_start(
            token_index, block_start
        ):
            continue
        if within_gap and not is_anchor and not _is_haystack_token_start(
            token_index, block_start
        ):
            prev_included_end = block_end
            continue
        fill_start = prev_included_end if within_gap else block_start
        assert fill_start is not None
        for token in token_index.tokens_in_range(fill_start, block_end):
            annotated.set_token_label(token, field_name, sub_field_name, instance_id)
        prev_included_end = block_end


def _fill_sub_field_gaps(
    annotated: JatsAnnotatedLayoutDocument,
    tokens: List[LayoutToken],
    field_name: str,
    sub_field_name: str,
) -> None:
    """Fill token gaps between consecutive sub_field spans of the same instance.

    When author names are aligned per-name, separator tokens (commas, semicolons,
    initials with periods) between consecutively matched names remain unlabeled.
    This merges them into a single contiguous span so that all author tokens end
    up in one <author> element — matching the grobid training data convention.

    Gaps wider than _MAX_AUTHOR_GAP_TOKENS tokens are left unfilled to avoid
    accidentally absorbing subsequent reference fields (title, year, etc.).
    """
    last_instance: Optional[int] = None
    pending: List[LayoutToken] = []

    for token in tokens:
        entry = annotated.token_label_by_id.get(id(token))
        if (
            entry is not None
            and entry[0] == field_name
            and entry[1] == sub_field_name
        ):
            if last_instance is not None and entry[2] == last_instance and pending:
                for pt in pending:
                    annotated.set_token_label(pt, field_name, sub_field_name, last_instance)
            last_instance = entry[2]
            pending = []
        elif last_instance is not None and (
            entry is None
            or (entry[0] == field_name and entry[2] == last_instance)
        ):
            # Include unlabeled tokens (entry is None) and same-instance reference tokens
            # in the gap between two author spans.  The parent bibl SW match does not cover
            # separators like "." between an initial and "et al.", so those tokens have no
            # label at this point.
            if len(pending) < _MAX_AUTHOR_GAP_TOKENS:
                pending.append(token)
            else:
                last_instance = None
                pending = []
        else:
            last_instance = None
            pending = []


def _attach_sub_field_trailing_periods(
    annotated: JatsAnnotatedLayoutDocument,
    tokens: List[LayoutToken],
    field_name: str,
    sub_field_name: str,
) -> None:
    """Relabel a bare '.' token that immediately follows a sub_field-labeled token.

    The PDF tokeniser splits 'D.' into 'D' and '.'.  The period is not in the
    JATS text, so SW and gap-fill logic miss it.  This pass attaches such periods,
    mirroring reference_annotator.get_suffix_extended_token_tags in the old tool.
    """
    last_was_sub = False
    last_instance = 0

    for token in tokens:
        entry = annotated.token_label_by_id.get(id(token))
        if (
            entry is not None
            and entry[0] == field_name
            and entry[1] == sub_field_name
        ):
            last_was_sub = True
            last_instance = entry[2]
        elif (
            last_was_sub
            and (entry is None or entry[0] == field_name)
            and normalize_for_alignment(token.text or '') == '.'
        ):
            annotated.set_token_label(token, field_name, sub_field_name, last_instance)
        else:
            last_was_sub = False


def _sort_reference_sub_fields_by_length(
    field_values: List[JatsFieldValue],
) -> List[JatsFieldValue]:
    """Within each reference's sub-field group, sort by descending normalized-needle length.

    Processing longer needles first means a short REFERENCE_LABEL "1" cannot claim a
    position that REFERENCE_YEAR "1987" should own: the year matches and masks first, so
    the label either lands on the true citation number or fails gracefully.  Authors and
    other sub-fields follow the same rule — within each group the longest needle wins the
    best position, and shorter needles fill remaining unmasked positions.
    """
    result: List[JatsFieldValue] = []
    pending: List[JatsFieldValue] = []

    def _flush() -> None:
        if pending:
            pending.sort(
                key=lambda fv: len(normalize_for_alignment(fv.text)),
                reverse=True,
            )
            result.extend(pending)
            pending.clear()

    for fv in field_values:
        if fv.sub_field_name is None:
            _flush()
            result.append(fv)
        else:
            pending.append(fv)
    _flush()
    return result


class LayoutDocumentJatsAligner:
    """Aligns JATS field values to LayoutDocument tokens via fuzzy text matching."""

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()

    def align(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        layout_document: LayoutDocument,
        field_values: List[JatsFieldValue],
    ) -> JatsAnnotatedLayoutDocument:
        annotated = JatsAnnotatedLayoutDocument(layout_document=layout_document)
        if not field_values:
            return annotated

        # Within each reference, longer sub-field needles are matched first.
        # This prevents short needles (e.g. REFERENCE_LABEL "1") from claiming a
        # position that a longer needle (e.g. REFERENCE_YEAR "1987") should own.
        field_values = _sort_reference_sub_fields_by_length(field_values)

        token_index = _build_token_index(layout_document)
        if not token_index.haystack:
            return annotated

        last_match_end = 0
        body_floor = 0
        body_content_end = 0
        front_matter_end = 0
        keywords_floor = 0
        reference_floor = 0
        parent_match_by_field: Dict[str, Tuple[int, int, int]] = {}
        missed_by_field: Dict[str, int] = {}
        matched_count = 0
        instance_by_field: Dict[str, int] = {}
        # Per-parent masked ranges: reset each time a new main-field match is
        # established so that sub-fields of one parent don't bleed into the next.
        sub_field_masked_ranges: Dict[str, List[Tuple[int, int]]] = {}
        # Furthest end of any DOI/PMID/PMCID match for the current reference instance.
        # Used to advance the backward-search floor past identifier URLs in the tail.
        ref_id_subfield_end: Dict[str, int] = {}

        for fv in field_values:
            search_start, search_end = _search_range(
                fv, last_match_end, body_floor, body_content_end,
                front_matter_end, keywords_floor, reference_floor,
                parent_match_by_field,
            )
            masked = (
                sub_field_masked_ranges.get(fv.field_name)
                if fv.sub_field_name is not None
                else None
            )
            match_range = _fuzzy_match_field_value(
                token_index, fv, self.config,
                search_start=search_start, search_end=search_end,
                masked_ranges=masked,
            )
            # If primary match relied on a mid-token within-gap block (e.g. 't'
            # inside 'staff' matching the initial 'T' in "Guardian T"), the SW
            # found a false-positive at a later occurrence.  Try the fallback
            # (surname only) which ignores the ambiguous initial; if it lands
            # earlier, prefer it.
            if (
                match_range is not None
                and fv.sub_field_name is not None
                and fv.fallback_text
                and _has_mid_token_within_gap_blocks(match_range[2], token_index)
            ):
                _fallback_fv = JatsFieldValue(
                    text=fv.fallback_text,
                    field_name=fv.field_name,
                    sub_field_name=fv.sub_field_name,
                )
                _earlier_match = _fuzzy_match_field_value(
                    token_index, _fallback_fv, self.config,
                    search_start=search_start, search_end=search_end,
                    masked_ranges=masked,
                )
                if _earlier_match is not None and _earlier_match[0] < match_range[0]:
                    match_range = _earlier_match
                    # Extend surname-only fallback match with given-names initial
                    # (e.g. "T" from "Guardian T") found within gap of surname end.
                    assert fv.fallback_text is not None
                    match_range = _extend_match_with_given_names_tail(
                        match_range, fv.text, fv.fallback_text, token_index,
                    )
            # Front-matter region constraint is soft: if a field value (e.g. an
            # affiliation that appears near the end of the paper) is not found
            # within the preferred region, fall back to a global search.  Sub-field
            # containment (search_end set because sub_field_name is not None) is a
            # hard constraint and does not get this fallback.
            if match_range is None and search_end is not None and fv.sub_field_name is None:
                match_range = _fuzzy_match_field_value(
                    token_index, fv, self.config, search_start=0, search_end=None,
                )
            # Body-content incremental constraint is also soft: body_content_end
            # can jump forward when a nested sub-section (e.g. a mathematical
            # appendix) matches at a later PDF position than subsequent paragraphs
            # of the parent section.  Fall back to searching from body_floor
            # (end of abstract) so those paragraphs are not permanently blocked.
            if (
                match_range is None
                and fv.sub_field_name is None
                and fv.field_name in _BODY_CONTENT_FIELDS
                and search_start > body_floor
            ):
                match_range = _fuzzy_match_field_value(
                    token_index, fv, self.config,
                    search_start=body_floor, search_end=None,
                )
            # Parent REFERENCE fallback: retry with a relaxed threshold when the
            # full-text parent match just misses 0.8.  JATS may concatenate initials
            # ("CA") or order publisher/place differently from the PDF reference list,
            # reducing quality without indicating a wrong match.  Only applied to
            # parent matches (sub_field_name is None) of the REFERENCE field so that
            # sub-field containment and other fields keep the stricter threshold.
            if (
                match_range is None
                and fv.sub_field_name is None
                and fv.field_name in _REFERENCE_FIELDS
                and _REFERENCE_PARENT_MIN_THRESHOLD < self.config.threshold
            ):
                _relaxed_config = AlignmentConfig(
                    threshold=_REFERENCE_PARENT_MIN_THRESHOLD,
                    max_window=self.config.max_window,
                )
                match_range = _fuzzy_match_field_value(
                    token_index, fv, _relaxed_config,
                    search_start=search_start, search_end=search_end,
                    masked_ranges=masked,
                )
            # Sub-field fallback: retry with fallback_text (e.g. surname only)
            # when the primary JATS name text does not match the PDF text.
            if match_range is None and fv.sub_field_name is not None and fv.fallback_text:
                fallback_fv = JatsFieldValue(
                    text=fv.fallback_text,
                    field_name=fv.field_name,
                    sub_field_name=fv.sub_field_name,
                )
                match_range = _fuzzy_match_field_value(
                    token_index, fallback_fv, self.config,
                    search_start=search_start, search_end=search_end,
                    masked_ranges=masked,
                )
                if match_range is not None:
                    # Extend surname-only match with given-names initial from
                    # the original needle (e.g. "Y.-H." after "HSIEH").
                    match_range = _extend_match_with_given_names_tail(
                        match_range, fv.text, fv.fallback_text, token_index,
                    )
            if match_range is None:
                if fv.sub_field_name is None:
                    missed_by_field[fv.field_name] = (
                        missed_by_field.get(fv.field_name, 0) + 1
                    )
                continue
            matched_count += 1
            a_start, a_end, block_ranges = match_range
            last_match_end = max(last_match_end, a_end)
            if fv.field_name in _ANCHOR_FIELDS:
                body_floor = max(body_floor, a_end)
            if fv.field_name in _FRONT_MATTER_END_FIELDS:
                front_matter_end = max(front_matter_end, a_end)
            if (
                fv.field_name in _KEYWORDS_SECTION_ANCHOR_FIELDS
                or fv.field_name in _KEYWORDS_FIELDS
            ):
                keywords_floor = max(keywords_floor, a_end)
            if fv.field_name in _BODY_CONTENT_FIELDS:
                body_content_end = max(body_content_end, a_end)
            if fv.field_name in _REFERENCE_ANCHOR_FIELDS or fv.field_name in _REFERENCE_FIELDS:
                reference_floor = max(reference_floor, a_end)
            if fv.sub_field_name is None:
                prev_parent_end = parent_match_by_field.get(fv.field_name, (0, 0, 0))[1]
                prev_id_end = ref_id_subfield_end.pop(fv.field_name, 0)
                # Advance the floor past identifier URLs (DOI/PMID/PMCID) that appear
                # in the tail of the previous reference, beyond its parent match.
                # Two-column guard: ignore either value if it falls at or after the
                # current reference's parent start (handles overlapping SW matches and
                # two-column layouts where the previous URL wraps past the current ref).
                effective_prev_end = max(
                    prev_parent_end if prev_parent_end <= a_start else 0,
                    prev_id_end if prev_id_end <= a_start else 0,
                )
                parent_match_by_field[fv.field_name] = (a_start, a_end, effective_prev_end)
                sub_field_masked_ranges[fv.field_name] = []
                instance_by_field[fv.field_name] = (
                    instance_by_field.get(fv.field_name, 0) + 1
                )
            else:
                sub_field_masked_ranges.setdefault(fv.field_name, []).append(
                    (a_start, a_end)
                )
                if fv.sub_field_name in _REFERENCE_IDENTIFIER_SUB_FIELDS:
                    ref_id_subfield_end[fv.field_name] = max(
                        ref_id_subfield_end.get(fv.field_name, 0), a_end
                    )
            instance_id = instance_by_field.get(fv.field_name, 0)
            _label_tokens_for_blocks(
                annotated, token_index, block_ranges,
                fv.field_name, fv.sub_field_name, instance_id,
            )

        # Merge per-name REFERENCE_AUTHOR spans into a single <author> element.
        # Separator tokens (commas, semicolons, initials with periods) between
        # consecutively matched names remain unlabeled after per-name SW; these
        # two passes fill the gaps and attach trailing periods on abbreviations.
        _fill_sub_field_gaps(
            annotated, token_index.tokens,
            JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR,
        )
        _attach_sub_field_trailing_periods(
            annotated, token_index.tokens,
            JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR,
        )

        total = len(field_values)
        if missed_by_field:
            missed = sum(missed_by_field.values())
            LOGGER.warning(
                'Unmatched fields (%d/%d): %s',
                missed, total,
                ', '.join('%s:%d' % (k, v) for k, v in sorted(missed_by_field.items())),
            )
        else:
            LOGGER.info('Aligned all %d field values', total)

        return annotated
