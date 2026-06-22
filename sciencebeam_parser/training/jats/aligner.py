import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from sciencebeam_alignment.align import LocalSequenceMatcher, SimpleScoring

from sciencebeam_parser.document.layout_document import LayoutDocument, LayoutToken
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.field_extractor import JatsFieldValue
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
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
_SUB_FIELD_PARENT_PRE_BUFFER = 0

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

# Type alias for the return value of _fuzzy_match_field_value:
#   (abs_start, abs_end, [(block_start, block_end), ...])
_MatchResult = Tuple[int, int, List[Tuple[int, int]]]


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


def _fuzzy_search_in_window(
    haystack: str,
    needle: str,
    window_start: int,
    window_end: int,
    threshold: float,
) -> Optional[_MatchResult]:
    """Try to find `needle` in haystack[window_start:window_end].

    Returns (abs_start, abs_end, [(block_start, block_end), ...]) if quality >=
    threshold, else None.  Block ranges are in absolute haystack coordinates.
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
    return a_start, a_end, abs_block_ranges


def _fuzzy_match_field_value(
    token_index: _TokenIndex,
    field_value: JatsFieldValue,
    config: AlignmentConfig,
    search_start: int,
    search_end: Optional[int] = None,
) -> Optional[_MatchResult]:
    needle = normalize_for_alignment(field_value.text)
    if not needle:
        return None

    haystack = token_index.haystack
    hay_end = len(haystack) if search_end is None else min(search_end, len(haystack))

    need_len = len(needle)

    window_size = max(
        _DEFAULT_MIN_WINDOW,
        min(config.max_window, need_len * _WINDOW_NEEDLE_MULTIPLIER),
    )
    stride = max(1, window_size - need_len - 20)

    start = search_start
    while start < hay_end:
        end = min(start + window_size, hay_end)
        result = _fuzzy_search_in_window(haystack, needle, start, end, config.threshold)
        if result is not None:
            return result
        if end >= hay_end:
            break
        start += stride

    return None


def _search_range(
    fv: JatsFieldValue,
    last_match_end: int,
    body_floor: int,
    body_content_end: int,
    front_matter_end: int,
    keywords_floor: int,
    reference_floor: int,
    parent_match_by_field: Dict[str, Tuple[int, int]],
) -> Tuple[int, Optional[int]]:
    """Return (search_start, search_end) for fv given current position state."""
    if fv.sub_field_name is not None and fv.field_name in parent_match_by_field:
        p_start, p_end = parent_match_by_field[fv.field_name]
        return p_start, p_end + _SUB_FIELD_PARENT_BUFFER
    if fv.field_name in _BODY_CONTENT_FIELDS:
        return max(0, max(body_floor, body_content_end) - 200), None
    if fv.field_name in _REFERENCE_ANCHOR_FIELDS or fv.field_name in _REFERENCE_FIELDS:
        # References use a dedicated floor that is independent of body_content_end.
        # This prevents appendix or late body content from advancing body_content_end
        # past the reference section, which would make the reference search start
        # skip over all reference positions.
        ref_start = max(0, reference_floor - 200) if reference_floor > 0 else max(0, body_floor)
        return ref_start, None
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


def _label_tokens_for_blocks(
    annotated: JatsAnnotatedLayoutDocument,
    token_index: _TokenIndex,
    block_ranges: List[Tuple[int, int]],
    field_name: str,
    sub_field_name: Optional[str],
    instance_id: int,
) -> None:
    """Label tokens using anchor+chain strategy (see module constants for rationale)."""
    has_anchor = any(be - bs >= _MIN_ANCHOR_BLOCK_SIZE for bs, be in block_ranges)
    prev_included_end: Optional[int] = None
    for block_start, block_end in block_ranges:
        is_anchor = (block_end - block_start) >= _MIN_ANCHOR_BLOCK_SIZE
        within_gap = (
            prev_included_end is not None
            and block_start - prev_included_end <= _MAX_HAYSTACK_GAP_TO_FILL
        )
        if has_anchor and not is_anchor and not within_gap:
            continue
        fill_start = block_start
        if within_gap:
            assert prev_included_end is not None
            fill_start = prev_included_end
        for token in token_index.tokens_in_range(fill_start, block_end):
            annotated.set_token_label(token, field_name, sub_field_name, instance_id)
        prev_included_end = block_end


class LayoutDocumentJatsAligner:
    """Aligns JATS field values to LayoutDocument tokens via fuzzy text matching."""

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()

    def align(  # pylint: disable=too-many-locals,too-many-branches
        self,
        layout_document: LayoutDocument,
        field_values: List[JatsFieldValue],
    ) -> JatsAnnotatedLayoutDocument:
        annotated = JatsAnnotatedLayoutDocument(layout_document=layout_document)
        if not field_values:
            return annotated

        token_index = _build_token_index(layout_document)
        if not token_index.haystack:
            return annotated

        last_match_end = 0
        body_floor = 0
        body_content_end = 0
        front_matter_end = 0
        keywords_floor = 0
        reference_floor = 0
        parent_match_by_field: Dict[str, Tuple[int, int]] = {}
        missed_by_field: Dict[str, int] = {}
        matched_count = 0
        instance_by_field: Dict[str, int] = {}

        for fv in field_values:
            search_start, search_end = _search_range(
                fv, last_match_end, body_floor, body_content_end,
                front_matter_end, keywords_floor, reference_floor,
                parent_match_by_field,
            )
            match_range = _fuzzy_match_field_value(
                token_index, fv, self.config,
                search_start=search_start, search_end=search_end,
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
                parent_match_by_field[fv.field_name] = (a_start, a_end)
                instance_by_field[fv.field_name] = (
                    instance_by_field.get(fv.field_name, 0) + 1
                )
            instance_id = instance_by_field.get(fv.field_name, 0)
            _label_tokens_for_blocks(
                annotated, token_index, block_ranges,
                fv.field_name, fv.sub_field_name, instance_id,
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
