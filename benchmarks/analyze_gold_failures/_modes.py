from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from ._types import FailureMode, GoldValueResult

DEFAULT_SIMILARITY_THRESHOLD = 0.8

# Map typographic characters to their plain-ASCII equivalents so that gold
# labels from JATS XML (which may use curly quotes, en-dashes, etc.) can be
# matched against raw TEI text (which often uses plain ASCII punctuation).
_UNICODE_REPLACEMENTS = str.maketrans({
    0x2018: "'",   # LEFT SINGLE QUOTATION MARK
    0x2019: "'",   # RIGHT SINGLE QUOTATION MARK
    0x201A: "'",   # SINGLE LOW-9 QUOTATION MARK
    0x02BC: "'",   # MODIFIER LETTER APOSTROPHE
    0x2032: "'",   # PRIME
    0x201C: '"',   # LEFT DOUBLE QUOTATION MARK
    0x201D: '"',   # RIGHT DOUBLE QUOTATION MARK
    0x201E: '"',   # DOUBLE LOW-9 QUOTATION MARK
    0x2033: '"',   # DOUBLE PRIME
    0x2013: '-',   # EN DASH
    0x2014: '-',   # EM DASH
    0x2010: '-',   # HYPHEN (Unicode)
    0x2011: '-',   # NON-BREAKING HYPHEN
    0x00AD: '',    # SOFT HYPHEN (remove)
    0x00A0: ' ',   # NON-BREAKING SPACE -> regular space
    0x200B: '',    # ZERO WIDTH SPACE
    0x200C: '',    # ZERO WIDTH NON-JOINER
    0x200D: '',    # ZERO WIDTH JOINER
    0xFEFF: '',    # ZERO WIDTH NO-BREAK SPACE (BOM)
})


def _normalize(text: str) -> str:
    # Decode XML/HTML entities: raw TEI is unprocessed file content so it
    # may contain &amp;, &lt;, &#x03B2; etc. Gold values from lxml are
    # already decoded, but html.unescape is idempotent on decoded text.
    text = html.unescape(text)
    # NFKC handles ligatures, superscripts, and other compatibility forms
    text = unicodedata.normalize('NFKC', text)
    # Map typographic punctuation to ASCII equivalents
    text = text.translate(_UNICODE_REPLACEMENTS)
    # Collapse all whitespace (including the now-regular space from NBSP)
    return re.sub(r'\s+', '', text)


def _normalize_for_presence(text: str) -> str:
    """Lowercase variant of _normalize used for presence detection only.

    Presence checks (is this value anywhere in the raw TEI?) should be
    case-insensitive so that ALL-CAPS PDF renderings of a title are not
    mistakenly classified as NOT_IN_RAW_TEXT.  Similarity scoring still
    uses _normalize (case-sensitive) so case differences surface in the
    PARTIAL_WRONG section and near-miss classification.
    """
    return _normalize(text).lower()


def _strip_xml_tags(text: str) -> str:
    """Replace each XML/HTML tag with a space to preserve word boundaries."""
    return re.sub(r'<[^>]+>', ' ', text)


def _in_raw_fuzzy(norm_value: str, norm_raw: str) -> bool:
    """Exact substring match, with a sliding-window fallback for single-char differences.

    Uses a fast character-mismatch counter (not SequenceMatcher) so the per-window
    cost is O(n) rather than O(n²). Allows 1 mismatch per 30 characters, minimum 1.
    Only activates for values of 8+ characters; shorter strings require exact match
    to avoid false positives.
    """
    if norm_value in norm_raw:
        return True
    n = len(norm_value)
    if n < 8:
        return False
    max_mismatches = max(1, n // 30)
    for start in range(len(norm_raw) - n + 1):
        window = norm_raw[start:start + n]
        mismatches = sum(a != b for a, b in zip(norm_value, window))
        if mismatches <= max_mismatches:
            return True
    return False


def _parse_pipe_separated(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [v.strip() for v in text.split(' | ') if v.strip()]


def _best_edit_sim(
    gold: str,
    candidates: List[str],
    case_insensitive: bool = False,
) -> Tuple[Optional[str], float]:
    if not candidates:
        return None, 0.0
    norm_gold = _normalize(gold)
    if case_insensitive:
        norm_gold = norm_gold.lower()
    best_match: Optional[str] = None
    best_sim = 0.0
    for candidate in candidates:
        norm_cand = _normalize(candidate)
        if case_insensitive:
            norm_cand = norm_cand.lower()
        sim = SequenceMatcher(None, norm_gold, norm_cand).ratio()
        if sim > best_sim:
            best_sim = sim
            best_match = candidate
    return best_match, best_sim


def assign_failure_modes(  # pylint: disable=too-many-locals
    gold_values: List[str],
    raw_tei_text: Optional[str],
    sb_field_text: Optional[str],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[GoldValueResult]:
    """Assign a FailureMode to each gold value.

    Args:
        gold_values: Expected field values from the JATS gold standard.
        raw_tei_text: Full text content of the ScienceBeam raw TEI prediction
            (used to detect values absent from the PDF text extraction).
        sb_field_text: Pipe-separated field values extracted from the ScienceBeam
            TEI via XPath (used to detect extraction failures).
        similarity_threshold: Minimum SequenceMatcher ratio to consider a value
            correctly extracted (default 0.8).
    """
    # Case-insensitive normalised strings for presence detection only.
    # ALL-CAPS PDF renderings of titles must not be classified as NOT_IN_RAW_TEXT.
    norm_raw = _normalize_for_presence(_strip_xml_tags(raw_tei_text)) if raw_tei_text else ''
    norm_sb_full = _normalize_for_presence(sb_field_text) if sb_field_text else ''
    sb_values = _parse_pipe_separated(sb_field_text)

    results = []
    for value in gold_values:
        norm_value_presence = _normalize_for_presence(value)

        in_raw = bool(norm_value_presence and _in_raw_fuzzy(norm_value_presence, norm_raw))
        in_sb_field = bool(
            norm_value_presence and _in_raw_fuzzy(norm_value_presence, norm_sb_full)
        )

        if not in_raw:
            # Even though the gold value was not found in the raw text, record whatever
            # was extracted into the target field so the report can show it.  The
            # similarity score is computed case-insensitively so that ALL-CAPS PDF
            # renderings still show high similarity rather than landing in "absent".
            nr_match, nr_sim = _best_edit_sim(value, sb_values, case_insensitive=True)
            results.append(GoldValueResult(
                value=value,
                mode=FailureMode.NOT_IN_RAW_TEXT,
                in_raw=False,
                in_sb_field=in_sb_field,
                best_sb_match=nr_match,
                best_sb_similarity=round(nr_sim, 3) if nr_match is not None else None,
            ))
        elif not in_sb_field:
            results.append(GoldValueResult(
                value=value,
                mode=FailureMode.EXTRACTION_FAILED,
                in_raw=True,
                in_sb_field=False,
            ))
        else:
            best_match, best_sim = _best_edit_sim(value, sb_values)
            mode = (
                FailureMode.CORRECT
                if best_sim >= similarity_threshold
                else FailureMode.PARTIAL_WRONG
            )
            results.append(GoldValueResult(
                value=value,
                mode=mode,
                in_raw=True,
                in_sb_field=True,
                best_sb_match=best_match,
                best_sb_similarity=round(best_sim, 3),
            ))

    return results
