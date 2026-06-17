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


def _find_best_raw_window(norm_value: str, norm_raw: str) -> Tuple[bool, float]:
    """Return (passes_threshold, best_similarity) for norm_value against norm_raw.

    Scans every fixed-length window of norm_raw using a fast character-mismatch
    counter (O(n·m), not SequenceMatcher) to find the best-matching position, then
    computes a proper SequenceMatcher ratio on that window to get a meaningful
    similarity score that handles insertions/deletions.

    passes_threshold: True when the best window has ≤ max(1, n//30) mismatches
                      (same tolerance as the old _in_raw_fuzzy).
    best_similarity:  SequenceMatcher ratio in [0, 1] of the best window found.
                      1.0 for an exact substring match; lower when content differs.
    """
    if not norm_value or not norm_raw:
        return False, 0.0
    if norm_value in norm_raw:
        return True, 1.0
    n = len(norm_value)
    if n < 8:
        return False, 0.0
    raw_len = len(norm_raw)
    if raw_len < n:
        sim = SequenceMatcher(None, norm_value, norm_raw).ratio()
        return False, round(sim, 3)
    max_mismatches = max(1, n // 30)
    best_start = 0
    best_mismatches = n
    for start in range(raw_len - n + 1):
        mismatches = sum(a != b for a, b in zip(norm_value, norm_raw[start:start + n]))
        if mismatches < best_mismatches:
            best_mismatches = mismatches
            best_start = start
    best_window = norm_raw[best_start:best_start + n]
    sim = SequenceMatcher(None, norm_value, best_window).ratio()
    return best_mismatches <= max_mismatches, round(sim, 3)


def _in_raw_fuzzy(norm_value: str, norm_raw: str) -> bool:
    """Presence-only wrapper over _find_best_raw_window."""
    in_raw, _ = _find_best_raw_window(norm_value, norm_raw)
    return in_raw


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

        in_raw, best_raw_sim = (
            _find_best_raw_window(norm_value_presence, norm_raw)
            if norm_value_presence else (False, 0.0)
        )
        in_sb_field = bool(
            norm_value_presence and _in_raw_fuzzy(norm_value_presence, norm_sb_full)
        )

        if not in_raw:
            # Record whatever was extracted so the report can show it alongside the
            # raw-text similarity.  Both similarity scores are case-insensitive so
            # ALL-CAPS PDF renderings appear close rather than unrelated.
            nr_match, nr_sim = _best_edit_sim(value, sb_values, case_insensitive=True)
            results.append(GoldValueResult(
                value=value,
                mode=FailureMode.NOT_IN_RAW_TEXT,
                in_raw=False,
                in_sb_field=in_sb_field,
                best_sb_match=nr_match,
                best_sb_similarity=round(nr_sim, 3) if nr_match is not None else None,
                best_raw_similarity=best_raw_sim,
            ))
        elif not in_sb_field:
            results.append(GoldValueResult(
                value=value,
                mode=FailureMode.EXTRACTION_FAILED,
                in_raw=True,
                in_sb_field=False,
            ))
        else:
            # Use case-insensitive similarity so ALL-CAPS extracted values score
            # correctly rather than appearing as near-zero matches.
            best_match, best_sim = _best_edit_sim(value, sb_values, case_insensitive=True)
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
