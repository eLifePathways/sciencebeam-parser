import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutLine,
    LayoutPageMeta,
    LayoutToken,
)
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.field_vocab import SEGMENTATION_LABEL_BY_FIELD


LOGGER = logging.getLogger(__name__)

# Segmentation label constants (mirror SegmentationTagNames in trainer-grobid-tools)
SEG_FRONT = '<header>'
SEG_BODY = '<body>'
SEG_REFERENCES = '<references>'
SEG_ACKNOWLEDGEMENT = '<acknowledgement>'
SEG_ANNEX = '<annex>'
SEG_PAGE = '<page>'
SEG_HEADNOTE = '<headnote>'
SEG_FOOTNOTE = '<footnote>'

# Fraction of page height: lines above this → headnote, below this → footnote candidate
_HEADNOTE_Y_RATIO = 0.08
_FOOTNOTE_Y_RATIO = 0.92

# Line index threshold: front blocks starting beyond this are cleared.
# ORE papers have a second front-matter page (author roles, competing interests,
# grant info, copyright) that can start at line ~60+, so the threshold is set high
# enough to preserve those blocks when they match JATS front-matter fields.
_DEFAULT_FRONT_MAX_START_LINE_INDEX = 80
# Headnotes are expected in the first few lines (index ≤ this)
_DEFAULT_PAGE_HEADER_MAX_FIRST_LINE_INDEX = 5


@dataclass
class SegmentationConfig:
    front_max_start_line_index: int = _DEFAULT_FRONT_MAX_START_LINE_INDEX
    page_header_max_first_line_index: int = _DEFAULT_PAGE_HEADER_MAX_FIRST_LINE_INDEX
    headnote_y_ratio: float = _HEADNOTE_Y_RATIO
    footnote_y_ratio: float = _FOOTNOTE_Y_RATIO


@dataclass
class _SegLine:
    layout_line: LayoutLine
    line_index: int
    seg_label: Optional[str] = None

    @property
    def text(self) -> str:
        return self.layout_line.text

    @property
    def first_token(self) -> Optional[LayoutToken]:
        tokens = self.layout_line.tokens
        return tokens[0] if tokens else None


def _majority_vote_label(
    tokens: List[LayoutToken],
    annotated: JatsAnnotatedLayoutDocument,
) -> Optional[str]:
    field_names: List[str] = [
        label
        for t in tokens
        if (label := annotated.get_token_field(t)) is not None
    ]
    if not field_names:
        return None
    most_common_field: str = Counter(field_names).most_common(1)[0][0]
    return SEGMENTATION_LABEL_BY_FIELD.get(most_common_field)


def _is_valid_page_number_candidate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        int(stripped)
        return True
    except ValueError:
        return False


def _parse_page_number(text: str) -> Optional[int]:
    try:
        return int(text.strip())
    except ValueError:
        return None


def _is_valid_headnote_candidate(text: str, count: int, min_count: int = 2) -> bool:
    if count < min_count:
        return False
    if re.match(r'^(\d|\s|\.)+$', text):
        return False
    if len(re.split(r'\s', text.strip())) < 2:
        return False
    return True


def _get_page_meta_by_page_number(
    layout_document: LayoutDocument,
) -> Mapping[int, LayoutPageMeta]:
    return {
        page.meta.page_number: page.meta
        for page in layout_document.pages
    }


def _get_line_y_ratio(
    seg_line: _SegLine,
    page_meta_by_number: Mapping[int, LayoutPageMeta],
) -> Optional[float]:
    token = seg_line.first_token
    if token is None or token.coordinates is None or not token.coordinates:
        return None
    coords = token.coordinates
    page_meta = page_meta_by_number.get(coords.page_number)
    if page_meta is None or page_meta.coordinates is None or not page_meta.coordinates:
        return None
    page_height = page_meta.coordinates.height
    if page_height <= 0:
        return None
    return coords.y / page_height


# ── Heuristic passes ──────────────────────────────────────────────────────────

def _tag_by_coordinates(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> None:
    """Use vertical position to label headnotes and footnotes for untagged lines."""
    for seg_line in seg_lines:
        if seg_line.seg_label is not None:
            continue
        y_ratio = _get_line_y_ratio(seg_line, page_meta_by_number)
        if y_ratio is None:
            continue
        if y_ratio < config.headnote_y_ratio:
            seg_line.seg_label = SEG_HEADNOTE
        elif y_ratio > config.footnote_y_ratio:
            if _is_valid_page_number_candidate(seg_line.text):
                seg_line.seg_label = SEG_PAGE
            else:
                seg_line.seg_label = SEG_FOOTNOTE


def _tag_headnotes_by_text_repetition(
    seg_lines: List[_SegLine],
    max_first_line_index: int,
) -> None:
    """Text repetition fallback for headnote detection (no coordinates)."""
    untagged_text_counts: Counter = Counter(
        sl.text for sl in seg_lines if sl.seg_label is None
    )
    if not untagged_text_counts:
        return
    min_count: Optional[int] = None
    for text, count in untagged_text_counts.most_common():
        if not _is_valid_headnote_candidate(text, count, min_count=min_count or 2):
            continue
        first_line_index = next(
            (sl.line_index for sl in seg_lines if sl.text == text), -1
        )
        if first_line_index >= max_first_line_index:
            continue
        if min_count is None:
            min_count = max(2, count - 1)
        for sl in seg_lines:
            if sl.text == text and sl.seg_label is None:
                sl.seg_label = SEG_HEADNOTE


def _find_missing_page_numbers(seg_lines: List[_SegLine]) -> None:
    """Label standalone numeric untagged lines that fit between known page-number lines."""

    @dataclass
    class _Candidate:
        seg_line: _SegLine
        page_number: int

    existing = [
        _Candidate(sl, _parse_page_number(sl.text))  # type: ignore[arg-type]
        for sl in seg_lines
        if sl.seg_label == SEG_PAGE and _parse_page_number(sl.text) is not None
    ]
    candidates = [
        _Candidate(sl, _parse_page_number(sl.text))  # type: ignore[arg-type]
        for sl in seg_lines
        if sl.seg_label is None and _is_valid_page_number_candidate(sl.text)
    ]
    if not existing or not candidates:
        return

    min_page_number = 1
    for known in existing:
        max_line_index = known.seg_line.line_index
        max_page_number = known.page_number - 1
        for cand in candidates:
            if cand.seg_line.line_index >= max_line_index:
                continue
            if cand.page_number < min_page_number or cand.page_number > max_page_number:
                continue
            cand.seg_line.seg_label = SEG_PAGE
        min_page_number = known.page_number + 1


def _clear_front_beyond_threshold(
    seg_lines: List[_SegLine],
    max_block_start_line_index: int,
) -> None:
    if not max_block_start_line_index:
        return
    block_label: Optional[str] = None
    block_start_idx = 0
    for sl in seg_lines:
        if sl.seg_label != block_label:
            block_label = sl.seg_label
            block_start_idx = sl.line_index
        if (
            block_label == SEG_FRONT
            and block_start_idx > max_block_start_line_index
        ):
            sl.seg_label = None


def _merge_gap_lines(
    seg_lines: List[_SegLine],
    enabled_labels: Set[str],
    enabled_tail_labels: Set[str],
) -> None:
    """Assign untagged gap lines to the surrounding region."""
    _IGNORED = {SEG_HEADNOTE, SEG_PAGE}
    candidate_gap: List[_SegLine] = []
    prev_label: Optional[str] = SEG_FRONT
    for sl in seg_lines:
        if sl.seg_label in _IGNORED:
            continue
        if sl.seg_label is not None:
            if prev_label == sl.seg_label and sl.seg_label in enabled_labels:
                for gap_sl in candidate_gap:
                    gap_sl.seg_label = sl.seg_label
            candidate_gap = []
            prev_label = sl.seg_label
        elif prev_label in enabled_labels:
            candidate_gap.append(sl)
        else:
            candidate_gap = []

    if candidate_gap and prev_label in enabled_tail_labels:
        for gap_sl in candidate_gap:
            gap_sl.seg_label = prev_label


# ── Public API ────────────────────────────────────────────────────────────────

class SegmentationLabelDeriver:
    """Derives one segmentation label per LayoutLine from token-level JATS annotations."""

    def __init__(self, config: Optional[SegmentationConfig] = None) -> None:
        self.config = config or SegmentationConfig()

    def derive_labels(
        self,
        layout_document: LayoutDocument,
        annotated: JatsAnnotatedLayoutDocument,
    ) -> Dict[int, str]:
        """Return a mapping from line_id → segmentation label string.

        Uses `LayoutLineMeta.line_id` as the key so callers can look up labels
        without holding LayoutLine references.
        """
        seg_lines = [
            _SegLine(layout_line=line, line_index=idx)
            for idx, line in enumerate(layout_document.iter_all_lines())
        ]

        # ── Tier 1: majority-vote from JATS token labels ──
        for sl in seg_lines:
            label = _majority_vote_label(sl.layout_line.tokens, annotated)
            if label:
                sl.seg_label = label

        # ── Tier 2: coordinate-based margin detection ──
        page_meta_by_number = _get_page_meta_by_page_number(layout_document)
        _tag_by_coordinates(seg_lines, page_meta_by_number, self.config)

        # ── Tier 3: heuristic passes ──
        _clear_front_beyond_threshold(
            seg_lines, self.config.front_max_start_line_index
        )
        _find_missing_page_numbers(seg_lines)
        _tag_headnotes_by_text_repetition(
            seg_lines, self.config.page_header_max_first_line_index
        )
        _merge_gap_lines(
            seg_lines,
            enabled_labels={SEG_FRONT, SEG_ANNEX, SEG_REFERENCES},
            enabled_tail_labels={SEG_ANNEX},
        )

        # ── Default remaining untagged lines → body ──
        for sl in seg_lines:
            if sl.seg_label is None:
                sl.seg_label = SEG_BODY

        # Build id(LayoutLine) → label mapping — safe because layout_document holds strong refs
        result: Dict[int, str] = {}
        for sl in seg_lines:
            if sl.seg_label:
                result[id(sl.layout_line)] = sl.seg_label
        return result
