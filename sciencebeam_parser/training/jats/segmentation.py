import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Set

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
SEG_AVAILABILITY = '<availability>'
SEG_ANNEX = '<annex>'
SEG_PAGE = '<page>'
SEG_HEADNOTE = '<headnote>'
SEG_FOOTNOTE = '<footnote>'
SEG_OTHER = '<other>'

# Fraction of page height: lines above this → headnote, below this → footnote candidate
_HEADNOTE_Y_RATIO = 0.08
_FOOTNOTE_Y_RATIO = 0.92
# A block of notes starts higher up the page than a single footer line does, so
# it needs a mark of its own.  Measured over both corpora, every run of back
# matter whose middle line falls below this is a numbered note block and none of
# the sections that follow a body reach it.
_FOOTNOTE_BLOCK_Y_RATIO = 0.65

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
    footnote_block_y_ratio: float = _FOOTNOTE_BLOCK_Y_RATIO


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


# A page number as it is printed: bare, or spelled out with a prefix and a total.
_PAGE_MARKER_PATTERN = re.compile(
    r'^(p(age|ág(ina)?)?\.?\s*)?(?P<number>\d{1,4})(\s*(of|de|/)\s*\d{1,4})?$',
    re.IGNORECASE,
)


def _parse_page_number(text: str) -> Optional[int]:
    match = _PAGE_MARKER_PATTERN.match(text.strip())
    return int(match.group('number')) if match else None


def _is_valid_page_number_candidate(text: str) -> bool:
    return _parse_page_number(text) is not None


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


_FURNITURE_LABELS = {SEG_HEADNOTE, SEG_FOOTNOTE, SEG_PAGE}


def _enclosing_label(
    seg_lines: List[_SegLine], index: int, step: int
) -> Optional[str]:
    """The label of the nearest line either side that is not page furniture."""
    position = index + step
    while 0 <= position < len(seg_lines) and seg_lines[position].seg_label in _FURNITURE_LABELS:
        position += step
    if not 0 <= position < len(seg_lines):
        return None
    return seg_lines[position].seg_label


def _release_furniture_inside_references(seg_lines: List[_SegLine]) -> None:
    """Give the reference list back a line the margin rules took from the middle of it.

    A reference list is one continuous run, so a line the coordinate rules called
    a headnote or a footnote from *inside* it is suspect: a reference whose
    journal name wraps onto the next page prints directly under the running
    header and lands in the header zone.

    Two things stay furniture, which is what makes this safe.  Text repeating
    elsewhere in the document is a running header or footer, and text that reads
    as a page marker is a page number however it is written.  Everything else
    between two reference lines is released, for the gap merge to take back into
    the region.

    Scoped to references because that is where the test is clean.  `<body>` is
    interrupted by figure captions and deposit boilerplate that are furniture on
    the page and would be released by the same rule.
    """
    # Counted over furniture only.  A running header is a line that repeats *as*
    # a header; counting every occurrence lets a common word protect itself, so
    # a reference wrapping onto "Infancia y Aprendizaje," keeps the fragments
    # that happen to appear elsewhere in the list.
    furniture_counts: Counter = Counter(
        sl.text for sl in seg_lines if sl.seg_label in (SEG_HEADNOTE, SEG_FOOTNOTE)
    )
    released = [
        seg_line
        for index, seg_line in enumerate(seg_lines)
        if seg_line.seg_label in (SEG_HEADNOTE, SEG_FOOTNOTE)
        and furniture_counts[seg_line.text] <= 1
        and not _is_valid_page_number_candidate(seg_line.text)
        and _enclosing_label(seg_lines, index, -1)
        == SEG_REFERENCES
        == _enclosing_label(seg_lines, index, 1)
    ]
    for seg_line in released:
        seg_line.seg_label = None


def _reclaim_repeated_headnotes(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> None:
    """Give a running header back to `<headnote>` where a region match took it.

    The same absorption the page-number pass undoes: a JATS match reaching across
    a page break labels the running header between its halves, although the
    field's own text does not contain it.  A line is reclaimed only when the same
    text is already headnote at least twice elsewhere in the document *and* the
    line itself sits in the header zone.  Repetition alone is not enough: a
    running header carrying the article's title repeats, and would otherwise take
    the title out of the front matter, where it prints below the header zone.
    """
    headnote_counts: Counter = Counter(
        sl.text for sl in seg_lines if sl.seg_label == SEG_HEADNOTE
    )
    for seg_line in seg_lines:
        if seg_line.seg_label in (SEG_HEADNOTE, None):
            continue
        count = headnote_counts.get(seg_line.text, 0)
        if count < 2 or not _is_valid_headnote_candidate(seg_line.text, count, min_count=2):
            continue
        if not _is_in_header_zone(seg_line, page_meta_by_number, config):
            continue
        seg_line.seg_label = SEG_HEADNOTE


def _get_page_number(seg_line: _SegLine) -> Optional[int]:
    token = seg_line.first_token
    if token is None or token.coordinates is None or not token.coordinates:
        return None
    return token.coordinates.page_number


def _is_grounded(seg_line: _SegLine, annotated: JatsAnnotatedLayoutDocument) -> bool:
    return any(
        annotated.get_token_field(token) is not None
        for token in seg_line.layout_line.tokens
    )


def _extend_region_to_page_start(
    seg_lines: List[_SegLine],
    label: str,
    annotated: JatsAnnotatedLayoutDocument,
) -> None:
    """Start a region at the top of the page its first line is on.

    A peer-review section is headed by its title, the report's date and the
    report's licence, none of which the JATS carries, so the region began
    partway down the page the section starts on.  Where nothing above it on that
    page is evidenced, the page break is the better boundary than the first line
    the aligner could ground.

    One evidenced line above stops it, because that is a page whose top still
    belongs to whatever came before.
    """
    first = next((i for i, sl in enumerate(seg_lines) if sl.seg_label == label), None)
    if first is None:
        return
    page_number = _get_page_number(seg_lines[first])
    if page_number is None:
        return
    start = first
    while start > 0 and _get_page_number(seg_lines[start - 1]) == page_number:
        start -= 1
    span = seg_lines[start:first]
    if any(_is_grounded(sl, annotated) for sl in span):
        return
    for seg_line in span:
        if seg_line.seg_label is None:
            seg_line.seg_label = label


def _iter_label_runs(
    seg_lines: List[_SegLine], label: str
) -> Iterator[List[int]]:
    """Each maximal run of `label`, by index, reading over the furniture inside it."""
    run: List[int] = []
    for index, seg_line in enumerate(seg_lines):
        if seg_line.seg_label == label:
            run.append(index)
        elif seg_line.seg_label not in _FURNITURE_LABELS:
            if run:
                yield run
            run = []
    if run:
        yield run


def _get_line_font_size(seg_line: _SegLine) -> Optional[float]:
    sizes = [
        token.font.font_size
        for token in seg_line.layout_line.tokens
        if token.font is not None and token.font.font_size
    ]
    if not sizes:
        return None
    return Counter(sizes).most_common(1)[0][0]


def _grow_run_over_matching_type(
    seg_lines: List[_SegLine],
    run: List[int],
    label: str,
) -> None:
    """Take in the lines either side that the block is evidently still setting.

    A note the aligner could not place is left unlabelled at the edge of the
    block, where the gap merge cannot reach it: it has the block on one side
    only.  The page says it belongs, because a note block is set smaller than
    the body it sits under, so the run grows over an unlabelled neighbour on the
    same page in the same size of type.
    """
    for step, edge in ((-1, run[0]), (1, run[-1])):
        # Both taken from the edge rather than from the run: a region spans the
        # pages its block is set over, and the type it is set in changes within
        # it -- a title, an abstract and a licence are one front matter and
        # three sizes.  What carries across the edge is the line beside it.
        page_number = _get_page_number(seg_lines[edge])
        edge_size = _get_line_font_size(seg_lines[edge])
        if edge_size is None:
            continue
        position = edge + step
        while 0 <= position < len(seg_lines):
            seg_line = seg_lines[position]
            if (
                seg_line.seg_label is not None
                or _get_page_number(seg_line) != page_number
                or _get_line_font_size(seg_line) != edge_size
            ):
                break
            seg_line.seg_label = label
            position += step


def _reclassify_page_foot_notes(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> None:
    """A note printed under the body of a page is a footnote, not an annex.

    An essay's numbered notes are back matter in the JATS and an appendix is too,
    so the field they come from cannot separate them; where they print can.  A
    run sitting in the lower part of its page is the block under the body, set
    once per page, rather than a section that follows the body and runs on.
    """
    for run in list(_iter_label_runs(seg_lines, SEG_ANNEX)):
        ratios = sorted(
            ratio for ratio in (
                _get_line_y_ratio(seg_lines[i], page_meta_by_number) for i in run
            ) if ratio is not None
        )
        if not ratios or ratios[len(ratios) // 2] <= config.footnote_block_y_ratio:
            continue
        for index in run:
            seg_lines[index].seg_label = SEG_FOOTNOTE
        _grow_run_over_matching_type(seg_lines, run, SEG_FOOTNOTE)


def _is_in_header_zone(
    seg_line: _SegLine,
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> bool:
    y_ratio = _get_line_y_ratio(seg_line, page_meta_by_number)
    return y_ratio is not None and y_ratio < config.headnote_y_ratio


def _is_in_footer_zone(
    seg_line: _SegLine,
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> bool:
    y_ratio = _get_line_y_ratio(seg_line, page_meta_by_number)
    return y_ratio is not None and y_ratio > config.footnote_y_ratio


def _find_missing_page_numbers(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> None:
    """Label footer lines that fit between known page-number lines.

    A JATS match can reach across a page break -- a figure caption continuing
    overleaf spans the page number between its halves -- and label the number by
    association, although the field's own text does not contain it.  Reclaiming
    it needs the number to fall between two page numbers already found and to
    print in the footer, so a numbered reference or a table row cannot be taken
    this way.
    """

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
        if _is_valid_page_number_candidate(sl.text)
        and _is_in_footer_zone(sl, page_meta_by_number, config)
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
    """Drop a block of front matter that starts too late to be front matter.

    The reference list is the boundary: nothing printed after it is the paper's
    own front matter, so a front block starting there is a field that matched
    somewhere else -- a reviewer's licence sentence carrying the article's
    copyright, say.  A line index stands in for the boundary where a document
    has no reference region to measure against.

    The index alone was cutting the front matter short.  A paper whose first
    page continues onto a second -- keywords, corresponding author, competing
    interests, grant information -- puts them past any index that a one-page
    front matter would suggest.
    """
    first_reference = next(
        (index for index, sl in enumerate(seg_lines) if sl.seg_label == SEG_REFERENCES),
        None,
    )
    if first_reference is None and not max_block_start_line_index:
        return
    block_label: Optional[str] = None
    block_start_idx = 0
    for index, sl in enumerate(seg_lines):
        if sl.seg_label != block_label:
            block_label = sl.seg_label
            block_start_idx = index
        if block_label != SEG_FRONT:
            continue
        beyond = (
            block_start_idx >= first_reference if first_reference is not None
            else block_start_idx > max_block_start_line_index
        )
        if beyond:
            sl.seg_label = None


def _merge_gap_lines(
    seg_lines: List[_SegLine],
    enabled_labels: Set[str],
    enabled_tail_labels: Set[str],
) -> None:
    """Assign untagged gap lines to the surrounding region.

    A running header or a page number interrupts a region without ending it, so
    it is stepped over rather than closing the gap.  A footer is not: it carries
    the deposit boilerplate that ends a cover page.
    """
    candidate_gap: List[_SegLine] = []
    prev_label: Optional[str] = SEG_FRONT
    for sl in seg_lines:
        if sl.seg_label in (SEG_HEADNOTE, SEG_PAGE):
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
        _find_missing_page_numbers(seg_lines, page_meta_by_number, self.config)
        _tag_headnotes_by_text_repetition(
            seg_lines, self.config.page_header_max_first_line_index
        )
        _release_furniture_inside_references(seg_lines)
        # `<other>` is peer-review sub-articles, which print as one run at the end
        # of the document.  Its values are short, repeated checklist fragments in
        # an order the page does not follow, so the aligner places only some of
        # them; bridging the gaps between those it does place, and running the
        # region to the end, recovers the rest without asking the aligner for
        # per-fragment precision it cannot give.
        _merge_gap_lines(
            seg_lines,
            enabled_labels={
                SEG_FRONT, SEG_ANNEX, SEG_AVAILABILITY, SEG_REFERENCES, SEG_OTHER
            },
            enabled_tail_labels={SEG_ANNEX, SEG_OTHER},
        )

        # Both regions run to the end of the document, so the page their first
        # evidenced line sits on is the page the region starts on.
        for tail_label in (SEG_ANNEX, SEG_OTHER):
            _extend_region_to_page_start(seg_lines, tail_label, annotated)

        _reclassify_page_foot_notes(seg_lines, page_meta_by_number, self.config)

        # The front matter ends in lines the renderer composes -- how to cite,
        # first published -- which no JATS field carries, so they sit at the end
        # of the run with the region on one side only and the gap merge cannot
        # reach them.  They are set in the same type as the block above them.
        for front_run in list(_iter_label_runs(seg_lines, SEG_FRONT)):
            _grow_run_over_matching_type(seg_lines, front_run, SEG_FRONT)

        # After the merge, not before: a line the merge uses as the anchor of a
        # region may itself be a running header, and taking it back first leaves
        # the lines after it with nothing to bridge from.
        _reclaim_repeated_headnotes(seg_lines, page_meta_by_number, self.config)

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
