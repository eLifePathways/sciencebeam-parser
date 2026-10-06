# pylint: disable=too-many-lines
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import (
    Dict, FrozenSet, Iterator, List, Mapping, Optional, Set, Tuple
)

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
SEG_COVER = '<cover>'
SEG_REVIEW = '<review>'
SEG_CONTRIBUTION = '<contribution>'
SEG_CONFLICT = '<conflict>'
SEG_FUNDING = '<funding>'

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
    block_index: int = -1
    seg_label: Optional[str] = None

    @property
    def text(self) -> str:
        return self.layout_line.text

    @property
    def first_token(self) -> Optional[LayoutToken]:
        tokens = self.layout_line.tokens
        return tokens[0] if tokens else None


def _build_seg_lines(layout_document: LayoutDocument) -> List[_SegLine]:
    """One record per line, in reading order, each knowing the block it is set in."""
    seg_lines: List[_SegLine] = []
    block_index = 0
    for page in layout_document.pages:
        for block in page.blocks:
            for line in block.lines:
                seg_lines.append(_SegLine(
                    layout_line=line, line_index=len(seg_lines), block_index=block_index,
                ))
            block_index += 1
    return seg_lines


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
    """Use vertical position to label headnotes and footnotes for untagged lines.

    The margin is read from the block, not the line alone: a column of text
    whose last line crosses into the footer zone is still text, because the rest
    of its block sits above the zone.  A running foot is a block of its own and
    lies in the zone entirely.

    At the top of a page, position alone is not enough either.  A running head
    is what every page carries -- 96% of them repeat across the document -- so a
    line that prints once, on one page, is the page's own content that happens
    to start at the top: a table caption, or the kind of article it is.  It is
    left for the regions to claim.  A page number is a page number wherever it
    prints, at the head as at the foot.
    """
    y_ratio_by_line = {
        id(seg_line): _get_line_y_ratio(seg_line, page_meta_by_number)
        for seg_line in seg_lines
    }
    pages_by_text: Dict[str, Set[int]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is not None:
            pages_by_text.setdefault(seg_line.text, set()).add(page_number)

    def is_in_margin(y_ratio: Optional[float]) -> bool:
        return y_ratio is not None and (
            y_ratio < config.headnote_y_ratio or y_ratio > config.footnote_y_ratio
        )

    block_is_margin: Dict[int, bool] = {}
    for seg_line in seg_lines:
        block_is_margin[seg_line.block_index] = (
            block_is_margin.get(seg_line.block_index, True)
            and is_in_margin(y_ratio_by_line[id(seg_line)])
        )
    for seg_line in seg_lines:
        if seg_line.seg_label is not None:
            continue
        y_ratio = y_ratio_by_line[id(seg_line)]
        if y_ratio is None or not block_is_margin.get(seg_line.block_index, True):
            continue
        if not (y_ratio < config.headnote_y_ratio or y_ratio > config.footnote_y_ratio):
            continue
        if _is_valid_page_number_candidate(seg_line.text):
            seg_line.seg_label = SEG_PAGE
        elif y_ratio > config.footnote_y_ratio:
            seg_line.seg_label = SEG_FOOTNOTE
        elif len(pages_by_text.get(seg_line.text, ())) > 1:
            seg_line.seg_label = SEG_HEADNOTE


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


_RUNNING_FOOT_MIN_Y_RATIO = 0.5
_RUNNING_FOOT_Y_TOLERANCE = 0.01


def _tag_running_feet_by_repetition(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
) -> None:
    """Take the foot a document prints on most of its pages, wherever it sits.

    The footer zone is a fixed share of the page, and a foot set a little above
    it falls outside -- taking the page number beside it with it.  Printing in
    the same place on most of the pages says the same thing the zone does, and
    says it without a threshold to fall the wrong side of.
    """
    placed = [
        (seg_line, page_number, y_ratio)
        for seg_line, page_number, y_ratio in (
            (
                seg_line,
                _get_page_number(seg_line),
                _get_line_y_ratio(seg_line, page_meta_by_number),
            )
            for seg_line in seg_lines
        )
        if page_number is not None and y_ratio is not None and seg_line.text.strip()
    ]
    rows: Dict[str, List[Tuple[_SegLine, int, float]]] = {}
    for seg_line, page_number, y_ratio in placed:
        rows.setdefault(seg_line.text, []).append((seg_line, page_number, y_ratio))
    threshold = len({page_number for _, page_number, _ in placed}) / 2
    feet: Dict[int, float] = {}
    for text, occurrences in rows.items():
        if len({page for _, page, _ in occurrences}) <= threshold:
            continue
        if not _is_valid_headnote_candidate(text, len(occurrences)):
            continue
        if any(y < _RUNNING_FOOT_MIN_Y_RATIO for _, _, y in occurrences):
            continue
        for seg_line, page_number, y_ratio in occurrences:
            if seg_line.seg_label is None:
                seg_line.seg_label = SEG_FOOTNOTE
            feet[page_number] = y_ratio
    for seg_line, page_number, y_ratio in placed:
        if seg_line.seg_label is not None or page_number not in feet:
            continue
        if abs(y_ratio - feet[page_number]) > _RUNNING_FOOT_Y_TOLERANCE:
            continue
        if _is_valid_page_number_candidate(seg_line.text):
            seg_line.seg_label = SEG_PAGE


# What a preprint server opens the page it deposits an article behind with.
_DEPOSIT_NOTICE_PATTERNS = (
    re.compile(
        r'^\s*(situa[\u00e7c][\u00e3a]o'
        r'|status'
        r'|estado\s+d(a\s+publica[\u00e7c][\u00e3a]o|e\s+la\s+publicaci[\u00f3o]n))\s*:',
        re.IGNORECASE,
    ),
    re.compile(
        r'sob\s+as\s+seguintes\s+condi[\u00e7c][\u00f5o]es'
        r'|bajo\s+las\s+siguientes\s+condiciones'
        r'|under\s+the\s+following\s+conditions',
        re.IGNORECASE,
    ),
)


def _is_deposit_notice(text: str) -> bool:
    return any(pattern.search(text) for pattern in _DEPOSIT_NOTICE_PATTERNS)


def _tag_cover_pages(seg_lines: List[_SegLine]) -> None:
    """Take the whole of a page the publisher wraps the article in.

    A cover page is the publisher's rather than the author's: it restates the
    title, the authors and the identifier, and adds the deposit conditions and
    the licence.  GROBID asks for the whole of such a page under one label and
    reads nothing back out of it, which is the point -- the deposit conditions
    are not the paper's header, and a header model trained on them learns that
    a page of licence terms is bibliographic data.

    The page is recognised by the notice the server opens it with, and only
    where nothing on it has already been read as part of the article.  It is
    not always the first page: one preprint carries two, and others print the
    conditions at the end instead.
    """
    pages: Dict[int, List[_SegLine]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is not None:
            pages.setdefault(page_number, []).append(seg_line)
    for page_lines in pages.values():
        first = next((sl for sl in page_lines if sl.text.strip()), None)
        if first is None or not _is_deposit_notice(first.text):
            continue
        if any(
            sl.seg_label not in (None, SEG_FRONT) and sl.seg_label not in _FURNITURE_LABELS
            for sl in page_lines
        ):
            continue
        for seg_line in page_lines:
            seg_line.seg_label = SEG_COVER


_APPENDIX_HEADING_PATTERN = re.compile(
    r'^\s*(ap[\u00eae]ndices?|appendix|appendices|anexos?|annexe?s?)\b',
    re.IGNORECASE,
)

_APPENDIX_HEADING_MAX_LENGTH = 80


def _claim_appendix_region(seg_lines: List[_SegLine]) -> None:
    """Open an annex where the page prints an appendix heading.

    A paper can print an appendix the JATS does not carry -- the section is
    declared as supplementary material and its text lives only in the PDF -- so
    nothing grounds it and it falls to the body.  GROBID's own corpus labels a
    printed appendix heading `<annex>` 69 times, before the reference list as
    often as after, so where it prints does not decide it.

    The heading has to be unclaimed, which is what keeps a sentence that merely
    mentions an appendix out: that sentence belongs to a paragraph the JATS
    carries, so it is already labelled.  The annex then runs on over what
    nothing else claims, and stops at whatever does.
    """
    index = 0
    while index < len(seg_lines):
        seg_line = seg_lines[index]
        if (
            seg_line.seg_label is not None
            or len(seg_line.text) > _APPENDIX_HEADING_MAX_LENGTH
            or not _APPENDIX_HEADING_PATTERN.match(seg_line.text)
        ):
            index += 1
            continue
        for following in seg_lines[index:]:
            if following.seg_label in _FURNITURE_LABELS:
                continue
            if following.seg_label is not None:
                break
            following.seg_label = SEG_ANNEX
        index += 1


_PUBLICATION_DATE_PATTERN = re.compile(
    r'^\s*(recebido|aprovado|aceito|submetido|revisado|enviado'
    r'|recibido|aceptado|presentado'
    r'|received|accepted|submitted|posted|revised|published)\b'
    r'[^0-9]{0,30}[0-9][0-9/.\u2013-]*\s*(\([^)]*\))?\s*$',
    re.IGNORECASE,
)


def _tag_publication_dates(seg_lines: List[_SegLine]) -> None:
    """Take a line that is nothing but a date the article was received or posted.

    These print wherever the publisher puts them -- a title page, a deposit
    page, or a block of their own after the reference list -- and no JATS field
    carries the ones printed at the end, so they were falling to the body.  They
    are part of the bibliographic record, which GROBID allows to be several
    areas in several places.  The whole line has to be the date: a sentence that
    mentions one is prose.
    """
    for seg_line in seg_lines:
        if seg_line.seg_label is None and _PUBLICATION_DATE_PATTERN.match(seg_line.text):
            seg_line.seg_label = SEG_FRONT


_SAME_ROW_Y_TOLERANCE = 3.0

# A column gutter runs to about two fifths of the page; a badge sits a fraction
# of that from the text it belongs to.  Anything further across the row is in
# the next column and says nothing about this line.
_SAME_ROW_MAX_X_GAP_RATIO = 0.15


def _get_line_position(seg_line: _SegLine) -> Optional[Tuple[float, float]]:
    token = seg_line.first_token
    if token is None or token.coordinates is None or not token.coordinates:
        return None
    return token.coordinates.x, token.coordinates.y


def _claim_lines_sharing_a_row(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
) -> None:
    """Give a line to the region it is printed beside.

    A revised article carries a "REVISED" badge in the corner of the box that
    says what changed.  The badge is a block of its own, so the reading order
    puts it at the foot of the page, far from the box -- but it prints on the
    same row as that box's heading, which says where it belongs whatever order
    it is read in.

    Beside, not merely level with: the other column of a two-column page is on
    the same rows throughout and belongs to whatever it belongs to.
    """
    rows: Dict[int, List[_SegLine]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is not None and _get_line_position(seg_line) is not None:
            rows.setdefault(page_number, []).append(seg_line)
    for page_number, page_lines in rows.items():
        page_meta = page_meta_by_number.get(page_number)
        if page_meta is None or page_meta.coordinates is None or not page_meta.coordinates:
            continue
        max_x_gap = page_meta.coordinates.width * _SAME_ROW_MAX_X_GAP_RATIO
        for seg_line in page_lines:
            if seg_line.seg_label is not None:
                continue
            x, y = _get_line_position(seg_line)  # type: ignore[misc]
            beside = sorted(
                (
                    (abs(position[0] - x), other.seg_label)
                    for other, position in (
                        (o, _get_line_position(o)) for o in page_lines
                    )
                    if other.seg_label is not None
                    and other.seg_label not in _FURNITURE_LABELS
                    and position is not None
                    and abs(position[1] - y) <= _SAME_ROW_Y_TOLERANCE
                ),
                key=lambda entry: entry[0],
            )
            if beside and beside[0][0] <= max_x_gap:
                seg_line.seg_label = beside[0][1]


_FOOTNOTE_MARKER_MAX_LINES = 2

_FURNITURE_LABELS = {SEG_HEADNOTE, SEG_FOOTNOTE, SEG_PAGE}


def _enclosing_label(
    seg_lines: List[_SegLine], index: int, step: int
) -> Optional[str]:
    """The region the nearest labelled line either side belongs to.

    Page furniture is read over, and so is a line nothing has claimed yet: an
    unlabelled line says no more about which region encloses this one than a
    running header does.
    """
    position = index + step
    while (
        0 <= position < len(seg_lines)
        and (seg_lines[position].seg_label is None
             or seg_lines[position].seg_label in _FURNITURE_LABELS)
    ):
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

    A cover page keeps everything it has: GROBID asks for the whole of one under
    the one label, and the banner across the top of it is the publisher's too.
    """
    headnote_counts: Counter = Counter(
        sl.text for sl in seg_lines if sl.seg_label == SEG_HEADNOTE
    )
    for seg_line in seg_lines:
        if seg_line.seg_label in (SEG_HEADNOTE, SEG_COVER, None):
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


_REFERENCE_HEADINGS = frozenset({
    'reference', 'references', 'reference list', 'references cited',
    'literature cited', 'works cited',
    'bibliography', 'bibliographie', 'bibliografia', 'bibliografía',
    'referencia', 'referencias', 'referências', 'references bibliographiques',
    'referencias bibliograficas', 'referencias bibliográficas',
    'referencia bibliograficas', 'referências bibliográficas',
    'referencias bibliografias', 'bibliografia referencias',
    'literatur', 'literaturverzeichnis',
})

_HEADING_NOISE_PATTERN = re.compile(r'[^a-z\u00c0-\u024f ]+')


def _normalised_heading(text: str) -> str:
    return ' '.join(_HEADING_NOISE_PATTERN.sub(' ', text.lower()).split())


def _claim_region_heading(
    seg_lines: List[_SegLine],
    label: str,
    headings: FrozenSet[str],
    annotated: JatsAnnotatedLayoutDocument,
) -> None:
    """Take the printed heading a region is introduced by.

    Most JATS gives `<ref-list>` no `<title>`, so the "References" the page
    prints above the list has nothing to match and the list starts one line too
    late.  The heading is only claimed where it sits directly above the region
    and nothing else has a claim on it.
    """
    for run in _iter_label_runs(seg_lines, label):
        index = run[0] - 1
        if index < 0:
            continue
        seg_line = seg_lines[index]
        if seg_line.seg_label is not None or _is_grounded(seg_line, annotated):
            continue
        if _normalised_heading(seg_line.text) in headings:
            seg_line.seg_label = label


def _get_running_texts(seg_lines: List[_SegLine]) -> Set[str]:
    """The lines printed on most of the pages: running heads and feet.

    Most of the pages, not merely more than one: a reference list repeats
    "Publisher Full Text" under every entry that has one, and that is part of
    the list rather than furniture around it.
    """
    pages_by_text: Dict[str, Set[int]] = {}
    page_numbers: Set[int] = set()
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is None or not seg_line.text.strip():
            continue
        page_numbers.add(page_number)
        pages_by_text.setdefault(seg_line.text, set()).add(page_number)
    threshold = len(page_numbers) / 2
    return {text for text, pages in pages_by_text.items() if len(pages) > threshold}


def _extend_region_to_page_end(
    seg_lines: List[_SegLine],
    label: str,
    annotated: JatsAnnotatedLayoutDocument,
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> None:
    """Run each region on to the end of the page over what nothing else claims.

    A reference list prints link labels -- "Publisher Full Text" and the rest --
    that the JATS does not carry, so the last one on a page is left over with the
    region on one side only and the gap merge cannot reach it.  A first page ends
    the same way, with which gateway or collection the article belongs to.

    Bounded to the page the run ends on and stopped by anything evidenced, so it
    only takes what is already inside the run's own page.  A running foot is
    stepped over rather than taken, the same way the gap merge steps over a
    running head.  Per run, because an article that records what changed since
    its previous version carries front matter on a later page too.
    """
    running = _get_running_texts(seg_lines)
    for run in _iter_label_runs(seg_lines, label):
        page_number = _get_page_number(seg_lines[run[-1]])
        for seg_line in seg_lines[run[-1] + 1:]:
            if _get_page_number(seg_line) != page_number:
                break
            if seg_line.seg_label in _FURNITURE_LABELS:
                continue
            if seg_line.seg_label is not None or _is_grounded(seg_line, annotated):
                break
            if _is_in_footer_zone(seg_line, page_meta_by_number, config):
                break
            if seg_line.text in running or _is_valid_page_number_candidate(seg_line.text):
                continue
            seg_line.seg_label = label


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


def _bridge_footnote_block(seg_lines: List[_SegLine]) -> None:
    """Keep a note's marker with the note.

    A numbered footnote prints its number on a line of its own when the note
    wraps, and the JATS runs number and text together, so the number matches
    nothing.  The gap merge cannot reach it: it treats a footnote as furniture
    to step over rather than as a region to bridge.

    A marker is a line or two, so only a gap that short is bridged.  A page can
    carry notes at its head and foot with an appendix between them, and that is
    not one note.
    """
    index = 0
    while index < len(seg_lines):
        if seg_lines[index].seg_label is not None:
            index += 1
            continue
        end = index
        while end < len(seg_lines) and seg_lines[end].seg_label is None:
            end += 1
        gap = seg_lines[index:end]
        sides = [seg_lines[index - 1] if index else None,
                 seg_lines[end] if end < len(seg_lines) else None]
        enclosed = all(sl is not None and sl.seg_label == SEG_FOOTNOTE for sl in sides)
        pages = {_get_page_number(sl) for sl in gap + sides if sl is not None}
        if enclosed and len(gap) <= _FOOTNOTE_MARKER_MAX_LINES and len(pages) == 1:
            for seg_line in gap:
                seg_line.seg_label = SEG_FOOTNOTE
        index = end + 1


def _merge_gap_lines(
    seg_lines: List[_SegLine],
    enabled_labels: Set[str],
    enabled_tail_labels: Set[str],
) -> None:
    """Assign untagged gap lines to the surrounding region.

    Page furniture interrupts a region without ending it, so it is stepped over
    rather than closing the gap.  That includes the running foot: a reference
    broken across a page break resumes under one, and the deposit boilerplate
    that a footer used to stand in for is now a cover page in its own right.
    """
    candidate_gap: List[_SegLine] = []
    prev_label: Optional[str] = SEG_FRONT
    for sl in seg_lines:
        if sl.seg_label in _FURNITURE_LABELS:
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
        seg_lines = _build_seg_lines(layout_document)

        # ── Tier 1: majority-vote from JATS token labels ──
        for sl in seg_lines:
            label = _majority_vote_label(sl.layout_line.tokens, annotated)
            if label:
                sl.seg_label = label

        # ── Tier 2: coordinate-based margin detection ──
        page_meta_by_number = _get_page_meta_by_page_number(layout_document)
        _tag_by_coordinates(seg_lines, page_meta_by_number, self.config)
        _tag_cover_pages(seg_lines)

        # ── Tier 3: heuristic passes ──
        _clear_front_beyond_threshold(
            seg_lines, self.config.front_max_start_line_index
        )
        _find_missing_page_numbers(seg_lines, page_meta_by_number, self.config)
        _tag_headnotes_by_text_repetition(
            seg_lines, self.config.page_header_max_first_line_index
        )
        _tag_running_feet_by_repetition(seg_lines, page_meta_by_number)
        _release_furniture_inside_references(seg_lines)
        _claim_region_heading(
            seg_lines, SEG_REFERENCES, _REFERENCE_HEADINGS, annotated
        )
        # `<review>` is the peer-review sub-articles, which print as one run at the end
        # of the document.  Its values are short, repeated checklist fragments in
        # an order the page does not follow, so the aligner places only some of
        # them; bridging the gaps between those it does place, and running the
        # region to the end, recovers the rest without asking the aligner for
        # per-fragment precision it cannot give.
        _bridge_footnote_block(seg_lines)
        _merge_gap_lines(
            seg_lines,
            enabled_labels={
                SEG_FRONT, SEG_ANNEX, SEG_AVAILABILITY, SEG_REFERENCES, SEG_REVIEW
            },
            enabled_tail_labels={SEG_ANNEX, SEG_REVIEW},
        )

        # Both regions run to the end of the document, so the page their first
        # evidenced line sits on is the page the region starts on.
        for tail_label in (SEG_ANNEX, SEG_REVIEW):
            _extend_region_to_page_start(seg_lines, tail_label, annotated)

        # With the cover page taken, the article's own title block is the first
        # thing above the front matter that nothing else claims: the title and
        # authors the cover restated, which the aligner matched on the cover and
        # so left unevidenced here.
        _extend_region_to_page_start(seg_lines, SEG_FRONT, annotated)

        _extend_region_to_page_end(
            seg_lines, SEG_REFERENCES, annotated, page_meta_by_number, self.config
        )

        _reclassify_page_foot_notes(seg_lines, page_meta_by_number, self.config)

        # The front matter ends in lines the renderer composes -- how to cite,
        # first published -- which no JATS field carries, so they sit at the end
        # of the run with the region on one side only and the gap merge cannot
        # reach them.  They are set in the same type as the block above them.
        for front_run in list(_iter_label_runs(seg_lines, SEG_FRONT)):
            _grow_run_over_matching_type(seg_lines, front_run, SEG_FRONT)
        _extend_region_to_page_end(
            seg_lines, SEG_FRONT, annotated, page_meta_by_number, self.config
        )

        # After the merge, not before: a line the merge uses as the anchor of a
        # region may itself be a running header, and taking it back first leaves
        # the lines after it with nothing to bridge from.
        _reclaim_repeated_headnotes(seg_lines, page_meta_by_number, self.config)

        _claim_appendix_region(seg_lines)
        _tag_publication_dates(seg_lines)
        _claim_lines_sharing_a_row(seg_lines, page_meta_by_number)

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
