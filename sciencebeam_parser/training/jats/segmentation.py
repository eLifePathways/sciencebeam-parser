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
from sciencebeam_parser.training.jats.field_vocab import (
    SEGMENTATION_LABEL_BY_FIELD,
    JatsFieldNames,
)


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
# A page marker is sometimes set off from what precedes it in the running foot
# ("Volpatto et al. | 3"), and sometimes gives the article's extent rather than
# the page ("1-13").
_PAGE_MARKER_PATTERN = re.compile(
    r'^[|\u2013\u2014\u00b7-]?\s*'
    r'(p(age|\u00e1g(ina)?)?\.?\s*)?'
    r'(?P<number>\d{1,4})'
    r'(\s*(of|de|/|[\u2013\u2014-])\s*\d{1,4})?$',
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

# What a title page names its corresponding author with, set as a note at the
# foot of the page.  GROBID's corpus labels such a line `<front>` 339 times and
# `<footnote>` never: the note is the front matter's, wherever it is printed.
_CORRESPONDING_AUTHOR_PATTERN = re.compile(
    r'(corresponding\s+author'
    r'|autor\s+(correspondente|de\s+correspond\w+)'
    r'|correspond[e\u00ea]ncia\s*:'
    r'|autor\s+para\s+correspond\w+)',
    re.IGNORECASE,
)


def _get_corresponding_author_lines(
    seg_lines: List[_SegLine],
    y_ratio_by_line: Mapping[int, Optional[float]],
    config: SegmentationConfig,
) -> Set[int]:
    """The note at the foot of the title page naming the corresponding author.

    Front matter, and so is what it carries on into: the addresses and the
    e-mails under it are set as blocks of their own.  It stops at a line
    opening with a digit, which is where a numbered footnote starts instead.
    """
    lines: Set[int] = set()
    in_note = False
    for seg_line in seg_lines:
        if _get_page_number(seg_line) != 1:
            continue
        y_ratio = y_ratio_by_line[id(seg_line)]
        if y_ratio is None or y_ratio <= config.footnote_y_ratio:
            continue
        if _CORRESPONDING_AUTHOR_PATTERN.search(seg_line.text):
            in_note = True
        elif in_note and re.match(r'^\s*\d', seg_line.text):
            in_note = False
        if in_note:
            lines.add(id(seg_line))
    return lines


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
    corresponding_author_lines = _get_corresponding_author_lines(
        seg_lines, y_ratio_by_line, config
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
            seg_line.seg_label = (
                SEG_FRONT if id(seg_line) in corresponding_author_lines
                else SEG_FOOTNOTE
            )
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


_PAGE_NUMBER_POSITION_PRECISION = 2


def _tag_page_numbers_by_position(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
) -> None:
    """Take the number a document sets in the same corner of most of its pages.

    A page number differs on every page, so text repetition cannot find it, and
    it is not always far enough into the margin for position alone to say.  What
    repeats is where it is set: the same spot, page after page, in a line of its
    own that reads as a number.  Two pages at least, since one page cannot show
    a position repeating, and a number in the middle of a table is not one.

    The spot carries over a region the JATS vote has already named: a number the
    page prints in its own corner is furniture wherever the article's text
    happens to have matched it.  A cover page is left alone, which keeps a whole
    page to one label.
    """
    by_spot: Dict[Tuple[float, float], List[_SegLine]] = {}
    page_numbers: Set[int] = set()
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        position = _get_line_position(seg_line)
        y_ratio = _get_line_y_ratio(seg_line, page_meta_by_number)
        if page_number is None or position is None or y_ratio is None:
            continue
        page_numbers.add(page_number)
        if not _is_valid_page_number_candidate(seg_line.text):
            continue
        spot = (
            round(position[0], -_PAGE_NUMBER_POSITION_PRECISION),
            round(y_ratio, _PAGE_NUMBER_POSITION_PRECISION),
        )
        by_spot.setdefault(spot, []).append(seg_line)
    threshold = max(len(page_numbers) / 2, 1)
    for spot_lines in by_spot.values():
        if len({_get_page_number(sl) for sl in spot_lines}) <= threshold:
            continue
        for seg_line in spot_lines:
            if seg_line.seg_label is None or seg_line.seg_label in _ARTICLE_CONTENT_LABELS:
                seg_line.seg_label = SEG_PAGE


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
    # The same line, shortened: "Estado: El preprint no ha sido enviado ...".
    re.compile(r'^\s*estado\s*:.*\bpreprint\b', re.IGNORECASE),
    re.compile(
        r'sob\s+as\s+seguintes\s+condi[\u00e7c][\u00f5o]es'
        r'|bajo\s+las\s+siguientes\s+condiciones'
        r'|under\s+the\s+following\s+conditions',
        re.IGNORECASE,
    ),
)


def _is_deposit_notice(text: str) -> bool:
    return any(pattern.search(text) for pattern in _DEPOSIT_NOTICE_PATTERNS)


# The status line with its value set somewhere the line does not reach, which
# leaves the label on its own.
_DEPOSIT_LABEL_PATTERN = re.compile(
    r'^\s*(situa[\u00e7c][\u00e3a]o|status|estado)\s*:\s*$', re.IGNORECASE
)

# What only the deposit form prints: the legend telling the depositor how to
# write a date, and the server's own identifier for the deposit.
_DEPOSIT_MARKER_PATTERNS = (
    re.compile(r'\(\s*(AAAA|YYYY)\s*[-/]\s*MM\s*[-/]\s*(DD|JJ)\s*\)', re.IGNORECASE),
    re.compile(r'doi\.org/10\.1590/SciELOPreprints', re.IGNORECASE),
)


# The sheet a submission system wraps a manuscript in for review.  It restates
# the title and carries the system's own identifier for the manuscript, which
# is the publisher's page rather than the article's.
_SUBMISSION_COVER_PATTERNS = (
    re.compile(r'manuscriptcentral\.com', re.IGNORECASE),
    re.compile(r'^\s*manuscript\s+id\b', re.IGNORECASE),
)


def _is_submission_cover_marker(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SUBMISSION_COVER_PATTERNS)


def _is_deposit_marker(text: str) -> bool:
    return any(pattern.search(text) for pattern in _DEPOSIT_MARKER_PATTERNS)


def _tag_cover_pages(seg_lines: List[_SegLine]) -> None:
    """Take the whole of a page the publisher wraps the article in.

    A cover page is the publisher's rather than the author's: it restates the
    title, the authors and the identifier, and adds the deposit conditions and
    the licence.  GROBID asks for the whole of such a page under one label and
    reads nothing back out of it, which is the point -- the deposit conditions
    are not the paper's header, and a header model trained on them learns that
    a page of licence terms is bibliographic data.

    The page is recognised by the notice the server opens it with, and only
    where nothing on it has already been read as part of the article's own
    narrative.  A back-matter statement matching here does not count: the
    conditions restate what the article declares, so a conflict-of-interest
    field matches the publisher's wording and would otherwise keep the page
    that wording belongs to from being recognised.  It is not always the first
    page: one preprint carries two, and others print the conditions at the end
    instead.
    """
    pages: Dict[int, List[_SegLine]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is not None:
            pages.setdefault(page_number, []).append(seg_line)
    cover_pages: Set[int] = set()
    for page_number in sorted(pages):
        page_lines = pages[page_number]
        first = next((sl for sl in page_lines if sl.text.strip()), None)
        if first is None:
            continue
        if any(sl.seg_label in _ARTICLE_CONTENT_LABELS for sl in page_lines):
            continue
        opens_a_deposit_page = _is_deposit_notice(first.text) or (
            _DEPOSIT_LABEL_PATTERN.match(first.text)
            and any(_is_deposit_marker(sl.text) for sl in page_lines)
        )
        # A submission sheet opens with the watermark printed across it, set a
        # letter to a line, so what names it is not the line the page starts
        # with.
        opens_a_deposit_page = opens_a_deposit_page or any(
            _is_submission_cover_marker(sl.text) for sl in page_lines
        )
        # The form can run past the page it starts on, leaving the dates and the
        # generator's footer by themselves overleaf.  That is the rest of the
        # cover rather than the article starting.
        continues_one = page_number - 1 in cover_pages and all(
            sl.seg_label in _FURNITURE_LABELS or _is_deposit_marker(sl.text)
            for sl in page_lines if sl.text.strip()
        )
        if not opens_a_deposit_page and not continues_one:
            continue
        cover_pages.add(page_number)
        for seg_line in page_lines:
            seg_line.seg_label = SEG_COVER


_AMENDMENTS_OPENING_PATTERN = re.compile(r'^\s*Amendments\s+from\s+Version\b', re.IGNORECASE)
_AMENDMENTS_CLOSING_PATTERN = re.compile(
    r'^\s*Any\s+further\s+responses\s+from\s+the\s+reviewers', re.IGNORECASE
)
_AMENDMENTS_MAX_LINES = 60


def _claim_amendments_box(seg_lines: List[_SegLine]) -> None:
    """Take the box a revised article says what changed in.

    The box is the publisher's, and it opens and closes with the publisher's
    own words, the same in every revised article.  The JATS carries it in
    `<front>/<notes>`, but not always the version the PDF prints -- one article
    describes one set of changes in its notes and another on its page -- and
    then nothing grounds it and it reads as the article's argument.

    Both ends have to be found: the opening line alone would run into whatever
    follows it on a page that does not close the box.  Between them the box
    takes a line the body matched as well as one nothing claimed, because the
    box paraphrases the article and a sentence of it can match the article's
    own -- and a body paragraph cannot be inside the box.
    """
    for index, seg_line in enumerate(seg_lines):
        if not _AMENDMENTS_OPENING_PATTERN.match(seg_line.text):
            continue
        end = next(
            (
                position
                for position in range(index, min(len(seg_lines), index + _AMENDMENTS_MAX_LINES))
                if _AMENDMENTS_CLOSING_PATTERN.match(seg_lines[position].text)
            ),
            None,
        )
        if end is None:
            continue
        for seg_line_in_box in seg_lines[index:end + 2]:
            if seg_line_in_box.seg_label in (None, SEG_BODY):
                seg_line_in_box.seg_label = SEG_FRONT


_APPENDIX_HEADING_PATTERN = re.compile(
    r'^\s*('
    r'ap[e\u00ea\u00e9]ndices?|appendix|appendices|anexos?|annexe?s?'
    # What GROBID's corpus most often opens an annex with: "Supporting
    # information" 73 times against "Appendix" 6.
    r'|supporting\s+information'
    r'|supplementary\s+(material|information|data|file)'
    r'|materi(al|ais|ales)\s+suplementar\w*'
    r')\b',
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

# Wide enough for an indent, far short of a column gutter.
_SAME_COLUMN_MAX_X_GAP = 90.0


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

# The article's own narrative, as against anything a cover page restates.
_ARTICLE_CONTENT_LABELS = {
    SEG_BODY, SEG_REFERENCES, SEG_REVIEW, SEG_ANNEX, SEG_ACKNOWLEDGEMENT,
}


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


_ENCLOSED_RUN_MAX_LINES = 10

_MAROONED_RUN_MAX_LINES = 4


def _release_references_inside_availability(seg_lines: List[_SegLine]) -> None:
    """Give the availability section back the datasets it cites.

    A data-availability statement names the datasets it describes, and the JATS
    lists those among the references, so the reference matcher claims them.  On
    the page they are set inside the availability block, between the heading
    and the description of what the dataset contains, and that is the region
    they belong to.
    """
    for run in _iter_label_runs(seg_lines, SEG_REFERENCES):
        if len(run) > _ENCLOSED_RUN_MAX_LINES:
            continue
        if (
            _enclosing_label(seg_lines, run[0], -1)
            == SEG_AVAILABILITY
            == _enclosing_label(seg_lines, run[-1], 1)
        ):
            for index in run:
                seg_lines[index].seg_label = SEG_AVAILABILITY


def _release_references_inside_body(seg_lines: List[_SegLine]) -> None:
    """Take back a reference that matched a citation in the argument.

    A bibliography entry and the inline citation of it share their distinctive
    words -- an author and a year -- so a long entry can match the handful of
    words a sentence spends naming it.  The reference list is one region at the
    end of the document, so a short run of it marooned in the body is that
    mistake rather than a second list.
    """
    for run in _iter_label_runs(seg_lines, SEG_REFERENCES):
        if len(run) > _MAROONED_RUN_MAX_LINES:
            continue
        if (
            _enclosing_label(seg_lines, run[0], -1)
            == SEG_BODY
            == _enclosing_label(seg_lines, run[-1], 1)
        ):
            for index in run:
                seg_lines[index].seg_label = None


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
        edge_position = _get_line_position(seg_lines[edge])
        if edge_size is None or edge_position is None:
            continue
        position = edge + step
        while 0 <= position < len(seg_lines):
            seg_line = seg_lines[position]
            line_position = _get_line_position(seg_line)
            if (
                seg_line.seg_label is not None
                or _get_page_number(seg_line) != page_number
                or _get_line_font_size(seg_line) != edge_size
                or line_position is None
                or abs(line_position[0] - edge_position[0]) > _SAME_COLUMN_MAX_X_GAP
            ):
                break
            seg_line.seg_label = label
            position += step


# Compared without spaces, because the extractor splits a word where the page
# hyphenates or kerns it -- "Refer ências" is one heading, not two words.
_REFERENCE_HEADINGS = frozenset(heading.replace(' ', '') for heading in (
    'reference', 'references', 'reference list', 'references cited',
    'literature cited', 'works cited',
    'bibliography', 'bibliographie', 'bibliografia', 'bibliografía',
    'referencia', 'referencias', 'referências', 'references bibliographiques',
    'referencias bibliograficas', 'referencias bibliográficas',
    'referencia bibliograficas', 'referências bibliográficas',
    'referencias bibliografias', 'bibliografia referencias',
    'literatur', 'literaturverzeichnis',
))

_HEADING_NOISE_PATTERN = re.compile(r'[^a-z\u00c0-\u024f ]+')


def _normalised_heading(text: str) -> str:
    return ''.join(_HEADING_NOISE_PATTERN.sub(' ', text.lower()).split())


_TRAILING_HEADING_MAX_LENGTH = 30
_TRAILING_HEADING_PATTERN = re.compile(r'^[^\W\d_][^\W\d_\s]*(\s+[^\W\d_][^\W\d_\s]*)*\s*:?$')


def _release_trailing_heading(seg_lines: List[_SegLine], label: str) -> None:
    """Give back the heading a region swallowed off the end of itself.

    A region grown to the foot of a page takes the line under it, and where
    that line is the heading of what comes next the region ends one line late
    -- "TABLAS" set under the last reference belongs to the tables, not to the
    list.  Letters only, so the DOI a reference ends with is not mistaken for
    one, and set in capitals, which is what tells a heading from a sentence
    that happens to be short.
    """
    runs = list(_iter_label_runs(seg_lines, label))
    for run in runs:
        last = seg_lines[run[-1]]
        text = last.text.strip()
        if (
            len(text) > _TRAILING_HEADING_MAX_LENGTH
            or not text.isupper()
            or not _TRAILING_HEADING_PATTERN.match(text)
        ):
            continue
        following = next(
            (
                sl for sl in seg_lines[run[-1] + 1:]
                if sl.seg_label not in _FURNITURE_LABELS
            ),
            None,
        )
        if following is None or following.seg_label in (None, label):
            continue
        last.seg_label = following.seg_label


_REGION_HEADING_MAX_GAP = 5


def _claim_region_heading(
    seg_lines: List[_SegLine],
    label: str,
    headings: FrozenSet[str],
    annotated: JatsAnnotatedLayoutDocument,
) -> None:
    """Take the printed heading a region is introduced by.

    Most JATS gives `<ref-list>` no `<title>`, so the "References" the page
    prints above the list has nothing to match and the list starts one line too
    late.  The heading is only claimed where nothing else has a claim on it.

    It is not always the line directly above.  A list whose first entry the
    JATS does not carry starts further down, leaving the heading and the entry
    between them unclaimed; both belong to the list.  Only unclaimed lines are
    read over, and only a few of them, so a heading further off than that is a
    heading for something else.
    """
    for run in _iter_label_runs(seg_lines, label):
        read_over = 0
        for index in range(run[0] - 1, -1, -1):
            seg_line = seg_lines[index]
            if seg_line.seg_label in _FURNITURE_LABELS:
                continue
            if seg_line.seg_label is not None or _is_grounded(seg_line, annotated):
                break
            if _normalised_heading(seg_line.text) in headings:
                for position in range(index, run[0]):
                    if seg_lines[position].seg_label not in _FURNITURE_LABELS:
                        seg_lines[position].seg_label = label
                break
            read_over += 1
            if read_over > _REGION_HEADING_MAX_GAP:
                break


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


# What tells a column from an indent.  A page's columns are set about 258pt
# apart in this corpus; a hanging indent or a list number moves a line 17 to
# 32pt.  An edge nearer than this to one already taken is the same column.
_MIN_COLUMN_SEPARATION = 150.0
_COLUMN_EDGE_TOLERANCE = 10.0
_MIN_COLUMN_LINES = 3


def _get_page_column_edges(
    seg_lines: List[_SegLine], page_number: int
) -> List[float]:
    """Where each column of a page starts, read from the lines it is set in.

    A column is a left edge that many lines share.  Comparing a line with the
    last line of a region does not say which column it is in: a region covers
    both columns of a page, and its last line can be a fragment set anywhere
    along a row.
    """
    counts: Counter = Counter()
    for seg_line in seg_lines:
        position = _get_line_position(seg_line)
        if position is None or _get_page_number(seg_line) != page_number:
            continue
        counts[round(position[0] / _COLUMN_EDGE_TOLERANCE) * _COLUMN_EDGE_TOLERANCE] += 1
    edges: List[float] = []
    for edge in sorted(counts):
        if counts[edge] < _MIN_COLUMN_LINES:
            continue
        if edges and edge - edges[-1] < _MIN_COLUMN_SEPARATION:
            continue
        edges.append(edge)
    return edges


def _get_column_index(x: float, edges: List[float]) -> int:
    """Which column a line starting at `x` is set in; 0 where there is one."""
    index = 0
    for position in range(1, len(edges)):
        boundary = (edges[position - 1] + edges[position]) / 2
        if x >= boundary:
            index = position
    return index


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

    Bounded to the page the run ends on, to the column the run is set in, and
    stopped by anything evidenced, so it only takes what is already inside the
    run's own column of its own page.  A running foot is
    stepped over rather than taken, the same way the gap merge steps over a
    running head.  Per run, because an article that records what changed since
    its previous version carries front matter on a later page too.
    """
    running = _get_running_texts(seg_lines)
    for run in _iter_label_runs(seg_lines, label):
        page_number = _get_page_number(seg_lines[run[-1]])
        edge_position = _get_line_position(seg_lines[run[-1]])
        column_edges = (
            _get_page_column_edges(seg_lines, page_number)
            if page_number is not None else []
        )
        edge_column = (
            _get_column_index(edge_position[0], column_edges)
            if edge_position is not None else None
        )
        for seg_line in seg_lines[run[-1] + 1:]:
            if _get_page_number(seg_line) != page_number:
                break
            position = _get_line_position(seg_line)
            if (
                edge_column is not None and position is not None
                and _get_column_index(position[0], column_edges) != edge_column
            ):
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


# Front matter a paper may print after its reference list.  An author note
# carries the corresponding author, which a preprint server sets at the end.
_LATE_FRONT_FIELDS = frozenset({JatsFieldNames.AUTHOR_NOTES})


def _is_grounded_in(
    seg_line: _SegLine,
    annotated: JatsAnnotatedLayoutDocument,
    fields: FrozenSet[str],
) -> bool:
    return any(
        annotated.get_token_field(token) in fields
        for token in seg_line.layout_line.tokens
    )


def _clear_front_beyond_threshold(
    seg_lines: List[_SegLine],
    max_block_start_line_index: int,
    annotated: JatsAnnotatedLayoutDocument,
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

    An author note is the exception the boundary has to make.  A preprint
    server prints the corresponding author's address at the end, after the
    references, and that block is the front matter wherever it is set -- unlike
    the licence sentence, which is a field that matched somewhere it does not
    belong.  The field the line is grounded in is what tells them apart.
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
        if beyond and not _is_grounded_in(sl, annotated, _LATE_FRONT_FIELDS):
            sl.seg_label = None


def _bridge_footnote_block(seg_lines: List[_SegLine]) -> None:
    """Keep a note's marker with the note.

    A numbered footnote prints its number on a line of its own when the note
    wraps, and the JATS runs number and text together, so the number matches
    nothing.  The gap merge cannot reach it: it treats a footnote as furniture
    to step over rather than as a region to bridge.

    A note can also have the note on one side only: a line of it the JATS does
    not carry, with nothing but a page number or a page break past it.  That is
    the same gap with no far side, and it is taken the same way.

    Reading order, not the page: the line taken is the one that follows the
    note in the file, which for a glyph the renderer emits last can be set
    above it.  Only where the note is what precedes the gap, though, because a
    line before a note belongs to whatever the note is set under.

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
        before = sides[0]
        if before is None or before.seg_label != SEG_FOOTNOTE:
            index = end + 1
            continue
        # Page furniture only: a footnote is what this is looking for, and
        # `_FURNITURE_LABELS` counts one as furniture to step over.
        after = next(
            (
                sl for sl in seg_lines[end:]
                if sl.seg_label not in (SEG_PAGE, SEG_HEADNOTE)
            ),
            None,
        )
        enclosed = (
            after is None
            or after.seg_label == SEG_FOOTNOTE
            or _get_page_number(after) != _get_page_number(before)
        )
        pages = {_get_page_number(sl) for sl in [before, *gap]}
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

FURNITURE_LABELS: FrozenSet[str] = frozenset({SEG_HEADNOTE, SEG_FOOTNOTE, SEG_PAGE})


def get_page_furniture_line_ids(
    layout_document: LayoutDocument,
    config: Optional[SegmentationConfig] = None,
) -> Set[int]:
    """The lines the page prints around its content, by `id(LayoutLine)`.

    Only the passes that need no JATS evidence run here, because this is what
    the aligner is given before it has any: a running head, a running foot and
    a page number are known from where they sit and from repeating across the
    document, and that pair is what reaches the furniture without taking a
    heading with it.  Position on its own does not, which is why no pass that
    reads only the margin is in this list.

    `derive_labels` runs the same passes again over the labels the aligner then
    produces, and decides the region a line belongs to; this says only that a
    line is furniture, which does not depend on the answer.

    The pass that reads text repetition alone is not among them.  It asks only
    that a line repeat and that its first copy print near the start of the
    document, which an author's name reprinted over their reviewer report
    satisfies -- in `derive_labels` the JATS vote has already claimed that line,
    and here there is nothing to stop it.

    The first page is left alone, as the noise filter leaves it.  Taking a line
    out joins what was printed either side of it, which is the point where a
    reference is printed around a running foot and the wrong thing where the
    title block is: an author list interrupted by a deposit notice reads as one
    affiliation running into the next, and the front matter has no page break
    for the furniture to be hiding.
    """
    seg_lines = _build_seg_lines(layout_document)
    page_meta_by_number = _get_page_meta_by_page_number(layout_document)
    _tag_by_coordinates(seg_lines, page_meta_by_number, config or SegmentationConfig())
    _tag_running_feet_by_repetition(seg_lines, page_meta_by_number)
    _tag_page_numbers_by_position(seg_lines, page_meta_by_number)
    # Repetition is required, not merely available.  Running the position pass
    # with nothing already labelled is not what `derive_labels` does -- there the
    # JATS vote has claimed the content first -- and on its own the margin takes
    # a reference whose last line is set low in the column with it.  A page
    # number is exempt because its text is different on every page, and the pass
    # that tags one has read its position and its shape already.
    pages_by_text: Dict[str, Set[int]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is not None:
            pages_by_text.setdefault(seg_line.text, set()).add(page_number)
    return {
        id(seg_line.layout_line)
        for seg_line in seg_lines
        if seg_line.seg_label in FURNITURE_LABELS
        and _get_page_number(seg_line) != 1
        and (
            seg_line.seg_label == SEG_PAGE
            or len(pages_by_text.get(seg_line.text) or ()) > 1
        )
    }


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
            seg_lines, self.config.front_max_start_line_index, annotated
        )
        _find_missing_page_numbers(seg_lines, page_meta_by_number, self.config)
        _tag_headnotes_by_text_repetition(
            seg_lines, self.config.page_header_max_first_line_index
        )
        _tag_running_feet_by_repetition(seg_lines, page_meta_by_number)
        _tag_page_numbers_by_position(seg_lines, page_meta_by_number)
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

        _release_references_inside_availability(seg_lines)
        _release_references_inside_body(seg_lines)

        for page_end_label in (SEG_REFERENCES, SEG_AVAILABILITY):
            _extend_region_to_page_end(
                seg_lines, page_end_label, annotated, page_meta_by_number, self.config
            )

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

        _claim_amendments_box(seg_lines)
        _claim_appendix_region(seg_lines)
        _tag_publication_dates(seg_lines)
        _claim_lines_sharing_a_row(seg_lines, page_meta_by_number)

        # ── Default remaining untagged lines → body ──
        for sl in seg_lines:
            if sl.seg_label is None:
                sl.seg_label = SEG_BODY

        # Last, because it reads the label of what follows the region, and what
        # follows is only settled once the default has been applied.
        _release_trailing_heading(seg_lines, SEG_REFERENCES)

        # Build id(LayoutLine) → label mapping — safe because layout_document holds strong refs
        result: Dict[int, str] = {}
        for sl in seg_lines:
            if sl.seg_label:
                result[id(sl.layout_line)] = sl.seg_label
        return result
