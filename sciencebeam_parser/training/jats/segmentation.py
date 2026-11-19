# pylint: disable=too-many-lines
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import (
    Dict, FrozenSet, Iterator, List, Mapping, NamedTuple, Optional, Pattern,
    Set, Tuple
)

from sciencebeam_parser.document.layout_document import (
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageMeta,
    LayoutToken,
)
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.back_matter import match_fn_keywords
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


# What a float prints at the top or the foot of a page it runs over: its label
# again, and the word saying the table goes on.  GROBID's corpus never files
# either as page furniture -- a bare "Table N" line sits in `<body>` 3 times and
# in an annex 5, which is the region its table is in, and "Table N (continued)"
# in `<body>` 66 times.
_FLOAT_LABEL_PATTERN = (
    r"(?:tab(?:le|\.|ela|la)?|fig(?:ure|\.|ura)?|quadro|gr[a\u00e1]fico)\s*\.?\s*\d+\s*\.?"
)
_FLOAT_CONTINUES_PATTERN = (
    r"\(?\s*(?:continued?|continuation|continua(?:[c\u00e7][a\u00e3]o)?"
    r"|continuaci[o\u00f3]n|cont'd|cont\.)\s*\)?\.?"
)
# Either on its own, or the label with the word after it, which is how a
# journal in two languages carries a "Tabela 2" over: "Tabela 2. Continuation".
_FLOAT_CARRY_OVER_PATTERN = re.compile(
    r"^\s*(?:"
    + _FLOAT_LABEL_PATTERN + r"(?:\s*" + _FLOAT_CONTINUES_PATTERN + r")?"
    + r"|" + _FLOAT_CONTINUES_PATTERN
    + r")\s*$",
    re.IGNORECASE,
)


def _carries_a_float_over_the_page(text: str) -> bool:
    """Whether the line is a float saying it goes on, rather than page furniture."""
    return bool(_FLOAT_CARRY_OVER_PATTERN.match(text))


def _is_valid_headnote_candidate(text: str, count: int, min_count: int = 2) -> bool:
    if count < min_count:
        return False
    if _carries_a_float_over_the_page(text):
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


def _running_in_step_with_the_pages(
    candidates: List[_SegLine], printed: List[_SegLine]
) -> List[_SegLine]:
    """The margin numbers left once a page printing two has chosen between them.

    An affiliation marker is set in the same margin as the page number, so the
    one that advances with the page wins -- at whatever number the journal
    starts its own sequence from.  A page printing one keeps it, whatever it
    reads: "Page 3 of 12" has nothing to be mistaken for.

    Two is what the page prints rather than what is left to label: a number a
    JATS field has already claimed is still the other one in that margin, and
    without it the marker beside it is alone and keeps whatever it reads.
    """
    def offset_of(seg_line: _SegLine) -> Optional[int]:
        page_number = _get_page_number(seg_line)
        printed = _parse_page_number(seg_line.text)
        if page_number is None or printed is None:
            return None
        return printed - page_number

    per_page: Counter = Counter(_get_page_number(seg_line) for seg_line in printed)
    offsets: Counter = Counter(
        offset for offset in map(offset_of, printed) if offset is not None
    )
    if not offsets:
        return list(candidates)
    in_step = min(offsets, key=lambda offset: (-offsets[offset], abs(offset), offset))
    return [
        seg_line for seg_line in candidates
        if per_page[_get_page_number(seg_line)] == 1 or offset_of(seg_line) == in_step
    ]


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

    Repeating is counted at the head and nowhere else, because a running head
    repeats *as* a head.  "Fuente: Elaboración propia." is printed under every
    table of a document and so repeats across its pages, but it stands at the
    top of one only where the table above it ran over, and there it is still
    the caption of that table.
    """
    y_ratio_by_line = {
        id(seg_line): _get_line_y_ratio(seg_line, page_meta_by_number)
        for seg_line in seg_lines
    }
    header_pages_by_text: Dict[str, Set[int]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        y_ratio = y_ratio_by_line[id(seg_line)]
        if page_number is None or y_ratio is None:
            continue
        if y_ratio < config.headnote_y_ratio:
            header_pages_by_text.setdefault(seg_line.text, set()).add(page_number)

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

    def is_margin_line(seg_line: _SegLine) -> bool:
        if id(seg_line) in corresponding_author_lines:
            return False
        if not block_is_margin.get(seg_line.block_index, True):
            return False
        return is_in_margin(y_ratio_by_line[id(seg_line)])

    def is_unclaimed_margin_line(seg_line: _SegLine) -> bool:
        return seg_line.seg_label is None and is_margin_line(seg_line)

    margin_numbers = [
        seg_line for seg_line in seg_lines
        if is_margin_line(seg_line) and _is_valid_page_number_candidate(seg_line.text)
    ]
    page_number_lines = {
        id(seg_line) for seg_line in _running_in_step_with_the_pages(
            [seg_line for seg_line in margin_numbers if seg_line.seg_label is None],
            margin_numbers,
        )
    }
    for seg_line in seg_lines:
        if id(seg_line) in corresponding_author_lines:
            # The note at the foot of the title page says what it is.  One
            # token of it that a paragraph of the body happens to use as well
            # carries the line otherwise, because a line is voted on by what is
            # labelled on it and three tokens can be carried by one.
            seg_line.seg_label = SEG_FRONT
            continue
        if not is_unclaimed_margin_line(seg_line):
            continue
        if _carries_a_float_over_the_page(seg_line.text):
            continue
        y_ratio = y_ratio_by_line[id(seg_line)]
        if id(seg_line) in page_number_lines:
            seg_line.seg_label = SEG_PAGE
        elif y_ratio is not None and y_ratio > config.footnote_y_ratio:
            seg_line.seg_label = (
                SEG_FRONT if id(seg_line) in corresponding_author_lines
                else SEG_FOOTNOTE
            )
        elif len(header_pages_by_text.get(seg_line.text, ())) > 1:
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


_RUNNING_HEAD_MAX_Y_RATIO = 0.5
_RUNNING_HEAD_Y_TOLERANCE = 0.01

# A running head prints on page after page, where a sentence the argument
# happens to repeat prints twice.
_RUNNING_HEAD_MIN_PAGES = 3


def _tag_running_heads_by_position(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
) -> None:
    """Take the head a document prints in the same spot on page after page.

    The headnote zone is a fixed share of the page and a journal can set its
    running head below it -- one sets it at an eighth of the way down -- where
    position alone says nothing.  Printing the same words in the same place on
    more than one page says it instead, which is what the page-number pass reads
    for a number that differs on every page.

    It carries over a region the JATS vote has already named, because a running
    head of the authors' surnames is matched by the byline on every page it
    prints on: the guidelines ask for a running title not to be labelled at all.
    """
    by_text: Dict[str, List[Tuple[_SegLine, int, float]]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        y_ratio = _get_line_y_ratio(seg_line, page_meta_by_number)
        if page_number is None or y_ratio is None or not seg_line.text.strip():
            continue
        if y_ratio > _RUNNING_HEAD_MAX_Y_RATIO:
            continue
        by_text.setdefault(seg_line.text, []).append((seg_line, page_number, y_ratio))
    for text, occurrences in by_text.items():
        pages = {page_number for _, page_number, _ in occurrences}
        if len(pages) < _RUNNING_HEAD_MIN_PAGES:
            continue
        if not _is_valid_headnote_candidate(text, len(occurrences)):
            continue
        spread = max(y for _, _, y in occurrences) - min(y for _, _, y in occurrences)
        if spread > _RUNNING_HEAD_Y_TOLERANCE:
            continue
        for seg_line, _, _ in occurrences:
            if seg_line.seg_label != SEG_COVER:
                seg_line.seg_label = SEG_HEADNOTE


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


def _get_seg_lines_by_page_number(
    seg_lines: List[_SegLine]
) -> Dict[int, List[_SegLine]]:
    pages: Dict[int, List[_SegLine]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is not None:
            pages.setdefault(page_number, []).append(seg_line)
    return pages


def _opens_a_deposit_page(page_lines: List[_SegLine]) -> bool:
    """Whether the page opens with the notice a server deposits an article behind.

    Text alone, so it holds before anything has been aligned.
    """
    first = next((sl for sl in page_lines if sl.text.strip()), None)
    if first is None:
        return False
    if _is_deposit_notice(first.text):
        return True
    if _DEPOSIT_LABEL_PATTERN.match(first.text) and any(
        _is_deposit_marker(sl.text) for sl in page_lines
    ):
        return True
    # A submission sheet opens with the watermark printed across it, set a
    # letter to a line, so what names it is not the line the page starts with.
    return any(_is_submission_cover_marker(sl.text) for sl in page_lines)


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
    pages = _get_seg_lines_by_page_number(seg_lines)
    cover_pages: Set[int] = set()
    for page_number in sorted(pages):
        page_lines = pages[page_number]
        if not any(sl.text.strip() for sl in page_lines):
            continue
        if any(sl.seg_label in _ARTICLE_CONTENT_LABELS for sl in page_lines):
            continue
        # The form can run past the page it starts on, leaving the dates and the
        # generator's footer by themselves overleaf.  That is the rest of the
        # cover rather than the article starting.
        continues_one = page_number - 1 in cover_pages and all(
            sl.seg_label in _FURNITURE_LABELS or _is_deposit_marker(sl.text)
            for sl in page_lines if sl.text.strip()
        )
        if not _opens_a_deposit_page(page_lines) and not continues_one:
            continue
        cover_pages.add(page_number)
        for seg_line in page_lines:
            seg_line.seg_label = SEG_COVER


_AMENDMENTS_OPENING_PATTERN = re.compile(r'^\s*Amendments\s+from\s+Version\b', re.IGNORECASE)
_AMENDMENTS_CLOSING_PATTERN = re.compile(
    r'^\s*Any\s+further\s+responses\s+from\s+the\s+reviewers', re.IGNORECASE
)
_AMENDMENTS_MAX_LINES = 60


def _iter_amendments_boxes(seg_lines: List[_SegLine]) -> Iterator[List[_SegLine]]:
    """The lines of each box a revised article says what changed in.

    Both ends have to be found: the opening line alone would run into whatever
    follows it on a page that does not close the box.
    """
    for index, seg_line in enumerate(seg_lines):
        if not _AMENDMENTS_OPENING_PATTERN.match(seg_line.text):
            continue
        end = next(
            (
                position
                for position in range(
                    index, min(len(seg_lines), index + _AMENDMENTS_MAX_LINES)
                )
                if _AMENDMENTS_CLOSING_PATTERN.match(seg_lines[position].text)
            ),
            None,
        )
        if end is not None:
            yield seg_lines[index:end + 2]


def get_amendments_box_line_ids(layout_document: LayoutDocument) -> Set[int]:
    """The lines of the box a revised article says what changed in.

    The box is the publisher's, and nothing the JATS declares of the article is
    printed inside it: a name there is one the amendment cites, not the byline
    again.  The article's own note is what the box is, and that is the one
    value looked for in it.
    """
    return {
        id(seg_line.layout_line)
        for box in _iter_amendments_boxes(_build_seg_lines(layout_document))
        for seg_line in box
    }


def _claim_amendments_box(seg_lines: List[_SegLine]) -> None:
    """Take the box a revised article says what changed in.

    The box is the publisher's, and it opens and closes with the publisher's
    own words, the same in every revised article.  The JATS carries it in
    `<front>/<notes>`, but not always the version the PDF prints -- one article
    describes one set of changes in its notes and another on its page -- and
    then nothing grounds it and it reads as the article's argument.

    The box takes a line the body matched as well as one nothing claimed,
    because the box paraphrases the article and a sentence of it can match the
    article's own -- and a body paragraph cannot be inside the box.
    """
    for box in _iter_amendments_boxes(seg_lines):
        for seg_line_in_box in box:
            if seg_line_in_box.seg_label in (None, SEG_BODY):
                seg_line_in_box.seg_label = SEG_FRONT


_ANNEX_HEADING_PATTERN = re.compile(
    r'^\s*('
    r'ap[e\u00ea\u00e9]ndices?|appendix|appendices|anexos?|annexe?s?'
    # What GROBID's corpus most often opens an annex with: "Supporting
    # information" 73 times against "Appendix" 6.
    r'|supporting\s+information'
    r'|supplementary\s+(material|information|data|file)'
    r'|materi(al|ais|ales)\s+suplementar\w*'
    # The authors' biographies, which GROBID's corpus sets in an annex.
    r'|about\s+the\s+authors?'
    r'|sobre\s+(os?|as?)\s+autor(es|as|a)?'
    r'|sobre\s+l(os|as)\s+autor(es|as)'
    r')\b',
    re.IGNORECASE,
)

_ANNEX_HEADING_MAX_LENGTH = 80


def _claim_annex_region(seg_lines: List[_SegLine]) -> None:
    """Open an annex where the page prints a heading that opens one.

    A paper can print an appendix the JATS does not carry -- the section is
    declared as supplementary material and its text lives only in the PDF -- so
    nothing grounds it and it falls to the body.  GROBID's own corpus labels a
    printed appendix heading `<annex>` 69 times, before the reference list as
    often as after, so where it prints does not decide it.  It labels the
    authors' biographies the same way, twice out of the three times it holds
    them.

    The heading has to be unclaimed, which is what keeps a sentence that merely
    mentions an appendix out: that sentence belongs to a paragraph the JATS
    carries, so it is already labelled.  The annex then runs on over what
    nothing else claims, and stops at whatever does -- or hands over, where
    the page names a declaration under it: the biographies an article closes
    with are followed by the statement of what each author did.  A heading as
    the page sets one, which is to say in bold -- a checklist asking whether
    there are "pias para atender o numero de colaboradores" names nothing.
    """
    index = 0
    while index < len(seg_lines):
        seg_line = seg_lines[index]
        if (
            seg_line.seg_label is not None
            or len(seg_line.text) > _ANNEX_HEADING_MAX_LENGTH
            or not _ANNEX_HEADING_PATTERN.match(seg_line.text)
        ):
            index += 1
            continue
        taken = SEG_ANNEX
        for following in seg_lines[index:]:
            if following.seg_label in _FURNITURE_LABELS:
                continue
            if following.seg_label is not None:
                break
            named = (
                None if following is seg_line or not _is_bold(following)
                else _declaration_named_by(following.text)
            )
            taken = named or taken
            following.seg_label = taken
        index += 1


# A date as a journal prints one, to the year.  A line ending in a bare number
# is a line of prose the page broke there.
_PRINTED_DATE_PATTERN = (
    r'(?:\d{1,4}\s*[/.\u2013-]\s*\d{1,2}\s*[/.\u2013-]\s*\d{2,4}'
    r'|\d{1,2}(?:\s*(?:st|nd|rd|th))?\s*,?\s*\d{4})'
)
_PUBLICATION_DATE_PATTERN = re.compile(
    r'^\s*(?:[^\W\d_]+\s+){0,2}'
    r'(recebido|recebida|aprovado|aprovada|aceito|aceita|submetido|revisado|enviado'
    r'|apresentado|apresentada'
    r'|recibido|aceptado|presentado'
    r'|received|accepted|approved|submitted|posted|revised|published|preprint)\b'
    r'[^0-9]{0,30}' + _PRINTED_DATE_PATTERN + r'\s*\.?\s*(\([^)]*\))?\s*$',
    re.IGNORECASE,
)

# The deposit page states the dates as its own, and a page number is structure.
_LABELS_A_DATE_DOES_NOT_TAKE = frozenset({SEG_COVER, SEG_PAGE})


def _tag_publication_dates(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Take a line that is nothing but a date the article was received or posted.

    These print wherever the publisher puts them -- a title page, a deposit
    page, or a block of their own after the reference list -- and no JATS field
    carries the ones printed at the end, so they fall to whichever region the
    page around them reaches into.  They are part of the bibliographic record,
    which GROBID allows to be several areas in several places.  The whole line
    has to be the date: a sentence that mentions one is prose, and a line a
    field does carry is where that field puts it -- until the reference list,
    past which a field carrying the line is the list reaching over a date it
    has no entry for and reading it as an entry's pages.
    """
    first_reference = _get_first_reference_index(seg_lines)
    for index, seg_line in enumerate(seg_lines):
        if seg_line.seg_label in _LABELS_A_DATE_DOES_NOT_TAKE:
            continue
        if _is_grounded(seg_line, annotated) and not (
            first_reference is not None and index > first_reference
        ):
            continue
        if _PUBLICATION_DATE_PATTERN.match(seg_line.text):
            seg_line.seg_label = SEG_FRONT


# The sentence a journal grants its licence in, from the words it opens with.
# A line that cites a licence in passing says something about it first.
_LICENCE_NOTICE_PATTERN = re.compile(
    r'^\s*(?:open\s+access\s+)?'
    r'(?:this\s+(?:is\s+an?|article\s+is|work\s+is)'
    r'|est[ae]\s+(?:\u00e9|e)\s+um'
    r'|est[ae]\s+art[\u00edi]culo)\b'
    r'[^.]{0,160}?'
    r'(?:creative\s+commons|open[\s-]access|acesso\s+aberto|cc[\s-]by)',
    re.IGNORECASE,
)


def _get_first_reference_index(seg_lines: List[_SegLine]) -> Optional[int]:
    """Where the reference list opens, if the document has one."""
    return next(
        (index for index, sl in enumerate(seg_lines)
         if sl.seg_label == SEG_REFERENCES),
        None,
    )


# The line a renderer composes to say how the article should be cited.
_HOW_TO_CITE_PATTERN = re.compile(
    r'^\s*(?:how\s+to\s+cite|como\s+citar|c[o\u00f3]mo\s+citar)\b',
    re.IGNORECASE,
)


def _claim_the_citation_printed_after_the_references(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Take the "how to cite" block an article closes with.

    It is the bibliographic record of the article itself, not of anything it
    cites and not part of whatever section it is printed under: GROBID's
    corpus holds it in `<front>` 11 times of the 12 it has after a reference
    list.  The citation runs under the heading to the end of the section the
    page sets it in.
    """
    opened = _get_first_reference_index(seg_lines)
    if opened is None:
        return
    for index in range(opened, len(seg_lines)):
        if not _HOW_TO_CITE_PATTERN.match(seg_lines[index].text):
            continue
        under = seg_lines[index].seg_label
        if under in _TRAILING_MATTER_LABELS:
            continue
        for seg_line in seg_lines[index:]:
            if seg_line is not seg_lines[index] and (
                seg_line.seg_label != under or _is_grounded(seg_line, annotated)
            ):
                break
            seg_line.seg_label = SEG_FRONT


def _claim_the_block_printed_after_the_references(
    seg_lines: List[_SegLine], pattern: Pattern[str]
) -> None:
    """Give the front matter each block past the list that the pattern opens."""
    opened = _get_first_reference_index(seg_lines)
    if opened is None:
        return
    for index in range(opened, len(seg_lines)):
        if seg_lines[index].seg_label in _TRAILING_MATTER_LABELS:
            continue
        if not pattern.match(seg_lines[index].text):
            continue
        block_index = seg_lines[index].block_index
        for seg_line in seg_lines[index:]:
            if seg_line.block_index != block_index:
                break
            if seg_line.seg_label in _TRAILING_MATTER_LABELS:
                continue
            seg_line.seg_label = SEG_FRONT


# What a journal credits the editor who handled the article on.  Never
# "Editora", which is what a reference calls its publisher.
_EDITOR_CREDIT_PATTERN = re.compile(
    r'^\s*(?:handling|section|academic|associate|managing|guest)?\s*'
    r'editor(?:es|s)?'
    r'(?:[\s\u2010-\u2015-]*(?:chefe|cient[i\u00ed]fico|in[\s-]chief|associad[oa]))?'
    r'\s*:',
    re.IGNORECASE,
)


def _claim_the_editor_credited_after_the_references(seg_lines: List[_SegLine]) -> None:
    """Take the line crediting the editor from the list it is printed under.

    GROBID's corpus holds such a line in `<front>` 83 times of 98, and 13 of
    the 15 it does not are inside a `listBibl`, which is this same list
    reaching over a line it has no entry for.
    """
    _claim_the_block_printed_after_the_references(seg_lines, _EDITOR_CREDIT_PATTERN)


def _claim_the_licence_printed_after_the_references(seg_lines: List[_SegLine]) -> None:
    """Take a licence notice printed past the reference list.

    A journal grants its licence at the foot of the last page, and the
    deposit need not carry the sentence at all, so the list above reaches
    over it.  GROBID's corpus holds the sentence printed after a reference
    list in `<front>` 10 times of 12.  Before the list it is as often the
    standing line a journal prints in a margin of every page, which the
    margins have already settled.

    Whatever the aligner made of the line: printed across the foot of a page
    the list is set in two columns down, it arrives in the middle of the list
    and S0102-76382018000400391 has it matched as an entry's article title.
    No article is titled this.

    The notice is as long as the block the page sets it in, the sentence
    running on over two or three lines of which only the first says what it is.
    """
    _claim_the_block_printed_after_the_references(seg_lines, _LICENCE_NOTICE_PATTERN)


# What the deposit files a note of the article's under.
_DEPOSITED_NOTE_FIELDS = frozenset({JatsFieldNames.BACK_FOOTNOTE})


def _opens_what_is_under_it(seg_lines: List[_SegLine], index: int) -> bool:
    """Whether the line below opens a sentence of its own.

    A heading does not wrap, where a statement that names its authors in bold
    runs on into the line under it.
    """
    below = next(
        (seg_line for seg_line in seg_lines[index + 1:] if seg_line.text.strip()),
        None,
    )
    return below is not None and not below.text.strip()[0].islower()


# A heading does not fill the measure: of the headings these journals print
# the widest runs to three quarters of it, where a line of the statement under
# one runs to the whole of it.
_HEADING_MAX_MEASURE_RATIO = 0.8

# The width to read as the measure, which is the widest line but for the few a
# table or a figure runs across the page.
_MEASURE_QUANTILE = 0.9


def _get_line_width(seg_line: _SegLine) -> Optional[float]:
    """How far across the page the line runs."""
    boxes = [
        token.coordinates for token in seg_line.layout_line.tokens
        if token.coordinates
    ]
    if not boxes:
        return None
    return max(box.x + box.width for box in boxes) - min(box.x for box in boxes)


def _get_measure_by_page(seg_lines: List[_SegLine]) -> Dict[int, float]:
    """The width each page sets its text to."""
    widths: Dict[int, List[float]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        width = _get_line_width(seg_line)
        if page_number is not None and width:
            widths.setdefault(page_number, []).append(width)
    measures = {}
    for page_number, page_widths in widths.items():
        ordered = sorted(page_widths)
        at = min(int(len(ordered) * _MEASURE_QUANTILE), len(ordered) - 1)
        measures[page_number] = ordered[at]
    return measures


def _is_set_as_a_heading(
    seg_lines: List[_SegLine],
    index: int,
    text_size: Optional[float],
    measure_by_page: Mapping[int, float],
) -> bool:
    """Whether the page sets the line as a heading over the text under it."""
    seg_line = seg_lines[index]
    if len(seg_line.text.strip()) > _STATEMENT_HEADING_MAX_LENGTH:
        return False
    page_number = _get_page_number(seg_line)
    width = _get_line_width(seg_line)
    measure = measure_by_page.get(page_number) if page_number is not None else None
    if width and measure and width > measure * _HEADING_MAX_MEASURE_RATIO:
        return False
    size = _get_line_font_size(seg_line)
    stands_out = _is_bold(seg_line) or (
        size is not None and text_size is not None and size > text_size
    )
    return stands_out and _opens_what_is_under_it(seg_lines, index)


def _claim_the_declarations_the_article_closes_with(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Open each declaration the pages the article closes on name.

    GROBID's corpus gives a section opening with one of these headings the
    declaration it names: contribution 148 times out of 148, conflict 123 of
    137, funding 113 of 168, and never the list it is printed among.  What the
    deposit filed the statement under does not decide it -- a journal files all
    four as back footnotes, and a region reaches over the ones it carries
    nothing for at all.

    A heading that names no declaration opens an annex, which is where GROBID
    files an unnamed one.  The run stops at the reference list, and at a line
    the deposit files under a region of its own: what the page says is read
    over what the regions made of it, never over what the deposit declares --
    an availability statement with sections of its own is one statement.
    """
    first_reference = _get_first_reference_index(seg_lines)
    if first_reference is None:
        return
    closing_page = _get_page_number(seg_lines[first_reference])
    text_size = _get_prevailing_type_size(seg_lines)
    measure_by_page = _get_measure_by_page(seg_lines)
    label: Optional[str] = None
    for index, seg_line in enumerate(seg_lines):
        page_number = _get_page_number(seg_line)
        if page_number is None or (
            closing_page is not None and page_number < closing_page
        ):
            continue
        if (
            _is_grounded_in(seg_line, annotated, _DECLARED_REGION_FIELDS)
            or seg_line.seg_label in {SEG_REVIEW, SEG_COVER}
        ):
            label = None
            continue
        if _is_set_as_a_heading(seg_lines, index, text_size, measure_by_page):
            label = _declaration_named_by(seg_line.text) or (
                SEG_ANNEX if label else None
            )
        if label is None:
            continue
        if (
            seg_line.seg_label in _FURNITURE_LABELS
            and not _is_grounded(seg_line, annotated)
        ):
            continue
        seg_line.seg_label = label


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

# What a declaration the argument restates runs to, where a stray reference
# fragment is a line or two.
_RESTATED_DECLARATION_MAX_LINES = 10


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
        # A dataset the statement cites is not introduced by the word
        # "References": where the run is that heading, it belongs to the list
        # it heads, which on a two-column page is read a long way after it.
        if any(
            _normalised_heading(seg_lines[index].text) in _REFERENCE_HEADINGS
            for index in run
        ):
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


_IMPRINT_BLOCK_MAX_LINES = 8


def _claim_the_rest_of_the_block_the_front_matter_fills(
    seg_lines: List[_SegLine]
) -> None:
    """Give the front matter the rest of the small block it is printed in.

    A journal sets the article's imprint in one block at the foot of its first
    page: the doi, who to write to, the dates it was received and accepted,
    and what paid for it.  The page sets them as one thing, and most of them
    are matched to the front matter, so the two that are not -- a doi no field
    carries, and a funding statement the deposit files as a declaration -- are
    left reading as the column above them.

    Only a block short enough to be an imprint, and only over what the regions
    have not claimed: a column where the front matter runs into the body is
    long, and says nothing about the lines it ends with.
    """
    taken = {None, SEG_BODY} | set(_STATEMENT_LABELS)
    by_block: Dict[int, List[_SegLine]] = {}
    for seg_line in seg_lines:
        by_block.setdefault(seg_line.block_index, []).append(seg_line)
    for block in by_block.values():
        if len(block) > _IMPRINT_BLOCK_MAX_LINES:
            continue
        front = [sl for sl in block if sl.seg_label == SEG_FRONT]
        if len(front) * 2 <= len(block):
            continue
        for seg_line in block:
            if seg_line.seg_label in taken:
                seg_line.seg_label = SEG_FRONT


def _release_body_inside_the_references(seg_lines: List[_SegLine]) -> None:
    """Take back the section heading that matched an entry of the reference list.

    A deposit declares the article's body once for every language it carries,
    and the journal prints one of them.  The bodies it does not print go
    looking for somewhere to be, and a section title lands on the entry that
    shares its words -- "Introdução" on "FARIAS, T. Introdução ao Direito
    Ambiental".  The list is one region at the end of the document, so a short
    run of body inside it is that mistake rather than the article resuming
    between two references.

    Before the gap merge, because what a false match costs is not the line it
    takes but the entries either side of it, which the merge can no longer
    carry across.

    The matches are read together, because one deposit supplies as many of
    them as it carries languages and the list is long enough to hold several:
    asked one at a time, each has another of them on one side and the list on
    the other, and neither is enclosed.
    """
    runs = list(_iter_label_runs(seg_lines, SEG_BODY))
    index = 0
    while index < len(runs):
        group = [runs[index]]
        while (
            index + len(group) < len(runs)
            and _enclosing_label(seg_lines, group[-1][-1], 1) == SEG_BODY
        ):
            group.append(runs[index + len(group)])
        index += len(group)
        if sum(len(run) for run in group) > _MAROONED_RUN_MAX_LINES:
            continue
        if (
            _enclosing_label(seg_lines, group[0][0], -1)
            == SEG_REFERENCES
            == _enclosing_label(seg_lines, group[-1][-1], 1)
        ):
            for run in group:
                for position in run:
                    seg_lines[position].seg_label = None


def _release_body_inside_front(seg_lines: List[_SegLine]) -> None:
    """Take back the abstract sub-heading that matched a section of the same name.

    A structured abstract is set under the headings the body goes on to use --
    Introdução, Métodos, Resultados -- and the aligner has only the text to go
    on, so it gives those lines to the body section that repeats them.  The
    front matter is one run, so a short stretch of body inside it is that
    mistake rather than the article beginning and then stopping again.
    """
    for run in _iter_label_runs(seg_lines, SEG_BODY):
        if len(run) > _MAROONED_RUN_MAX_LINES:
            continue
        if (
            _enclosing_label(seg_lines, run[0], -1)
            == SEG_FRONT
            == _enclosing_label(seg_lines, run[-1], 1)
        ):
            for index in run:
                seg_lines[index].seg_label = SEG_FRONT


def _iter_statement_blocks(seg_lines: List[_SegLine]) -> Iterator[List[int]]:
    """Consecutive declarations, read as one block.

    An article prints them together, one after another, so what the page sets
    above them it sets above all of them: the lines between two declarations are
    the heading of the second, which nothing has claimed.
    """
    block: List[int] = []
    for index, seg_line in enumerate(seg_lines):
        if seg_line.seg_label in _STATEMENT_LABELS:
            block.append(index)
            continue
        if seg_line.seg_label is None or seg_line.seg_label in _FURNITURE_LABELS:
            continue
        if block:
            yield block
            block = []
    if block:
        yield block


def _release_statements_inside_front(seg_lines: List[_SegLine]) -> None:
    """A declaration the front matter prints is part of the front matter.

    GROBID's corpora carry it both ways -- a `<div type="conflict">` region on
    366 of its segmentation documents and a conflict note inside the header on
    113 of its header ones -- and the JATS says nothing about which, so where
    the page sets it is what decides: a declaration the page sets before the
    article begins, with the front matter above it, is the front matter's.

    Where the body begins rather than what follows the declaration, because what
    follows it is not always matched: a preprint that sets its declarations
    between the affiliations and the abstract would be cut in two, and the pages
    of front matter after them left with no run to be merged into.
    """
    body_start = next(
        (
            index for index, seg_line in enumerate(seg_lines)
            if seg_line.seg_label == SEG_BODY
        ),
        len(seg_lines),
    )
    for block in _iter_statement_blocks(seg_lines):
        if (
            block[-1] < body_start
            and _enclosing_label(seg_lines, block[0], -1) == SEG_FRONT
        ):
            for index in block:
                seg_lines[index].seg_label = SEG_FRONT


# A wrapped line follows the one above it by a line of type and the leading
# under it; three times its height covers that and a paragraph break.
_PARAGRAPH_PITCH_RATIO = 3.0


def _follows_on_from(seg_line: _SegLine, neighbour: _SegLine) -> bool:
    """Whether the page sets the two as consecutive lines of one paragraph."""
    token = seg_line.first_token
    position = _get_line_position(seg_line)
    neighbour_position = _get_line_position(neighbour)
    if token is None or token.coordinates is None:
        return False
    if position is None or neighbour_position is None:
        return False
    return (
        _get_page_number(seg_line) == _get_page_number(neighbour)
        and abs(position[0] - neighbour_position[0]) <= _SAME_COLUMN_MAX_X_GAP
        and abs(position[1] - neighbour_position[1])
        <= token.coordinates.height * _PARAGRAPH_PITCH_RATIO
    )


def _continues_the_lines_around_it(seg_lines: List[_SegLine], run: List[int]) -> bool:
    """Whether the run reads as more of the text set either side of it."""
    if run[0] == 0 or run[-1] + 1 >= len(seg_lines):
        return False
    return (
        _follows_on_from(seg_lines[run[0]], seg_lines[run[0] - 1])
        and _follows_on_from(seg_lines[run[-1]], seg_lines[run[-1] + 1])
    )


def _release_annex_inside_body(seg_lines: List[_SegLine]) -> None:
    """Take back a body line that matched the appendix restating it.

    An appendix prints the instrument the article describes, so a sentence of
    the body -- the aim of the study, the name of the questionnaire -- occurs
    in both and the appendix match wins the line.  The appendix is one run at
    the end of the document, so a short stretch of it inside the body is that
    match rather than a second appendix.

    Only where the stretch is part of the paragraph around it: the page sets
    it in the same column at the pitch the lines either side are set at.  A
    statement an article prints at the foot of a page stands apart from the
    argument above it, and there the back matter the JATS named it is what it
    is, wherever it prints.

    A declaration runs longer than a restated sentence does -- an ethics
    statement the methods set out in full takes six lines -- so the length a
    marooned reference is held to is not the length to hold this to.  The
    paragraph test is what decides either way.
    """
    for run in _iter_label_runs(seg_lines, SEG_ANNEX):
        if len(run) > _RESTATED_DECLARATION_MAX_LINES:
            continue
        if not _continues_the_lines_around_it(seg_lines, run):
            continue
        if (
            _enclosing_label(seg_lines, run[0], -1)
            == SEG_BODY
            == _enclosing_label(seg_lines, run[-1], 1)
        ):
            for index in run:
                seg_lines[index].seg_label = None


def _without_the_numbers(text: str) -> str:
    """What the line says apart from the numbers that change with the page."""
    return re.sub(r'[\d\W_]+', ' ', text).strip()


def _reclaim_repeated_headnotes(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> None:
    """Give a running header back to `<headnote>` where a region match took it.

    The same absorption the page-number pass undoes: a JATS match reaching across
    a page break labels the running header between its halves, although the
    field's own text does not contain it.  A line is reclaimed only when the same
    text is set in the header zone on at least two pages *and* the line itself
    sits in that zone.  The zone rather than an existing headnote, because a
    running head of the authors' surnames is matched by the byline on every
    page it prints on, and so is never left for the coordinate rules to name.
    Repetition alone is not enough: a running header carrying the article's
    title repeats, and would otherwise take the title out of the front matter,
    where it prints below the header zone.

    A cover page keeps everything it has: GROBID asks for the whole of one under
    the one label, and the banner across the top of it is the publisher's too.

    What the line says apart from its numbers, because a journal prints the
    article's page range in the head of the page it opens on and the volume
    alone thereafter.
    """
    pages_by_text: Dict[str, Set[int]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is None:
            continue
        if seg_line.seg_label == SEG_HEADNOTE or _is_in_header_zone(
            seg_line, page_meta_by_number, config
        ):
            pages_by_text.setdefault(
                _without_the_numbers(seg_line.text), set()
            ).add(page_number)
    for seg_line in seg_lines:
        if seg_line.seg_label in (SEG_HEADNOTE, SEG_COVER, None):
            continue
        count = len(pages_by_text.get(_without_the_numbers(seg_line.text), ()))
        if count < 2 or not _is_valid_headnote_candidate(seg_line.text, count, min_count=2):
            continue
        if not _is_in_header_zone(seg_line, page_meta_by_number, config):
            continue
        seg_line.seg_label = SEG_HEADNOTE


def _reclaim_repeated_running_feet(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
) -> None:
    """Give a running foot back to `<footnote>` where something else took it.

    The counterpart of the headnote reclaim.  A journal sets its own citation
    along every foot, page range and all, so the `<page>` vote takes the line on
    the page the article opens and the front merge takes it on the next.  It is
    the same furniture on every page it prints on.
    """
    foot_counts: Counter = Counter(
        sl.text for sl in seg_lines if sl.seg_label == SEG_FOOTNOTE
    )
    for seg_line in seg_lines:
        if seg_line.seg_label in (SEG_FOOTNOTE, SEG_COVER, None):
            continue
        count = foot_counts.get(seg_line.text, 0)
        if count < 2 or not _is_valid_headnote_candidate(seg_line.text, count, min_count=2):
            continue
        y_ratio = _get_line_y_ratio(seg_line, page_meta_by_number)
        if y_ratio is None or y_ratio < _RUNNING_FOOT_MIN_Y_RATIO:
            continue
        seg_line.seg_label = SEG_FOOTNOTE


# A line short enough to occur anywhere says nothing about what it is: a
# document sets "R" and "2023)." at the end of a line as readily as anywhere.
_RECOGNISABLE_LINE_MIN_LENGTH = 20


def _says_enough_to_be_recognised(text: str) -> bool:
    stripped = text.strip()
    return (
        len(stripped) >= _RECOGNISABLE_LINE_MIN_LENGTH
        and len(stripped.split()) > 1
    )


def _reclaim_margin_lines_the_deposit_places_elsewhere(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """A margin line the deposit carries as content is content.

    A table prints the same note under each of its pages -- "Abbreviation - PD:
    Peritoneal Dialysis." -- and the aligner places most of them and loses one
    to the ambiguity.  The one it loses is set where the footer zone begins and
    reads as a running foot.  What the document itself says is that this text is
    the table's: the aligner grounded it elsewhere under a region label, and no
    running foot is ever grounded anywhere.
    """
    label_by_text: Dict[str, str] = {}
    for seg_line in seg_lines:
        if (
            seg_line.seg_label is not None
            and seg_line.seg_label not in _FURNITURE_LABELS
            and _is_grounded(seg_line, annotated)
            and _says_enough_to_be_recognised(seg_line.text)
        ):
            label_by_text.setdefault(seg_line.text, seg_line.seg_label)
    running = _get_running_texts(seg_lines)
    for seg_line in seg_lines:
        if seg_line.seg_label != SEG_FOOTNOTE or seg_line.text in running:
            continue
        label = label_by_text.get(seg_line.text)
        if label is not None:
            seg_line.seg_label = label


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


def _resumes_on_the_next_page(
    seg_lines: List[_SegLine], previous: List[int], run: List[int]
) -> bool:
    """Whether the run carries the previous one over a page break."""
    ends = _get_page_number(seg_lines[previous[-1]])
    starts = _get_page_number(seg_lines[run[0]])
    return ends is not None and starts == ends + 1


def _extend_region_to_page_start(
    seg_lines: List[_SegLine],
    label: str,
    annotated: JatsAnnotatedLayoutDocument,
    including_runs_resuming_overleaf: bool = False,
) -> None:
    """Start a region at the top of the page its first line is on.

    A peer-review section is headed by its title, the report's date and the
    report's licence, none of which the JATS carries, so the region began
    partway down the page the section starts on.  Where nothing above it on that
    page is evidenced, the page break is the better boundary than the first line
    the aligner could ground.

    One evidenced line above stops it, because that is a page whose top still
    belongs to whatever came before.

    The first run of the region, and -- for the front matter, which is the only
    region with a column that runs overleaf -- any that carries on from the page
    before: an affiliation column set down the side of the title page continues
    at the top of the next one, and what the page breaks there is a column
    rather than a region.  A run that starts further into the document stands on
    its own and takes nothing above it.
    """
    runs = list(_iter_label_runs(seg_lines, label))
    for position, run in enumerate(runs):
        first = run[0]
        page_number = _get_page_number(seg_lines[first])
        if page_number is None:
            continue
        if position and not (
            including_runs_resuming_overleaf
            and _resumes_on_the_next_page(seg_lines, runs[position - 1], run)
        ):
            continue
        start = first
        while start > 0 and _get_page_number(seg_lines[start - 1]) == page_number:
            start -= 1
        span = seg_lines[start:first]
        if any(_is_grounded(sl, annotated) for sl in span):
            continue
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

# What a journal calls the notes it gathers at the end of the article.
_NOTE_HEADINGS = frozenset(heading.replace(' ', '') for heading in (
    'note', 'notes', 'endnotes', 'end notes',
    'nota', 'notas', 'notas finais', 'notas de rodape', 'notas de rodapé',
    'notes de fin', 'anmerkungen',
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


# "Keywords:", "Palabras clave:" -- the word or two the page labels a field
# with, before the colon that separates it from the values.
_FRONT_FIELD_LABEL_PATTERN = re.compile(
    r'^\s*([^\W\d_][^\W\d_\s]*(?:\s+[^\W\d_][^\W\d_\s]*)?)\s*:'
)


def _front_field_label(text: str) -> Optional[str]:
    match = _FRONT_FIELD_LABEL_PATTERN.match(text)
    return match.group(1).lower() if match else None


def _claim_repeated_front_field(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Take the second copy of a front-matter field the gold carries once.

    A preprint prints its keywords in both of the languages it is written in,
    and the two lists do not always hold the same terms, so the JATS can carry
    one of them and leave the other unmatched -- directly under the front
    matter, at the head of the page the argument begins on, where it reads as
    the argument beginning.  It opens with the word the matched list opens
    with, and that is what says it is the same field over again.
    """
    for run in _iter_label_runs(seg_lines, SEG_FRONT):
        labels = {
            label
            for label in (
                _front_field_label(seg_lines[index].text)
                for index in run
                if _is_grounded(seg_lines[index], annotated)
            )
            if label
        }
        if not labels:
            continue
        position = run[-1] + 1
        while (
            position < len(seg_lines)
            and seg_lines[position].seg_label in _FURNITURE_LABELS
        ):
            position += 1
        if position >= len(seg_lines):
            continue
        seg_line = seg_lines[position]
        if seg_line.seg_label is not None or _is_grounded(seg_line, annotated):
            continue
        if _front_field_label(seg_line.text) not in labels:
            continue
        seg_line.seg_label = SEG_FRONT
        while position + 1 < len(seg_lines):
            following = seg_lines[position + 1]
            if (
                following.seg_label is not None
                or _is_grounded(following, annotated)
                or not _follows_on_from(following, seg_lines[position])
            ):
                break
            following.seg_label = SEG_FRONT
            position += 1


def _release_front_alone_at_the_page_top(seg_lines: List[_SegLine]) -> None:
    """End the front matter where the page the body starts on begins.

    The front matter does not resume for one line at the top of a page and
    then give way to the argument: where it reaches that far, the line is the
    body's own opening, matched against the front because the gold ran two
    values together -- a keyword list whose last term is the heading that
    follows it, with no separator between them.

    Over the reviewed corpus 55 front runs end in a line short enough to read
    as a heading and one of them stands alone at the top of its page, which is
    the one this is about.
    """
    for run in _iter_label_runs(seg_lines, SEG_FRONT):
        last = seg_lines[run[-1]]
        page_number = _get_page_number(last)
        if page_number is None or len(run) < 2:
            continue
        if _get_page_number(seg_lines[run[-2]]) == page_number:
            continue
        following = next(
            (
                sl for sl in seg_lines[run[-1] + 1:]
                if sl.seg_label not in _FURNITURE_LABELS
            ),
            None,
        )
        if following is None or following.seg_label != SEG_BODY:
            continue
        if _get_page_number(following) != page_number:
            continue
        last.seg_label = SEG_BODY


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


# A heading is set above the region it introduces, within the space a blank
# line and a line of type take up.
_HEADING_ABOVE_MAX_GAP = 60.0


def _claim_region_heading_by_position(
    seg_lines: List[_SegLine],
    label: str,
    headings: FrozenSet[str],
) -> None:
    """Take the heading the page sets above a region that reading order puts elsewhere.

    A two-column page whose reference list runs the full width underneath is
    read column by column first, so the heading printed directly above the list
    arrives sixteen lines before it with two other regions in between, and the
    backward scan in reading order never gets near it.  What the page says is
    unambiguous: the heading sits in the same column, just above the first line
    of the list.
    """
    # Where the region sits is a question about the page, so it is asked of the
    # page: the aligner places a reference list as many runs with gaps between
    # them, and the line the heading stands over is the highest of them rather
    # than the first in reading order.
    highest_by_page: Dict[int, Tuple[float, float]] = {}
    for seg_line in seg_lines:
        if seg_line.seg_label != label:
            continue
        page_number = _get_page_number(seg_line)
        position = _get_line_position(seg_line)
        if page_number is None or position is None:
            continue
        highest = highest_by_page.get(page_number)
        if highest is None or position[1] < highest[1]:
            highest_by_page[page_number] = position

    for seg_line in seg_lines:
        if seg_line.seg_label is not None:
            continue
        if _normalised_heading(seg_line.text) not in headings:
            continue
        page_number = _get_page_number(seg_line)
        position = _get_line_position(seg_line)
        if page_number is None or position is None:
            continue
        highest = highest_by_page.get(page_number)
        if highest is None:
            continue
        if not 0 < highest[1] - position[1] <= _HEADING_ABOVE_MAX_GAP:
            continue
        if abs(position[0] - highest[0]) > _SAME_COLUMN_MAX_X_GAP:
            continue
        seg_line.seg_label = label


# The page sets a statement's heading in bold where the JATS carries no title
# for it, which is most of the time.
_STATEMENT_LABELS = (
    SEG_CONFLICT, SEG_CONTRIBUTION, SEG_FUNDING, SEG_AVAILABILITY,
    SEG_ACKNOWLEDGEMENT,
)

# What the deposit names a region of its own with -- the reference list among
# them, which is the one thing on these pages that is nobody's declaration.
# The two notes a deposit declares a statement as belong here as well, though
# they are filed with the front matter: the deposit has already said what the
# line is.
_DECLARED_REGION_FIELDS = frozenset(
    field for field, label in SEGMENTATION_LABEL_BY_FIELD.items()
    if label in _STATEMENT_LABELS + (SEG_ANNEX, SEG_REFERENCES)
) | {
    JatsFieldNames.CONFLICT_NOTE,
    JatsFieldNames.CONTRIBUTION_NOTE,
}

_STATEMENT_HEADING_MAX_LENGTH = 80


def _is_bold(seg_line: _SegLine) -> bool:
    token = seg_line.first_token
    return token is not None and token.font is not None and bool(token.font.is_bold)


def _claim_statement_heading(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Take the printed heading a statement section is introduced by.

    A JATS carries a conflict or a contribution statement as a bare paragraph
    more often than as a titled section, so the line the page sets above it has
    nothing to match and the region starts one line late.  Bold is what decides
    it, and only against the region below: a statement set in the same type as
    its heading has no heading to take.

    A heading also prints once.  "Publisher Full Text" is set in bold under
    every reference the list carries, and one of those happens to be the line
    above the availability statement.
    """
    text_counts: Counter = Counter(seg_line.text for seg_line in seg_lines)

    def is_heading_of(above: _SegLine, first: _SegLine) -> bool:
        if above.seg_label is not None or _is_grounded(above, annotated):
            return False
        if len(above.text.strip()) > _STATEMENT_HEADING_MAX_LENGTH:
            return False
        return (
            text_counts[above.text] == 1
            and _is_bold(above)
            and not _is_bold(first)
        )

    for label in _STATEMENT_LABELS:
        for run in _iter_label_runs(seg_lines, label):
            if run[0] == 0:
                continue
            above = seg_lines[run[0] - 1]
            if is_heading_of(above, seg_lines[run[0]]):
                above.seg_label = label


_STATEMENT_LABEL_BY_KIND = {
    'conflict': SEG_CONFLICT,
    'contribution': SEG_CONTRIBUTION,
    'funding': SEG_FUNDING,
    'availability': SEG_AVAILABILITY,
    'acknowledgement': SEG_ACKNOWLEDGEMENT,
}


def _starts_a_column_of_its_own(seg_lines: List[_SegLine], index: int) -> bool:
    """Whether the page sets this line where the one above it does not reach."""
    if index == 0:
        return False
    position = _get_line_position(seg_lines[index])
    above = _get_line_position(seg_lines[index - 1])
    if position is None or above is None:
        return False
    return abs(position[0] - above[0]) > _SAME_COLUMN_MAX_X_GAP


def _declarations_are_beside_it(seg_lines: List[_SegLine], index: int) -> bool:
    """Whether the run of declarations the article closes with is on either side.

    Below as well as above, because the run has a first declaration as well as a
    last and that one has the argument above it -- but only where the page sets
    the heading in a column of its own.  A heading the argument runs straight
    into is the argument naming a declaration rather than making one.
    """
    if _enclosing_label(seg_lines, index, -1) in _STATEMENT_LABELS:
        return True
    return (
        _enclosing_label(seg_lines, index, 1) in _STATEMENT_LABELS
        and _starts_a_column_of_its_own(seg_lines, index)
    )


def _claim_statement_by_heading(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Take a statement whose text the aligner matched somewhere else entirely.

    A deposit cover declares that the manuscript states its conflicts of
    interest, in much the words the statement itself uses, and the aligner can
    take the cover's sentence for the article's.  The statement then prints
    with nothing matched against it at all.

    The page still says what it is: a bold heading naming a declaration, in
    the run of declarations the article closes with.  That last part is what
    keeps it out of the argument, where a paragraph may well open by naming
    one.
    """
    text_counts: Counter = Counter(seg_line.text for seg_line in seg_lines)
    for index, seg_line in enumerate(seg_lines):
        if seg_line.seg_label is not None or _is_grounded(seg_line, annotated):
            continue
        text = seg_line.text.strip()
        if len(text) > _STATEMENT_HEADING_MAX_LENGTH or text_counts[seg_line.text] > 1:
            continue
        if not _is_bold(seg_line):
            continue
        label = _STATEMENT_LABEL_BY_KIND.get(match_fn_keywords(text) or '')
        if label is None or not _declarations_are_beside_it(seg_lines, index):
            continue
        for position in range(index, len(seg_lines)):
            if seg_lines[position].seg_label in _FURNITURE_LABELS:
                continue
            if position > index and seg_lines[position].seg_label is not None:
                break
            seg_lines[position].seg_label = label


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
    run's own column of its own page.  On a page set in columns, to where the
    run is set as well as to the column it was put in: a narrow column of
    affiliations beside two of text reads as two columns, and the affiliations
    share the first of them with the text column next to it.  A page set in one
    column has no such second reading, and there a line further along the row --
    the DOI closing a reference -- is the same column still.  A running foot is
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
            if (
                len(column_edges) > 1
                and edge_position is not None and position is not None
                and abs(position[0] - edge_position[0]) > _SAME_COLUMN_MAX_X_GAP
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


# Front matter a paper may print after its reference list.  A correspondence
# block names who to write to, which a preprint server sets at the end, and an
# author note carries the rest.  A journal in several languages closes with
# the abstract in each of them, a page of front matter at a time.
_LATE_FRONT_FIELDS = frozenset({
    JatsFieldNames.ABSTRACT,
    JatsFieldNames.AUTHOR_NOTES,
    JatsFieldNames.CORRESPONDENCE,
    JatsFieldNames.SUBJECT_CLASSIFICATION,
})


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


_NOTE_BLOCK_MIN_LINES = 3


def _claim_the_note_a_block_of_notes_opens_with(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Keep the line a block of notes opens with in the block.

    A note opens on the words the paragraph it annotates uses, so the aligner
    can match its first line to the argument as well as to the note, and the
    vote then splits that line off the block the page sets it in.  What the
    rest of the block is settles it.
    """
    by_block: Dict[int, List[_SegLine]] = {}
    for seg_line in seg_lines:
        by_block.setdefault(seg_line.block_index, []).append(seg_line)
    for block in by_block.values():
        if len(block) < _NOTE_BLOCK_MIN_LINES or block[0].seg_label == SEG_FOOTNOTE:
            continue
        if not all(seg_line.seg_label == SEG_FOOTNOTE for seg_line in block[1:]):
            continue
        if _is_grounded_in(block[0], annotated, _DEPOSITED_NOTE_FIELDS):
            block[0].seg_label = SEG_FOOTNOTE


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


_TRAILING_MATTER_LABELS = _FURNITURE_LABELS | {SEG_COVER}


# A note in the other column sits in the same band as the one already found,
# give or take the leading between two lines of it.
_FOOTNOTE_BAND_TOLERANCE = 2.0


def _claim_footnotes_sharing_a_band(seg_lines: List[_SegLine]) -> None:
    """Take the note the other column sets beside a note already placed.

    A two-column page sets its footnotes under both columns, in one band
    across the foot of the page, but reading order takes the right column's
    notes with the right column's text -- twenty-five lines before the left
    column's notes arrive.  The page says they are one block: the same band,
    in the same size of type, which is smaller than the text above them.
    """
    bands: Dict[int, Tuple[float, float, Set[float]]] = {}
    for seg_line in seg_lines:
        if seg_line.seg_label != SEG_FOOTNOTE:
            continue
        page_number = _get_page_number(seg_line)
        position = _get_line_position(seg_line)
        font_size = _get_line_font_size(seg_line)
        if page_number is None or position is None or font_size is None:
            continue
        top, bottom, sizes = bands.get(page_number, (position[1], position[1], set()))
        bands[page_number] = (
            min(top, position[1]), max(bottom, position[1]), sizes | {font_size}
        )

    for seg_line in seg_lines:
        if seg_line.seg_label is not None:
            continue
        band = bands.get(_get_page_number(seg_line) or -1)
        position = _get_line_position(seg_line)
        if band is None or position is None:
            continue
        top, bottom, sizes = band
        if not top - _FOOTNOTE_BAND_TOLERANCE <= position[1] <= bottom + _FOOTNOTE_BAND_TOLERANCE:
            continue
        if _get_line_font_size(seg_line) not in sizes:
            continue
        seg_line.seg_label = SEG_FOOTNOTE


def _get_block_type_size(block: List[_SegLine]) -> Optional[float]:
    """The size the block is mostly set in, a superscript marker aside."""
    sizes = [size for size in map(_get_line_font_size, block) if size is not None]
    return max(sizes) if sizes else None


def _iter_page_foot_blocks(
    seg_lines: List[_SegLine], running_texts: Set[str]
) -> Iterator[List[_SegLine]]:
    """The block each page sets below everything else, the margins aside."""
    by_page: Dict[int, List[_SegLine]] = {}
    for seg_line in seg_lines:
        page_number = _get_page_number(seg_line)
        if page_number is None or not seg_line.text.strip():
            continue
        if seg_line.text in running_texts:
            continue
        if seg_line.seg_label in {SEG_HEADNOTE, SEG_PAGE}:
            continue
        by_page.setdefault(page_number, []).append(seg_line)
    for page_lines in by_page.values():
        last = page_lines[-1].block_index
        yield [sl for sl in page_lines if sl.block_index == last]


def _claim_the_notes_a_page_sets_below_its_text(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """Take a block set under the text where the document sets its notes.

    A note broken over a page break is one value with the page furniture
    between its halves, so neither half matches it and the body above reaches
    over both.  The page still says what they are: set last, under the text,
    in the type the notes the deposit does carry are set in, and smaller than
    the type that page sets its own text in -- which a reference list, set
    small down a page of nothing else, is not.
    """
    blocks = list(_iter_page_foot_blocks(seg_lines, _get_running_texts(seg_lines)))
    # The notes the deposit carries, never a running foot: that repeats in the
    # same small type at the foot of every page and is nobody's note.
    note_sizes = {
        size
        for block in blocks
        if all(sl.seg_label == SEG_FOOTNOTE for sl in block)
        and any(_is_grounded(sl, annotated) for sl in block)
        for size in [_get_block_type_size(block)]
        if size is not None
    }
    for block in blocks:
        if any(sl.seg_label == SEG_FOOTNOTE for sl in block):
            continue
        if any(_is_grounded(sl, annotated) for sl in block):
            continue
        size = _get_block_type_size(block)
        if size not in note_sizes:
            continue
        page_size = _get_prevailing_type_size(
            [sl for sl in seg_lines if _get_page_number(sl) == _get_page_number(block[0])]
        )
        if page_size is None or size >= page_size:
            continue
        for seg_line in block:
            seg_line.seg_label = SEG_FOOTNOTE


def _get_prevailing_type_size(seg_lines: List[_SegLine]) -> Optional[float]:
    """The size most of the document's lines are set in."""
    sizes = Counter(
        size for size in map(_get_line_font_size, seg_lines) if size is not None
    )
    return sizes.most_common(1)[0][0] if sizes else None


def _is_set_smaller_than(seg_line: _SegLine, neighbour: _SegLine) -> bool:
    """Whether the page sets the line in smaller type than its neighbour."""
    size = _get_line_font_size(seg_line)
    neighbour_size = _get_line_font_size(neighbour)
    return size is not None and neighbour_size is not None and size < neighbour_size


def _nearest_line_above_in_the_column(
    seg_lines: List[_SegLine], index: int
) -> Optional[int]:
    """The line the page sets directly above this one in the same column.

    Reading order is not the page's vertical order once a table is in it: the
    line handed over before a row's first cell is the last cell of the row
    above, three columns to the right.
    """
    position = _get_line_position(seg_lines[index])
    page_number = _get_page_number(seg_lines[index])
    if position is None:
        return None
    nearest: Optional[int] = None
    nearest_y = float('-inf')
    for other in range(index):
        other_position = _get_line_position(seg_lines[other])
        if other_position is None or _get_page_number(seg_lines[other]) != page_number:
            continue
        if other_position[1] >= position[1] or other_position[1] <= nearest_y:
            continue
        if abs(other_position[0] - position[0]) > _SAME_COLUMN_MAX_X_GAP:
            continue
        nearest, nearest_y = other, other_position[1]
    return nearest


def _release_margin_lines_continuing_a_column(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """A margin line the column above runs straight into is not a running foot.

    A publisher who sets the affiliations down the side of the title page runs
    them past the footer threshold, and a block lying beyond it entirely reads
    as furniture.  So does the last row of a table whose rows the page carries
    over the threshold one at a time.  What the page says instead is that it
    goes on with the line above it, in the same column at the pitch of the lines
    around it.  Released rather than relabelled, because the column it continues
    may not be labelled yet either -- the gap merge is what settles both.

    A foot the document repeats is left alone: repeating across the pages is
    what a running foot does, and no reading of one line can say otherwise.  So
    is a foot that goes on with another one -- a note wrapping onto a second
    line is a note still, and only the column the page sets above the block can
    say the block is not one -- whether it is the one handed over before it or
    the one the page sets above it.  So is a foot set in smaller type than the
    column: a translator's note comes one line's leading under the paragraph it
    annotates, and the type is the only thing that separates them.  Only where
    the line carries enough to be a note -- the marker above an affiliation is
    a superscript, smaller than everything, and says nothing by being so.  And
    a line the deposit carries as a note of the article's is one wherever the
    page sets it, which is what a journal that runs its notes down the outer
    margin needs: the column beside them is not the column above them.
    """
    running = _get_running_texts(seg_lines)
    released: Set[int] = set()
    for index in range(1, len(seg_lines)):
        seg_line = seg_lines[index]
        if seg_line.seg_label != SEG_FOOTNOTE or seg_line.text in running:
            continue
        if _is_grounded_in(seg_line, annotated, _DEPOSITED_NOTE_FIELDS):
            continue
        if (
            seg_lines[index - 1].seg_label in _FURNITURE_LABELS
            and index - 1 not in released
        ):
            continue
        above_index = _nearest_line_above_in_the_column(seg_lines, index)
        if above_index is None:
            continue
        above = seg_lines[above_index]
        if above.seg_label in _FURNITURE_LABELS and above_index not in released:
            continue
        if (
            _says_enough_to_be_recognised(seg_line.text)
            and _is_set_smaller_than(seg_line, above)
        ):
            continue
        if _follows_on_from(seg_line, above):
            seg_line.seg_label = None
            released.add(index)


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
    # A preprint server appends its deposit cover to the end of the file, so the
    # last line of the document is not the last line of the article: a region
    # running to the foot of the article has the cover beyond it and would
    # otherwise stop at the page before.
    tail = len(seg_lines)
    while tail and seg_lines[tail - 1].seg_label in _TRAILING_MATTER_LABELS:
        tail -= 1

    candidate_gap: List[_SegLine] = []
    prev_label: Optional[str] = SEG_FRONT
    for sl in seg_lines[:tail]:
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


def get_cover_page_line_ids(layout_document: LayoutDocument) -> Set[int]:
    """The lines of a page the publisher wraps the article in, by `id(LayoutLine)`.

    The notice the server opens such a page with is text, so this holds before
    anything has been aligned -- which is what the aligner needs, because the
    cover restates the article's title and authors and a value is matched once.
    Matched on the cover, the article's own front matter has nothing left to
    match it, and the header region is cut away from the cover.

    `derive_labels` runs the same test again over the labels the aligner then
    produces, where it can also see a page carrying article content and a cover
    running onto a second page. Neither is available here, and neither is needed
    to say that a page opens with a deposit notice.
    """
    seg_lines = _build_seg_lines(layout_document)
    return {
        id(seg_line.layout_line)
        for page_lines in _get_seg_lines_by_page_number(seg_lines).values()
        if _opens_a_deposit_page(page_lines)
        for seg_line in page_lines
    }


# What tells a column the page sets beside the article from the article's own.
# The gutter measured between ORE's panel and the column beside it is 11 points,
# and the rule reads the same over the twenty documents anywhere from 8 to 10.
_SIDE_COLUMN_MIN_GUTTER = 10.0
# A panel is narrower than what it is set beside.  Two columns of body text are
# the same width, which is what this keeps the rule away from.
_SIDE_COLUMN_MAX_WIDTH_SHARE = 0.7
_SIDE_COLUMN_MIN_LINES = 3
_SIDE_COLUMN_MIN_Y_OVERLAP = 50.0


def _get_line_box(seg_line: _SegLine) -> Optional[Tuple[float, float, float]]:
    """Where a line starts, where it ends and how far down the page it sits."""
    boxes = [
        token.coordinates for token in (seg_line.layout_line.tokens or [])
        if token.coordinates is not None and token.coordinates
    ]
    if not boxes:
        return None
    return (
        min(box.x for box in boxes),
        max(box.x + box.width for box in boxes),
        min(box.y for box in boxes),
    )


def _iter_side_column_lines(
    page_lines: List[Tuple[_SegLine, Tuple[float, float, float]]]
) -> Iterator[_SegLine]:
    """The lines of the narrower of two columns the page sets side by side.

    Read from the lines themselves rather than from where they start: a page is
    full width at the top and splits lower down, so the question is asked of
    each band in turn -- below this line, does every line fall on one side of a
    gutter or the other?
    """
    for band_start in sorted({box[2] for _, box in page_lines}):
        below = [(line, box) for line, box in page_lines if box[2] >= band_start]
        if len(below) < 2 * _SIDE_COLUMN_MIN_LINES:
            continue
        for boundary in sorted({box[1] for _, box in below}):
            left = [(line, box) for line, box in below if box[1] <= boundary]
            right = [(line, box) for line, box in below if box[0] > boundary]
            if len(left) + len(right) != len(below):
                continue
            if min(len(left), len(right)) < _SIDE_COLUMN_MIN_LINES:
                continue
            gutter = min(box[0] for _, box in right) - max(box[1] for _, box in left)
            if gutter < _SIDE_COLUMN_MIN_GUTTER:
                continue
            left_width = max(b[1] for _, b in left) - min(b[0] for _, b in left)
            right_width = max(b[1] for _, b in right) - min(b[0] for _, b in right)
            if (
                min(left_width, right_width) / max(left_width, right_width)
                > _SIDE_COLUMN_MAX_WIDTH_SHARE
            ):
                continue
            left_y = [box[2] for _, box in left]
            right_y = [box[2] for _, box in right]
            overlap = min(max(left_y), max(right_y)) - max(min(left_y), min(right_y))
            if overlap < _SIDE_COLUMN_MIN_Y_OVERLAP:
                continue
            for line, _box in (right if right_width < left_width else left):
                yield line
            return


def _get_line_band(line: LayoutLine) -> Optional[Tuple[float, float]]:
    """How far down the page a line starts and ends."""
    boxes = [
        token.coordinates for token in (line.tokens or [])
        if token.coordinates is not None and token.coordinates
    ]
    if not boxes:
        return None
    return (
        min(box.y for box in boxes),
        max(box.y + box.height for box in boxes),
    )


def _get_block_band(lines: List[LayoutLine]) -> Optional[Tuple[float, float]]:
    """How far down the page a block of lines starts and ends."""
    bands = [_get_line_band(line) for line in lines]
    if not bands or any(band is None for band in bands):
        return None
    found = [band for band in bands if band is not None]
    return min(band[0] for band in found), max(band[1] for band in found)


def _get_block_sides(lines: List[LayoutLine]) -> Optional[Tuple[float, float]]:
    """How far across the page a block of lines reaches."""
    boxes = [
        token.coordinates for line in lines for token in (line.tokens or [])
        if token.coordinates is not None
    ]
    if not boxes:
        return None
    return min(box.x for box in boxes), max(box.x + box.width for box in boxes)


def _overlap(one: Tuple[float, float], other: Tuple[float, float]) -> bool:
    return one[0] < other[1] and other[0] < one[1]


def _blocks_in_the_order_the_page_sets_them(
    bands: List[Tuple[float, float]], sides: List[Tuple[float, float]]
) -> List[int]:
    """The blocks placed by where the page sets them, as far as it says.

    Sharing the width is what says a block has been handed over late: a block
    set wholly above one it shares the width with was printed before it,
    whereas a panel set beside a column says nothing about which of them the
    page sets first -- read down the page alone the two are shuffled into each
    other.  Where the page says nothing the order the blocks came in stands,
    which is also what keeps two columns of one text in the order they were
    read.

    Where it does say so, the block goes above everything already placed that
    the page sets below it, whatever width that was set in.  `ore` sets its
    reference list in two columns under two columns of text: the second column
    of text shares its width with the second column of the list, which is what
    says it was handed over late, and it belongs above the whole list rather
    than between its halves.
    """
    placed: List[int] = []
    for index, band in enumerate(bands):
        handed_over_late = any(
            _overlap(sides[index], sides[other]) and bands[other][0] >= band[1]
            for other in placed
        )
        below = next(
            (
                position for position, other in enumerate(placed)
                if handed_over_late and bands[other][0] >= band[1]
            ),
            len(placed),
        )
        placed.insert(below, index)
    return placed


# How much a block has to carry before the page is read as setting it
# somewhere.  One line says less than a block does -- a heading, a standing
# line, an identifier -- so it goes where the block it waits for goes.
_MIN_LINES_TO_PLACE_A_BLOCK = 2


def _lies_in_a_margin(band: Tuple[float, float], height: Optional[float]) -> bool:
    """Whether the page sets the block above or below everything it prints.

    Where its middle falls, not where it begins or ends: a standing line at the
    foot of a page opens a line above the threshold, and a column of body text
    runs a line below it.
    """
    if not height:
        return False
    middle = (band[0] + band[1]) / 2
    return middle < height * _HEADNOTE_Y_RATIO or middle > height * _FOOTNOTE_Y_RATIO


class _PlaceableBlocks(NamedTuple):
    lines: List[List[LayoutLine]]
    bands: List[Tuple[float, float]]
    sides: List[Tuple[float, float]]


def _get_page_height(page: LayoutPage) -> Optional[float]:
    coordinates = page.meta.coordinates if page.meta else None
    return coordinates.height if coordinates else None


def _get_placeable_blocks(
    page: LayoutPage, furniture_line_ids: Set[int]
) -> _PlaceableBlocks:
    """The page's content blocks, each with the band and the width it is set in."""
    height = _get_page_height(page)
    blocks = _PlaceableBlocks([], [], [])
    waiting: List[LayoutLine] = []
    for block in page.blocks:
        block_lines = [
            line for line in block.lines if id(line) not in furniture_line_ids
        ]
        band = _get_block_band(block_lines) if block_lines else None
        side = _get_block_sides(block_lines) if block_lines else None
        if band is None or side is None or _lies_in_a_margin(band, height):
            continue
        if len(block_lines) < _MIN_LINES_TO_PLACE_A_BLOCK:
            waiting.extend(block_lines)
            continue
        blocks.lines.append(waiting + block_lines)
        blocks.bands.append(band)
        blocks.sides.append(side)
        waiting = []
    if waiting and blocks.lines:
        blocks.lines[-1].extend(waiting)
    return blocks


def _page_lines_in_printed_order(
    page: LayoutPage, furniture_line_ids: Set[int]
) -> List[LayoutLine]:
    """One page's lines, its content moved into the order the page sets it."""
    page_lines = [line for block in page.blocks for line in block.lines]
    content, bands, sides = _get_placeable_blocks(page, furniture_line_ids)
    if not content:
        return page_lines
    placed = {id(line) for block_lines in content for line in block_lines}
    positions = [
        index for index, line in enumerate(page_lines) if id(line) in placed
    ]
    reordered = [
        line for index in _blocks_in_the_order_the_page_sets_them(bands, sides)
        for line in content[index]
    ]
    for index, line in zip(positions, reordered):
        page_lines[index] = line
    return page_lines


def get_lines_in_printed_order(
    layout_document: LayoutDocument, furniture_line_ids: Set[int]
) -> List[LayoutLine]:
    """The document's lines in the order the page sets them.

    pdfalto reads a page column by column, and a block running the full width
    under two columns belongs to neither: `ore` hands over the metadata block at
    the foot of its second page before the column above it, so a thousand
    characters stand between the two halves of an abstract that the page sets
    nothing between.  Another journal hands over the affiliations under its
    byline after the whole of the introduction.

    A block is placed before one the page sets wholly below it, and only where
    the two share the width they are set in.  Two columns share no width, and
    neither does a panel set beside a column, so the order those came in
    stands; a block set under the full width of the page shares it with
    everything, and there the page says which it sets first.

    Of the page's content, and a block the page sets in a margin is not that:
    a running head, a page number or a journal's standing line is printed
    around the page rather than in it, and any of them would deny the page the
    question -- `ore` sets its page number below the block that is out of
    place.  They keep the place they were handed over in; nothing reads them.
    """
    lines: List[LayoutLine] = []
    for page in layout_document.pages:
        lines.extend(_page_lines_in_printed_order(page, furniture_line_ids))
    return lines


def get_side_column_line_ids(layout_document: LayoutDocument) -> Set[int]:
    """The lines of a column the page sets beside the one the article runs down.

    `ore` sets its peer review panel down the side of the first page.  The panel
    is nobody's value, so nothing else says where it is, and a title or an
    abstract that runs down the column beside it can take a line of it by
    coincidence -- `Open Peer` reads enough like the words of an abstract for a
    few characters to match there.
    """
    seg_lines = _build_seg_lines(layout_document)
    side_column_line_ids: Set[int] = set()
    for page_lines in _get_seg_lines_by_page_number(seg_lines).values():
        with_boxes = [
            (seg_line, box) for seg_line, box in (
                (seg_line, _get_line_box(seg_line)) for seg_line in page_lines
            ) if box is not None
        ]
        side_column_line_ids.update(
            id(seg_line.layout_line) for seg_line in _iter_side_column_lines(with_boxes)
        )
    return side_column_line_ids


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


# What a declaration the JATS files with the front matter is, where the page
# prints it after the article instead.  `<annex>` is what GROBID gives a
# declaration it has no label for: 60 of the 60 segmentation documents opening
# an ethics or consent statement hold it under that type.
_BACK_LABEL_BY_FRONT_FIELD = {
    JatsFieldNames.CONFLICT_NOTE: SEG_CONFLICT,
    JatsFieldNames.CONTRIBUTION_NOTE: SEG_CONTRIBUTION,
    JatsFieldNames.FUNDING: SEG_FUNDING,
    JatsFieldNames.AUTHOR_NOTES: SEG_ANNEX,
}

# `ARTICLE_NOTES` is not among them.  A revised preprint prints the box saying
# what changed after its body, and `_claim_amendments_box` has already decided
# that the box is the publisher's front matter.

# The front matter a paper does print after its body: a preprint server sets the
# correspondence block at the end, and it is the front matter wherever it is set.
_FRONT_AFTER_THE_BODY_FIELDS = frozenset({
    JatsFieldNames.CORRESPONDENCE,
    JatsFieldNames.CORRESP_AFF,
    JatsFieldNames.CORRESP_ADDRESS,
    JatsFieldNames.CORRESP_PHONE,
})


def _declaration_named_by(text: str) -> Optional[str]:
    """The declaration a line heads, where it is a heading that names one."""
    heading = text.strip()
    if len(heading) > _STATEMENT_HEADING_MAX_LENGTH:
        return None
    return _STATEMENT_LABEL_BY_KIND.get(match_fn_keywords(heading) or '')


def _names_another_declaration(text: str, label: str) -> bool:
    """Whether the line heads a declaration other than the one running on."""
    named = _declaration_named_by(text)
    return named is not None and named != label


def _release_front_after_the_body(
    seg_lines: List[_SegLine], annotated: JatsAnnotatedLayoutDocument
) -> None:
    """A declaration the page prints after the article is not the front matter.

    Where the page sets a declaration is what decides, which is the rule
    `_release_statements_inside_front` reads from the other end.

    Only a line the JATS names a declaration: the front matter itself can print
    past the first line of body, where an affiliation column runs onto the page
    the article opens on.  A run the declarations close over is let go
    rather than relabelled, the regions claiming it on their own terms -- a
    table printed among them, and the byline the aligner reads into its
    caption.  By the run rather than the line, because a correspondence block
    printed after the declarations opens with a heading of its own that nothing
    grounds, and what keeps the block is what keeps its heading.
    """
    body_start = next(
        (
            index for index, seg_line in enumerate(seg_lines)
            if seg_line.seg_label == SEG_BODY
        ),
        None,
    )
    if body_start is None:
        return

    def fields_of(seg_line: _SegLine) -> Set[Optional[str]]:
        return {
            annotated.get_token_field(token)
            for token in seg_line.layout_line.tokens
        }

    for seg_line in seg_lines[body_start:]:
        if seg_line.seg_label != SEG_FRONT:
            continue
        label = next(
            (
                _BACK_LABEL_BY_FRONT_FIELD[field] for field in fields_of(seg_line)
                if field in _BACK_LABEL_BY_FRONT_FIELD
            ),
            None,
        )
        if label is not None:
            seg_line.seg_label = label
    back_labels = set(_BACK_LABEL_BY_FRONT_FIELD.values()) | {SEG_CONTRIBUTION}
    for run in list(_iter_label_runs(seg_lines, SEG_FRONT)):
        if run[0] < body_start:
            continue
        if any(
            fields_of(seg_lines[index]) & _FRONT_AFTER_THE_BODY_FIELDS
            for index in run
        ):
            continue
        above = _enclosing_label(seg_lines, run[0], -1)
        if above not in back_labels:
            continue
        # A contribution statement names each author and what they did, so the
        # byline matches inside it and carries the line into the front matter.
        # Under that heading the run is the statement's rather than nothing's;
        # GROBID's corpus has 296 contribution sections opening this way.
        names_the_authors = any(
            JatsFieldNames.AUTHOR in fields_of(seg_lines[index]) for index in run
        )
        taken = (
            SEG_CONTRIBUTION
            if above == SEG_CONTRIBUTION and names_the_authors
            else None
        )
        for index in run:
            # The statement ends where the page names the next declaration:
            # an article closes with them one after another, and the heading
            # of the one that follows is not the contributions'.
            if taken is not None and _names_another_declaration(
                seg_lines[index].text, taken
            ):
                taken = None
            seg_lines[index].seg_label = taken


def _is_across_from(block_lines: List[_SegLine], rows: List[float]) -> bool:
    """Whether the page sets the block on a row it already prints furniture on."""
    band = [
        position[1] for position in map(_get_line_position, block_lines) if position
    ]
    if not band:
        return False
    return any(
        min(band) - _SAME_ROW_Y_TOLERANCE <= row <= max(band) + _SAME_ROW_Y_TOLERANCE
        for row in rows
    )


def _claim_the_banner_above_the_front_matter(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
) -> None:
    """The block a page sets in its top margin above the front matter.

    A journal prints its licence there -- "This is an open access article under
    the CC BY license" -- and GROBID's corpus holds that sentence in `<front>`
    363 times against 91 as a headnote and 74 as a foot.  What is in the top
    margin of the page the front matter begins on belongs to it -- unless the
    page is already printing furniture across that row.  A journal that runs
    its head over the first page as well as the rest has made the row its
    margin, and the block beside the head is printed around the page rather
    than in it: S0034-737X2019000600486 sets its licence in the third corner
    of a row whose other two hold the page number and the running head.

    The block, not the line: a column whose first line starts in the zone is
    the page's own content carrying on below it.
    """
    first = next((sl for sl in seg_lines if sl.seg_label == SEG_FRONT), None)
    if first is None:
        return
    page_number = _get_page_number(first)
    page_lines = [sl for sl in seg_lines if _get_page_number(sl) == page_number]
    in_zone: Dict[int, bool] = {}
    blocks: Dict[int, List[_SegLine]] = {}
    furniture_rows: List[float] = []
    for seg_line in page_lines:
        in_zone[seg_line.block_index] = (
            in_zone.get(seg_line.block_index, True)
            and _is_in_header_zone(seg_line, page_meta_by_number, config)
        )
        if seg_line.seg_label == SEG_HEADNOTE:
            position = _get_line_position(seg_line)
            if position:
                furniture_rows.append(position[1])
        if seg_line.seg_label is None:
            blocks.setdefault(seg_line.block_index, []).append(seg_line)
    for block_index, block_lines in blocks.items():
        if not in_zone.get(block_index):
            continue
        taken = (
            SEG_HEADNOTE if _is_across_from(block_lines, furniture_rows)
            else SEG_FRONT
        )
        for seg_line in block_lines:
            seg_line.seg_label = taken


def _tag_the_margins_and_the_headings(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    annotated: JatsAnnotatedLayoutDocument,
    config: SegmentationConfig,
) -> None:
    """What the page prints around its content, and what opens a region."""
    _clear_front_beyond_threshold(
        seg_lines, config.front_max_start_line_index, annotated
    )
    _find_missing_page_numbers(seg_lines, page_meta_by_number, config)
    _tag_headnotes_by_text_repetition(
        seg_lines, config.page_header_max_first_line_index
    )
    _tag_running_feet_by_repetition(seg_lines, page_meta_by_number)
    _tag_page_numbers_by_position(seg_lines, page_meta_by_number)
    _release_furniture_inside_references(seg_lines)
    _claim_region_heading(seg_lines, SEG_REFERENCES, _REFERENCE_HEADINGS, annotated)
    _claim_region_heading_by_position(seg_lines, SEG_REFERENCES, _REFERENCE_HEADINGS)
    _claim_region_heading(seg_lines, SEG_FOOTNOTE, _NOTE_HEADINGS, annotated)
    _claim_statement_heading(seg_lines, annotated)
    _claim_statement_by_heading(seg_lines, annotated)


def _take_back_the_page_furniture(
    seg_lines: List[_SegLine],
    page_meta_by_number: Mapping[int, LayoutPageMeta],
    config: SegmentationConfig,
    annotated: JatsAnnotatedLayoutDocument,
) -> None:
    """The margins, read again over the regions the merge has settled."""
    _reclaim_repeated_headnotes(seg_lines, page_meta_by_number, config)
    _reclaim_repeated_running_feet(seg_lines, page_meta_by_number)
    _tag_running_heads_by_position(seg_lines, page_meta_by_number)
    _reclaim_margin_lines_the_deposit_places_elsewhere(seg_lines, annotated)


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
        _tag_the_margins_and_the_headings(
            seg_lines, page_meta_by_number, annotated, self.config
        )
        # `<review>` is the peer-review sub-articles, which print as one run at the end
        # of the document.  Its values are short, repeated checklist fragments in
        # an order the page does not follow, so the aligner places only some of
        # them; bridging the gaps between those it does place, and running the
        # region to the end, recovers the rest without asking the aligner for
        # per-fragment precision it cannot give.
        _bridge_footnote_block(seg_lines)
        _claim_footnotes_sharing_a_band(seg_lines)
        # Before the merge: a declaration left as a region of its own cuts the
        # front run in two, and the merge then has no run to carry across the
        # page break to the front matter printed after it.
        _release_statements_inside_front(seg_lines)
        _release_body_inside_the_references(seg_lines)
        _release_margin_lines_continuing_a_column(seg_lines, annotated)
        _claim_the_note_a_block_of_notes_opens_with(seg_lines, annotated)
        # After the release, which reads a note set at the foot of a page as the
        # column above it carrying on and lets it go.
        _claim_the_notes_a_page_sets_below_its_text(seg_lines, annotated)
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
        _extend_region_to_page_start(
            seg_lines, SEG_FRONT, annotated, including_runs_resuming_overleaf=True
        )

        _claim_repeated_front_field(seg_lines, annotated)
        _release_references_inside_availability(seg_lines)
        _release_references_inside_body(seg_lines)
        _release_body_inside_front(seg_lines)
        _release_annex_inside_body(seg_lines)

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
        _take_back_the_page_furniture(
            seg_lines, page_meta_by_number, self.config, annotated
        )

        _claim_amendments_box(seg_lines)
        _claim_annex_region(seg_lines)
        # Before the three passes below, each of which claims one thing the
        # page prints among the declarations and says what that one thing is.
        _claim_the_declarations_the_article_closes_with(seg_lines, annotated)
        _tag_publication_dates(seg_lines, annotated)
        _claim_the_licence_printed_after_the_references(seg_lines)
        _claim_lines_sharing_a_row(seg_lines, page_meta_by_number)

        _claim_the_banner_above_the_front_matter(
            seg_lines, page_meta_by_number, self.config
        )

        # After every pass that builds the front run, because what it reads is
        # the run as it ends up rather than the lines the JATS vote claimed.
        _release_front_after_the_body(seg_lines, annotated)
        # Again, because the run of declarations an article closes with is only
        # a run of declarations once the pass above has released it.
        _claim_statement_by_heading(seg_lines, annotated)
        # After the release above, which reads the citation as front matter
        # printed past the body and lets it go.
        _claim_the_citation_printed_after_the_references(seg_lines, annotated)
        # Beside the citation, and after the release for the same reason: the
        # deposit files the editor's name as a note on the authors.
        _claim_the_editor_credited_after_the_references(seg_lines)
        # And after it too, which is what decides where a declaration printed
        # in the imprint block belongs.
        _claim_the_rest_of_the_block_the_front_matter_fills(seg_lines)

        # ── Default remaining untagged lines → body ──
        for sl in seg_lines:
            if sl.seg_label is None:
                sl.seg_label = SEG_BODY

        # Last, because it reads the label of what follows the region, and what
        # follows is only settled once the default has been applied.
        _release_trailing_heading(seg_lines, SEG_REFERENCES)
        _release_front_alone_at_the_page_top(seg_lines)

        # Build id(LayoutLine) → label mapping — safe because layout_document holds strong refs
        result: Dict[int, str] = {}
        for sl in seg_lines:
            if sl.seg_label:
                result[id(sl.layout_line)] = sl.seg_label
        return result
