import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.document.page_main_area import calculate_page_main_areas
from sciencebeam_parser.utils.bounding_box import BoundingBox

LOGGER = logging.getLogger(__name__)

# Fraction of the page height at the top or bottom that names an outside-main-area
# block a running head or foot rather than a margin note
_EDGE_ZONE_FRACTION = 0.25


class _BlockOccurrence(NamedTuple):
    page_index: int
    y_relative: Optional[float]
    height_relative: Optional[float]
    block: LayoutBlock


@dataclass
class LayoutNoiseFilterConfig:
    enabled: bool = False
    repetition_fraction: float = 0.5
    # Fraction of occurrences that must fall in the top/bottom quartile zone
    position_consistency_fraction: float = 0.8
    # Max standard deviation of y_relative across qualifying occurrences
    max_position_stddev: float = 0.05
    # Occurrences whose height exceeds this multiple of the group median are not filtered
    # (catches e.g. a large title on page 1 that also repeats as a small footer)
    max_height_ratio: float = 2.0
    # Never filter running-head blocks on page 1.
    # Only needed when the page-1 title has the same height as the running head
    # (otherwise the height check already preserves it). Defaults to False because
    # setting True will also preserve genuine running-head blocks on page 1.
    preserve_first_page_head: bool = False
    # Never filter running-foot blocks on page 1 (footers have no special status on page 1)
    preserve_first_page_foot: bool = False
    # Also filter a block outside the page's main text area whose text either repeats
    # across pages as a letters-only pattern, or carries no letters at all.
    # Reaches furniture the repetition rule cannot group, such as a `Page 3 of 13`
    # footer or a bare page number, whose text differs on every page.
    filter_outside_main_area: bool = False
    # Shortest letters-only pattern that may count as repeating. `Page 3 of 13`
    # reduces to `pageof`, so GROBID's own limit of 8 is too long here.
    min_repeating_pattern_length: int = 3
    # Longest text carrying no letters that counts as furniture on position alone
    max_letterless_length: int = 12


@dataclass
class TaggedNoiseBlock:
    block: LayoutBlock
    note_type: str  # "running-head" | "running-foot" | "marginnote"


def _get_block_y_relative(block: LayoutBlock, page: LayoutPage) -> Optional[float]:
    page_height = (
        page.meta.coordinates.height
        if page.meta and page.meta.coordinates
        else None
    )
    if not page_height:
        return None
    y_values = [
        token.coordinates.y
        for token in block.iter_all_tokens()
        if token.coordinates
    ]
    if not y_values:
        return None
    return min(y_values) / page_height


def _get_block_height_relative(block: LayoutBlock, page: LayoutPage) -> Optional[float]:
    page_height = (
        page.meta.coordinates.height
        if page.meta and page.meta.coordinates
        else None
    )
    if not page_height:
        return None
    extents = [
        (token.coordinates.y, token.coordinates.y + token.coordinates.height)
        for token in block.iter_all_tokens()
        if token.coordinates
    ]
    if not extents:
        return None
    return (max(y1 for _, y1 in extents) - min(y0 for y0, _ in extents)) / page_height


def _stddev(values: List[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _compute_page_quartiles(
    page_block_y_rels: Dict[int, List[float]]
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Return (q1_per_page, q3_per_page) — 25th and 75th percentile of y_relative."""
    q1_map: Dict[int, float] = {}
    q3_map: Dict[int, float] = {}
    for page_index, ys in page_block_y_rels.items():
        if not ys:
            continue
        sorted_ys = sorted(ys)
        n = len(sorted_ys)
        q1_map[page_index] = sorted_ys[n // 4]
        q3_map[page_index] = sorted_ys[min(3 * n // 4, n - 1)]
    return q1_map, q3_map


def _in_zone(
    page_idx: int,
    y_rel: float,
    note_type: str,
    page_q1_y: Dict[int, float],
    page_q3_y: Dict[int, float],
) -> bool:
    if note_type == 'running-head':
        return page_idx in page_q1_y and y_rel < page_q1_y[page_idx]
    return page_idx in page_q3_y and y_rel > page_q3_y[page_idx]


def _classify_repetition_group(
    occurrences: List[_BlockOccurrence],
    page_q1_y: Dict[int, float],
    page_q3_y: Dict[int, float],
    consistency_fraction: float,
    max_position_stddev: float,
) -> Optional[str]:
    classifiable = sum(
        1 for occ in occurrences
        if occ.y_relative is not None and occ.page_index in page_q1_y
    )
    if not classifiable:
        return None
    for note_type in ('running-head', 'running-foot'):
        zone = [
            occ for occ in occurrences
            if occ.y_relative is not None
            and _in_zone(occ.page_index, occ.y_relative, note_type, page_q1_y, page_q3_y)
        ]
        if len(zone) / classifiable < consistency_fraction:
            continue
        y_rels = [occ.y_relative for occ in zone if occ.y_relative is not None]
        if len(y_rels) > 1 and _stddev(y_rels) > max_position_stddev:
            continue
        return note_type
    return None


def _tag_noise_occurrences(
    occurrences: List[_BlockOccurrence],
    note_type: str,
    page_q1_y: Dict[int, float],
    page_q3_y: Dict[int, float],
    max_height_ratio: float,
    preserve_first_page: bool,
) -> List[TaggedNoiseBlock]:
    zone_heights = [
        occ.height_relative for occ in occurrences
        if occ.height_relative is not None and occ.y_relative is not None
        and _in_zone(occ.page_index, occ.y_relative, note_type, page_q1_y, page_q3_y)
    ]
    median_height = sorted(zone_heights)[len(zone_heights) // 2] if zone_heights else None
    result = []
    for occ in occurrences:
        if preserve_first_page and occ.page_index == 0:
            continue
        if occ.y_relative is None:
            continue
        if not _in_zone(occ.page_index, occ.y_relative, note_type, page_q1_y, page_q3_y):
            continue
        if (median_height and occ.height_relative is not None
                and occ.height_relative > max_height_ratio * median_height):
            continue
        result.append(TaggedNoiseBlock(block=occ.block, note_type=note_type))
    return result


def _collect_blocks(
    layout_document: LayoutDocument,
) -> Tuple[Dict[str, List[_BlockOccurrence]], Dict[int, List[float]]]:
    text_to_occurrences: Dict[str, List[_BlockOccurrence]] = defaultdict(list)
    page_block_y_rels: Dict[int, List[float]] = defaultdict(list)
    for page_index, page in enumerate(layout_document.pages):
        for block in page.blocks:
            text = block.text.strip().casefold()
            if not text:
                continue
            y_rel = _get_block_y_relative(block, page)
            h_rel = _get_block_height_relative(block, page)
            text_to_occurrences[text].append(
                _BlockOccurrence(page_index, y_rel, h_rel, block)
            )
            if y_rel is not None:
                page_block_y_rels[page_index].append(y_rel)
    return dict(text_to_occurrences), dict(page_block_y_rels)


def _get_letters_only_pattern(text: str) -> str:
    return re.sub(r'[^a-zA-Z]', '', text).lower()


def _has_no_letters(text: str, max_length: int) -> bool:
    stripped = text.strip()
    return bool(stripped) and len(stripped) <= max_length and not re.search(r'[^\W\d_]', stripped)


def _get_pages_by_letters_only_pattern(
    layout_document: LayoutDocument,
    min_pattern_length: int
) -> Dict[str, Set[int]]:
    pages_by_pattern: Dict[str, Set[int]] = defaultdict(set)
    for page_index, page in enumerate(layout_document.pages):
        for block in page.blocks:
            pattern = _get_letters_only_pattern(block.text)
            if len(pattern) >= min_pattern_length:
                pages_by_pattern[pattern].add(page_index)
    return pages_by_pattern


def _is_outside_main_area(
    block: LayoutBlock,
    main_area: Optional[BoundingBox]
) -> bool:
    """Unlike the `is_main_area` feature, a page without a main area keeps all its blocks."""
    if main_area is None:
        return False
    coordinates_list = block.get_merged_coordinates_list()
    if not coordinates_list:
        return False
    return not main_area.intersection(coordinates_list[0].bounding_box)


def _get_outside_main_area_note_type(y_relative: Optional[float]) -> str:
    if y_relative is None:
        return 'marginnote'
    if y_relative < _EDGE_ZONE_FRACTION:
        return 'running-head'
    if y_relative > 1 - _EDGE_ZONE_FRACTION:
        return 'running-foot'
    return 'marginnote'


def _is_preserved_first_page(
    page_index: int,
    note_type: str,
    config: LayoutNoiseFilterConfig
) -> bool:
    if page_index != 0:
        return False
    if note_type == 'running-head':
        return config.preserve_first_page_head
    if note_type == 'running-foot':
        return config.preserve_first_page_foot
    return False


def _get_outside_main_area_noise_blocks(
    layout_document: LayoutDocument,
    config: LayoutNoiseFilterConfig,
) -> List[TaggedNoiseBlock]:
    total_pages = len(layout_document.pages)
    pages_by_pattern = _get_pages_by_letters_only_pattern(
        layout_document, config.min_repeating_pattern_length
    )
    page_threshold = max(2.0, config.repetition_fraction * total_pages)
    main_area_by_page_number = calculate_page_main_areas(layout_document)
    result: List[TaggedNoiseBlock] = []
    for page_index, page in enumerate(layout_document.pages):
        main_area = main_area_by_page_number.get(page.meta.page_number)
        for block in page.blocks:
            if not _is_outside_main_area(block, main_area):
                continue
            pattern = _get_letters_only_pattern(block.text)
            is_repeating = (
                len(pattern) >= config.min_repeating_pattern_length
                and len(pages_by_pattern[pattern]) >= page_threshold
            )
            if not is_repeating and not _has_no_letters(
                block.text, config.max_letterless_length
            ):
                continue
            note_type = _get_outside_main_area_note_type(
                _get_block_y_relative(block, page)
            )
            if _is_preserved_first_page(page_index, note_type, config):
                continue
            result.append(TaggedNoiseBlock(block=block, note_type=note_type))
    return result


def _get_repeating_text_noise_blocks(
    layout_document: LayoutDocument,
    config: LayoutNoiseFilterConfig,
) -> List[TaggedNoiseBlock]:
    total_pages = len(layout_document.pages)
    text_to_occurrences, page_block_y_rels = _collect_blocks(layout_document)
    page_q1_y, page_q3_y = _compute_page_quartiles(page_block_y_rels)
    threshold = max(2.0, config.repetition_fraction * total_pages)
    noise_blocks: List[TaggedNoiseBlock] = []
    for _text, occurrences in text_to_occurrences.items():
        if len(occurrences) < threshold:
            continue
        note_type = _classify_repetition_group(
            occurrences, page_q1_y, page_q3_y,
            config.position_consistency_fraction,
            config.max_position_stddev,
        )
        if not note_type:
            continue
        preserve = (
            config.preserve_first_page_head
            if note_type == 'running-head'
            else config.preserve_first_page_foot
        )
        noise_blocks.extend(_tag_noise_occurrences(
            occurrences, note_type, page_q1_y, page_q3_y,
            config.max_height_ratio, preserve
        ))
    return noise_blocks


def get_noise_blocks(
    layout_document: LayoutDocument,
    config: LayoutNoiseFilterConfig,
) -> Sequence[TaggedNoiseBlock]:
    if not config.enabled:
        return []
    if len(layout_document.pages) < 2:
        return []
    noise_blocks = list(_get_repeating_text_noise_blocks(layout_document, config))
    if config.filter_outside_main_area:
        already_tagged: Set[int] = {id(item.block) for item in noise_blocks}
        noise_blocks.extend(
            item
            for item in _get_outside_main_area_noise_blocks(layout_document, config)
            if id(item.block) not in already_tagged
        )
    LOGGER.debug('found %d layout noise blocks', len(noise_blocks))
    return noise_blocks


def remove_noise_blocks(
    layout_document: LayoutDocument,
    noise_blocks: Sequence[TaggedNoiseBlock],
) -> LayoutDocument:
    if not noise_blocks:
        return layout_document
    excluded_ids: Set[int] = {id(nb.block) for nb in noise_blocks}
    return LayoutDocument(pages=[
        page.replace(blocks=[
            block for block in page.blocks
            if id(block) not in excluded_ids
        ])
        for page in layout_document.pages
    ])
