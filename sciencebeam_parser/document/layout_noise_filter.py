import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)

LOGGER = logging.getLogger(__name__)

# Occurrence: (page_index, y_relative, height_relative, block)
_Occurrence = Tuple[int, Optional[float], Optional[float], LayoutBlock]


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


@dataclass
class TaggedNoiseBlock:
    block: LayoutBlock
    note_type: str  # "running-head" | "running-foot"


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
    occurrences: List[_Occurrence],
    page_q1_y: Dict[int, float],
    page_q3_y: Dict[int, float],
    consistency_fraction: float,
    max_position_stddev: float,
) -> Optional[str]:
    classifiable = sum(
        1 for idx, y_rel, _h, _b in occurrences
        if y_rel is not None and idx in page_q1_y
    )
    if not classifiable:
        return None
    for note_type in ('running-head', 'running-foot'):
        zone = [
            (idx, y_rel) for idx, y_rel, _h, _b in occurrences
            if y_rel is not None
            and _in_zone(idx, y_rel, note_type, page_q1_y, page_q3_y)
        ]
        if len(zone) / classifiable < consistency_fraction:
            continue
        y_rels = [y for _, y in zone]
        if len(y_rels) > 1 and _stddev(y_rels) > max_position_stddev:
            continue
        return note_type
    return None


def _tag_noise_occurrences(
    occurrences: List[_Occurrence],
    note_type: str,
    page_q1_y: Dict[int, float],
    page_q3_y: Dict[int, float],
    max_height_ratio: float,
) -> List[TaggedNoiseBlock]:
    zone_heights = [
        h for idx, y_rel, h, _ in occurrences
        if h is not None and y_rel is not None
        and _in_zone(idx, y_rel, note_type, page_q1_y, page_q3_y)
    ]
    median_height = sorted(zone_heights)[len(zone_heights) // 2] if zone_heights else None
    result = []
    for page_idx, y_rel, height_rel, block in occurrences:
        if y_rel is None or not _in_zone(page_idx, y_rel, note_type, page_q1_y, page_q3_y):
            continue
        if (median_height and height_rel is not None
                and height_rel > max_height_ratio * median_height):
            continue
        result.append(TaggedNoiseBlock(block=block, note_type=note_type))
    return result


def _collect_blocks(
    layout_document: LayoutDocument,
) -> Tuple[Dict[str, List[_Occurrence]], Dict[int, List[float]]]:
    text_to_occurrences: Dict[str, List[_Occurrence]] = defaultdict(list)
    page_block_y_rels: Dict[int, List[float]] = defaultdict(list)
    for page_index, page in enumerate(layout_document.pages):
        for block in page.blocks:
            text = block.text.strip().casefold()
            if not text:
                continue
            y_rel = _get_block_y_relative(block, page)
            h_rel = _get_block_height_relative(block, page)
            text_to_occurrences[text].append((page_index, y_rel, h_rel, block))
            if y_rel is not None:
                page_block_y_rels[page_index].append(y_rel)
    return dict(text_to_occurrences), dict(page_block_y_rels)


def get_noise_blocks(
    layout_document: LayoutDocument,
    config: LayoutNoiseFilterConfig,
) -> Sequence[TaggedNoiseBlock]:
    if not config.enabled:
        return []
    total_pages = len(layout_document.pages)
    if total_pages < 2:
        return []
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
        noise_blocks.extend(_tag_noise_occurrences(
            occurrences, note_type, page_q1_y, page_q3_y, config.max_height_ratio
        ))
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
