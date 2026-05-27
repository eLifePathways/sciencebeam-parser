import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class LayoutNoiseFilterConfig:
    enabled: bool = False
    repetition_fraction: float = 0.5


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


def _classify_repetition_group(
    occurrences: List[Tuple[int, Optional[float], LayoutBlock]]
) -> Optional[str]:
    y_values = [y for _, y, _ in occurrences if y is not None]
    if not y_values:
        return None
    median_y = sorted(y_values)[len(y_values) // 2]
    if median_y < 0.2:
        return 'running-head'
    if median_y > 0.8:
        return 'running-foot'
    return None


def get_noise_blocks(
    layout_document: LayoutDocument,
    config: LayoutNoiseFilterConfig,
) -> Sequence[TaggedNoiseBlock]:
    if not config.enabled:
        return []
    total_pages = len(layout_document.pages)
    if total_pages < 2:
        return []

    text_to_occurrences: Dict[str, List[Tuple[int, Optional[float], LayoutBlock]]] = (
        defaultdict(list)
    )
    for page_index, page in enumerate(layout_document.pages):
        for block in page.blocks:
            text = block.text.strip().casefold()
            if not text:
                continue
            y_rel = _get_block_y_relative(block, page)
            text_to_occurrences[text].append((page_index, y_rel, block))

    threshold = max(2.0, config.repetition_fraction * total_pages)
    noise_blocks: List[TaggedNoiseBlock] = []

    for _text, occurrences in text_to_occurrences.items():
        if len(occurrences) < threshold:
            continue
        note_type = _classify_repetition_group(occurrences)
        if not note_type:
            continue
        for _, _, block in occurrences:
            noise_blocks.append(TaggedNoiseBlock(block=block, note_type=note_type))

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
