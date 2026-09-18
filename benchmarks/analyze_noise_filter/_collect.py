import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sciencebeam_parser.app.parser import ScienceBeamParser
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.document.layout_noise_filter import (
    LayoutNoiseFilterConfig,
    TaggedNoiseBlock,
    get_noise_blocks,
)
from sciencebeam_parser.utils.media_types import MediaTypes

from ._jats import JatsTextIndex
from ._types import DocumentSummary, RemovedBlock

LOGGER = logging.getLogger(__name__)


def get_layout_document(
    pdf_filename: str,
    sciencebeam_parser: ScienceBeamParser
) -> LayoutDocument:
    with sciencebeam_parser.get_new_session() as session:
        source = session.get_source(pdf_filename, MediaTypes.PDF, source_name=pdf_filename)
        return source.get_layout_document()


def _get_block_relative_geometry(
    block: LayoutBlock,
    page: LayoutPage
) -> Tuple[Optional[float], Optional[float]]:
    page_coordinates = page.meta.coordinates if page.meta else None
    page_height = page_coordinates.height if page_coordinates else None
    coordinates_list = block.get_merged_coordinates_list()
    if not page_height or not coordinates_list:
        return None, None
    coordinates = coordinates_list[0]
    return coordinates.y / page_height, coordinates.height / page_height


def _index_pages_by_block_id(
    layout_document: LayoutDocument
) -> Dict[int, Tuple[int, LayoutPage]]:
    return {
        id(block): (page_index, page)
        for page_index, page in enumerate(layout_document.pages)
        for block in page.blocks
    }


def _get_removed_block(
    noise_block: TaggedNoiseBlock,
    page_index: int,
    page: LayoutPage,
    document: DocumentSummary,
    jats_index: Optional[JatsTextIndex]
) -> RemovedBlock:
    y_relative, height_relative = _get_block_relative_geometry(noise_block.block, page)
    text = noise_block.block.text
    return RemovedBlock(
        document_id=document.document_id,
        corpus=document.corpus,
        page_number=page_index + 1,
        page_count=document.page_count,
        note_type=noise_block.note_type,
        y_relative=y_relative,
        height_relative=height_relative,
        line_count=len(noise_block.block.lines),
        text=text,
        in_jats=jats_index.contains(text) if jats_index is not None else None
    )


def analyze_document(
    pdf_filename: str,
    corpus: str,
    layout_document: LayoutDocument,
    noise_filter_config: LayoutNoiseFilterConfig,
    jats_index: Optional[JatsTextIndex] = None
) -> Tuple[DocumentSummary, List[RemovedBlock]]:
    document_id = Path(pdf_filename).stem
    page_count = len(layout_document.pages)
    blocks = [block for page in layout_document.pages for block in page.blocks]
    page_by_block_id = _index_pages_by_block_id(layout_document)
    document = DocumentSummary(
        document_id=document_id, corpus=corpus, page_count=page_count,
        block_count=len(blocks), line_count=sum(len(block.lines) for block in blocks),
        removed_block_count=0, removed_line_count=0, in_jats_count=0
    )
    removed = [
        _get_removed_block(
            noise_block,
            *page_by_block_id[id(noise_block.block)],
            document=document,
            jats_index=jats_index
        )
        for noise_block in get_noise_blocks(layout_document, noise_filter_config)
    ]
    return dataclasses.replace(
        document,
        removed_block_count=len(removed),
        removed_line_count=sum(item.line_count for item in removed),
        in_jats_count=sum(1 for item in removed if item.in_jats)
    ), removed


def get_corpus_for_pdf(pdf_filename: str) -> str:
    """The corpus is the directory the PDF sits in, as in benchmarks/data/<split>/<corpus>."""
    return Path(pdf_filename).parent.name


def _get_failed_document_summary(
    document_id: str, corpus: str, error: str
) -> DocumentSummary:
    return DocumentSummary(
        document_id=document_id, corpus=corpus, page_count=0, block_count=0,
        line_count=0, removed_block_count=0, removed_line_count=0,
        in_jats_count=0, error=error
    )


def _analyze_pdf(
    pdf_filename: str,
    sciencebeam_parser: ScienceBeamParser,
    noise_filter_config: LayoutNoiseFilterConfig,
    jats_filename: Optional[str]
) -> Tuple[DocumentSummary, List[RemovedBlock]]:
    corpus = get_corpus_for_pdf(pdf_filename)
    try:
        return analyze_document(
            pdf_filename,
            corpus=corpus,
            layout_document=get_layout_document(pdf_filename, sciencebeam_parser),
            noise_filter_config=noise_filter_config,
            jats_index=JatsTextIndex.from_file(jats_filename) if jats_filename else None
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception('failed to process %r', pdf_filename)
        return _get_failed_document_summary(Path(pdf_filename).stem, corpus, str(exc)), []


def analyze_documents(
    pdf_filenames: Sequence[str],
    sciencebeam_parser: ScienceBeamParser,
    noise_filter_config: LayoutNoiseFilterConfig,
    jats_filename_by_stem: Optional[Dict[str, str]] = None
) -> Tuple[List[DocumentSummary], List[RemovedBlock]]:
    summaries: List[DocumentSummary] = []
    removed: List[RemovedBlock] = []
    for index, pdf_filename in enumerate(pdf_filenames):
        document_id = Path(pdf_filename).stem
        LOGGER.info('[%d/%d] %s', index + 1, len(pdf_filenames), document_id)
        jats_filename = (jats_filename_by_stem or {}).get(document_id)
        if jats_filename_by_stem is not None and not jats_filename:
            LOGGER.warning('no JATS found for %r', document_id)
        summary, document_removed = _analyze_pdf(
            pdf_filename,
            sciencebeam_parser=sciencebeam_parser,
            noise_filter_config=noise_filter_config,
            jats_filename=jats_filename
        )
        summaries.append(summary)
        removed.extend(document_removed)
    return summaries, removed
