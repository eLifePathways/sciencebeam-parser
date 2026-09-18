from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken,
)
from sciencebeam_parser.document.layout_noise_filter import LayoutNoiseFilterConfig

from benchmarks.analyze_noise_filter._collect import analyze_document, get_corpus_for_pdf
from benchmarks.analyze_noise_filter._jats import JatsTextIndex, normalize_text

PAGE_HEIGHT = 1000
PAGE_WIDTH = 600
ENABLED_CONFIG = LayoutNoiseFilterConfig(enabled=True, repetition_fraction=0.5)
_BODY_Y_POSITIONS = [200, 300, 500, 600, 700]


def _block_at_y(text: str, y: float, page_number: int, height: float = 20) -> LayoutBlock:
    token = LayoutToken(
        text=text,
        coordinates=LayoutPageCoordinates(
            x=10, y=y, width=200, height=height, page_number=page_number
        )
    )
    return LayoutBlock(lines=[LayoutLine(tokens=[token])])


def _page_with_body(extra_blocks: list, page_number: int) -> LayoutPage:
    body_blocks = [
        _block_at_y(f'body p{page_number} i{i}', y=y, page_number=page_number)
        for i, y in enumerate(_BODY_Y_POSITIONS)
    ]
    return LayoutPage(
        blocks=extra_blocks + body_blocks,
        meta=LayoutPageMeta(
            page_number=page_number,
            coordinates=LayoutPageCoordinates(
                x=0, y=0, width=PAGE_WIDTH, height=PAGE_HEIGHT, page_number=page_number
            )
        )
    )


def _document_with_running_head() -> LayoutDocument:
    return LayoutDocument(pages=[
        _page_with_body([_block_at_y('Journal Name', y=10, page_number=page_number)],
                        page_number=page_number)
        for page_number in (1, 2)
    ])


class TestGetCorpusForPdf:
    def test_should_use_the_parent_directory(self):
        assert get_corpus_for_pdf('benchmarks/data/train/biorxiv/x.pdf') == 'biorxiv'


class TestAnalyzeDocument:
    def test_should_report_the_removed_block_with_its_page_and_position(self):
        summary, removed = analyze_document(
            'benchmarks/data/train/biorxiv/doc-1.pdf',
            corpus='biorxiv',
            layout_document=_document_with_running_head(),
            noise_filter_config=ENABLED_CONFIG
        )
        assert summary.document_id == 'doc-1'
        assert summary.page_count == 2
        assert summary.removed_block_count == 2
        assert summary.removed_line_count == 2
        assert [item.text for item in removed] == ['Journal Name', 'Journal Name']
        assert [item.page_number for item in removed] == [1, 2]
        assert removed[0].note_type == 'running-head'
        assert removed[0].y_relative == 10 / PAGE_HEIGHT
        assert removed[0].height_relative == 20 / PAGE_HEIGHT

    def test_should_count_lines_and_blocks_of_the_whole_document(self):
        summary, _ = analyze_document(
            'x/biorxiv/doc-1.pdf',
            corpus='biorxiv',
            layout_document=_document_with_running_head(),
            noise_filter_config=ENABLED_CONFIG
        )
        assert summary.block_count == 12
        assert summary.line_count == 12

    def test_should_remove_nothing_when_the_filter_is_disabled(self):
        summary, removed = analyze_document(
            'x/biorxiv/doc-1.pdf',
            corpus='biorxiv',
            layout_document=_document_with_running_head(),
            noise_filter_config=LayoutNoiseFilterConfig(enabled=False)
        )
        assert not removed
        assert summary.removed_block_count == 0

    def test_should_leave_in_jats_unset_without_a_jats_index(self):
        _, removed = analyze_document(
            'x/biorxiv/doc-1.pdf',
            corpus='biorxiv',
            layout_document=_document_with_running_head(),
            noise_filter_config=ENABLED_CONFIG
        )
        assert all(item.in_jats is None for item in removed)

    def test_should_flag_a_removal_whose_text_is_in_the_jats(self):
        long_head = 'Mortality in the period from January 2020 to February 2021'
        layout_document = LayoutDocument(pages=[
            _page_with_body([_block_at_y(long_head, y=10, page_number=page_number)],
                            page_number=page_number)
            for page_number in (1, 2)
        ])
        summary, removed = analyze_document(
            'x/biorxiv/doc-1.pdf',
            corpus='biorxiv',
            layout_document=layout_document,
            noise_filter_config=ENABLED_CONFIG,
            jats_index=JatsTextIndex(normalize_text(f'Title. {long_head}. Body.'))
        )
        assert [item.in_jats for item in removed] == [True, True]
        assert summary.in_jats_count == 2
