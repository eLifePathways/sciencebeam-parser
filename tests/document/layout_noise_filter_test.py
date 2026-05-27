from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken,
)
from sciencebeam_parser.document.layout_noise_filter import (
    LayoutNoiseFilterConfig,
    TaggedNoiseBlock,
    get_noise_blocks,
    remove_noise_blocks,
)


PAGE_HEIGHT = 1000
PAGE_WIDTH = 600

ENABLED_CONFIG = LayoutNoiseFilterConfig(enabled=True, repetition_fraction=0.5)


def _page_meta(page_number: int = 1) -> LayoutPageMeta:
    return LayoutPageMeta(
        page_number=page_number,
        coordinates=LayoutPageCoordinates(
            x=0, y=0, width=PAGE_WIDTH, height=PAGE_HEIGHT, page_number=page_number
        )
    )


def _block_at_y(text: str, y: float, page_number: int = 1) -> LayoutBlock:
    token = LayoutToken(
        text=text,
        coordinates=LayoutPageCoordinates(
            x=10, y=y, width=200, height=20, page_number=page_number
        )
    )
    return LayoutBlock(lines=[LayoutLine(tokens=[token])])


def _page(blocks: list, page_number: int = 1) -> LayoutPage:
    return LayoutPage(blocks=blocks, meta=_page_meta(page_number))


def _doc(*pages: LayoutPage) -> LayoutDocument:
    return LayoutDocument(pages=list(pages))


class TestGetNoiseBlocks:
    def test_returns_empty_when_disabled(self):
        header = _block_at_y('Journal Name', y=10)
        doc = _doc(
            _page([header], page_number=1),
            _page([_block_at_y('Journal Name', y=10)], page_number=2),
        )
        result = get_noise_blocks(doc, LayoutNoiseFilterConfig(enabled=False))
        assert not result

    def test_returns_empty_for_single_page(self):
        doc = _doc(_page([_block_at_y('Journal Name', y=10)]))
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not result

    def test_detects_running_head_at_top(self):
        # y=10 / PAGE_HEIGHT=1000 → y_relative=0.01 < 0.2
        header_text = 'Journal of Something'
        doc = _doc(
            _page([_block_at_y(header_text, y=10)], page_number=1),
            _page([_block_at_y(header_text, y=10)], page_number=2),
            _page([_block_at_y(header_text, y=10)], page_number=3),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        note_types = {nb.note_type for nb in result}
        assert note_types == {'running-head'}
        assert len(result) == 3

    def test_detects_running_foot_at_bottom(self):
        # y=900 / PAGE_HEIGHT=1000 → y_relative=0.9 > 0.8
        footer_text = 'Copyright 2024'
        doc = _doc(
            _page([_block_at_y(footer_text, y=900)], page_number=1),
            _page([_block_at_y(footer_text, y=900)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        note_types = {nb.note_type for nb in result}
        assert note_types == {'running-foot'}

    def test_does_not_flag_non_repeating_block(self):
        doc = _doc(
            _page([_block_at_y('Unique text page 1', y=10)], page_number=1),
            _page([_block_at_y('Unique text page 2', y=10)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not result

    def test_does_not_flag_repeating_block_in_middle(self):
        # y=500 / PAGE_HEIGHT=1000 → y_relative=0.5 — middle of page, not noise
        mid_text = 'Section Title'
        doc = _doc(
            _page([_block_at_y(mid_text, y=500)], page_number=1),
            _page([_block_at_y(mid_text, y=500)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not result

    def test_respects_repetition_fraction(self):
        # 2 out of 5 pages = 0.4, below default threshold of 0.5 → not flagged
        header_text = 'Sparse Header'
        doc = _doc(
            _page([_block_at_y(header_text, y=10)], page_number=1),
            _page([_block_at_y('body text', y=400)], page_number=2),
            _page([_block_at_y('body text', y=400)], page_number=3),
            _page([_block_at_y('body text', y=400)], page_number=4),
            _page([_block_at_y(header_text, y=10)], page_number=5),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not any(nb.block.text.strip() == header_text for nb in result)

    def test_case_insensitive_normalisation(self):
        doc = _doc(
            _page([_block_at_y('JOURNAL NAME', y=10)], page_number=1),
            _page([_block_at_y('journal name', y=10)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert len(result) == 2


class TestRemoveNoiseBlocks:
    def test_returns_same_document_when_no_noise(self):
        doc = _doc(_page([_block_at_y('body', y=400)]))
        assert remove_noise_blocks(doc, []) is doc

    def test_removes_tagged_blocks(self):
        body = _block_at_y('body text', y=400)
        header = _block_at_y('Journal Name', y=10)
        page = _page([body, header])
        doc = _doc(page)
        noise = [TaggedNoiseBlock(block=header, note_type='running-head')]
        result = remove_noise_blocks(doc, noise)
        remaining = list(result.iter_all_blocks())
        assert body in remaining
        assert header not in remaining

    def test_preserves_page_structure(self):
        b1 = _block_at_y('block 1', y=400, page_number=1)
        b2 = _block_at_y('header', y=10, page_number=1)
        b3 = _block_at_y('block 3', y=400, page_number=2)
        doc = _doc(_page([b1, b2], page_number=1), _page([b3], page_number=2))
        noise = [TaggedNoiseBlock(block=b2, note_type='running-head')]
        result = remove_noise_blocks(doc, noise)
        assert len(result.pages) == 2
        assert len(result.pages[0].blocks) == 1
        assert len(result.pages[1].blocks) == 1
