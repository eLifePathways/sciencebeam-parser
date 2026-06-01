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

# Body blocks at these y positions give a spread across the middle of the page.
# y_rels: 0.2, 0.3, 0.5, 0.6, 0.7  →  q1≈0.3, q3≈0.6 when combined with a
# header at y_rel≈0.01 or footer at y_rel≈0.95.
_BODY_Y_POSITIONS = [200, 300, 500, 600, 700]


def _page_meta(page_number: int = 1) -> LayoutPageMeta:
    return LayoutPageMeta(
        page_number=page_number,
        coordinates=LayoutPageCoordinates(
            x=0, y=0, width=PAGE_WIDTH, height=PAGE_HEIGHT, page_number=page_number
        )
    )


def _block_at_y(
    text: str, y: float, page_number: int = 1, height: float = 20
) -> LayoutBlock:
    token = LayoutToken(
        text=text,
        coordinates=LayoutPageCoordinates(
            x=10, y=y, width=200, height=height, page_number=page_number
        )
    )
    return LayoutBlock(lines=[LayoutLine(tokens=[token])])


def _page(blocks: list, page_number: int = 1) -> LayoutPage:
    return LayoutPage(blocks=blocks, meta=_page_meta(page_number))


def _page_with_body(extra_blocks: list, page_number: int) -> LayoutPage:
    """Build a realistic page with unique body content plus any extra blocks."""
    body_blocks = [
        _block_at_y(f'body p{page_number} i{i}', y=y, page_number=page_number)
        for i, y in enumerate(_BODY_Y_POSITIONS)
    ]
    return _page(extra_blocks + body_blocks, page_number=page_number)


def _doc(*pages: LayoutPage) -> LayoutDocument:
    return LayoutDocument(pages=list(pages))


class TestGetNoiseBlocks:
    def test_returns_empty_when_disabled(self):
        doc = _doc(
            _page_with_body([_block_at_y('Journal Name', y=10)], page_number=1),
            _page_with_body([_block_at_y('Journal Name', y=10)], page_number=2),
        )
        result = get_noise_blocks(doc, LayoutNoiseFilterConfig(enabled=False))
        assert not result

    def test_returns_empty_for_single_page(self):
        doc = _doc(_page_with_body([_block_at_y('Journal Name', y=10)], page_number=1))
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not result

    def test_detects_running_head_at_top(self):
        # y=10 → y_rel=0.01, well below q1 of each page
        header_text = 'Journal of Something'
        doc = _doc(
            _page_with_body([_block_at_y(header_text, y=10)], page_number=1),
            _page_with_body([_block_at_y(header_text, y=10)], page_number=2),
            _page_with_body([_block_at_y(header_text, y=10)], page_number=3),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert {nb.note_type for nb in result} == {'running-head'}
        assert len(result) == 3

    def test_detects_running_foot_at_bottom(self):
        # y=950 → y_rel=0.95, well above q3 of each page
        footer_text = 'Copyright 2024'
        doc = _doc(
            _page_with_body([_block_at_y(footer_text, y=950)], page_number=1),
            _page_with_body([_block_at_y(footer_text, y=950)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert {nb.note_type for nb in result} == {'running-foot'}

    def test_does_not_flag_non_repeating_block(self):
        doc = _doc(
            _page_with_body([_block_at_y('Unique text page 1', y=10)], page_number=1),
            _page_with_body([_block_at_y('Unique text page 2', y=10)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not result

    def test_does_not_flag_repeating_block_in_middle(self):
        # y=400 → y_rel=0.4, between q1 and q3 → neither zone
        mid_text = 'Section Title'
        doc = _doc(
            _page_with_body([_block_at_y(mid_text, y=400)], page_number=1),
            _page_with_body([_block_at_y(mid_text, y=400)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not any(nb.block.text.strip() == mid_text for nb in result)

    def test_does_not_flag_block_with_inconsistent_position(self):
        # Same text but moves between top and bottom across pages → high stddev → not noise
        wandering_text = 'Wandering Block'
        doc = _doc(
            _page_with_body([_block_at_y(wandering_text, y=10)], page_number=1),
            _page_with_body([_block_at_y(wandering_text, y=950)], page_number=2),
            _page_with_body([_block_at_y(wandering_text, y=10)], page_number=3),
            _page_with_body([_block_at_y(wandering_text, y=950)], page_number=4),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not any(nb.block.text.strip() == wandering_text for nb in result)

    def test_respects_repetition_fraction(self):
        # 2 out of 5 pages = 0.4, below default threshold of 0.5 → not flagged
        header_text = 'Sparse Header'
        doc = _doc(
            _page_with_body([_block_at_y(header_text, y=10)], page_number=1),
            _page_with_body([], page_number=2),
            _page_with_body([], page_number=3),
            _page_with_body([], page_number=4),
            _page_with_body([_block_at_y(header_text, y=10)], page_number=5),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert not any(nb.block.text.strip() == header_text for nb in result)

    def test_case_insensitive_normalisation(self):
        doc = _doc(
            _page_with_body([_block_at_y('JOURNAL NAME', y=10)], page_number=1),
            _page_with_body([_block_at_y('journal name', y=10)], page_number=2),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        assert len(result) == 2

    def test_large_title_on_first_page_not_filtered_when_repeated_as_small_footer(self):
        # Regression (scielo_br): a paper title appears large (height=60) near the
        # top of page 1, and the same text repeats as a small footer (height=8) at
        # the bottom of pages 2+. The height check must preserve the page-1 title.
        title_text = 'A radicalização do debate sobre inclusão escolar no Brasil'
        doc = _doc(
            # page 1: large title near the top
            _page_with_body([_block_at_y(title_text, y=50, height=60)], page_number=1),
            # pages 2-5: same text as small footer at the bottom
            _page_with_body([_block_at_y(title_text, y=950, height=8)], page_number=2),
            _page_with_body([_block_at_y(title_text, y=950, height=8)], page_number=3),
            _page_with_body([_block_at_y(title_text, y=950, height=8)], page_number=4),
            _page_with_body([_block_at_y(title_text, y=950, height=8)], page_number=5),
        )
        result = get_noise_blocks(doc, ENABLED_CONFIG)
        # Footer occurrences on pages 2-5 are filtered
        assert all(nb.note_type == 'running-foot' for nb in result)
        assert len(result) == 4
        # The large page-1 title must not be among the filtered blocks
        filtered_ids = {id(nb.block) for nb in result}
        page1_title_block = doc.pages[0].blocks[0]
        assert id(page1_title_block) not in filtered_ids


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
