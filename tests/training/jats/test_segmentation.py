from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken,
)
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_BODY,
    SEG_FRONT,
    SEG_HEADNOTE,
    SEG_PAGE,
    SEG_REFERENCES,
    SegmentationConfig,
    SegmentationLabelDeriver,
)


def _make_page_meta(page_number: int = 1, height: float = 1000.0) -> LayoutPageMeta:
    return LayoutPageMeta(
        page_number=page_number,
        coordinates=LayoutPageCoordinates(
            x=0, y=0, width=600, height=height, page_number=page_number
        ),
    )


def _make_token(
    text: str,
    page_number: int = 1,
    y: float = 500.0,
) -> LayoutToken:
    return LayoutToken(
        text=text,
        coordinates=LayoutPageCoordinates(
            x=10, y=y, width=50, height=12, page_number=page_number
        ),
    )


def _make_line(*texts: str, y: float = 500.0, page_number: int = 1) -> LayoutLine:
    tokens = [_make_token(t, page_number=page_number, y=y) for t in texts]
    return LayoutLine(tokens=tokens)


def _make_doc_with_page(
    *blocks: LayoutBlock, page_height: float = 1000.0, page_number: int = 1
) -> LayoutDocument:
    page_meta = _make_page_meta(page_number=page_number, height=page_height)
    page = LayoutPage(blocks=list(blocks), meta=page_meta)
    return LayoutDocument(pages=[page])


def _annotate(doc: LayoutDocument, field_by_line_index: dict) -> JatsAnnotatedLayoutDocument:
    annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
    lines = list(doc.iter_all_lines())
    for line_idx, field_name in field_by_line_index.items():
        for token in lines[line_idx].tokens:
            annotated.set_token_label(token, field_name)
    return annotated


def _derive_labels(doc, annotated, **config_kwargs):
    config = SegmentationConfig(**config_kwargs) if config_kwargs else None
    return SegmentationLabelDeriver(config).derive_labels(doc, annotated)


class TestMajorityVoteLabeling:
    def test_title_tokens_give_header_label(self):
        line = _make_line('My', 'Title')
        doc = _make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = _annotate(doc, {0: JatsFieldNames.TITLE})
        labels = _derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_FRONT

    def test_body_paragraph_gives_body_label(self):
        line = _make_line('Some', 'body', 'text')
        doc = _make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = _annotate(doc, {0: JatsFieldNames.BODY_SECTION_PARAGRAPH})
        labels = _derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_BODY

    def test_reference_tokens_give_references_label(self):
        line = _make_line('Smith', '2020')
        doc = _make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = _annotate(doc, {0: JatsFieldNames.REFERENCE})
        labels = _derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_REFERENCES

    def test_unannotated_line_defaults_to_body(self):
        line = _make_line('Unknown', 'content')
        doc = _make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = _derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_BODY

    def test_majority_vote_mixed_line(self):
        # 3 tokens labeled as TITLE, 1 as BODY_SECTION_PARAGRAPH → should give FRONT (header)
        line = _make_line('My', 'Title', 'Here', 'text')
        doc = _make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        tokens = line.tokens
        for t in tokens[:3]:
            annotated.set_token_label(t, JatsFieldNames.TITLE)
        annotated.set_token_label(tokens[3], JatsFieldNames.BODY_SECTION_PARAGRAPH)
        labels = _derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_FRONT


class TestCoordinateBasedDetection:
    def test_line_at_top_of_page_becomes_headnote(self):
        line = _make_line('Running', 'header', y=20.0)  # 20/1000 = 2% < 8%
        doc = _make_doc_with_page(LayoutBlock(lines=[line]), page_height=1000.0)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = _derive_labels(doc, annotated, headnote_y_ratio=0.08)
        assert labels[id(line)] == SEG_HEADNOTE

    def test_line_in_middle_of_page_is_not_headnote(self):
        line = _make_line('Normal', 'content', y=500.0)  # 50% of page
        doc = _make_doc_with_page(LayoutBlock(lines=[line]), page_height=1000.0)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = _derive_labels(doc, annotated)
        assert labels[id(line)] != SEG_HEADNOTE

    def test_numeric_line_at_bottom_becomes_page(self):
        line = _make_line('42', y=950.0)  # 95% of page > 92%
        doc = _make_doc_with_page(LayoutBlock(lines=[line]), page_height=1000.0)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = _derive_labels(doc, annotated, footnote_y_ratio=0.92)
        assert labels[id(line)] == SEG_PAGE


class TestGapMerge:
    def test_untagged_line_between_front_lines_gets_front(self):
        line_front1 = _make_line('Title', 'text', y=200.0)
        line_gap = _make_line('Some', 'untagged', 'stuff', y=220.0)
        line_front2 = _make_line('More', 'header', y=240.0)
        block = LayoutBlock(lines=[line_front1, line_gap, line_front2])
        doc = _make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        for t in line_front1.tokens:
            annotated.set_token_label(t, JatsFieldNames.TITLE)
        for t in line_front2.tokens:
            annotated.set_token_label(t, JatsFieldNames.AUTHOR)
        labels = _derive_labels(doc, annotated)
        assert labels[id(line_front1)] == SEG_FRONT
        assert labels[id(line_front2)] == SEG_FRONT
        assert labels[id(line_gap)] == SEG_FRONT

    def test_untagged_line_after_body_stays_body(self):
        line_body = _make_line('Body', 'paragraph', y=400.0)
        line_gap = _make_line('More', 'stuff', y=420.0)
        block = LayoutBlock(lines=[line_body, line_gap])
        doc = _make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        for t in line_body.tokens:
            annotated.set_token_label(t, JatsFieldNames.BODY_SECTION_PARAGRAPH)
        labels = _derive_labels(doc, annotated)
        assert labels[id(line_body)] == SEG_BODY
        # gap after body → body (default)
        assert labels[id(line_gap)] == SEG_BODY


class TestFrontThreshold:
    def test_front_block_starting_within_threshold_is_kept(self):
        # A front block starting at line 0 should not be cleared
        lines = [_make_line(f'word{i}') for i in range(5)]
        block = LayoutBlock(lines=lines)
        doc = _make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        for t in lines[0].tokens:
            annotated.set_token_label(t, JatsFieldNames.TITLE)
        for t in lines[4].tokens:
            annotated.set_token_label(t, JatsFieldNames.AUTHOR)
        labels = _derive_labels(doc, annotated, front_max_start_line_index=80)
        assert labels[id(lines[0])] == SEG_FRONT
        assert labels[id(lines[4])] == SEG_FRONT

    def test_front_block_starting_beyond_threshold_is_cleared(self):
        # A front block starting at line 100 (> 80) should be cleared → defaults to body
        lines = [_make_line(f'word{i}') for i in range(3)]
        block = LayoutBlock(lines=lines)
        doc = _make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        # Annotate only the last line as front — its block starts at index 2 which is
        # not cleared.  Use a config with a very low threshold to test the clearing logic.
        for t in lines[2].tokens:
            annotated.set_token_label(t, JatsFieldNames.AUTHOR_NOTES)
        labels = _derive_labels(doc, annotated, front_max_start_line_index=1)
        # line 2 starts its own front block at index 2 > threshold 1 → cleared → body
        assert labels[id(lines[2])] == SEG_BODY


class TestTextRepetitionHeadnote:
    def test_repeated_line_near_top_becomes_headnote(self):
        # No coordinates → will fall through to text-repetition detection
        lines_no_coords = [LayoutLine(tokens=[LayoutToken(text=t)]) for t in ['Journal Name'] * 3]
        block2 = LayoutBlock(lines=lines_no_coords)
        doc = LayoutDocument(pages=[LayoutPage(blocks=[block2])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = _derive_labels(
            doc, annotated, page_header_max_first_line_index=10
        )
        for line in lines_no_coords:
            assert labels.get(id(line)) == SEG_HEADNOTE
