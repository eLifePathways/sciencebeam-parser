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
    SEG_ANNEX,
    SEG_BODY,
    SEG_FOOTNOTE,
    SEG_REVIEW,
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

    def test_spelled_out_page_marker_at_bottom_becomes_page(self):
        line = _make_line('Page', '3', 'of', '12', y=950.0)
        doc = _make_doc_with_page(LayoutBlock(lines=[line]), page_height=1000.0)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = _derive_labels(doc, annotated, footnote_y_ratio=0.92)
        assert labels[id(line)] == SEG_PAGE

    def test_numeric_line_away_from_the_footer_is_not_a_page_number(self):
        page_marker = _make_line('Page', '3', 'of', '12', y=950.0)
        row_number = _make_line('2', y=500.0)
        doc = _make_doc_with_page(LayoutBlock(lines=[row_number, page_marker]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        assert _derive_labels(doc, annotated)[id(row_number)] != SEG_PAGE


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


class TestGapMergeAcrossPageFurniture:
    """A page break interrupts a region without ending it, footer and all."""

    def _make_doc(self, middle_text: str):
        first = _make_line('Smith,', 'J.', '(2020).', 'A', 'title', page_number=1)
        middle = _make_line(*middle_text.split(), y=950.0, page_number=1)
        gap = _make_line('and', 'the', 'rest', 'of', 'it', page_number=2)
        last = _make_line('Jones,', 'K.', '(2021).', 'Another', page_number=2)
        # The foot of the second page too: a running foot is what repeats, and
        # that is what tells it apart from a line the margin rules took by
        # accident.
        trailing = _make_line(*middle_text.split(), y=950.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[first]), LayoutBlock(lines=[middle])],
                meta=_make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=[gap, last]), LayoutBlock(lines=[trailing])],
                meta=_make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        annotated = _annotate(doc, {
            lines.index(first): JatsFieldNames.REFERENCE,
            lines.index(last): JatsFieldNames.REFERENCE,
        })
        return doc, annotated, middle, gap

    def test_a_page_number_between_two_reference_lines_is_stepped_over(self):
        doc, annotated, middle, gap = self._make_doc('Page 3 of 12')
        labels = _derive_labels(doc, annotated)
        assert labels.get(id(middle)) == SEG_PAGE
        assert labels.get(id(gap)) == SEG_REFERENCES

    def test_a_footer_between_two_reference_lines_is_stepped_over(self):
        doc, annotated, middle, gap = self._make_doc('Powered by TCPDF')
        labels = _derive_labels(doc, annotated)
        assert labels.get(id(middle)) == SEG_FOOTNOTE
        assert labels.get(id(gap)) == SEG_REFERENCES


class TestReclaimRepeatedHeadnote:
    def test_a_running_header_a_match_absorbed_is_taken_back(self):
        header_lines = [
            _make_line('Journal', 'of', 'Things', y=20.0, page_number=page)
            for page in (1, 2, 3)
        ]
        absorbed = header_lines[2]
        body = _make_line('a', 'paragraph', 'spanning', 'the', 'break', page_number=3)
        pages = [
            LayoutPage(blocks=[LayoutBlock(lines=[header_lines[0]])], meta=_make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[header_lines[1]])], meta=_make_page_meta(2)),
            LayoutPage(blocks=[LayoutBlock(lines=[absorbed, body])], meta=_make_page_meta(3)),
        ]
        doc = LayoutDocument(pages=pages)
        lines = list(doc.iter_all_lines())
        annotated = _annotate(doc, {
            lines.index(absorbed): JatsFieldNames.BODY_SECTION_PARAGRAPH,
            lines.index(body): JatsFieldNames.BODY_SECTION_PARAGRAPH,
        })
        assert _derive_labels(doc, annotated)[id(absorbed)] == SEG_HEADNOTE


class TestRegionStartsAtThePageTop:
    """A section's heading is above the first line the JATS can evidence."""

    def _make_doc(self, heading_is_grounded: bool):
        body = _make_line('The', 'article', 'body', 'ends', 'here', page_number=1)
        heading = _make_line('Open', 'Peer', 'Review', y=80.0, page_number=2)
        report = _make_line('Thank', 'you', 'for', 'addressing', 'the', 'comments',
                            y=300.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[body])], meta=_make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[heading, report])], meta=_make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(body): JatsFieldNames.BODY_SECTION_PARAGRAPH,
            lines.index(report): JatsFieldNames.SUB_ARTICLE,
        }
        if heading_is_grounded:
            fields[lines.index(heading)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, _annotate(doc, fields), heading

    def test_the_region_takes_the_unevidenced_heading_above_it(self):
        doc, annotated, heading = self._make_doc(heading_is_grounded=False)
        assert _derive_labels(doc, annotated)[id(heading)] == SEG_REVIEW

    def test_an_evidenced_line_above_keeps_the_region_where_it_was(self):
        doc, annotated, heading = self._make_doc(heading_is_grounded=True)
        assert _derive_labels(doc, annotated)[id(heading)] == SEG_BODY


class TestPageFootNotes:
    """Back matter printed under the body of a page is a note, not an appendix."""

    def _make_doc(self, note_y: float):
        body = _make_line('The', 'argument', 'continues', y=300.0, page_number=1)
        note = _make_line('1', 'Marks,', 'Robert', 'B.', 'Exhausting', 'the', 'Earth',
                          y=note_y, page_number=1)
        later = _make_line('More', 'of', 'the', 'argument', y=300.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[body, note])], meta=_make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[later])], meta=_make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        annotated = _annotate(doc, {
            lines.index(body): JatsFieldNames.BODY_SECTION_PARAGRAPH,
            lines.index(note): JatsFieldNames.BACK_SECTION_PARAGRAPH,
            lines.index(later): JatsFieldNames.BODY_SECTION_PARAGRAPH,
        })
        return doc, annotated, note

    def test_a_note_low_on_the_page_is_a_footnote(self):
        doc, annotated, note = self._make_doc(note_y=780.0)
        assert _derive_labels(doc, annotated)[id(note)] == SEG_FOOTNOTE

    def test_back_matter_higher_up_stays_an_annex(self):
        doc, annotated, note = self._make_doc(note_y=420.0)
        assert _derive_labels(doc, annotated)[id(note)] == SEG_ANNEX


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


class TestFloatPlacement:
    """A float is body wherever it prints, as GROBID's own corpus has it."""

    def test_float_after_the_references_is_still_body(self):
        reference, table = _make_line('Smith', '2020'), _make_line('Table', '1.', 'Measures')
        doc = _make_doc_with_page(LayoutBlock(lines=[reference, table]))
        annotated = _annotate(doc, {
            0: JatsFieldNames.REFERENCE,
            1: JatsFieldNames.FLOAT_TABLE,
        })
        assert _derive_labels(doc, annotated)[id(table)] == SEG_BODY

    def test_float_before_the_references_is_body(self):
        figure, reference = _make_line('Figure', '1.', 'Model'), _make_line('Smith', '2020')
        doc = _make_doc_with_page(LayoutBlock(lines=[figure, reference]))
        annotated = _annotate(doc, {
            0: JatsFieldNames.FLOAT_FIGURE,
            1: JatsFieldNames.REFERENCE,
        })
        assert _derive_labels(doc, annotated)[id(figure)] == SEG_BODY


class TestFurnitureInsideReferences:
    """A reference wrapping over a page break prints in the header zone."""

    def _make_doc(self, *header_zone_texts: str, banner_on_first_page: bool = False):
        first_page_lines = [_make_line('Smith,', 'J.', '(2020).', 'Journal', 'of')]
        if banner_on_first_page:
            first_page_lines.insert(0, _make_line(*header_zone_texts[0].split(), y=50.0))
        second_page_lines = [
            _make_line(*text.split(), y=50.0, page_number=2) for text in header_zone_texts
        ] + [_make_line('38(2),', '279-294.', page_number=2)]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=first_page_lines)], meta=_make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=second_page_lines)], meta=_make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        annotated = _annotate(doc, {
            lines.index(first_page_lines[-1]): JatsFieldNames.REFERENCE,
            lines.index(second_page_lines[-1]): JatsFieldNames.REFERENCE,
        })
        return doc, annotated, second_page_lines

    def test_every_line_of_a_wrapped_reference_is_released(self):
        doc, annotated, second_page_lines = self._make_doc(
            'Development,', 'Infancia', 'y', 'Aprendizaje,'
        )
        labels = _derive_labels(doc, annotated)
        for line in second_page_lines:
            assert labels.get(id(line)) == SEG_REFERENCES

    def test_a_repeated_running_header_stays_a_headnote(self):
        doc, annotated, second_page_lines = self._make_doc(
            'Preprints - this document is a preprint', banner_on_first_page=True
        )
        labels = _derive_labels(doc, annotated)
        assert labels.get(id(second_page_lines[0])) == SEG_HEADNOTE


class TestFrontMatterBoundary:
    """The reference list bounds the front matter; a line index stands in for it."""

    def _make_doc(self, front_after_references: bool):
        texts = (
            [('A', 'title', 'of', 'the', 'paper')]
            + [('body', f'line{index}') for index in range(100)]
            + ([('Smith,', 'J.', '(2020).', 'A', 'reference'),
                ('Grant', 'information:', 'funded', 'by', 'a', 'grant')]
               if front_after_references
               else [('Grant', 'information:', 'funded', 'by', 'a', 'grant'),
                     ('Smith,', 'J.', '(2020).', 'A', 'reference')])
        )
        # One row per line, down the text area of the page, as a page sets them.
        lines = [
            _make_line(*text, y=100.0 + 5.0 * row) for row, text in enumerate(texts)
        ]
        title = lines[0]
        grant = lines[-1] if front_after_references else lines[-2]
        reference = lines[-2] if front_after_references else lines[-1]
        doc = _make_doc_with_page(LayoutBlock(lines=lines))
        index_of = {id(line): i for i, line in enumerate(lines)}
        annotated = _annotate(doc, {
            index_of[id(title)]: JatsFieldNames.TITLE,
            index_of[id(grant)]: JatsFieldNames.FUNDING,
            index_of[id(reference)]: JatsFieldNames.REFERENCE,
        })
        return doc, annotated, grant

    def test_front_matter_past_the_index_is_kept_before_the_references(self):
        doc, annotated, grant = self._make_doc(front_after_references=False)
        labels = _derive_labels(doc, annotated, front_max_start_line_index=80)
        assert labels[id(grant)] == SEG_FRONT

    def test_a_front_match_after_the_references_is_dropped(self):
        doc, annotated, grant = self._make_doc(front_after_references=True)
        labels = _derive_labels(doc, annotated, front_max_start_line_index=80)
        assert labels[id(grant)] != SEG_FRONT
