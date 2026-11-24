"""Furniture a page sets in its margins, and what separates it from content.

A page number, a running foot and the column an article happens to set beside
them are all read from where they are printed, so each of these says what the
page does and asks which of the two it is.
"""
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_BODY,
    SEG_FOOTNOTE,
    SEG_FRONT,
    SEG_HEADNOTE,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
    page_number_of,
)


class TestRunningFootTakenByAMatch:
    """A journal sets its own citation, page range and all, along every foot."""

    def _make_doc(self):
        feet = [
            make_line('Rev.', 'Ceres,', 'v.', '66,', 'p.', '486-492',
                      y=950.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[
                        make_line('The', 'argument', y=400.0, page_number=page_number)
                    ]),
                    LayoutBlock(lines=[feet[page_number - 1]]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3)
        ])
        lines = list(doc.iter_all_lines())
        return doc, annotate(doc, {lines.index(feet[0]): JatsFieldNames.PAGE_NO}), feet

    def test_the_foot_the_page_range_matched_is_still_a_running_foot(self):
        doc, annotated, feet = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(foot)] for foot in feet] == [SEG_FOOTNOTE] * 3


class TestATableRowCarriedPastTheFooterThreshold:
    """A table whose rows the page sets one at a time across the threshold."""

    def _make_doc(self, note_in_smaller_type: bool = False):
        body = [
            make_line('The', 'argument', f'of{page_number}', f'runs{index}',
                      y=200.0 + 20 * index, page_number=page_number, font_size=10.0)
            for page_number in (1, 2, 3)
            for index in range(8)
        ]
        by_page = {
            page_number: [line for line in body
                          if page_number_of(line) == page_number]
            for page_number in (1, 2, 3)
        }
        rows = [
            [make_line('Cell', f'one{index}', y=y, x=10.0, page_number=2,
                       font_size=8.0),
             make_line('Cell', f'two{index}', y=y, x=300.0, page_number=2,
                       font_size=8.0)]
            for index, y in enumerate((890.0, 905.0, 930.0, 945.0))
        ]
        note = make_line('1', 'A', 'note', 'under', 'the', 'column', y=930.0,
                         x=10.0, page_number=2, font_size=6.0)
        second_page = (
            [LayoutBlock(lines=by_page[2])]
            + [LayoutBlock(lines=row) for row in rows[:2]]
            + ([LayoutBlock(lines=[note])] if note_in_smaller_type
               else [LayoutBlock(lines=row) for row in rows[2:]])
        )
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=by_page[1])],
                       meta=make_page_meta(1)),
            LayoutPage(blocks=second_page, meta=make_page_meta(2)),
            LayoutPage(blocks=[LayoutBlock(lines=by_page[3])],
                       meta=make_page_meta(3)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        return doc, annotate(doc, fields), rows, note

    def test_the_rows_beyond_it_are_the_table_still(self):
        doc, annotated, rows, _note = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for row in rows[2:] for line in row] == [SEG_BODY] * 4

    def test_a_note_set_smaller_under_the_column_is_a_note(self):
        doc, annotated, _rows, note = self._make_doc(note_in_smaller_type=True)
        labels = derive_labels(doc, annotated)
        assert labels[id(note)] == SEG_FOOTNOTE


class TestANoteTheArticleCarriesOverAPageBreak:
    """A note split by the page break matches no value, and nothing grounds it.

    The page still says what it is: set last, under the text, in the type the
    notes it does carry are set in.
    """

    def _make_doc(self, carried_size: float):
        def body(page_number: int, count: int):
            return [
                make_line(
                    'The', 'marsh', 'held', f'herons{index}',
                    page_number=page_number, y=100.0 + 20 * index, font_size=10.0,
                )
                for index in range(count)
            ]

        placed_note = make_line(
            '1', 'From', 'the', 'ringing', 'scheme', "'", 's', 'returns',
            page_number=1, y=900.0, font_size=8.0,
        )
        carried_note = make_line(
            'counted', 'again', 'at', 'the', 'second', 'marsh',
            page_number=2, y=900.0, font_size=carried_size,
        )
        first, second = body(1, 8), body(2, 8)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=first), LayoutBlock(lines=[placed_note])],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=second), LayoutBlock(lines=[carried_note])],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in first + second
        }
        fields[lines.index(placed_note)] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), carried_note

    def test_the_half_on_the_next_page_goes_with_the_note(self):
        doc, annotated, carried = self._make_doc(carried_size=8.0)
        assert derive_labels(doc, annotated)[id(carried)] == SEG_FOOTNOTE

    def test_a_block_set_in_the_type_of_the_text_is_not_a_note(self):
        doc, annotated, carried = self._make_doc(carried_size=10.0)
        assert derive_labels(doc, annotated)[id(carried)] != SEG_FOOTNOTE


class TestNotesRunDownTheOuterMargin:
    """A journal that sets its numbered notes beside the text, not under it."""

    def _make_page(self, page_number: int):
        column = [
            make_line('The', 'marshes', 'held', f'birds{page_number}{index}',
                      x=100.0, y=86.0 + 14 * index, page_number=page_number)
            for index in range(6)
        ]
        note = [
            make_line(str(page_number), '.', 'The', 'marshes', 'were', 'drained',
                      x=25.0, y=86.0, page_number=page_number),
            make_line('in', 'the', f'spring{page_number}', 'of', 'that', 'year',
                      x=25.0, y=100.0, page_number=page_number),
            make_line('and', 'never', f'reflooded{page_number}', '.',
                      x=25.0, y=114.0, page_number=page_number),
        ]
        return column, note

    def _make_doc(self):
        pages = [self._make_page(page_number) for page_number in (1, 2, 3)]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=column), LayoutBlock(lines=note)],
                meta=make_page_meta(page_number),
            )
            for page_number, (column, note) in enumerate(pages, start=1)
        ])
        lines = list(doc.iter_all_lines())
        fields = {}
        for column, note in pages:
            for line in column:
                fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in note[1:]:
                fields[lines.index(line)] = JatsFieldNames.BACK_FOOTNOTE
            # The note opens on the words of the paragraph it annotates, so the
            # aligner matches its first line to both.
            fields[lines.index(note[0])] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        annotated = annotate(doc, fields)
        for _, note in pages:
            annotated.set_token_label(
                note[0].tokens[0], JatsFieldNames.BACK_FOOTNOTE
            )
        return doc, annotated, pages[1][1]

    def test_a_note_beside_the_column_is_not_the_column_carrying_on(self):
        doc, annotated, note = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in note[1:]] == [SEG_FOOTNOTE] * 2

    def test_the_line_the_note_opens_with_stays_with_it(self):
        doc, annotated, note = self._make_doc()
        assert derive_labels(doc, annotated)[id(note[0])] == SEG_FOOTNOTE


class TestAffiliationColumnRunningPastTheFooter:
    """A publisher sets the affiliations down the side of the title page."""

    def _make_doc(self):
        title = make_line('A', 'title', y=100.0)
        affiliation = make_line('A', 'university', 'somewhere', y=900.0)
        late = make_line('Another', 'university', 'elsewhere', y=930.0)
        last = make_line('A', 'last', 'university', 'nearby', y=950.0)
        overleaf = make_line(
            'A', 'third', 'university', 'elsewhere', y=100.0, page_number=2
        )
        grounded_overleaf = make_line(
            'A', 'fourth', 'university', 'far', 'away', y=120.0, page_number=2
        )
        body = [
            make_line('The', 'argument', f'runs{index}', y=200.0 + 20 * index,
                      page_number=2)
            for index in range(8)
        ]
        # The column is set beside the article, so the page emits the text of
        # it after the whole column rather than between its lines.
        beside_the_column = [
            make_line('The', 'argument', 'opens', f'here{index}',
                      y=200.0 + 20 * index, x=300.0)
            for index in range(6)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[title, affiliation]),
                    LayoutBlock(lines=[late]),
                    LayoutBlock(lines=[last]),
                    LayoutBlock(lines=beside_the_column),
                ],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=[overleaf, grounded_overleaf] + body
                )],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(title): JatsFieldNames.TITLE,
            lines.index(affiliation): JatsFieldNames.AUTHOR_AFF,
            lines.index(last): JatsFieldNames.AUTHOR_AFF,
            lines.index(grounded_overleaf): JatsFieldNames.AUTHOR_AFF,
        }
        for line in body + beside_the_column:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), {
            'late': late, 'overleaf': overleaf,
        }

    def test_the_column_is_front_matter_past_the_footer_threshold(self):
        doc, annotated, lines = self._make_doc()
        assert derive_labels(doc, annotated)[id(lines['late'])] == SEG_FRONT

    def test_it_carries_on_at_the_top_of_the_next_page(self):
        doc, annotated, lines = self._make_doc()
        assert derive_labels(doc, annotated)[id(lines['overleaf'])] == SEG_FRONT


class TestAMarginLineTheDepositPlacesElsewhere:
    """A table prints the same note under each page it runs over."""

    def _make_doc(self):
        note = 'Abbreviation - PD: Peritoneal Dialysis.'
        placed = make_line(*note.split(' '), y=400.0)
        in_the_margin = make_line(*note.split(' '), y=940.0, page_number=4)
        body = [
            make_line('The', 'argument', 'of', f'page{page_number}', f'line{index}',
                      y=500.0 + 20 * index, page_number=page_number)
            for page_number in (1, 2, 3, 4)
            for index in range(6)
        ]
        by_page = {
            page_number: [line for line in body
                          if page_number_of(line) == page_number]
            for page_number in (1, 2, 3, 4)
        }
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[placed] + by_page[1])],
                       meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=by_page[2])],
                       meta=make_page_meta(2)),
            LayoutPage(blocks=[LayoutBlock(lines=by_page[3])],
                       meta=make_page_meta(3)),
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=by_page[4]),
                    LayoutBlock(lines=[in_the_margin]),
                ],
                meta=make_page_meta(4),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(placed): JatsFieldNames.BODY_TABLE}
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), in_the_margin

    def test_the_one_in_the_margin_is_content_too(self):
        doc, annotated, in_the_margin = self._make_doc()
        assert derive_labels(doc, annotated)[id(in_the_margin)] == SEG_BODY


class TestAFootnoteWrappingOntoASecondLine:
    """A note at the foot of a page runs over the line it starts on."""

    def _make_doc(self):
        body = [
            make_line('The', 'argument', f'runs{index}', y=200.0 + 20 * index)
            for index in range(8)
        ]
        first = make_line('1', 'The', 'note', 'opens', 'here', y=930.0)
        second = make_line('and', 'carries', 'on', 'here', y=945.0)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=body), LayoutBlock(lines=[first, second])],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    make_line('and', 'goes', f'on{index}', y=200.0 + 20 * index,
                              page_number=2)
                    for index in range(4)
                ])],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        return doc, annotate(doc, fields), first, second

    def test_both_lines_stay_at_the_foot(self):
        doc, annotated, first, second = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(first)], labels[id(second)]] == [SEG_FOOTNOTE] * 2


class TestAShortLineAtTheFootOfAPage:
    """A one-word fragment the page sets at a foot says nothing about itself."""

    def _make_doc(self):
        fragment = 'R'
        in_the_body = make_line(fragment, y=400.0)
        in_the_margin = make_line(fragment, y=940.0, page_number=4)
        body = [
            make_line('The', 'argument', 'of', f'page{page_number}', f'line{index}',
                      y=500.0 + 20 * index, page_number=page_number)
            for page_number in (1, 2, 3, 4)
            for index in range(6)
        ]
        by_page = {
            page_number: [line for line in body
                          if page_number_of(line) == page_number]
            for page_number in (1, 2, 3, 4)
        }
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[in_the_body] + by_page[1])],
                       meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=by_page[2])],
                       meta=make_page_meta(2)),
            LayoutPage(blocks=[LayoutBlock(lines=by_page[3])],
                       meta=make_page_meta(3)),
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=by_page[4]),
                    LayoutBlock(lines=[in_the_margin]),
                ],
                meta=make_page_meta(4),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(in_the_body): JatsFieldNames.BODY_SECTION_PARAGRAPH}
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), in_the_margin

    def test_it_is_not_claimed_by_the_same_fragment_elsewhere(self):
        doc, annotated, in_the_margin = self._make_doc()
        assert derive_labels(doc, annotated)[id(in_the_margin)] == SEG_FOOTNOTE


class TestARunningFootSetAboveTheFooterZone:
    """A foot the page prints a little above where the footer zone begins."""

    def _make_doc(self):
        feet = [
            make_line('Journal', 'of', 'Marsh', 'Birds,', 'v.', '12,', '2024',
                      y=900.0, page_number=page_number)
            for page_number in (1, 2, 3, 4)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[feet[page_number - 1]]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4)
        ])
        return doc, annotate(doc, {}), feet

    def test_the_foot_the_page_repeats_is_read_from_where_it_is_printed(self):
        doc, annotated, feet = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(foot)] for foot in feet] == [SEG_FOOTNOTE] * 4


class TestTheMarginOfTheTitlePage:
    """What a journal prints at the foot of the page it opens the article on."""

    def _make_doc(
        self,
        front_matter_on_the_later_page: bool = False,
        foot: str = 'Journal of Marsh Birds, v. 12, 2024',
    ):
        title = make_line('Herons', 'of', 'the', 'northern', 'marshes', y=100.0)
        author = make_line('A', 'Nilsson', y=130.0)
        note = make_line(
            '1', 'This', 'survey', 'was', 'carried', 'out', 'on', 'the',
            'Varanger', 'marshes.', y=950.0,
        )
        feet = [
            make_line(*foot.split(), y=965.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        later_note = make_line(
            '2', 'The', 'count', 'was', 'repeated', 'in', 'the', 'autumn.',
            y=950.0, page_number=3,
        )
        amendments = make_line(
            'This', 'version', 'answers', 'the', 'reviewers.', y=150.0,
            page_number=3,
        )
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[title, author]),
                    LayoutBlock(lines=[note]),
                    LayoutBlock(lines=[feet[0]]),
                ],
                meta=make_page_meta(1),
            ),
        ] + [
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, page_number=page_number,
                    )]),
                ] + ([LayoutBlock(lines=[amendments])]
                     if page_number == 3 and front_matter_on_the_later_page
                     else []) + (
                    [LayoutBlock(lines=[later_note])] if page_number == 3 else []
                ) + [
                    LayoutBlock(lines=[feet[page_number - 1]]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (2, 3)
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(title): JatsFieldNames.TITLE,
            lines.index(author): JatsFieldNames.AUTHOR,
            lines.index(note): JatsFieldNames.BACK_FOOTNOTE,
            lines.index(later_note): JatsFieldNames.BACK_FOOTNOTE,
        }
        if front_matter_on_the_later_page:
            fields[lines.index(amendments)] = JatsFieldNames.ABSTRACT
        annotated = annotate(doc, fields)
        return doc, annotated, note, feet, later_note

    def test_the_note_printed_once_beside_the_front_matter_is_front_matter(self):
        doc, annotated, note, _, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(note)] == SEG_FRONT

    def test_the_foot_the_opening_page_prints_is_the_front_matter(self):
        doc, annotated, _, feet, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(foot)] for foot in feet] == [
            SEG_FRONT, SEG_FOOTNOTE, SEG_FOOTNOTE
        ]

    def test_the_stamp_the_file_carries_is_not_the_front_matter(self):
        doc, annotated, _, feet, _ = self._make_doc(
            foot='Powered by TCPDF (www.tcpdf.org)'
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(foot)] for foot in feet] == [SEG_FOOTNOTE] * 3

    def test_a_note_on_a_page_the_front_matter_does_not_reach_is_a_footnote(self):
        doc, annotated, _, _, later_note = self._make_doc()
        assert derive_labels(doc, annotated)[id(later_note)] == SEG_FOOTNOTE

    def test_front_matter_printed_further_in_does_not_make_a_title_page(self):
        doc, annotated, _, _, later_note = self._make_doc(
            front_matter_on_the_later_page=True
        )
        assert derive_labels(doc, annotated)[id(later_note)] == SEG_FOOTNOTE


class TestARunningFootCarryingThePageNumber:
    """A verso sets the page number beside the journal's name, a recto does not."""

    def _make_doc(self):
        feet = [
            make_line(
                *(([str(page_number)] if page_number % 2 == 0 else [])
                  + ['Journal', 'of', 'Marsh', 'Birds,', 'v.', '12,', '2024']),
                y=950.0, page_number=page_number,
            )
            for page_number in (1, 2, 3, 4, 5)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, page_number=page_number,
                    )]),
                    LayoutBlock(lines=[feet[page_number - 1]]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5)
        ])
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            lines.index(feet[3]): JatsFieldNames.BODY_SECTION_PARAGRAPH,
        })
        return doc, annotated, feet

    def test_the_foot_a_paragraph_matched_is_still_a_running_foot(self):
        doc, annotated, feet = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(foot)] for foot in feet] == [SEG_FOOTNOTE] * 5


class TestAFootThatIsNothingButItsNumbers:
    """A journal whose running foot is the year alone, and a reference entry
    that ends low on the page in a page range and a year."""

    def _make_doc(self):
        feet = [
            make_line('(', '2024', ')', y=950.0, page_number=page_number)
            for page_number in (1, 2, 3, 4, 5)
        ]
        tail = make_line('15', '-', '58', ',', '2005', '.', y=700.0, x=40.0,
                         page_number=3)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[make_line(
                        'Egret', 'A.', f'Wading{page_number}', 'in', 'the',
                        'reeds.', 'Marsh', 'Press.', y=200.0, x=40.0,
                        page_number=page_number,
                    )]),
                ] + ([LayoutBlock(lines=[tail])] if page_number == 3 else []) + [
                    LayoutBlock(lines=[feet[page_number - 1]]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5)
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(tail): JatsFieldNames.REFERENCE}
        for page in doc.pages:
            fields[lines.index(page.blocks[0].lines[0])] = JatsFieldNames.REFERENCE
        return doc, annotate(doc, fields), tail, feet

    def test_the_end_of_an_entry_is_not_the_foot(self):
        doc, annotated, tail, feet = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert labels[id(tail)] != SEG_FOOTNOTE
        assert [labels[id(foot)] for foot in feet] == [SEG_FOOTNOTE] * 5


class TestAChartWhoseLastRowCrossesTheFooter:
    """A chart staggers the lines of a row, so one cell of it falls past the
    threshold while the cell beside it does not."""

    def _make_doc(self, with_a_cell_beside_it: bool):
        feet = [
            make_line('Journal', 'of', 'Marsh', 'Birds,', 'v.', '12,', '2024',
                      y=960.0, x=40.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        cell = make_line('Egret', 'et', 'al.', '14', y=926.0, x=200.0,
                         page_number=2, font_size=7.0)
        beside = make_line('NS', ':', 'not', 'stated', ';', 'IMTG', ':', 'the',
                           'training', 'group', y=922.0, x=320.0,
                           page_number=2, font_size=7.0)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=40.0, page_number=page_number,
                    )]),
                ] + ([
                    LayoutBlock(lines=[cell]),
                ] + ([LayoutBlock(lines=[beside])] if with_a_cell_beside_it else [])
                    if page_number == 2 else []
                ) + [LayoutBlock(lines=[feet[page_number - 1]])],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3)
        ])
        order = list(doc.iter_all_lines())
        fields = (
            {order.index(beside): JatsFieldNames.BODY_TABLE}
            if with_a_cell_beside_it else {}
        )
        return doc, annotate(doc, fields), cell, feet

    def test_the_cell_past_the_threshold_is_not_a_foot(self):
        doc, annotated, cell, feet = self._make_doc(with_a_cell_beside_it=True)
        labels = derive_labels(doc, annotated)
        assert labels[id(cell)] != SEG_FOOTNOTE
        assert [labels[id(foot)] for foot in feet] == [SEG_FOOTNOTE] * 3

    def test_a_foot_with_nothing_beside_it_is_a_foot(self):
        doc, annotated, cell, _feet = self._make_doc(with_a_cell_beside_it=False)
        labels = derive_labels(doc, annotated)
        assert labels[id(cell)] == SEG_FOOTNOTE


class TestTheTabAJournalPrintsUpItsEdge:
    """A journal that names the kind of article up the outer edge of a page."""

    def _make_doc(self, in_the_margin: bool = True):
        pages = range(1, 4)
        columns = {
            page_number: [
                make_line('The', 'marshes', 'held', f'birds{page_number}{index}',
                          x=100.0, y=100.0 + 14 * index, page_number=page_number)
                for index in range(6)
            ]
            for page_number in pages
        }
        tabs = [
            make_line(
                'Research',
                x=(20.0 if page_number % 2 else 560.0) if in_the_margin else 110.0,
                y=500.0, page_number=page_number,
            )
            for page_number in pages
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=columns[page_number]),
                    LayoutBlock(lines=[tabs[page_number - 1]]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in pages
        ])
        order = list(doc.iter_all_lines())
        fields = {
            order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for lines in columns.values() for line in lines
        }
        return doc, annotate(doc, fields), tabs

    def test_the_tab_is_the_furniture_of_every_page(self):
        doc, annotated, tabs = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(tab)] for tab in tabs] == [SEG_HEADNOTE] * 3

    def test_the_same_word_set_in_the_column_is_not(self):
        doc, annotated, tabs = self._make_doc(in_the_margin=False)
        labels = derive_labels(doc, annotated)
        assert [labels[id(tab)] for tab in tabs] != [SEG_HEADNOTE] * 3


class TestTheLinesAPageSetsAmongItsNotes:
    """A marker the deposit carries nothing for, and a note matched as body."""

    def _make_doc(self):
        body = [
            make_line('The', 'marshes', 'held', f'birds{index}', x=100.0,
                      y=100.0 + 14 * index, page_number=1, font_size=12.0)
            for index in range(6)
        ]
        marker = make_line('6', x=90.0, y=740.0, page_number=1, font_size=6.0)
        note = [
            make_line('Taken', 'from', 'the', 'count', 'of', 'that', 'spring',
                      x=100.0, y=741.0, page_number=1, font_size=8.0),
            make_line('7', 'The', 'herons', 'were', 'counted', 'at', 'dawn',
                      x=100.0, y=755.0, page_number=1, font_size=8.0),
            make_line('and', 'again', 'at', 'dusk', '.', x=100.0, y=769.0,
                      page_number=1, font_size=8.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=body),
                    LayoutBlock(lines=[marker]),
                    LayoutBlock(lines=note),
                ],
                meta=make_page_meta(1),
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {
            order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in body
        }
        for line in note:
            fields[order.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        # The note gives the same count as a sentence of the body, so the
        # deposit matches one of its lines to the paragraph instead.
        fields[order.index(note[1])] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), marker, note

    def test_the_marker_on_the_row_the_note_opens_is_the_note(self):
        doc, annotated, marker, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(marker)] == SEG_FOOTNOTE

    def test_the_line_set_in_the_block_of_notes_is_a_note(self):
        doc, annotated, _, note = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in note] == [SEG_FOOTNOTE] * 3


class TestTheSourceLineAFigureIsPrintedUnder:
    """A caption's source line carrying the address a note gives as well."""

    def _make_doc(self):
        caption = make_line('Figure', '4', '-', 'The', 'marsh', 'at', 'dawn',
                            x=100.0, y=300.0, page_number=1, font_size=12.0)
        source = make_line('Source', ':', 'the', 'count', 'of', 'that',
                           'spring', x=100.0, y=320.0, page_number=1,
                           font_size=9.0)
        body = [
            make_line('The', 'marshes', 'held', f'birds{index}', x=100.0,
                      y=340.0 + 14 * index, page_number=1, font_size=12.0)
            for index in range(6)
        ]
        note = [
            make_line('7', 'The', 'herons', 'were', 'counted', 'at', 'dawn',
                      x=100.0, y=740.0, page_number=1, font_size=8.0),
            make_line('and', 'again', 'at', 'dusk', '.', x=100.0, y=754.0,
                      page_number=1, font_size=8.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[caption]),
                    LayoutBlock(lines=[source]),
                    LayoutBlock(lines=body),
                    LayoutBlock(lines=note),
                ],
                meta=make_page_meta(1),
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {order.index(caption): JatsFieldNames.BODY_FIGURE}
        for line in body:
            fields[order.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        for line in note:
            fields[order.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        # The note gives the same source as the line under the caption.
        fields[order.index(source)] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), source, note

    def test_the_line_among_the_text_is_the_text(self):
        doc, annotated, source, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(source)] == SEG_BODY

    def test_the_notes_at_the_foot_are_left_alone(self):
        doc, annotated, _, note = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in note] == [SEG_FOOTNOTE] * 2


class TestANoteSetInTheOtherColumn:
    """A page that sets its notes beside an address block, not under it."""

    def _make_doc(self):
        body = [
            make_line('The', 'marshes', 'held', f'birds{index}', x=100.0,
                      y=100.0 + 14 * index, page_number=1, font_size=12.0)
            for index in range(6)
        ]
        address = [
            make_line('Marsh', 'Institute', ',', 'Department', 'of', 'Wading',
                      x=100.0, y=400.0, page_number=1, font_size=10.0),
            make_line('Birds', x=100.0, y=414.0, page_number=1, font_size=10.0),
        ]
        note = [
            make_line('7', 'The', 'herons', 'were', 'counted', 'at', 'dawn',
                      x=400.0, y=400.0, page_number=1, font_size=10.0),
            make_line('and', 'again', 'at', 'dusk', '.', x=400.0, y=414.0,
                      page_number=1, font_size=10.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=body),
                    LayoutBlock(lines=address),
                    LayoutBlock(lines=note),
                ],
                meta=make_page_meta(1),
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {
            order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in body
        }
        for line in address:
            fields[order.index(line)] = JatsFieldNames.AUTHOR_NOTES
        for line in note:
            fields[order.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), address, note

    def test_the_line_level_with_it_is_not_the_note_s(self):
        doc, annotated, address, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert SEG_FOOTNOTE not in [labels[id(line)] for line in address]

    def test_the_note_beside_it_is_still_a_note(self):
        doc, annotated, _, note = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in note] == [SEG_FOOTNOTE] * 2


class TestTheQuotationAPageSetsAmongItsText:
    """A deposit files the author's translation of a quotation as a note."""

    def _make_doc(self):
        argument = [
            make_line('The', 'warden', 'of', 'the', 'reserve', f'wrote{index}',
                      y=100.0 + 15 * index, x=57.0, font_size=11.5)
            for index in range(3)
        ]
        quoted = [
            make_line('Rather', 'than', 'count', 'the', 'nests', 'from', 'the',
                      f'hide{index}', y=160.0 + 15 * index, x=117.0,
                      font_size=11.0)
            for index in range(3)
        ]
        after = [
            make_line('The', 'wardens', 'kept', 'to', 'that', f'rule{index}',
                      y=220.0 + 15 * index, x=57.0, font_size=11.5)
            for index in range(2)
        ]
        note = make_line('8', 'A', 'hide', 'is', 'a', 'hut', 'the', 'wardens',
                         'watch', 'from', y=900.0, x=57.0, font_size=8.5)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=argument),
                    LayoutBlock(lines=quoted),
                    LayoutBlock(lines=after),
                    LayoutBlock(lines=[note]),
                ],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(note): JatsFieldNames.BACK_FOOTNOTE}
        for line in argument + after:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        for line in quoted:
            fields[lines.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), quoted, note

    def test_the_block_set_in_among_the_text_is_the_body(self):
        doc, annotated, quoted, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in quoted] == [SEG_BODY] * 3

    def test_the_note_the_page_sets_at_its_foot_is_left_alone(self):
        doc, annotated, _, note = self._make_doc()
        assert derive_labels(doc, annotated)[id(note)] == SEG_FOOTNOTE
