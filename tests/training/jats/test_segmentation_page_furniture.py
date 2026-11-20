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
    SEG_PAGE,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
    page_number_of,
)


class TestTwoNumbersInOneMargin:
    """An affiliation marker is set where the page number is."""

    def _make_doc(self):
        marker = make_line('10', y=950.0, page_number=1)
        affiliation = make_line('A', 'university', 'somewhere', y=960.0, page_number=1)
        numbers = [
            make_line(str(page_number), y=950.0, x=300.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[make_line('The', 'argument', y=400.0)]),
                    LayoutBlock(lines=[marker, affiliation]),
                    LayoutBlock(lines=[numbers[0]]),
                ],
                meta=make_page_meta(1),
            ),
        ] + [
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[
                        make_line('more', 'argument', y=400.0, page_number=page_number)
                    ]),
                    LayoutBlock(lines=[numbers[page_number - 1]]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (2, 3)
        ])
        return doc, annotate(doc, {}), marker, numbers

    def test_the_number_that_counts_the_pages_is_the_page_number(self):
        doc, annotated, _, numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 3

    def test_the_number_that_counts_something_else_is_not(self):
        doc, annotated, marker, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(marker)] != SEG_PAGE


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


class TestAHeadCarryingThePageRangeOnTheOpeningPage:
    """A journal that prints the article's pages in the head of page one."""

    def _make_doc(self):
        heads = [
            make_line('Marsh', 'Bird', 'Review', ',', 'Vol', '.', '12', '(', '3', ')',
                      ':', '41', '-', '48', ',', 'May', '2019',
                      y=40.0, page_number=1),
            *[
                make_line('Marsh', 'Bird', 'Review', ',', 'Vol', '.', '12', '(', '3',
                          ')', ',', 'May', '2019', y=40.0, page_number=page_number)
                for page_number in (2, 3)
            ],
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[heads[page_number - 1]]),
                    LayoutBlock(lines=[
                        make_line('Herons', 'on', 'the', 'northern', 'marshes',
                                  y=400.0, page_number=page_number)
                    ]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3)
        ])
        lines = list(doc.iter_all_lines())
        # The journal's own city is the one the authors are affiliated to.
        return doc, annotate(doc, {
            lines.index(heads[0]): JatsFieldNames.AUTHOR_AFF,
        }), heads

    def test_the_head_the_affiliation_matched_is_still_a_running_head(self):
        doc, annotated, heads = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(head)] for head in heads] == [SEG_HEADNOTE] * 3


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


class TestAPageNumberInTheOuterCorner:
    """A journal sets the number in the outer corner, which swaps side."""

    def _make_doc(self):
        numbers = [
            make_line(
                str(page_number), y=100.0, page_number=page_number,
                x=20.0 if page_number % 2 == 0 else 560.0,
            )
            for page_number in (1, 2, 3, 4, 5)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[numbers[page_number - 1]]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5)
        ])
        return doc, annotate(doc, {}), numbers

    def test_the_number_is_the_page_number_on_a_verso_as_on_a_recto(self):
        doc, annotated, numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 5


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

    def _make_doc(self, front_matter_on_the_later_page: bool = False):
        title = make_line('Herons', 'of', 'the', 'northern', 'marshes', y=100.0)
        author = make_line('A', 'Nilsson', y=130.0)
        note = make_line(
            '1', 'This', 'survey', 'was', 'carried', 'out', 'on', 'the',
            'Varanger', 'marshes.', y=950.0,
        )
        feet = [
            make_line('Journal', 'of', 'Marsh', 'Birds,', 'v.', '12,', '2024',
                      y=965.0, page_number=page_number)
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

    def test_the_foot_printed_on_every_page_is_still_a_running_foot(self):
        doc, annotated, _, feet, _ = self._make_doc()
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


class TestThePageNumberSetSidewaysInTheMargin:
    """A number set up the side sits at one inset and at no one height."""

    def _make_doc(self, printed):
        numbers = [
            make_line(
                printed[page_number - 1], y=height, x=40.0, page_number=page_number,
            )
            for page_number, height in enumerate((620.0, 710.0, 680.0, 655.0, 700.0), 1)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[numbers[page_number - 1]]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=200.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5)
        ])
        return doc, annotate(doc, {}), numbers

    def test_a_number_that_advances_with_the_page_is_the_page_number(self):
        doc, annotated, numbers = self._make_doc(
            ['Página35', 'Página36', 'Página37', 'Página38', 'Página39']
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 5

    def test_a_number_that_counts_something_else_is_left_alone(self):
        doc, annotated, numbers = self._make_doc(['7', '2', '9', '4', '6'])
        labels = derive_labels(doc, annotated)
        assert SEG_PAGE not in [labels[id(number)] for number in numbers]


class TestAPageNumberPrintedInsideTheRunningHead:
    """A journal sets the number in the head row, beside what runs across it."""

    def _make_doc(self):
        heads = [
            make_line('Journal', 'of', 'Marsh', 'Birds,', 'v.', '12,', '2024',
                      y=30.0, x=40.0, page_number=page_number)
            for page_number in (1, 2, 3, 4, 5)
        ]
        numbers = [
            make_line(str(page_number), 'de', '5', y=30.0, x=520.0,
                      page_number=page_number)
            for page_number in (1, 2, 3, 4, 5)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[heads[page_number - 1]]),
                    LayoutBlock(lines=[numbers[page_number - 1]]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=200.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5)
        ])
        return doc, annotate(doc, {}), heads, numbers

    def test_the_number_beside_the_head_is_the_page_number(self):
        doc, annotated, _heads, numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 5

    def test_what_runs_across_the_head_is_still_the_head(self):
        doc, annotated, heads, _numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(head)] for head in heads] == [SEG_HEADNOTE] * 5


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


class TestAHeadThePageRangeMatched:
    """The deposit's page range is printed in the head, beside the number."""

    def _make_doc(self):
        heads = [
            make_line('Marsh', 'Bird', 'Review', ',', 'Vol', '.', '12', '(', '3', ')',
                      ':', '41', '-', '48', ',', 'May', '2019',
                      y=40.0, x=40.0, page_number=page_number)
            for page_number in (1, 2, 3, 4)
        ]
        numbers = [
            make_line(str(40 + page_number), y=40.0, x=520.0, page_number=page_number)
            for page_number in (1, 2, 3, 4)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[heads[page_number - 1]]),
                    LayoutBlock(lines=[numbers[page_number - 1]]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=200.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4)
        ])
        lines = list(doc.iter_all_lines())
        return doc, annotate(doc, {
            lines.index(head): JatsFieldNames.PAGE_NO for head in heads
        }), heads, numbers

    def test_the_head_the_range_matched_is_still_a_running_head(self):
        doc, annotated, heads, _numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(head)] for head in heads] == [SEG_HEADNOTE] * 4

    def test_the_number_beside_it_is_the_page_number(self):
        doc, annotated, _heads, numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 4


class TestATableOfNumbersBesideThePageNumber:
    """A table counts its rows, and one row happens to count as the pages do."""

    def _make_doc(self, cell_x: float, folio_on_the_table_page: bool):
        folio_pages = [
            page_number for page_number in (1, 2, 3, 4, 5)
            if page_number != 3 or folio_on_the_table_page
        ]
        numbers = {
            page_number: make_line(
                str(page_number), y=40.0, x=520.0, page_number=page_number
            )
            for page_number in folio_pages
        }
        cell = make_line('3', y=300.0, x=cell_x, page_number=3)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=(
                    [LayoutBlock(lines=[numbers[page_number]])]
                    if page_number in numbers else []
                ) + [
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=200.0, page_number=page_number,
                    )]),
                ] + ([LayoutBlock(lines=[cell])] if page_number == 3 else []),
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5)
        ])
        return doc, annotate(doc, {}), cell, list(numbers.values())

    def test_a_cell_set_across_the_measure_is_not_the_page_number(self):
        doc, annotated, cell, numbers = self._make_doc(
            cell_x=460.0, folio_on_the_table_page=False
        )
        labels = derive_labels(doc, annotated)
        assert labels[id(cell)] != SEG_PAGE
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 4

    def test_a_cell_set_where_the_number_is_leaves_that_page_unread(self):
        doc, annotated, cell, numbers = self._make_doc(
            cell_x=524.0, folio_on_the_table_page=True
        )
        labels = derive_labels(doc, annotated)
        assert labels[id(cell)] != SEG_PAGE
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 5


class TestAHeadTheBylineMatchedOnHalfThePages:
    """A journal runs the authors' surnames across its versos and nothing
    across its rectos."""

    def _make_doc(self):
        heads = [
            make_line('Egret', 'A', 'et', 'al.', y=40.0, x=40.0,
                      page_number=page_number)
            for page_number in (2, 4)
        ]
        carried = [
            make_line('the', 'marshes', 'of', 'the', 'north', 'in', 'the',
                      'shortening', 'days', y=70.0 + 14.0 * index, x=40.0,
                      page_number=4)
            for index in range(3)
        ]
        argument = {
            page_number: make_line(
                'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                y=200.0, x=40.0, page_number=page_number,
            )
            for page_number in (1, 2, 3, 4)
        }
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=(
                    [LayoutBlock(lines=[heads[page_number // 2 - 1]])]
                    if page_number % 2 == 0 else []
                ) + (
                    [LayoutBlock(lines=carried)] if page_number == 4 else []
                ) + [LayoutBlock(lines=[argument[page_number]])],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4)
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(head): JatsFieldNames.AUTHOR for head in heads}
        fields.update({
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in argument.values()
        })
        return doc, annotate(doc, fields), carried

    def test_the_page_under_it_is_not_the_front_matter(self):
        doc, annotated, carried = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert SEG_FRONT not in [labels[id(line)] for line in carried]


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
