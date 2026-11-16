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
