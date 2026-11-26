"""The number a page is counted by, and what else reads like one.

A page number is a digit in a margin, which is also where a journal sets the
marker beside an author's name and where a table sets a column of figures, so
each of these says what the page does and asks which of them it is.
"""
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_HEADNOTE,
    SEG_PAGE,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
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


class TestAMarkerSetInTheMarginTheNumberUses:
    """A verso numbers its pages at the left, where a recto sets the superscript
    marker beside an author's name."""

    def _make_doc(self):
        folios = {
            page_number: make_line(
                str(page_number), y=950.0,
                x=40.0 if page_number % 2 == 0 else 520.0,
                page_number=page_number,
            )
            for page_number in (1, 2, 3, 4, 5, 6)
        }
        marker = make_line('1', y=200.0, x=40.0, page_number=1, font_size=5.0)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[folios[page_number]])]
                + ([LayoutBlock(lines=[marker])] if page_number == 1 else [])
                + [LayoutBlock(lines=[make_line(
                    'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                    y=400.0, x=200.0, page_number=page_number,
                )])],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5, 6)
        ])
        return doc, annotate(doc, {}), marker, folios

    def test_the_marker_is_not_the_number(self):
        doc, annotated, marker, folios = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert labels[id(marker)] != SEG_PAGE
        assert [labels[id(folios[n])] for n in (2, 4, 6)] == [SEG_PAGE] * 3


class TestANumberTheDepositCallsAnAffiliation:
    """The affiliation marker and the page number read alike, and the deposit
    says which it declared."""

    def _make_doc(self):
        folios = [
            make_line(str(page_number), y=300.0 + 40.0 * page_number, x=40.0,
                      page_number=page_number)
            for page_number in (1, 2, 3, 4, 5)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[folios[page_number - 1]]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=200.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5)
        ])
        lines = list(doc.iter_all_lines())
        return doc, annotate(doc, {
            lines.index(folios[0]): JatsFieldNames.AUTHOR_AFF,
        }), folios

    def test_the_one_the_deposit_declares_is_left_alone(self):
        doc, annotated, folios = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert labels[id(folios[0])] != SEG_PAGE
        assert [labels[id(folio)] for folio in folios[1:]] == [SEG_PAGE] * 4


class TestAPageNumberInAMirroredMargin:
    """A journal whose verso margin is a little wider than its recto."""

    def _make_doc(self):
        numbers = [
            make_line(
                str(page_number), y=100.0, page_number=page_number,
                x=45.0 if page_number % 2 else 545.0,
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

    def test_the_two_margins_are_one_corner(self):
        doc, annotated, numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 5


class TestNumbersSetAtADriftingInset:
    """A column of figures a chart steps a little further in on every spread."""

    # A step short enough that each figure is within a corner's width of the
    # one before it, while the first and the last are the measure apart.
    _SET = (
        (1, 20.0, '31'), (2, 20.0, '52'), (3, 44.0, '18'), (4, 44.0, '47'),
        (5, 68.0, '63'), (6, 68.0, '25'), (7, 92.0, '59'), (8, 92.0, '14'),
    )

    def _make_doc(self):
        numbers = [
            make_line(figure, y=100.0, x=inset, page_number=page_number)
            for page_number, inset, figure in self._SET
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[number]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for (page_number, _inset, _figure), number in zip(self._SET, numbers)
        ])
        return doc, annotate(doc, {}), numbers

    def test_no_corner_runs_across_the_page(self):
        doc, annotated, numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert SEG_PAGE not in [labels[id(number)] for number in numbers]
