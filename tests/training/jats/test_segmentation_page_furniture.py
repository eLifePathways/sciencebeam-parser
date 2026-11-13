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
    SEG_FOOTNOTE,
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
