from typing import Sequence

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
)
from sciencebeam_parser.training.jats.segmentation import get_lines_in_printed_order

from tests.training.jats.segmentation_fixtures import make_line, make_page_meta


class TestLinesInPrintedOrder:
    """The order the page sets a page's lines in, where it is not the order given.

    pdfalto reads a page column by column, so a block running the full width
    under two columns can arrive before the column above it.  Said without a
    column model: blocks that overlap each other down the page are the page's
    columns and keep the order they came in, and the rest the page sets one
    below the next.
    """

    @staticmethod
    def _page(*blocks: Sequence[LayoutLine]) -> LayoutDocument:
        return LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=list(lines)) for lines in blocks],
                       meta=make_page_meta(1, height=1000.0)),
        ])

    def _texts(self, doc: LayoutDocument, furniture=frozenset()):
        return [line.text for line in get_lines_in_printed_order(doc, set(furniture))]

    def test_a_block_the_page_sets_under_the_column_above_it(self):
        # The foot of the page, handed over before the column it sits below.
        doc = self._page(
            [make_line('Grant information and who it came from', y=800.0),
             make_line('Copyright and the licence it is under', y=820.0)],
            [make_line('The last words of a statement above it', y=100.0),
             make_line('and the heading of what follows them', y=130.0)],
        )
        assert self._texts(doc)[:2] == [
            'The last words of a statement above it',
            'and the heading of what follows them',
        ]

    def test_a_block_the_page_sets_above_the_ones_handed_over_before_it(self):
        # One journal hands over the affiliations under its byline last of all.
        doc = self._page(
            [make_line('A title the page carries', y=100.0),
             make_line('and the byline under it', y=120.0)],
            [make_line('The argument of the article begins here', y=300.0),
             make_line('and carries on down the page', y=320.0)],
            [make_line('The affiliations set under the byline', y=150.0),
             make_line('and where the second author works', y=170.0)],
        )
        assert self._texts(doc)[2:4] == [
            'The affiliations set under the byline',
            'and where the second author works',
        ]

    def test_a_block_of_one_line_goes_where_its_section_goes(self):
        # A heading is one line and says less about the page than the block it
        # stands over, so it is not placed on its own: the section takes it.
        doc = self._page(
            [make_line('A heading the page sets above it', y=270.0)],
            [make_line('The argument of the article begins here', y=300.0),
             make_line('and carries on down the page', y=320.0)],
            [make_line('A title the page carries', y=100.0),
             make_line('and the byline under it', y=120.0)],
        )
        assert self._texts(doc) == [
            'A title the page carries',
            'and the byline under it',
            'A heading the page sets above it',
            'The argument of the article begins here',
            'and carries on down the page',
        ]

    def test_a_page_handed_over_the_way_it_is_set(self):
        doc = self._page(
            [make_line('The first line the page sets', y=100.0),
             make_line('the second line under it', y=130.0)],
            [make_line('and the third under that', y=160.0),
             make_line('and the fourth below it', y=190.0)],
        )
        assert self._texts(doc) == [
            'The first line the page sets',
            'the second line under it',
            'and the third under that',
            'and the fourth below it',
        ]

    def test_a_panel_set_beside_the_column_keeps_the_order_it_came_in(self):
        # `ore` sets its peer review panel beside the first column, and the
        # blocks of the two interleave down the page without touching: read by
        # where they sit alone, the panel is shuffled into the abstract.
        doc = self._page(
            [make_line('A title the page sets over both', y=100.0, x=50.0),
             make_line('and the byline under it', y=120.0, x=50.0)],
            [make_line('The abstract begins here', y=300.0, x=50.0),
             make_line('and carries on to here', y=320.0, x=50.0)],
            [make_line('and after a break it resumes', y=400.0, x=50.0),
             make_line('and finishes here', y=420.0, x=50.0)],
            [make_line('Open Peer Review', y=350.0, x=350.0),
             make_line('and who approved it', y=370.0, x=350.0)],
        )
        assert self._texts(doc) == [
            'A title the page sets over both',
            'and the byline under it',
            'The abstract begins here',
            'and carries on to here',
            'and after a break it resumes',
            'and finishes here',
            'Open Peer Review',
            'and who approved it',
        ]

    def test_two_columns_the_second_of_which_reaches_higher(self):
        # The columns overlap each other, so nothing says the page sets either
        # after the other and they keep the order they came in.
        doc = self._page(
            [make_line('The left column runs down', y=100.0, x=50.0),
             make_line('to the foot of the page', y=800.0, x=50.0)],
            [make_line('and the right column', y=100.0, x=350.0),
             make_line('beside it does the same', y=800.0, x=350.0)],
        )
        assert self._texts(doc) == [
            'The left column runs down',
            'to the foot of the page',
            'and the right column',
            'beside it does the same',
        ]

    def test_the_page_number_under_everything_is_not_asked_about(self):
        # The page number is handed over last and set below the block that is
        # out of place, so counted as content it would deny the page the
        # question.
        doc = self._page(
            [make_line('Grant information and who it came from', y=800.0),
             make_line('Copyright and the licence it is under', y=820.0)],
            [make_line('The last words of a statement above it', y=100.0),
             make_line('and the heading of what follows them', y=130.0)],
            [make_line('Page 2 of 29', y=950.0)],
        )
        number = doc.pages[0].blocks[2].lines[0]
        assert self._texts(doc, {id(number)})[0] == 'The last words of a statement above it'
