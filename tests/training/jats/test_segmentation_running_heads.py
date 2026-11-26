"""What a journal runs across the top of its pages, and what reads like it.

A running head repeats in one place page after page, which an article's own
short title does too where the journal uses it as the head: each of these says
what the page does and asks which of them a line is.
"""
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_FRONT,
    SEG_HEADNOTE,
    SEG_PAGE,
    SEG_REFERENCES,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
)


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


class TestARunningHeadOfOneWord:
    """The journal's name heads its rectos and the authors' names its versos,
    and the title it runs across a verso breaks onto a one-word line."""

    def _make_doc(self):
        name = [
            make_line('MARSHES', y=140.0, x=380.0, page_number=page_number)
            for page_number in (1, 3, 5, 7)
        ]
        byline = [
            make_line('Egret', 'A', 'et', 'al.', y=140.0, x=40.0,
                      page_number=page_number)
            for page_number in (2, 4, 6, 8)
        ]
        carried = [
            make_line('Herons', 'wading', 'on', 'the', 'northern', y=164.0, x=40.0,
                      page_number=page_number)
            for page_number in (2, 4, 6, 8)
        ]
        broken = [
            make_line('marshes', y=178.0, x=40.0, page_number=page_number)
            for page_number in (2, 4, 6, 8)
        ]
        heads = {
            page_number: (
                [name[page_number // 2]] if page_number % 2 else
                [byline[page_number // 2 - 1]]
            )
            for page_number in (1, 2, 3, 4, 5, 6, 7, 8)
        }
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=heads[page_number])]
                + ([LayoutBlock(lines=[
                    carried[page_number // 2 - 1], broken[page_number // 2 - 1]
                ])] if page_number % 2 == 0 else [])
                + [LayoutBlock(lines=[make_line(
                    'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                    y=400.0, x=200.0, page_number=page_number,
                )])],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4, 5, 6, 7, 8)
        ])
        return doc, annotate(doc, {}), name, broken

    def test_the_name_on_the_row_the_byline_uses_is_a_head(self):
        doc, annotated, name, _broken = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in name] == [SEG_HEADNOTE] * 4

    def test_the_word_the_title_breaks_onto_goes_with_it(self):
        doc, annotated, _name, broken = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in broken] == [SEG_HEADNOTE] * 4


class TestTheWordATableIsCarriedOverBy:
    """A table says it goes on at the top of the page it goes on to, where the
    running head is."""

    def _make_doc(self, word: str):
        markers = [
            make_line(word, y=40.0, x=500.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[markers[page_number - 1]]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=40.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3)
        ])
        return doc, annotate(doc, {}), markers

    def test_it_is_not_the_running_head_however_it_is_spelt(self):
        for word in ('(continued)', '(continues)'):
            doc, annotated, markers = self._make_doc(word)
            labels = derive_labels(doc, annotated)
            assert SEG_HEADNOTE not in [labels[id(line)] for line in markers], word


class TestAHeadThatIsTheArticlesOwnTitle:
    """A journal runs the article's short title across its pages, and prints it
    again lower on the first page as the title."""

    def _make_doc(self):
        title = make_line('A', 'cartilaginous', 'tumour', ':', 'a', 'case',
                          y=74.0, x=40.0, page_number=1)
        heads = [
            make_line('A', 'cartilaginous', 'tumour', ':', 'a', 'case',
                      y=40.0, x=40.0, page_number=page_number)
            for page_number in (2, 3, 4)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[
                        title if page_number == 1 else heads[page_number - 2]
                    ]),
                    LayoutBlock(lines=[make_line(
                        'Herons', 'wade', f'on{page_number}', 'the', 'marsh',
                        y=400.0, x=40.0, page_number=page_number,
                    )]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3, 4)
        ])
        lines = list(doc.iter_all_lines())
        return doc, annotate(doc, {
            lines.index(title): JatsFieldNames.TITLE,
        }), title, heads

    def test_the_one_set_off_the_head_s_row_is_the_title(self):
        doc, annotated, title, _heads = self._make_doc()
        assert derive_labels(doc, annotated)[id(title)] == SEG_FRONT

    def test_the_rest_are_the_running_head(self):
        doc, annotated, _title, heads = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(head)] for head in heads] == [SEG_HEADNOTE] * 3


class TestAnEntryEndingAtTheTopOfAPage:
    """A reference entry whose page range and year end high on the page, where
    the number counting the pages is also nothing but its numbers."""

    def _make_doc(self):
        numbers = [
            make_line(str(40 + page_number), y=40.0, x=520.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        tail = make_line('15', '-', '58', ',', '2005', '.',
                         y=60.0, x=40.0, page_number=3)
        entries = [
            make_line('Teal', ',', 'R', '.', f'201{page_number}', '.', 'Marsh',
                      'birds', '.', y=200.0, x=40.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[numbers[page_number - 1]]),
                ] + ([LayoutBlock(lines=[tail])] if page_number == 3 else []) + [
                    LayoutBlock(lines=[entries[page_number - 1]]),
                ],
                meta=make_page_meta(page_number),
            )
            for page_number in (1, 2, 3)
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(entry): JatsFieldNames.REFERENCE for entry in entries}
        fields[lines.index(tail)] = JatsFieldNames.REFERENCE
        return doc, annotate(doc, fields), tail, numbers

    def test_the_end_of_an_entry_is_not_the_running_head(self):
        doc, annotated, tail, _numbers = self._make_doc()
        assert derive_labels(doc, annotated)[id(tail)] == SEG_REFERENCES

    def test_the_numbers_counting_the_pages_are_still_the_page(self):
        doc, annotated, _tail, numbers = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(number)] for number in numbers] == [SEG_PAGE] * 3
