"""What the heading a page prints says about the region under it.

Where the deposit places little or nothing in a region -- a reference list it
carries none of, an abstract in the other language it carries once -- the
heading the page sets above it is the evidence there is.
"""
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_ANNEX,
    SEG_BODY,
    SEG_CONTRIBUTION,
    SEG_FOOTNOTE,
    SEG_FRONT,
    SEG_REFERENCES,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
)


class TestTheHeadingTheBodyOpensWith:
    """A journal setting its front matter in one column prints the body beside it."""

    def _make_doc(self):
        front = [
            make_line('Nesting', 'herons', 'of', 'the', 'Fenwick', 'marshes',
                      y=100.0, x=40.0),
            make_line('We', 'counted', 'the', 'nests', 'of', 'four', 'colonies',
                      y=120.0, x=40.0),
        ]
        received = make_line('Received', 'on', 'March', '3', 'rd', ',', '2024',
                             y=300.0, x=40.0)
        heading = make_line('INTRODUCTION', y=100.0, x=320.0)
        opening = make_line('Herons', 'have', 'nested', 'in', 'the', 'marshes',
                            y=120.0, x=320.0)
        matched = make_line('since', 'the', 'reserve', 'was', 'drained', ',',
                            y=132.0, x=320.0)
        running_on = make_line('and', 'the', 'colonies', 'have', 'grown',
                               y=144.0, x=320.0)
        statement = make_line('Each', 'author', 'counted', 'one', 'colony',
                              y=100.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=front),
                    LayoutBlock(lines=[received]),
                    LayoutBlock(lines=[heading]),
                    LayoutBlock(lines=[opening, matched, running_on]),
                ],
                meta=make_page_meta(1),
            ),
            LayoutPage(blocks=[LayoutBlock(lines=[statement])],
                       meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(front[0]): JatsFieldNames.TITLE,
            lines.index(front[1]): JatsFieldNames.ABSTRACT,
            lines.index(matched): JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH,
            lines.index(statement): JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH,
        }
        return doc, annotate(doc, fields), heading, matched, running_on

    def test_the_heading_opens_the_body(self):
        doc, annotated, heading, _, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_BODY

    def test_the_argument_under_it_is_the_body(self):
        doc, annotated, _, _, running_on = self._make_doc()
        assert derive_labels(doc, annotated)[id(running_on)] == SEG_BODY

    def test_the_line_the_statement_matched_is_the_body_too(self):
        doc, annotated, _, matched, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(matched)] == SEG_BODY


class TestTheHeadingATrailingAbstractIsPrintedUnder:
    """A journal closing on its abstracts in the other languages."""

    def _make_doc(self):
        title = make_line('Nesting', 'herons', 'of', 'the', 'Fenwick', 'marshes',
                          y=100.0)
        abstract = make_line('We', 'counted', 'the', 'nests', 'of', 'four',
                             'colonies', y=120.0)
        body = [
            make_line('The', 'colonies', f'grew{index}', y=100.0 + 13 * index,
                      page_number=2)
            for index in range(3)
        ]
        entries = [
            make_line('Teal', ',', 'R', '.', f'201{index}', '.', 'Marsh',
                      'birds', '.', y=100.0 + 13 * index, page_number=3)
            for index in range(3)
        ]
        heading = make_line('RESUMO', y=100.0, page_number=4)
        printed_again = make_line('Ninhos', 'de', 'garças', 'nos', 'pântanos',
                                  'de', 'Fenwick', y=115.0, page_number=4)
        translated = [
            make_line('Contamos', 'os', 'ninhos', f'de{index}', 'quatro',
                      'colônias', y=140.0 + 13 * index, page_number=4)
            for index in range(3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[title, abstract])],
                       meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=body)], meta=make_page_meta(2)),
            LayoutPage(blocks=[LayoutBlock(lines=entries)],
                       meta=make_page_meta(3)),
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[heading, printed_again]),
                    LayoutBlock(lines=translated),
                ],
                meta=make_page_meta(4),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(title): JatsFieldNames.TITLE,
            lines.index(abstract): JatsFieldNames.ABSTRACT,
            lines.index(printed_again): JatsFieldNames.REFERENCE,
        }
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        for line in entries:
            fields[lines.index(line)] = JatsFieldNames.REFERENCE
        for line in translated:
            fields[lines.index(line)] = JatsFieldNames.ABSTRACT
        return doc, annotate(doc, fields), heading, printed_again, entries[-1]

    def test_the_heading_is_the_front_matter(self):
        doc, annotated, heading, _, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_FRONT

    def test_the_title_printed_beside_it_is_the_front_matter(self):
        doc, annotated, _, printed_again, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(printed_again)] == SEG_FRONT

    def test_the_list_above_it_is_left_alone(self):
        doc, annotated, _, _, entry = self._make_doc()
        assert derive_labels(doc, annotated)[id(entry)] == SEG_REFERENCES


class TestTheListAReferenceHeadingOpens:
    """A deposit whose reference list the aligner places almost none of."""

    def _make_doc(self):
        body = [
            make_line('The', 'colonies', f'grew{index}', y=100.0 + 13 * index)
            for index in range(3)
        ]
        statement = make_line('Each', 'author', 'counted', 'one', 'colony',
                              y=200.0)
        wrapped = make_line('references', y=213.0)
        heading = make_line('References', y=100.0, page_number=2)
        unplaced = [
            make_line('Teal', ',', 'R', '.', f'201{index}', '.', 'Marsh',
                      'birds', '.', y=120.0 + 13 * index, page_number=2)
            for index in range(4)
        ]
        placed = make_line('Wren', ',', 'P', '.', '2022', '.', 'Reed', 'beds',
                           '.', y=180.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body),
                               LayoutBlock(lines=[statement, wrapped])],
                       meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[heading] + unplaced + [placed])],
                       meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(statement): JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH,
            lines.index(wrapped): JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH,
            lines.index(placed): JatsFieldNames.REFERENCE,
        }
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), heading, unplaced, wrapped

    def test_the_heading_opens_the_reference_list(self):
        doc, annotated, heading, _, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_REFERENCES

    def test_the_entries_the_aligner_missed_are_the_reference_list(self):
        doc, annotated, _, unplaced, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in unplaced] == [SEG_REFERENCES] * 4

    def test_the_sentence_the_statement_ends_on_is_not_a_heading(self):
        doc, annotated, _, _, wrapped = self._make_doc()
        assert derive_labels(doc, annotated)[id(wrapped)] == SEG_CONTRIBUTION


class TestALineLeftAloneBetweenTwoRegions:
    """What the page repeats, the deposit carries once."""

    def _make_doc(self, alone_is_grounded: bool = False):
        body = [
            make_line('The', 'colonies', f'grew{index}', y=100.0 + 13 * index)
            for index in range(3)
        ]
        editors = [
            make_line('Teal', ',', 'R', '.', '(', 'Fenwick', 'Reserve', ')',
                      y=100.0 + 13 * index, page_number=2)
            for index in range(2)
        ]
        alone = make_line('Wren', ',', 'P', '.', '(', 'Fenwick', 'Reserve', ')',
                          y=126.0, page_number=2)
        after = [
            make_line('Snipe', ',', 'A', '.', '(', 'Marsh', 'Trust', ')',
                      y=139.0 + 13 * index, page_number=2)
            for index in range(2)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body)], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=editors + [alone] + after)],
                       meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {}
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        for line in editors + after:
            fields[lines.index(line)] = JatsFieldNames.APPENDIX
        if alone_is_grounded:
            fields[lines.index(alone)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), alone

    def test_the_line_takes_the_label_the_lines_around_it_agree_on(self):
        doc, annotated, alone = self._make_doc()
        assert derive_labels(doc, annotated)[id(alone)] == SEG_ANNEX

    def test_a_line_the_deposit_grounds_is_left_alone(self):
        doc, annotated, alone = self._make_doc(alone_is_grounded=True)
        assert derive_labels(doc, annotated)[id(alone)] == SEG_BODY


class TestTheLineAnEntryOfTheListWrapsOnto:
    """A reference entry naming its article in the words the argument used."""

    def _make_doc(self, wrapped: bool = True):
        body = [
            make_line('The', 'colonies', f'grew{index}', y=100.0 + 13 * index)
            for index in range(3)
        ]
        opening = make_line('12', '.', 'Teal', 'R', '.', 'The', 'wardens',
                            'counted', 'the', 'herons', 'of', 'the',
                            x=320.0, y=100.0, page_number=2)
        matched = make_line(
            *(('marshes', 'the', 'reserve', 'was', 'drained', 'from') if wrapped
              else ('13', '.', 'Snipe', 'A', '.', 'Marshes', 'of', 'Fenwick')),
            x=320.0, y=113.0, page_number=2,
        )
        under = make_line(
            *(('between', '1961', 'and', '1964', '.', 'J', '.', 'Marsh',
               'Birds', '.') if wrapped
              else ('Pipit', 'W', '.', '1972', '.', 'Wardens', '.')),
            x=320.0, y=126.0, page_number=2,
        )
        carried = [opening, matched, under]
        heading = make_line('References', x=20.0, y=100.0, page_number=2)
        unplaced = [
            make_line('Teal', ',', 'R', '.', f'201{index}', '.', 'Marsh',
                      'birds', '.', x=20.0, y=113.0 + 13 * index, page_number=2)
            for index in range(3)
        ]
        placed = make_line('Wren', ',', 'P', '.', '2022', '.', 'Reed', 'beds',
                           '.', x=20.0, y=152.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body)], meta=make_page_meta(1)),
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=carried),
                    LayoutBlock(lines=[heading] + unplaced + [placed]),
                ],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        fields[lines.index(placed)] = JatsFieldNames.REFERENCE
        fields[lines.index(matched)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), matched

    def test_the_line_the_entry_wrapped_onto_goes_back_to_the_list(self):
        doc, annotated, matched = self._make_doc()
        assert derive_labels(doc, annotated)[id(matched)] == SEG_REFERENCES

    def test_an_entry_that_opens_a_line_of_its_own_is_left_alone(self):
        doc, annotated, matched = self._make_doc(wrapped=False)
        assert derive_labels(doc, annotated)[id(matched)] == SEG_BODY


class TestTheNotesASectionGathersAfterTheBody:
    """A journal that prints its notes together at the end of the article."""

    def _make_doc(self, heading_text: str = 'NOTAS'):
        body = [
            make_line('The', 'colonies', f'grew{index}', y=100.0 + 13 * index)
            for index in range(3)
        ]
        foot = make_line('Journal', 'of', 'Marsh', 'Birds', ',', 'v', '.', '12',
                         y=965.0)
        heading = make_line(heading_text, y=300.0, page_number=2)
        notes = [
            make_line(f'{index + 1}', '.', 'The', 'reserve', 'was', 'drained',
                      'in', '1964', y=320.0 + 13 * index, page_number=2)
            for index in range(3)
        ]
        foot_again = make_line('Journal', 'of', 'Marsh', 'Birds', ',', 'v', '.',
                               '12', y=965.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body), LayoutBlock(lines=[foot])],
                       meta=make_page_meta(1)),
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[heading] + notes),
                    LayoutBlock(lines=[foot_again]),
                ],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {}
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        for line in notes:
            fields[lines.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), heading, notes, foot_again

    def test_the_heading_opens_an_appended_section(self):
        doc, annotated, heading, _, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_ANNEX

    def test_the_notes_under_it_are_the_section(self):
        doc, annotated, _, notes, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in notes] == [SEG_ANNEX] * 3

    def test_the_foot_the_page_prints_across_it_is_left_alone(self):
        doc, annotated, _, _, foot_again = self._make_doc()
        assert derive_labels(doc, annotated)[id(foot_again)] == SEG_FOOTNOTE
