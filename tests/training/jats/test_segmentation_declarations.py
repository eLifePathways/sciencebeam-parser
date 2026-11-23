"""What a journal declares on the pages it closes on.

The contributions, the conflict, the copyright and who the authors are print
one after another where the article has ended, and each is read from what the
page names it rather than from what the deposit filed it under.
"""
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_CONFLICT,
    SEG_CONTRIBUTION,
    SEG_FRONT,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
)


class TestTheAddressesAJournalClosesOn:
    """A journal that heads its block of author addresses "Authorship"."""

    def _make_doc(self, deposited_as_a_contribution: bool = False):
        body = [
            make_line('The', 'herons', f'waded{index}', y=40.0 + 10 * index)
            for index in range(4)
        ]
        references = [
            make_line('A', 'reference', f'listed{index}', y=100.0 + 20 * index)
            for index in range(6)
        ]
        authors = [
            make_line('Authorship', y=240.0, bold=True),
            make_line('Wren', 'Halloway', y=260.0, bold=True),
            make_line('Marsh', 'Institute', ',', 'Department', 'of', 'Wading',
                      'Birds', y=280.0),
            make_line('12', 'Reedbed', 'Lane', ',', 'Fenwick', y=300.0),
            make_line('https://orcid.org/0000-0002-0000-0001', y=320.0),
            make_line('Piet', 'Vansittart', y=340.0, bold=True),
            make_line('Coastal', 'Lagoon', 'Observatory', y=360.0),
            make_line('4', 'Saltpan', 'Row', ',', 'Tidewater', y=380.0),
            make_line('https://orcid.org/0000-0002-0000-0002', y=400.0),
        ]
        contributions = [
            make_line('Authors', "'", 'Contributions', y=430.0, bold=True),
            make_line('1', 'st', 'author', ':', 'conceptualization', '(', 'lead',
                      ')', ';', 'investigation', '(', 'lead', ')', '.', y=450.0),
            make_line('2', 'nd', 'author', ':', 'validation', '(', 'equal', ')',
                      ';', 'writing', '(', 'equal', ')', '.', y=470.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=body + references + authors + contributions
                )],
                meta=make_page_meta(1),
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.REFERENCE for line in references}
        for line in body:
            fields[order.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        # The deposit files the whole block among the notes on the authors, and
        # names only some of the lines in it.
        for line in authors:
            fields[order.index(line)] = JatsFieldNames.AUTHOR_NOTES
        fields[order.index(authors[1])] = JatsFieldNames.AUTHOR
        fields[order.index(authors[2])] = JatsFieldNames.AUTHOR_AFF
        fields[order.index(authors[4])] = JatsFieldNames.ORCID
        fields[order.index(authors[5])] = JatsFieldNames.AUTHOR
        fields[order.index(authors[8])] = JatsFieldNames.ORCID
        if deposited_as_a_contribution:
            # Printed directly under the addresses, so both are one run of
            # front matter until the deposit's own names part them.
            for line in contributions:
                fields[order.index(line)] = JatsFieldNames.CONTRIBUTION_NOTE
        return doc, annotate(doc, fields), authors, contributions

    def test_the_people_are_the_front_matter_under_their_heading(self):
        doc, annotated, authors, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in authors] == [SEG_FRONT] * 9

    def test_the_contributions_under_their_own_heading_are_a_statement(self):
        doc, annotated, _, contributions = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in contributions] == [SEG_CONTRIBUTION] * 3

    def test_what_the_deposit_calls_a_statement_is_one(self):
        doc, annotated, _, contributions = self._make_doc(
            deposited_as_a_contribution=True
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in contributions] == [SEG_CONTRIBUTION] * 3


class TestTheStandardsTheJournalPublishesUnder:
    """A journal that closes its last page on the ethics body it belongs to."""

    def _make_doc(self, statement: str):
        references = [
            make_line('A', 'reference', f'listed{index}', y=100.0 + 20 * index)
            for index in range(6)
        ]
        standing = make_line(*statement.split(), y=300.0, font_size=8.0)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=references),
                    LayoutBlock(lines=[standing]),
                ],
                meta=make_page_meta(1),
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.REFERENCE for line in references}
        return doc, annotate(doc, fields), standing

    def test_the_line_naming_the_body_is_the_front_matter(self):
        doc, annotated, standing = self._make_doc(
            'The Journal of Marsh Birds is a member of, and subscribes to the'
            ' principles of the Committee on Publication Ethics (COPE).'
        )
        assert derive_labels(doc, annotated)[id(standing)] == SEG_FRONT

    def test_a_line_that_names_no_body_is_left_to_the_list(self):
        doc, annotated, standing = self._make_doc(
            'The herons were counted twice a week through the wet season.'
        )
        assert derive_labels(doc, annotated)[id(standing)] != SEG_FRONT


class TestTheContributionsPrintedBeforeTheList:
    """A journal that gives the contributions a page of their own, set bold."""

    def _make_doc(self):
        body = [
            make_line('The', 'herons', f'waded{index}', y=100.0 + 20 * index,
                      page_number=1, font_size=12.0)
            for index in range(6)
        ]
        heading = make_line("AUTHORS'", 'CONTRIBUTIONS', y=100.0, page_number=2,
                            bold=True, font_size=10.0)
        statement = [
            make_line('Wren', 'Halloway', '(', 'WH', ')', y=140.0, page_number=2,
                      bold=True, font_size=10.0),
            make_line('Piet', 'Vansittart', '(', 'PV', ')', y=160.0, page_number=2,
                      bold=True, font_size=10.0),
            make_line('Conception', 'of', 'the', 'study', ':', y=180.0,
                      page_number=2, bold=True, font_size=10.0),
            make_line('WH', ',', 'PV', y=200.0, page_number=2, bold=True,
                      font_size=10.0),
            make_line('Data', 'acquisition', ',', 'analysis', 'and',
                      'interpretation', 'of', 'the', 'counts', ':', y=220.0,
                      page_number=2, bold=True, font_size=10.0),
            make_line('WH', ',', 'PV', y=240.0, page_number=2, bold=True,
                      font_size=10.0),
        ]
        references = [
            make_line('A', 'reference', f'listed{index}', y=100.0 + 20 * index,
                      page_number=3, font_size=12.0)
            for index in range(6)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body)], meta=make_page_meta(1)),
            LayoutPage(
                blocks=[LayoutBlock(lines=[heading])]
                + [LayoutBlock(lines=[line]) for line in statement],
                meta=make_page_meta(2),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=references)], meta=make_page_meta(3)
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {
            order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in body
        }
        for line in references:
            fields[order.index(line)] = JatsFieldNames.REFERENCE
        for line in statement[:2]:
            fields[order.index(line)] = JatsFieldNames.AUTHOR
        return doc, annotate(doc, fields), heading, statement

    def test_the_heading_opens_the_declaration_it_names(self):
        doc, annotated, heading, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_CONTRIBUTION

    def test_the_page_set_in_the_same_bold_is_all_of_it(self):
        doc, annotated, _, statement = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in statement] == [SEG_CONTRIBUTION] * 6


class TestTheConflictDeclaredWithoutAHeading:
    """A journal that closes on the sentence alone, under no heading at all."""

    def _make_doc(self, sentence: str):
        body = [
            make_line('The', 'herons', f'waded{index}', y=100.0 + 20 * index)
            for index in range(4)
        ]
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        opening, closing = sentence.split(' | ')
        declaration = [
            make_line(*opening.split(), y=360.0),
            make_line(*closing.split(), y=380.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=body + references),
                    LayoutBlock(lines=[declaration[0]]),
                    LayoutBlock(lines=[declaration[1]]),
                ],
                meta=make_page_meta(1),
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {
            order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in body
        }
        for line in references:
            fields[order.index(line)] = JatsFieldNames.REFERENCE
        for line in declaration:
            fields[order.index(line)] = JatsFieldNames.AUTHOR_NOTES
        return doc, annotate(doc, fields), declaration

    def test_the_sentence_and_the_line_it_runs_into_are_the_declaration(self):
        doc, annotated, declaration = self._make_doc(
            'The authors report no commercial, proprietary or financial'
            ' interest in the devices or | companies named in this article.'
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in declaration] == [SEG_CONFLICT] * 2

    def test_a_sentence_declaring_nothing_is_left_alone(self):
        doc, annotated, declaration = self._make_doc(
            'The herons were counted twice a week through the wet season and'
            ' again | at the end of it.'
        )
        labels = derive_labels(doc, annotated)
        assert labels[id(declaration[0])] != SEG_CONFLICT
