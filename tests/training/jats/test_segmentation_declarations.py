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
    SEG_ACKNOWLEDGEMENT,
    SEG_BODY,
    SEG_CONFLICT,
    SEG_CONTRIBUTION,
    SEG_FOOTNOTE,
    SEG_FUNDING,
    SEG_FRONT,
    SEG_REFERENCES,
    SEG_REVIEW,
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

    def test_an_entry_that_cites_the_body_is_left_to_the_list(self):
        entries = [
            make_line('A', 'reference', f'listed{index}', y=100.0 + 20 * index)
            for index in range(2)
        ]
        cited = make_line('Committee', 'on', 'Publication', 'Ethics', '.',
                          'Guidelines', 'on', 'good', 'practice', '.', '2003', '.',
                          y=140.0)
        after = [
            make_line('A', 'reference', f'listed{index}', y=160.0 + 20 * index)
            for index in range(3)
        ]
        listed = entries + [cited] + after
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=listed)], meta=make_page_meta(1)),
        ])
        order = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            order.index(line): JatsFieldNames.REFERENCE for line in listed
        })
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in listed] == [SEG_REFERENCES] * 6

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


class TestTheThanksTheOpeningPagePrints:
    """A journal that calls its thanks from the title and sets them at the foot."""

    def _make_doc(self, opening: str, deposited: bool = False):
        title = make_line('Herons', 'of', 'the', 'northern', 'marshes', '*', y=100.0)
        author = make_line('A', 'Nilsson', y=130.0)
        body = [
            make_line('The', 'herons', f'waded{index}', y=200.0 + 20 * index)
            for index in range(4)
        ]
        thanks = [
            make_line(*opening.split(), y=900.0, font_size=8.0),
            make_line('on', 'an', 'earlier', 'draft', 'of', 'this', 'paper', '.',
                      y=912.0, font_size=8.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[title, author]),
                    LayoutBlock(lines=body),
                    LayoutBlock(lines=thanks),
                ],
                meta=make_page_meta(1),
            ),
        ])
        order = list(doc.iter_all_lines())
        fields = {
            order.index(title): JatsFieldNames.TITLE,
            order.index(author): JatsFieldNames.AUTHOR,
        }
        for line in body:
            fields[order.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        if deposited:
            for line in thanks:
                fields[order.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), body, thanks

    def test_the_note_of_thanks_is_an_acknowledgement(self):
        doc, annotated, _, thanks = self._make_doc(
            '* We thank the wardens of the Fenwick reserve for their comments'
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in thanks] == [SEG_ACKNOWLEDGEMENT] * 2

    def test_the_argument_the_note_is_printed_under_is_left_alone(self):
        doc, annotated, body, _ = self._make_doc(
            '* We thank the wardens of the Fenwick reserve for their comments'
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in body] == [SEG_BODY] * 4

    def test_the_thanks_the_deposit_carries_as_a_note_are_an_acknowledgement(self):
        doc, annotated, _, thanks = self._make_doc(
            '* We thank the wardens of the Fenwick reserve for their comments',
            deposited=True,
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in thanks] == [SEG_ACKNOWLEDGEMENT] * 2

    def test_a_note_that_thanks_nobody_is_left_alone(self):
        doc, annotated, _, thanks = self._make_doc(
            '* The counts were made from the hide at the north end'
        )
        labels = derive_labels(doc, annotated)
        assert labels[id(thanks[0])] != SEG_ACKNOWLEDGEMENT


class TestTheAuthorsAClosingStatementRanOnOver:
    """A journal that thanks its helpers and then names its authors."""

    def _make_doc(self):
        body = [
            make_line('The', 'colonies', f'grew{index}', y=100.0 + 13 * index)
            for index in range(3)
        ]
        thanks = [
            make_line('We', 'thank', 'the', 'wardens', 'of', 'the', 'Fenwick',
                      f'reserve{index}', y=100.0 + 13 * index, page_number=2)
            for index in range(2)
        ]
        named = make_line('Alida', 'Teal', y=140.0, page_number=2)
        orcid = make_line('ORCID', ':', 'https://orcid.org/0000-0002-1825-0097',
                          y=153.0, page_number=2)
        works_at = make_line('Reader', 'in', 'Wetland', 'Ecology', ',',
                             'Fenwick', 'Institute', y=166.0, page_number=2)
        writes_to = make_line('E', '-', 'mail', ':', 'a.teal@example.org',
                              y=179.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body)], meta=make_page_meta(1)),
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=thanks),
                    LayoutBlock(lines=[named, orcid, works_at, writes_to]),
                ],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(named): JatsFieldNames.AUTHOR,
            lines.index(orcid): JatsFieldNames.ORCID,
            lines.index(writes_to): JatsFieldNames.EMAIL,
        }
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        for line in thanks:
            fields[lines.index(line)] = JatsFieldNames.ACK_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), thanks, [named, orcid, works_at, writes_to]

    def test_the_authors_are_the_front_matter(self):
        doc, annotated, _, authors = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in authors] == [SEG_FRONT] * 4

    def test_the_thanks_above_them_are_left_alone(self):
        doc, annotated, thanks, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in thanks] == [SEG_ACKNOWLEDGEMENT] * 2


class TestThePeopleAJournalCreditsOnItsClosingPage:
    """A journal naming its editors and its reviewers where the article ends."""

    def _make_doc(self, on_the_opening_page: bool = False):
        page = 1 if on_the_opening_page else 2
        top = 100.0 if on_the_opening_page else 300.0
        credited = [
            make_line('EDITOR', '-', 'IN', '-', 'CHIEF', y=top, page_number=page),
            make_line('Alida', 'Teal', '(', 'Fenwick', 'Institute', ')',
                      y=top + 15, page_number=page),
            make_line('GUEST', 'EDITORS', y=top + 30, page_number=page),
            make_line('Peregrine', 'Snipe', '(', 'Marsh', 'Trust', ')',
                      y=top + 45, page_number=page),
        ]
        reviewed = [
            make_line('REVIEWERS', y=top + 60, page_number=page),
            make_line('Wren', 'Pipit', '(', 'Fenwick', 'Institute', ')',
                      y=top + 75, page_number=page),
            make_line('PEER', 'REVIEW', 'REPORT', y=top + 90, page_number=page),
            make_line('The', 'report', 'is', 'at', 'https://example.org/r/1',
                      y=top + 105, page_number=page),
        ]
        body = [
            make_line('The', 'colonies', f'grew{index}', y=500.0 + 13 * index)
            for index in range(3)
        ]
        pages = [LayoutPage(blocks=[LayoutBlock(lines=body)], meta=make_page_meta(1))]
        shown = [LayoutBlock(lines=[line]) for line in credited + reviewed]
        if on_the_opening_page:
            pages[0] = LayoutPage(blocks=shown + [LayoutBlock(lines=body)],
                                  meta=make_page_meta(1))
        else:
            pages.append(LayoutPage(blocks=shown, meta=make_page_meta(2)))
        doc = LayoutDocument(pages=pages)
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        for line in credited + reviewed:
            fields[lines.index(line)] = JatsFieldNames.AUTHOR_NOTES
        return doc, annotate(doc, fields), credited, reviewed

    def test_the_editors_are_the_front_matter(self):
        doc, annotated, credited, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in credited] == [SEG_FRONT] * 4

    def test_the_reviewers_and_their_report_are_the_review(self):
        doc, annotated, _, reviewed = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in reviewed] == [SEG_REVIEW] * 4

    def test_a_box_the_opening_page_prints_is_left_to_the_front_matter(self):
        doc, annotated, _, reviewed = self._make_doc(on_the_opening_page=True)
        labels = derive_labels(doc, annotated)
        assert SEG_REVIEW not in [labels[id(line)] for line in reviewed]


class TestTheHeadingACreditBlockIsPrintedUnder:
    """A journal heading the editors it names "Editorial responsibility"."""

    def _make_doc(self, with_a_footer: bool = False):
        heading = make_line('EDITORIAL', 'RESPONSIBILITY',
                            y=300.0, page_number=2, bold=True)
        credited = [
            make_line('Deputy', 'Editor', ':', 'Peregrine', 'Snipe', '.',
                      y=315.0, page_number=2),
            make_line('Associate', 'Editor', ':', 'Alida', 'Teal', '.',
                      y=330.0, page_number=2),
        ]
        feet = [
            make_line('J.', 'Marsh', 'Birds,', 'v.', '12,', 'p.', '486-492',
                      x=400.0, y=950.0, page_number=page_number, bold=True)
            for page_number in (1, 2)
        ]
        titled = make_line('Herons', 'of', 'the', 'Fenwick', 'marshes',
                           y=100.0, bold=True)
        body = [
            make_line('The', 'colonies', 'of', 'herons', f'grew{index}',
                      y=500.0 + 13 * index)
            for index in range(3)
        ]
        listed = [
            make_line('1', '.', 'Teal', 'A', '.', 'Herons', 'of', 'the',
                      'Fenwick', 'marshes', '.', f'{1990 + index}', '.',
                      y=360.0 + 13 * index, page_number=2)
            for index in range(3)
        ]
        above = feet[1:] if with_a_footer else [heading]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[titled]),
                    LayoutBlock(lines=body),
                    LayoutBlock(lines=[feet[0]]),
                ],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=(
                    [LayoutBlock(lines=[line]) for line in above + credited]
                    + [LayoutBlock(lines=listed)]
                ),
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        fields[lines.index(titled)] = JatsFieldNames.TITLE
        for line in listed:
            fields[lines.index(line)] = JatsFieldNames.REFERENCE
        for line in above + credited:
            if line is not feet[1]:
                fields[lines.index(line)] = JatsFieldNames.AUTHOR_NOTES
        return doc, annotate(doc, fields), heading, credited, feet[1]

    def test_the_editors_are_the_front_matter(self):
        doc, annotated, _, credited, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in credited] == [SEG_FRONT] * 2

    def test_the_heading_goes_with_the_editors_it_heads(self):
        doc, annotated, heading, _, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert labels[id(heading)] == SEG_FRONT

    def test_the_foot_the_page_prints_beside_them_is_not_their_heading(self):
        doc, annotated, _, _, foot = self._make_doc(with_a_footer=True)
        labels = derive_labels(doc, annotated)
        assert labels[id(foot)] == SEG_FOOTNOTE


class TestTheBlockAClosingCreditSharesWithADeclaration:
    """A journal that sets its editor and what funded the work in one block."""

    def _make_doc(self):
        body = [
            make_line('The', 'herons', f'waded{index}', y=40.0 + 10 * index)
            for index in range(4)
        ]
        references = [
            make_line('A', 'reference', f'listed{index}', y=100.0 + 20 * index)
            for index in range(4)
        ]
        credited = [
            make_line('EDITOR', '-', 'IN', '-', 'CHIEF', y=300.0, page_number=2),
            make_line('Alida', 'Teal', '(', 'Fenwick', 'Institute', ')',
                      y=315.0, page_number=2),
        ]
        funding = make_line('The', 'marsh', 'trust', 'paid', 'for', 'the', 'survey',
                            '.', y=330.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=body + references)],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=credited + [funding])],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        for line in references:
            fields[lines.index(line)] = JatsFieldNames.REFERENCE
        for line in credited:
            fields[lines.index(line)] = JatsFieldNames.AUTHOR_NOTES
        fields[lines.index(funding)] = JatsFieldNames.FUNDING_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), credited, funding

    def test_the_editor_is_the_front_matter(self):
        doc, annotated, credited, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in credited] == [SEG_FRONT] * 2

    def test_what_funded_the_work_is_left_where_the_deposit_files_it(self):
        doc, annotated, _, funding = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert labels[id(funding)] == SEG_FUNDING
