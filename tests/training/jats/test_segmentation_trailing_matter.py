"""What a journal prints after the article ends.

The submission history and the authors' biographies are set below the
reference list, where the region above them reaches over anything no field
carries.
"""
from typing import Tuple

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_ANNEX,
    SEG_AVAILABILITY,
    SEG_BODY,
    SEG_CONFLICT,
    SEG_CONTRIBUTION,
    SEG_FOOTNOTE,
    SEG_FRONT,
    SEG_REVIEW,
    SEG_FUNDING,
    SEG_REFERENCES,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_doc_with_page,
    make_line,
    make_page_meta,
)


class TestTheSubmissionHistoryAtTheFootOfTheLastPage:
    """A journal that prints when it received the article under the references."""

    def _make_doc(self, date_line: str, matched_as_a_reference: bool = False):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        printed = make_line(*date_line.split(), y=340.0)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=references + [printed])],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        if matched_as_a_reference:
            # The list has no entry for the date, and reads it as one's pages.
            fields[lines.index(printed)] = JatsFieldNames.REFERENCE
        return doc, annotate(doc, fields), printed

    def test_it_is_taken_out_of_the_reference_list(self):
        doc, annotated, printed = self._make_doc('Artigo apresentado em 25/06/2015')
        assert derive_labels(doc, annotated)[id(printed)] == SEG_FRONT

    def test_the_list_matching_it_does_not_keep_it(self):
        doc, annotated, printed = self._make_doc(
            'Approved: 01/04/2024', matched_as_a_reference=True
        )
        assert derive_labels(doc, annotated)[id(printed)] == SEG_FRONT

    def test_a_line_of_prose_broken_at_a_number_is_not(self):
        doc, annotated, printed = self._make_doc('recebido durante a pandemia 8 .')
        assert derive_labels(doc, annotated)[id(printed)] != SEG_FRONT


class TestTheAuthorsBiographiesAfterTheReferences:
    """A journal that closes with "About the authors"."""

    def _make_doc(self, heading_in_bold: bool = True):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        heading = make_line('About', 'the', 'authors', y=340.0)
        biography = [
            make_line('A', 'name', 'and', 'where', f'they{index}', 'teach',
                      y=360.0 + 20 * index)
            for index in range(3)
        ]
        statement = [
            make_line('Authors', 'participation', y=420.0, bold=heading_in_bold),
            make_line('Both', 'authors', 'counted', 'and', 'wrote', y=440.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=references + [heading] + biography + statement
                )],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        return doc, annotate(doc, fields), [heading] + biography, statement

    def test_the_heading_opens_an_annex(self):
        doc, annotated, block, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in block] == [SEG_ANNEX] * 4

    def test_the_annex_hands_over_where_a_declaration_is_named(self):
        doc, annotated, _, statement = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in statement] == [SEG_CONTRIBUTION] * 2

    def test_a_line_that_is_not_a_heading_names_nothing(self):
        doc, annotated, _, statement = self._make_doc(heading_in_bold=False)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in statement] == [SEG_ANNEX] * 2


class TestTheLicenceNoticeAtTheFootOfTheLastPage:
    """A journal that grants its licence alone under the reference list."""

    def _make_doc(self, notice: str):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        printed = make_line(*notice.split(), y=340.0)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=references + [printed])],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        return doc, annotate(doc, fields), printed

    def test_it_is_taken_out_of_the_reference_list(self):
        doc, annotated, printed = self._make_doc(
            'This is an open-access article distributed under the terms of the'
            ' Creative Commons Attribution License.'
        )
        assert derive_labels(doc, annotated)[id(printed)] == SEG_FRONT

    def test_the_notice_names_what_it_licenses_first(self):
        doc, annotated, printed = self._make_doc(
            'Este texto está licenciado com uma Licença Creative Commons'
            ' Atribuição 4.0 Internacional.'
        )
        assert derive_labels(doc, annotated)[id(printed)] == SEG_FRONT

    def test_a_sentence_that_cites_a_licence_is_not(self):
        doc, annotated, printed = self._make_doc(
            'The heron counts of both marshes were released under a Creative'
            ' Commons licence by the ringing scheme.'
        )
        assert derive_labels(doc, annotated)[id(printed)] != SEG_FRONT

    def test_the_rest_of_the_sentence_goes_with_it(self):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        notice = [
            make_line('This', 'is', 'an', 'open-access', 'article', 'under', 'the',
                      'terms', 'of', 'the', 'Creative', 'Commons', 'License', ',',
                      y=340.0),
            make_line('which', 'permits', 'use', 'in', 'any', 'medium', ',',
                      'provided', 'the', 'original', 'is', 'cited', '.', y=360.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=references), LayoutBlock(lines=notice)],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        })
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in notice] == [SEG_FRONT] * 2


class TestTheCitationTheRendererComposes:
    """A journal that closes with how the article should be cited."""

    def _make_doc(self):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        biography = make_line('About', 'the', 'authors', y=340.0)
        heading = make_line('How', 'to', 'cite', 'this', 'article', ':', y=360.0)
        citation = [
            make_line('Heron', 'A', '.', 'Counting', 'herons', '.', f'Part{index}',
                      y=380.0 + 20 * index)
            for index in range(2)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=references + [biography, heading] + citation
                )],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        return doc, annotate(doc, fields), biography, heading, citation

    def test_the_block_goes_to_the_front_matter(self):
        doc, annotated, _, heading, citation = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in [heading, *citation]] == [SEG_FRONT] * 3

    def test_the_section_above_it_is_left_where_it_is(self):
        doc, annotated, biography, _, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(biography)] == SEG_ANNEX


class TestTheDeclarationsTheArticleClosesWith:
    """A journal that prints its declarations under the reference list."""

    def _make_doc(
        self,
        names_authors_in_bold: bool = False,
        heading: Tuple[str, ...] = ('AUTHOR', 'CONTRIBUTIONS'),
    ):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        contribution = [make_line(*heading, y=340.0, bold=True)]
        if names_authors_in_bold:
            contribution += [
                make_line('Heron', 'A', 'and', 'Heron', 'B', ',', 'counted', 'the',
                          y=360.0, bold=True),
                make_line('nests', 'and', 'wrote', 'the', 'paper', '.', y=380.0),
            ]
        else:
            contribution += [
                make_line('Heron', 'A', 'counted', 'the', 'nests', ';', y=360.0),
                make_line('Heron', 'B', 'wrote', 'the', 'paper', '.', y=380.0),
            ]
        availability = [
            make_line('DATA', 'AVAILABILITY', 'STATEMENT', 'OF', 'THIS', 'ARTICLE',
                      'AND', 'ITS', 'SUPPORTING', 'FILES', y=400.0, bold=True),
            make_line('The', 'counts', 'are', 'in', 'the', 'article', 'itself', '.',
                      y=420.0),
        ]
        conflict = [
            make_line('CONFLICT', 'OF', 'INTEREST', y=440.0, bold=True),
            make_line('The', 'authors', 'declare', 'none', '.', y=460.0),
        ]
        funding = [
            make_line('FUNDING', y=480.0, bold=True),
            make_line('The', 'marsh', 'trust', 'paid', 'for', 'the', 'survey', '.',
                      y=500.0),
        ]
        unnamed = [
            make_line('Study', 'association', y=520.0, bold=True),
            make_line('No', 'thesis', 'is', 'associated', 'with', 'this', 'work', '.',
                      y=540.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=(
                        references + contribution + availability + conflict
                        + funding + unnamed
                    )
                )],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        # The journal deposits the last two of them as back footnotes, which is
        # what puts them in the margins the page never set them in.
        for line in [funding[1], *unnamed]:
            fields[lines.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        return (
            doc, annotate(doc, fields),
            references, contribution, conflict, funding, availability, unnamed,
        )

    def test_each_heading_opens_the_declaration_it_names(self):
        doc, annotated, _, contribution, conflict, funding, _, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in contribution] == [SEG_CONTRIBUTION] * 3
        assert [labels[id(line)] for line in conflict] == [SEG_CONFLICT] * 2
        assert [labels[id(line)] for line in funding] == [SEG_FUNDING] * 2

    def test_a_heading_that_fills_its_measure_still_opens_its_declaration(self):
        doc, annotated, _, _, _, _, availability, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in availability] == [SEG_AVAILABILITY] * 2

    def test_a_heading_that_names_nothing_opens_an_annex(self):
        doc, annotated, _, _, _, _, _, unnamed = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in unnamed] == [SEG_ANNEX] * 2

    def test_the_list_above_them_is_left_alone(self):
        doc, annotated, references, _, _, _, _, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in references] == [SEG_REFERENCES] * 6

    def test_a_statement_a_journal_heads_contributors_is_the_contributions(self):
        doc, annotated, _, contribution, _, _, _, _ = self._make_doc(
            heading=('CONTRIBUTORS',)
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in contribution] == [SEG_CONTRIBUTION] * 3

    def test_a_bold_line_the_statement_wraps_from_is_not_a_heading(self):
        doc, annotated, _, contribution, _, _, _, _ = self._make_doc(
            names_authors_in_bold=True
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in contribution] == [SEG_CONTRIBUTION] * 3


class TestTheNotesGatheredUnderAHeading:
    """A journal that gathers its numbered notes under "Notas" at the end."""

    def _make_doc(self):
        body = [
            make_line('The', 'argument', f'closes{index}', y=100.0 + 20 * index)
            for index in range(4)
        ]
        heading = make_line('NOTAS', y=200.0)
        notes = [
            make_line(str(index), '.', 'A', 'note', 'on', f'point{index}',
                      y=220.0 + 20 * index)
            for index in range(3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=body + [heading] + notes)],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        for line in notes:
            fields[lines.index(line)] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), heading

    def test_the_heading_is_what_the_notes_under_it_are(self):
        doc, annotated, heading = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_FOOTNOTE


class TestTheAbstractPrintedAgainInAnotherLanguage:
    """A journal that closes with its abstract in each language it carries."""

    def _make_doc(self):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        translated = [
            make_line('Herons', 'were', 'counted', 'on', f'marshes{index}',
                      y=100.0 + 20 * index, page_number=2)
            for index in range(5)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=references)], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=translated)], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        for line in translated:
            fields[lines.index(line)] = JatsFieldNames.ABSTRACT
        return doc, annotate(doc, fields), translated

    def test_a_page_of_it_past_the_references_is_still_front_matter(self):
        doc, annotated, translated = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in translated] == [SEG_FRONT] * 5


class TestAnAvailabilityStatementWithSectionsOfItsOwn:
    """A platform that sets "Underlying data" under "Data availability"."""

    def _make_doc(self):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        heading = make_line('Data', 'availability', y=340.0, bold=True)
        statement = [
            make_line('The', 'counts', 'are', 'held', 'by', 'the', 'repository', '.',
                      y=360.0),
            make_line('Underlying', 'data', y=380.0, bold=True),
            make_line('Heron', 'counts', 'by', 'marsh', 'and', 'by', 'year', '.',
                      y=400.0),
            make_line('Extended', 'data', y=420.0, bold=True),
            make_line('The', 'ringing', 'records', 'behind', 'them', '.', y=440.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=references + [heading] + statement)],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        for line in statement:
            fields[lines.index(line)] = JatsFieldNames.AVAILABILITY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), heading, statement

    def test_a_heading_inside_it_does_not_open_an_annex(self):
        doc, annotated, heading, statement = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in [heading, *statement]] == (
            [SEG_AVAILABILITY] * 6
        )


class TestTheEditorCreditedUnderTheReferences:
    """A journal that names the editor who handled the article at the end."""

    def _make_doc(self, credit: str):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        printed = make_line(*credit.split(), y=340.0)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=references), LayoutBlock(lines=[printed])],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        })
        return doc, annotated, printed

    def test_it_is_taken_out_of_the_reference_list(self):
        doc, annotated, printed = self._make_doc('Editor-chefe: A Name Here')
        assert derive_labels(doc, annotated)[id(printed)] == SEG_FRONT

    def test_the_qualifier_may_come_first_or_last(self):
        for credit in ('Responsible Editor: A Name Here',
                       'Editora Responsável: A Name Here'):
            doc, annotated, printed = self._make_doc(credit)
            assert derive_labels(doc, annotated)[id(printed)] == SEG_FRONT

    def test_a_publisher_an_entry_names_is_not(self):
        doc, annotated, printed = self._make_doc('Editora: A Publishing House')
        assert derive_labels(doc, annotated)[id(printed)] != SEG_FRONT


class TestTheHeadingAnAppendedSectionOpensWith:
    """A table of abbreviations the deposit carries without its title."""

    def _make_doc(self, heading_text='Abbreviations, acronyms & symbols',
                  heading_is_a_body_title: bool = False):
        body = [
            make_line('Herons', 'wade', f'on{index}', 'the', 'northern',
                      'marshes', 'through', 'the', 'shortening', 'days', 'of',
                      'the', 'season', y=100.0 + 20 * index)
            for index in range(8)
        ]
        heading = make_line(*heading_text.split(' '), y=300.0, bold=True)
        entries = [
            make_line(name, 'is', 'what', 'the', 'table', 'expands', 'it', 'to',
                      'for', 'the', 'reader', 'of', 'the', 'count',
                      y=320.0 + 20 * index)
            for index, name in enumerate(('BTO', 'WWT', 'RSPB'))
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=body),
                    LayoutBlock(lines=[heading]),
                    LayoutBlock(lines=entries),
                ],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
            for line in body
        }
        fields.update({
            lines.index(line): JatsFieldNames.BACK_SECTION_PARAGRAPH
            for line in entries
        })
        if heading_is_a_body_title:
            fields[lines.index(heading)] = JatsFieldNames.BODY_SECTION_TITLE
        return doc, annotate(doc, fields), heading

    def test_the_heading_above_it_belongs_to_it(self):
        doc, annotated, heading = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_ANNEX

    def test_a_line_that_fills_the_measure_is_not_a_heading(self):
        doc, annotated, heading = self._make_doc(
            'The counts below are set out as a table of the abbreviations used'
        )
        assert derive_labels(doc, annotated)[id(heading)] != SEG_ANNEX

    def test_a_heading_the_deposit_carries_as_the_body_s_is_left_alone(self):
        doc, annotated, heading = self._make_doc(
            'Conclusion', heading_is_a_body_title=True
        )
        assert derive_labels(doc, annotated)[id(heading)] == SEG_BODY


class TestADeclarationFiledWithTheAuthorNotes:
    """A deposit files the conflict statement under `author-notes` and the
    journal prints it under the reference list."""

    def _make_doc(self):
        lines = {
            'title': make_line('How', 'herons', 'read', 'a', 'marsh', y=60.0),
            'body': make_line('The', 'argument', 'runs', 'here', y=120.0),
            'reference': make_line(
                'Egret,', 'A.', '(2024).', 'Wading.', 'Marsh', 'Press.', y=180.0
            ),
            'heading': make_line('Conflict', 'of', 'Interests', y=240.0),
            'statement': make_line(
                'The', 'authors', 'declare', 'no', 'conflict', 'of', 'interest.',
                y=260.0,
            ),
        }
        doc = make_doc_with_page(LayoutBlock(lines=list(lines.values())))
        order = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            order.index(lines['title']): JatsFieldNames.TITLE,
            order.index(lines['body']): JatsFieldNames.BODY_SECTION_PARAGRAPH,
            order.index(lines['reference']): JatsFieldNames.REFERENCE,
            order.index(lines['heading']): JatsFieldNames.CONFLICT_NOTE,
            order.index(lines['statement']): JatsFieldNames.CONFLICT_NOTE,
        })
        return doc, annotated, lines

    def test_the_statement_printed_past_the_list_is_a_conflict_region(self):
        doc, annotated, lines = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(lines[key])] for key in ('heading', 'statement')] == (
            [SEG_CONFLICT] * 2
        )


class TestTheSubmissionHistoryTheDepositCarries:
    """A journal prints when it received and accepted the article under the
    last thing on its last page."""

    def _make_doc(self):
        body = [
            make_line('The', 'argument', f'runs{index}', 'here', y=100.0 + 20 * index)
            for index in range(4)
        ]
        references = [
            make_line('Egret', 'A.', f'Wading{index}', 'in', 'the', 'reeds.',
                      y=200.0 + 20 * index)
            for index in range(4)
        ]
        biography = [
            make_line('Egret', 'A', 'is', 'a', 'reader', 'of', 'marshes',
                      y=300.0 + 20 * index)
            for index in range(3)
        ]
        history = make_line(
            'Received', '30', 'September', '2024', 'and', 'accepted', '2', 'July',
            '2025.', y=400.0,
        )
        doc = make_doc_with_page(
            LayoutBlock(lines=body),
            LayoutBlock(lines=references),
            LayoutBlock(lines=biography),
            LayoutBlock(lines=[history]),
        )
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
                  for line in body}
        fields.update({order.index(line): JatsFieldNames.REFERENCE
                       for line in references})
        fields.update({order.index(line): JatsFieldNames.AUTHOR_NOTES
                       for line in biography})
        fields[order.index(history)] = JatsFieldNames.HISTORY_DATE
        return doc, annotate(doc, fields), history

    def test_it_is_the_front_matter_wherever_it_is_printed(self):
        doc, annotated, history = self._make_doc()
        assert derive_labels(doc, annotated)[id(history)] == SEG_FRONT


class TestTheRuleBetweenTheListAndTheAddress:
    """A journal rules a line between its reference list and the address to
    write to."""

    def _make_doc(self):
        references = [
            make_line('Egret', 'A.', f'Wading{index}', 'in', 'the', 'reeds.',
                      y=100.0 + 14 * index)
            for index in range(6)
        ]
        rule = make_line('______________________________', y=200.0)
        correspondence = [
            make_line('Correspondence', 'address', ':', y=220.0),
            make_line('A', 'Egret,', 'Marsh', 'Institute,', 'Tubarao.', y=234.0),
        ]
        doc = make_doc_with_page(
            LayoutBlock(lines=references),
            LayoutBlock(lines=[rule]),
            LayoutBlock(lines=correspondence),
        )
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.REFERENCE for line in references}
        fields.update({
            order.index(line): JatsFieldNames.CORRESPONDENCE
            for line in correspondence
        })
        return doc, annotate(doc, fields), rule

    def test_the_rule_goes_with_what_it_introduces(self):
        doc, annotated, rule = self._make_doc()
        assert derive_labels(doc, annotated)[id(rule)] == SEG_FRONT


class TestADateAReportIsHeadedWith:
    """A reviewer's report prints the day it was written over it, and the
    article's own history carries a date that reads like one."""

    def _make_doc(self):
        body = [
            make_line('The', 'argument', f'runs{index}', 'here', y=60.0 + 14 * index)
            for index in range(4)
        ]
        report = [
            make_line('I', 'have', 'read', 'this', 'submission', 'and', 'believe',
                      y=140.0),
            make_line('it', 'is', 'of', 'an', 'acceptable', 'standard.', y=154.0),
        ]
        dated = make_line('Author', 'Response', '13', 'March', '2023', y=168.0)
        answer = [
            make_line('Thank', 'you', 'for', 'the', 'comprehensive', 'review.',
                      y=182.0),
            make_line('We', 'have', 'counted', 'the', 'herons', 'again.', y=196.0),
        ]
        doc = make_doc_with_page(
            LayoutBlock(lines=body),
            LayoutBlock(lines=report + [dated] + answer),
        )
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
                  for line in body}
        fields.update({order.index(line): JatsFieldNames.SUB_ARTICLE
                       for line in report + answer})
        fields[order.index(dated)] = JatsFieldNames.HISTORY_DATE
        return doc, annotate(doc, fields), dated

    def test_the_date_stays_with_the_report(self):
        doc, annotated, dated = self._make_doc()
        assert derive_labels(doc, annotated)[id(dated)] == SEG_REVIEW


class TestADatePrintedAfterTheDeclarations:
    """A journal sets when it received the article beside the statement of who
    contributed what, and no field carries the dates."""

    def _make_doc(self, split_across_two_lines: bool):
        body = [
            make_line('The', 'argument', f'runs{index}', 'here', y=60.0 + 14 * index)
            for index in range(4)
        ]
        references = [
            make_line('Egret', 'A.', f'Wading{index}', 'in', 'the', 'reeds.',
                      y=140.0 + 14 * index)
            for index in range(4)
        ]
        statement = [
            make_line('Participation', 'of', 'the', 'authors', y=220.0),
            make_line('A', 'Egret', 'counted', 'and', 'B', 'Bittern', 'wrote.',
                      y=234.0),
        ]
        dates = (
            [make_line('Received', ':', y=260.0, x=40.0),
             make_line('2', '.', '5', '.', '2016', y=260.0, x=140.0)]
            if split_across_two_lines
            else [make_line('Received', ':', '2', '.', '5', '.', '2016', y=260.0)]
        )
        doc = make_doc_with_page(
            LayoutBlock(lines=body),
            LayoutBlock(lines=references),
            LayoutBlock(lines=statement),
            LayoutBlock(lines=dates),
        )
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
                  for line in body}
        fields.update({order.index(line): JatsFieldNames.REFERENCE
                       for line in references})
        fields.update({order.index(line): JatsFieldNames.CONTRIBUTION_NOTE
                       for line in statement})
        return doc, annotate(doc, fields), dates

    def test_the_date_is_front_matter_under_the_declarations(self):
        doc, annotated, dates = self._make_doc(split_across_two_lines=False)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in dates] == [SEG_FRONT]

    def test_the_word_and_the_date_beside_it_are_one_row(self):
        doc, annotated, dates = self._make_doc(split_across_two_lines=True)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in dates] == [SEG_FRONT] * 2


class TestAnAppendixCellThatMatchedAnEntry:
    """An appendix tabulating what each study found names its authors in a cell
    of its own."""

    def _make_doc(self):
        body = [
            make_line('The', 'argument', f'runs{index}', 'here', y=60.0 + 14 * index)
            for index in range(4)
        ]
        references = [
            make_line('Egret', 'A', 'and', 'colleagues', f'(201{index})',
                      'Wading.', y=140.0 + 14 * index)
            for index in range(4)
        ]
        appendix = [
            make_line('Appendix', 'A.', 'What', 'each', 'count', 'found',
                      y=220.0),
            make_line('Egret', 'A', 'and', y=234.0),
        ]
        cell = make_line('colleagues', y=248.0)
        more = [
            make_line('(2016)', 'counted', 'on', 'the', 'northern', 'marsh',
                      y=262.0),
        ]
        doc = make_doc_with_page(
            LayoutBlock(lines=body),
            LayoutBlock(lines=references),
            LayoutBlock(lines=appendix + [cell] + more),
        )
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
                  for line in body}
        fields.update({order.index(line): JatsFieldNames.REFERENCE
                       for line in references})
        fields.update({order.index(line): JatsFieldNames.APPENDIX
                       for line in appendix + more})
        fields[order.index(cell)] = JatsFieldNames.REFERENCE
        return doc, annotate(doc, fields), cell

    def test_the_cell_stays_with_the_appendix(self):
        doc, annotated, cell = self._make_doc()
        assert derive_labels(doc, annotated)[id(cell)] == SEG_ANNEX


class TestWhatAJournalDeclaresAboutThePublication:
    """A journal closes with who holds the copyright and how the article was
    reviewed, and the deposit files each among the notes of the back matter."""

    def _make_doc(self, heading: str, runs_on: bool = False):
        references = [
            make_line('Egret', 'A.', f'Wading{index}', 'in', 'the', 'reeds.',
                      y=100.0 + 14 * index)
            for index in range(5)
        ]
        under = (
            ('elsewhere', ',', 'where', 'the', 'reviewer', 'read', 'it', '.')
            if runs_on else
            ('The', 'journal', 'checks', 'every', 'article', 'it', 'takes.')
        )
        declaration = [
            make_line(*heading.split(' '), y=200.0),
            make_line(*under, y=214.0),
        ]
        foot = make_line('Journal', 'of', 'Marsh', 'Birds,', 'v.', '12,', '2024',
                         y=228.0)
        doc = make_doc_with_page(
            LayoutBlock(lines=references),
            LayoutBlock(lines=declaration + [foot]),
        )
        order = list(doc.iter_all_lines())
        fields = {order.index(line): JatsFieldNames.REFERENCE for line in references}
        fields[order.index(declaration[1])] = JatsFieldNames.BACK_FOOTNOTE
        return doc, annotate(doc, fields), declaration

    def test_a_sentence_that_opens_with_the_word_is_not_a_heading(self):
        doc, annotated, declaration = self._make_doc(
            'Copyright protection in the USA does not apply to this work, which'
            ' may be protected elsewhere.'
        )
        labels = derive_labels(doc, annotated)
        assert labels[id(declaration[0])] != SEG_FRONT

    def test_a_short_line_the_sentence_runs_on_from_is_not_a_heading(self):
        doc, annotated, declaration = self._make_doc(
            'Copyright protection in the USA does not apply', runs_on=True
        )
        labels = derive_labels(doc, annotated)
        assert labels[id(declaration[0])] != SEG_FRONT

    def test_the_declaration_and_its_prose_are_front_matter(self):
        for heading in ('Copyrights', 'Peer Review Method', 'Plagiarism Check'):
            doc, annotated, declaration = self._make_doc(heading)
            labels = derive_labels(doc, annotated)
            assert [labels[id(line)] for line in declaration] == (
                [SEG_FRONT] * 2
            ), heading
