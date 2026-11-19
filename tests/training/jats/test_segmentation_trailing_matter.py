"""What a journal prints after the article ends.

The submission history and the authors' biographies are set below the
reference list, where the region above them reaches over anything no field
carries.
"""
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutPage,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_ANNEX,
    SEG_AVAILABILITY,
    SEG_CONFLICT,
    SEG_CONTRIBUTION,
    SEG_FOOTNOTE,
    SEG_FRONT,
    SEG_FUNDING,
    SEG_REFERENCES,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
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

    def _make_doc(self, names_authors_in_bold: bool = False):
        references = [
            make_line('A', 'reference', f'listed{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        contribution = [make_line('AUTHOR', 'CONTRIBUTIONS', y=340.0, bold=True)]
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
        conflict = [
            make_line('CONFLICT', 'OF', 'INTEREST', y=400.0, bold=True),
            make_line('The', 'authors', 'declare', 'none', '.', y=420.0),
        ]
        funding = [
            make_line('FUNDING', y=440.0, bold=True),
            make_line('The', 'marsh', 'trust', 'paid', 'for', 'the', 'survey', '.',
                      y=460.0),
        ]
        unnamed = [
            make_line('Study', 'association', y=480.0, bold=True),
            make_line('No', 'thesis', 'is', 'associated', 'with', 'this', 'work', '.',
                      y=500.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=references + contribution + conflict + funding + unnamed
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
            references, contribution, conflict, funding, unnamed,
        )

    def test_each_heading_opens_the_declaration_it_names(self):
        doc, annotated, _, contribution, conflict, funding, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in contribution] == [SEG_CONTRIBUTION] * 3
        assert [labels[id(line)] for line in conflict] == [SEG_CONFLICT] * 2
        assert [labels[id(line)] for line in funding] == [SEG_FUNDING] * 2

    def test_a_heading_that_names_nothing_opens_an_annex(self):
        doc, annotated, _, _, _, _, unnamed = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in unnamed] == [SEG_ANNEX] * 2

    def test_the_list_above_them_is_left_alone(self):
        doc, annotated, references, _, _, _, _ = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in references] == [SEG_REFERENCES] * 6

    def test_a_bold_line_the_statement_wraps_from_is_not_a_heading(self):
        doc, annotated, _, contribution, _, _, _ = self._make_doc(
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

    def test_a_publisher_an_entry_names_is_not(self):
        doc, annotated, printed = self._make_doc('Editora: A Publishing House')
        assert derive_labels(doc, annotated)[id(printed)] != SEG_FRONT
