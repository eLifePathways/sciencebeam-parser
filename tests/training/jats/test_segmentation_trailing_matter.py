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
    SEG_CONTRIBUTION,
    SEG_FRONT,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
)


class TestTheSubmissionHistoryAtTheFootOfTheLastPage:
    """A journal that prints when it received the article under the references."""

    def _make_doc(self, date_line: str):
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
        return doc, annotate(doc, fields), printed

    def test_it_is_taken_out_of_the_reference_list(self):
        doc, annotated, printed = self._make_doc('Artigo apresentado em 25/06/2015')
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
