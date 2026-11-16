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

    def _make_doc(self):
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
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=references + [heading] + biography)],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.REFERENCE for line in references
        }
        return doc, annotate(doc, fields), [heading] + biography

    def test_the_heading_opens_an_annex(self):
        doc, annotated, block = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in block] == [SEG_ANNEX] * 4
