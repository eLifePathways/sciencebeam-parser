"""Where a region ends when the page and the JATS do not agree.

Each of these is a mismatch the aligner cannot settle on its own: a heading
the abstract and the body share, a sentence the appendix restates, a list the
page sets below the columns that are read before it.
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
    SEG_CONFLICT,
    SEG_COVER,
    SEG_FRONT,
    SEG_HEADNOTE,
    SEG_REFERENCES,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
)


class TestStructuredAbstractHeading:
    """A structured abstract repeats the headings the body goes on to use."""

    def _make_doc(self, body_lines_in_abstract: int):
        title = make_line('A', 'title', y=100.0)
        abstract_before = make_line('Resumen', y=120.0)
        matched = [
            make_line('Introduccion:', f'the{index}', 'abstract', 'opens', y=140.0 + 10 * index)
            for index in range(body_lines_in_abstract)
        ]
        abstract_after = [
            make_line('Results:', f'what{index}', 'the', 'study', 'found',
                      y=240.0 + 10 * index)
            for index in range(3)
        ]
        body = make_line('The', 'argument', 'begins', y=300.0, page_number=2)
        page_one = [title, abstract_before] + matched + abstract_after
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=page_one)], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[body])], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(title): JatsFieldNames.TITLE,
            lines.index(abstract_before): JatsFieldNames.ABSTRACT,
            lines.index(body): JatsFieldNames.BODY_SECTION_PARAGRAPH,
        }
        for line in matched:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_TITLE
        for line in abstract_after:
            fields[lines.index(line)] = JatsFieldNames.ABSTRACT
        return doc, annotate(doc, fields), matched, body

    def test_a_heading_matched_inside_the_abstract_is_front_matter(self):
        doc, annotated, matched, _ = self._make_doc(body_lines_in_abstract=1)
        assert derive_labels(doc, annotated)[id(matched[0])] == SEG_FRONT

    def test_the_body_itself_is_left_alone(self):
        doc, annotated, _, body = self._make_doc(body_lines_in_abstract=1)
        assert derive_labels(doc, annotated)[id(body)] == SEG_BODY

    def test_a_long_stretch_of_body_is_not_released(self):
        doc, annotated, matched, _ = self._make_doc(body_lines_in_abstract=5)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in matched] == [SEG_BODY] * 5


class TestAppendixMatchInsideTheBody:
    """An appendix restates a sentence of the body, and can win the line."""

    def _make_doc(self, at_the_page_foot: bool):
        before = [
            make_line('The', 'argument', f'continues{index}', y=100.0 + 10 * index)
            for index in range(3)
        ]
        matched = make_line('the', 'aim', 'of', 'the', 'study',
                            y=780.0 if at_the_page_foot else 130.0)
        after = [
            make_line('and', 'carries', f'on{index}',
                      y=140.0 + 10 * index,
                      page_number=2 if at_the_page_foot else 1)
            for index in range(3)
        ]
        appendix = make_line('Appendix', '1:', 'the', 'instrument',
                             y=100.0, page_number=3)
        page_one = before + ([matched] if at_the_page_foot else [matched] + after)
        pages = [LayoutPage(blocks=[LayoutBlock(lines=page_one)],
                            meta=make_page_meta(1))]
        if at_the_page_foot:
            pages.append(LayoutPage(blocks=[LayoutBlock(lines=after)],
                                    meta=make_page_meta(2)))
        pages.append(LayoutPage(blocks=[LayoutBlock(lines=[appendix])],
                                meta=make_page_meta(3)))
        doc = LayoutDocument(pages=pages)
        lines = list(doc.iter_all_lines())
        fields = {lines.index(matched): JatsFieldNames.APPENDIX,
                  lines.index(appendix): JatsFieldNames.APPENDIX}
        for line in before + after:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), matched, appendix

    def test_a_line_of_the_paragraph_around_it_is_released(self):
        doc, annotated, matched, _ = self._make_doc(at_the_page_foot=False)
        assert derive_labels(doc, annotated)[id(matched)] == SEG_BODY

    def test_a_statement_at_the_foot_of_the_page_stays_an_annex(self):
        doc, annotated, matched, _ = self._make_doc(at_the_page_foot=True)
        assert derive_labels(doc, annotated)[id(matched)] == SEG_ANNEX

    def test_the_appendix_itself_is_left_alone(self):
        doc, annotated, _, appendix = self._make_doc(at_the_page_foot=False)
        assert derive_labels(doc, annotated)[id(appendix)] == SEG_ANNEX


class TestStatementHeading:
    """A JATS carries a declaration without a title more often than with one."""

    def _make_doc(self, heading_is_bold: bool, statement_is_bold: bool = False):
        body = [
            make_line('The', 'argument', f'ends{index}', y=100.0 + 10 * index)
            for index in range(3)
        ]
        heading = make_line('DECLARACION', 'DE', 'CONFLICTOS', y=140.0,
                            bold=heading_is_bold)
        statement = make_line('The', 'authors', 'declare', 'none', y=160.0,
                              bold=statement_is_bold)
        later = make_line('More', 'of', 'the', 'back', 'matter',
                          y=100.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body + [heading, statement])],
                       meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[later])], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(statement): JatsFieldNames.CONFLICT_SECTION_PARAGRAPH,
            lines.index(later): JatsFieldNames.BACK_SECTION_PARAGRAPH,
        }
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), heading

    def test_a_bold_line_above_the_statement_is_its_heading(self):
        doc, annotated, heading = self._make_doc(heading_is_bold=True)
        assert derive_labels(doc, annotated)[id(heading)] == SEG_CONFLICT

    def test_a_line_set_like_the_body_is_not(self):
        doc, annotated, heading = self._make_doc(heading_is_bold=False)
        assert derive_labels(doc, annotated)[id(heading)] != SEG_CONFLICT

    def test_a_statement_set_in_bold_itself_takes_no_heading(self):
        doc, annotated, heading = self._make_doc(
            heading_is_bold=True, statement_is_bold=True
        )
        assert derive_labels(doc, annotated)[id(heading)] != SEG_CONFLICT


class TestReferenceHeadingAbovePageColumns:
    """A list set below two columns is read after them, its heading before."""

    def _make_doc(self, heading_x: float = 60.0, heading_y: float = 250.0):
        first_page = [
            make_line('The', 'argument', f'of{index}', y=100.0 + 10 * index)
            for index in range(3)
        ]
        # Reading order takes the heading with the column above the list, and
        # the statement that follows it with the column beside that.
        heading = make_line('References', y=heading_y, x=heading_x, page_number=2)
        statement = [
            make_line('Data', 'are', f'available{index}', y=90.0 + 10 * index,
                      x=56.0, page_number=2)
            for index in range(3)
        ]
        entries = [
            make_line('Smith', 'J:', f'A{index}', 'title.', '2020;', '1:', '2-3.',
                      y=280.0 + 10 * index, x=73.0, page_number=2)
            for index in range(3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=first_page)], meta=make_page_meta(1)),
            LayoutPage(
                blocks=[LayoutBlock(lines=[heading]), LayoutBlock(lines=statement),
                        LayoutBlock(lines=entries)],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
                  for line in first_page}
        for line in statement:
            fields[lines.index(line)] = JatsFieldNames.AVAILABILITY_SECTION_PARAGRAPH
        for line in entries:
            fields[lines.index(line)] = JatsFieldNames.REFERENCE
        return doc, annotate(doc, fields), heading

    def test_the_heading_above_the_list_belongs_to_it(self):
        doc, annotated, heading = self._make_doc()
        assert derive_labels(doc, annotated)[id(heading)] == SEG_REFERENCES

    def test_a_heading_further_up_the_page_is_left_alone(self):
        doc, annotated, heading = self._make_doc(heading_y=120.0)
        assert derive_labels(doc, annotated)[id(heading)] != SEG_REFERENCES

    def test_a_heading_in_the_other_column_is_left_alone(self):
        doc, annotated, heading = self._make_doc(heading_x=330.0)
        assert derive_labels(doc, annotated)[id(heading)] != SEG_REFERENCES


class TestTailRegionPastTheDepositCover:
    """A preprint server appends its cover to the end of the file."""

    def _make_doc(self, with_cover: bool):
        appendix = make_line('Apendice', '1:', 'the', 'checklist', y=100.0)
        unlabelled = [
            make_line('A', 'checklist', f'item{index}', y=110.0 + 10 * index)
            for index in range(3)
        ]
        cover = [
            make_line('Situacao:', 'O', 'preprint', 'nao', 'foi', 'submetido',
                      'para', 'publicacao', y=100.0, page_number=2),
            make_line('Submetido', 'em', '(AAAA-MM-DD):', '2021-03-22',
                      y=120.0, page_number=2),
        ]
        pages = [LayoutPage(
            blocks=[LayoutBlock(lines=[appendix] + unlabelled)],
            meta=make_page_meta(1),
        )]
        if with_cover:
            pages.append(LayoutPage(blocks=[LayoutBlock(lines=cover)],
                                    meta=make_page_meta(2)))
        doc = LayoutDocument(pages=pages)
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {lines.index(appendix): JatsFieldNames.APPENDIX})
        return doc, annotated, unlabelled, cover

    def test_the_appendix_runs_to_the_foot_of_the_article(self):
        doc, annotated, unlabelled, _ = self._make_doc(with_cover=True)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in unlabelled] == [SEG_ANNEX] * 3

    def test_the_cover_itself_is_left_alone(self):
        doc, annotated, _, cover = self._make_doc(with_cover=True)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in cover] == [SEG_COVER] * 2

    def test_a_document_with_no_cover_behaves_the_same(self):
        doc, annotated, unlabelled, _ = self._make_doc(with_cover=False)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in unlabelled] == [SEG_ANNEX] * 3


class TestRunningHeadRepeatsAtTheHead:
    """A caption printed under every table repeats, but not at the head."""

    def _make_doc(self, *caption_y: float):
        def _page(page_number, y):
            # Six lines of argument before the table, so the repeated text is
            # not within the few opening lines a running head is looked for in.
            opening = [
                make_line('The', 'argument', f'continues{page_number}x{index}',
                          y=100.0 + 10 * index, page_number=page_number)
                for index in range(6)
            ] + [make_line('Table', f'{page_number}:', 'the', 'numbers',
                           y=300.0, page_number=page_number)]
            caption = make_line('Fuente:', 'Elaboracion', 'propia.',
                                y=y, page_number=page_number)
            return LayoutPage(
                blocks=[LayoutBlock(lines=opening), LayoutBlock(lines=[caption])],
                meta=make_page_meta(page_number),
            ), caption

        pages, captions = zip(*(
            _page(number, y) for number, y in enumerate(caption_y, start=1)
        ))
        doc = LayoutDocument(pages=list(pages))
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            index: JatsFieldNames.BODY_SECTION_PARAGRAPH
            for index, line in enumerate(lines) if 'argument' in line.text
        })
        return doc, annotated, captions[-1]

    def test_a_caption_that_ran_over_to_the_top_is_not_a_running_head(self):
        # Under the table on page one, at the top of page two because the
        # table above it ran over.
        doc, annotated, caption = self._make_doc(320.0, 60.0)
        assert derive_labels(doc, annotated)[id(caption)] != SEG_HEADNOTE

    def test_a_line_printed_at_the_head_of_every_page_is_one(self):
        doc, annotated, caption = self._make_doc(60.0, 60.0)
        assert derive_labels(doc, annotated)[id(caption)] == SEG_HEADNOTE
