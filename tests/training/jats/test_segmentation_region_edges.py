"""Where a region ends when the page and the JATS do not agree.

Each of these is a mismatch the aligner cannot settle on its own: a heading
the abstract and the body share, a sentence the appendix restates, a list the
page sets below the columns that are read before it.
"""
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutFont,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutToken,
)
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_ANNEX,
    SEG_BODY,
    SEG_CONFLICT,
    SEG_COVER,
    SEG_FOOTNOTE,
    SEG_FRONT,
    SEG_HEADNOTE,
    SEG_REFERENCES,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_line,
    make_page_meta,
    page_number_of,
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

    def _make_doc(self, at_the_page_foot: bool, restated_lines: int = 1):
        before = [
            make_line('The', 'argument', f'continues{index}', y=100.0 + 10 * index)
            for index in range(3)
        ]
        restated = [
            make_line('the', 'aim', 'of', 'the', f'study{index}',
                      y=(780.0 if at_the_page_foot else 130.0) + 10 * index)
            for index in range(restated_lines)
        ]
        matched = restated[0]
        after = [
            make_line('and', 'carries', f'on{index}',
                      y=140.0 + 10 * restated_lines + 10 * index,
                      page_number=2 if at_the_page_foot else 1)
            for index in range(3)
        ]
        appendix = make_line('Appendix', '1:', 'the', 'instrument',
                             y=100.0, page_number=3)
        page_one = before + (restated if at_the_page_foot else restated + after)
        pages = [LayoutPage(blocks=[LayoutBlock(lines=page_one)],
                            meta=make_page_meta(1))]
        if at_the_page_foot:
            pages.append(LayoutPage(blocks=[LayoutBlock(lines=after)],
                                    meta=make_page_meta(2)))
        pages.append(LayoutPage(blocks=[LayoutBlock(lines=[appendix])],
                                meta=make_page_meta(3)))
        doc = LayoutDocument(pages=pages)
        lines = list(doc.iter_all_lines())
        fields = {lines.index(appendix): JatsFieldNames.APPENDIX}
        for line in restated:
            fields[lines.index(line)] = JatsFieldNames.APPENDIX
        for line in before + after:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), matched, appendix

    def test_a_line_of_the_paragraph_around_it_is_released(self):
        doc, annotated, matched, _ = self._make_doc(at_the_page_foot=False)
        assert derive_labels(doc, annotated)[id(matched)] == SEG_BODY

    def test_a_statement_at_the_foot_of_the_page_stays_an_annex(self):
        doc, annotated, matched, _ = self._make_doc(at_the_page_foot=True)
        assert derive_labels(doc, annotated)[id(matched)] == SEG_ANNEX

    def test_a_declaration_the_argument_restates_in_full_is_released(self):
        doc, annotated, matched, _ = self._make_doc(
            at_the_page_foot=False, restated_lines=6
        )
        assert derive_labels(doc, annotated)[id(matched)] == SEG_BODY

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


class TestFootnotesAcrossTwoColumns:
    """A two-column page sets its notes in one band under both columns."""

    def _make_doc(self, right_note_y: float = 700.0):
        def _note(*texts, x, y, size):
            return LayoutLine(tokens=[
                LayoutToken(
                    text=word,
                    font=LayoutFont(font_id=f'f{size}', font_size=size),
                    coordinates=LayoutPageCoordinates(
                        x=x, y=y, width=200, height=8, page_number=1,
                    ),
                )
                for word in texts
            ])

        # Reading order takes the right column first, so its note arrives
        # before the left column's text, let alone the left column's notes.
        right_text = [
            _note('The', 'argument', f'continues{index}', x=314.0,
                  y=600.0 + 11 * index, size=9.0)
            for index in range(3)
        ]
        right_note = _note('9', 'The', 'guidelines', 'are', 'in', 'File', '4.',
                           x=314.0, y=right_note_y, size=7.8)
        left_note = [
            _note('7', 'DARIAH-Campus', 'is', 'a', 'platform.', x=57.0,
                  y=690.0, size=7.8),
            _note('learning', 'resources', 'are', 'linked.', x=57.0,
                  y=710.0, size=7.8),
        ]
        later = make_line('More', 'of', 'the', 'argument', y=300.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=right_text + [right_note]),
                        LayoutBlock(lines=left_note)],
                meta=make_page_meta(1),
            ),
            LayoutPage(blocks=[LayoutBlock(lines=[later])], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(line): JatsFieldNames.BACK_FOOTNOTE for line in left_note}
        for line in right_text:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        fields[lines.index(later)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), right_note, right_text

    def test_a_note_in_the_band_is_claimed(self):
        doc, annotated, right_note, _ = self._make_doc()
        assert derive_labels(doc, annotated)[id(right_note)] == SEG_FOOTNOTE

    def test_the_text_above_the_band_is_left_alone(self):
        doc, annotated, _, right_text = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in right_text] == [SEG_BODY] * 3

    def test_a_line_above_the_band_is_left_alone(self):
        doc, annotated, right_note, _ = self._make_doc(right_note_y=500.0)
        assert derive_labels(doc, annotated)[id(right_note)] != SEG_FOOTNOTE


class TestStatementMatchedOnTheCover:
    """A deposit cover declares, in the statement's own words, that it exists."""

    def _make_doc(self, after_the_declarations: bool, heading_x: float = 10.0):
        argument = [
            make_line('The', 'argument', f'ends{index}', y=100.0 + 10 * index)
            for index in range(3)
        ]
        contribution = make_line('Both', 'authors', 'wrote', 'it', y=140.0)
        heading = make_line('Declaracion', 'de', 'conflicto', 'de', 'intereses:',
                            y=160.0, x=heading_x, bold=True)
        statement = make_line('The', 'authors', 'declare', 'none', y=180.0,
                              x=heading_x)
        funding = make_line('This', 'work', 'was', 'self-funded', y=200.0)
        lines = (
            argument + [contribution, heading, statement, funding]
            if after_the_declarations
            else argument + [heading, statement, contribution, funding]
        )
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=lines)], meta=make_page_meta(1)),
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    make_line('Los', 'autores', 'declaran', 'que', 'los',
                              'conflictos', 'de', 'intereses', 'se', 'incluyen',
                              y=100.0, page_number=2),
                ])],
                meta=make_page_meta(2),
            ),
        ])
        index_of = {id(line): i for i, line in enumerate(doc.iter_all_lines())}
        fields = {
            index_of[id(contribution)]: JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH,
            index_of[id(funding)]: JatsFieldNames.FUNDING_SECTION_PARAGRAPH,
        }
        for line in argument:
            fields[index_of[id(line)]] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), heading, statement

    def test_a_declaration_among_the_declarations_is_claimed(self):
        doc, annotated, heading, statement = self._make_doc(
            after_the_declarations=True
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(heading)], labels[id(statement)]] == [SEG_CONFLICT] * 2

    def test_a_heading_the_argument_runs_into_is_left_alone(self):
        doc, annotated, heading, _ = self._make_doc(after_the_declarations=False)
        assert derive_labels(doc, annotated)[id(heading)] != SEG_CONFLICT

    def test_a_heading_opening_a_column_of_its_own_is_claimed(self):
        doc, annotated, heading, statement = self._make_doc(
            after_the_declarations=False, heading_x=300.0
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(heading)], labels[id(statement)]] == [SEG_CONFLICT] * 2


class TestFrontMatterReachingTheNextPage:
    """A keyword list whose last term is the heading that follows it."""

    def _make_doc(self, keywords_run_on: bool):
        title = make_line('A', 'title', 'of', 'the', 'paper', y=100.0)
        abstract = [
            make_line('What', 'the', 'study', f'found{index}', y=120.0 + 10 * index)
            for index in range(3)
        ]
        keywords = make_line('Key', 'words:', 'one,', 'two,', 'three', y=160.0)
        # The gold ran the keyword list into the heading under it, so the
        # heading is matched as the last of the keywords.
        heading = make_line('Introduccion.',
                            y=100.0 if keywords_run_on else 170.0,
                            page_number=2 if keywords_run_on else 1)
        body = [
            make_line('The', 'argument', f'begins{index}',
                      y=110.0 + 10 * index, page_number=2)
            for index in range(3)
        ]
        page_one = [title] + abstract + [keywords] + ([] if keywords_run_on else [heading])
        page_two = ([heading] if keywords_run_on else []) + body
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=page_one)], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=page_two)], meta=make_page_meta(2)),
        ])
        index_of = {id(line): i for i, line in enumerate(doc.iter_all_lines())}
        fields = {
            index_of[id(title)]: JatsFieldNames.TITLE,
            index_of[id(keywords)]: JatsFieldNames.KEYWORDS,
            index_of[id(heading)]: JatsFieldNames.KEYWORDS,
        }
        for line in abstract:
            fields[index_of[id(line)]] = JatsFieldNames.ABSTRACT
        for line in body:
            fields[index_of[id(line)]] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), heading

    def test_a_line_alone_at_the_top_of_the_body_page_is_the_body(self):
        doc, annotated, heading = self._make_doc(keywords_run_on=True)
        assert derive_labels(doc, annotated)[id(heading)] == SEG_BODY

    def test_a_line_on_the_front_matter_page_is_left_alone(self):
        doc, annotated, heading = self._make_doc(keywords_run_on=False)
        assert derive_labels(doc, annotated)[id(heading)] == SEG_FRONT


class TestASecondKeywordList:
    """A preprint prints its keywords in both of its languages."""

    def _make_doc(self, second_list_text: str):
        title = make_line('A', 'title', 'of', 'the', 'paper', y=100.0)
        abstract = [
            make_line('What', 'the', 'study', f'found{index}', y=120.0 + 10 * index)
            for index in range(3)
        ]
        keywords = make_line('Keywords:', 'one,', 'two,', 'three', y=160.0)
        second = make_line(*second_list_text.split(' '), y=100.0, page_number=2)
        wrapped = make_line('and', 'five.', y=112.0, page_number=2)
        body = [
            make_line('The', 'argument', f'begins{index}',
                      y=140.0 + 10 * index, page_number=2)
            for index in range(3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[title] + abstract + [keywords])],
                       meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[second, wrapped] + body)],
                       meta=make_page_meta(2)),
        ])
        index_of = {id(line): i for i, line in enumerate(doc.iter_all_lines())}
        fields = {
            index_of[id(title)]: JatsFieldNames.TITLE,
            index_of[id(keywords)]: JatsFieldNames.KEYWORDS,
        }
        for line in abstract:
            fields[index_of[id(line)]] = JatsFieldNames.ABSTRACT
        for line in body:
            fields[index_of[id(line)]] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), second, wrapped, body

    def test_the_second_list_is_front_matter(self):
        doc, annotated, second, wrapped, _ = self._make_doc(
            'Keywords: one, two, four'
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(second)], labels[id(wrapped)]] == [SEG_FRONT] * 2

    def test_the_argument_under_it_is_left_alone(self):
        doc, annotated, _, _, body = self._make_doc('Keywords: one, two, four')
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in body] == [SEG_BODY] * 3

    def test_a_line_opening_with_another_word_is_left_alone(self):
        doc, annotated, second, _, _ = self._make_doc(
            'Introduction: the argument begins'
        )
        assert derive_labels(doc, annotated)[id(second)] != SEG_FRONT


class TestDeclarationPrintedAfterTheArticle:
    """A typeset journal files its declarations in the front matter and prints
    them at the end."""

    def _make_doc(self):
        front = {
            'title': make_line('A', 'title', y=100.0),
            'affiliation': make_line('A', 'university', 'somewhere', y=120.0),
            'late_affiliation': make_line(
                'Another', 'university', 'elsewhere', y=100.0, page_number=2
            ),
        }
        body = [
            make_line('The', 'argument', f'runs{index}', y=200.0 + 20 * index,
                      page_number=page_number)
            for page_number in (1, 2)
            for index in range(8)
        ]
        closing = {
            'conflict': make_line(
                'The', 'authors', 'declare', 'none', y=400.0, page_number=2
            ),
            'table': make_line('Macrodomain', 'Proposal', y=420.0, page_number=2),
            'caption': make_line(
                'Coordinators', ':', 'Ada', 'Lovelace', y=430.0, page_number=2
            ),
            'ethics': make_line(
                'The', 'study', 'needed', 'no', 'approval', y=440.0, page_number=2
            ),
            'correspondence_heading': make_line(
                'CORRESPOND\u00caNCIA', ':', y=460.0, page_number=2
            ),
            'correspondence': make_line(
                'Ada', 'Lovelace', ':', 'ada@example.org', y=470.0, page_number=2
            ),
        }
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=[front['title'], front['affiliation']]
                    + [line for line in body if page_number_of(line) == 1]
                )],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(
                    lines=[front['late_affiliation']]
                    + [line for line in body if page_number_of(line) == 2]
                    + list(closing.values())
                )],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(front['title']): JatsFieldNames.TITLE,
            lines.index(front['affiliation']): JatsFieldNames.AUTHOR_AFF,
            lines.index(front['late_affiliation']): JatsFieldNames.AUTHOR_AFF,
            lines.index(closing['conflict']): JatsFieldNames.CONFLICT_NOTE,
            lines.index(closing['ethics']): JatsFieldNames.AUTHOR_NOTES,
            lines.index(closing['caption']): JatsFieldNames.AUTHOR,
            lines.index(closing['correspondence']): JatsFieldNames.CORRESPONDENCE,
        }
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), {**front, **closing}

    def test_a_declaration_takes_the_label_it_names(self):
        doc, annotated, lines = self._make_doc()
        assert derive_labels(doc, annotated)[id(lines['conflict'])] == SEG_CONFLICT

    def test_a_note_naming_no_declaration_is_back_matter(self):
        doc, annotated, lines = self._make_doc()
        assert derive_labels(doc, annotated)[id(lines['ethics'])] == SEG_ANNEX

    def test_what_the_page_sets_between_two_declarations_is_let_go(self):
        doc, annotated, lines = self._make_doc()
        assert derive_labels(doc, annotated)[id(lines['table'])] == SEG_BODY

    def test_a_byline_read_into_a_caption_is_let_go_with_it(self):
        doc, annotated, lines = self._make_doc()
        assert derive_labels(doc, annotated)[id(lines['caption'])] == SEG_BODY

    def test_a_correspondence_block_after_them_keeps_its_heading(self):
        doc, annotated, lines = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [
            labels[id(lines['correspondence_heading'])],
            labels[id(lines['correspondence'])],
        ] == [SEG_FRONT] * 2

    def test_front_matter_printed_past_the_first_body_line_is_left_alone(self):
        doc, annotated, lines = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert labels[id(lines['late_affiliation'])] == SEG_FRONT


class TestTheBannerAboveTheFrontMatter:
    """A journal prints its licence in the top margin of the first page."""

    def _make_doc(self, in_the_top_margin: bool, beside_a_running_head: bool = False):
        banner = [
            make_line('This', 'is', 'an', 'open', 'access', 'article',
                      y=20.0 if in_the_top_margin else 420.0, x=400.0),
            make_line('under', 'the', 'CC', 'BY', 'license',
                      y=40.0 if in_the_top_margin else 440.0, x=400.0),
        ]
        running_head = [
            make_line('The', 'journal', 'and', 'the', 'issue', y=40.0, x=100.0,
                      page_number=page_number)
            for page_number in (1, 2)
        ] if beside_a_running_head else []
        title = make_line('A', 'title', y=100.0)
        abstract = make_line('What', 'the', 'article', 'found', y=120.0)
        body = [
            make_line('The', 'argument', f'runs{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        overleaf = [
            make_line('and', 'goes', f'on{index}', y=200.0 + 20 * index,
                      page_number=2)
            for index in range(4)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[title, abstract]),
                    LayoutBlock(lines=body),
                    LayoutBlock(lines=banner),
                ] + [LayoutBlock(lines=running_head[:1])] * bool(running_head),
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=overleaf)]
                + [LayoutBlock(lines=running_head[1:])] * bool(running_head),
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(title): JatsFieldNames.TITLE,
            lines.index(abstract): JatsFieldNames.ABSTRACT,
        }
        for line in body:
            fields[lines.index(line)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), banner

    def test_it_is_the_front_matter_s(self):
        doc, annotated, banner = self._make_doc(in_the_top_margin=True)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in banner] == [SEG_FRONT] * 2

    def test_a_block_below_the_margin_is_left_to_the_regions(self):
        doc, annotated, banner = self._make_doc(in_the_top_margin=False)
        labels = derive_labels(doc, annotated)
        assert labels[id(banner[0])] != SEG_FRONT

    def test_a_row_the_page_runs_its_head_across_is_a_margin(self):
        doc, annotated, banner = self._make_doc(
            in_the_top_margin=True, beside_a_running_head=True
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in banner] == [SEG_HEADNOTE] * 2


class TestATableCarriedOverThePage:
    """A table running over a page break repeats its label and says so."""

    def _make_doc(self, says_so_on_the_label_line: bool = False):
        body = [
            make_line('The', 'argument', 'of', f'page{page_number}', f'line{index}',
                      y=200.0 + 20 * index, page_number=page_number)
            for page_number in (1, 2, 3)
            for index in range(6)
        ]
        by_page = {
            page_number: [line for line in body
                          if page_number_of(line) == page_number]
            for page_number in (1, 2, 3)
        }
        carried = [
            make_line(
                *(('Table', '9', '.', 'Continuation') if says_so_on_the_label_line
                  else ('Table', '9')),
                y=20.0, page_number=page_number,
            )
            for page_number in (2, 3)
        ]
        marker = [
            make_line('(continued)', y=40.0, page_number=page_number)
            for page_number in (2, 3)
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=by_page[1])],
                       meta=make_page_meta(1)),
        ] + [
            LayoutPage(
                blocks=[
                    LayoutBlock(lines=[carried[index], marker[index]]),
                    LayoutBlock(lines=by_page[page_number]),
                ],
                meta=make_page_meta(page_number),
            )
            for index, page_number in enumerate((2, 3))
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        return doc, annotate(doc, fields), carried + marker

    def test_its_label_and_its_marker_are_not_page_furniture(self):
        doc, annotated, carried = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in carried] == [SEG_BODY] * 4

    def test_a_label_that_says_so_on_one_line_is_not_either(self):
        doc, annotated, carried = self._make_doc(says_so_on_the_label_line=True)
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in carried] == [SEG_BODY] * 4


class TestASectionTitleThatMatchedAReferenceEntry:
    """A deposit carrying a body the page never prints, over a reference list."""

    def _make_doc(self, long_run: bool = False):
        body = [
            make_line('The', 'argument', f'runs{index}', y=200.0 + 20 * index)
            for index in range(6)
        ]
        entries = [
            make_line('Autor', 'A', f'A{index}', 'title', 'and', 'a', 'year',
                      y=200.0 + 20 * index, page_number=2)
            for index in range(10)
        ]
        taken = [entries[3], entries[4]] + (entries[5:7] if long_run else [])
        heading = entries[7]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=body)], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=entries)], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH
                  for line in body}
        fields.update({
            lines.index(entries[index]): JatsFieldNames.REFERENCE
            for index in (0, 1, 9)
        })
        fields.update({
            lines.index(line): JatsFieldNames.BODY_SECTION_TITLE
            for line in taken + [heading]
        })
        return doc, annotate(doc, fields), taken + [heading]

    def test_the_list_takes_the_entries_back(self):
        doc, annotated, taken = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in taken] == [SEG_REFERENCES] * 3

    def test_a_run_too_long_to_be_a_mistake_is_left_alone(self):
        doc, annotated, taken = self._make_doc(long_run=True)
        labels = derive_labels(doc, annotated)
        assert SEG_BODY in [labels[id(line)] for line in taken]


class TestTheImprintBlockAtTheFootOfThePage:
    """A journal that sets the doi, the funding and the dates in one block."""

    def _make_doc(self):
        body = [
            make_line('The', 'argument', 'of', 'the', f'article{index}',
                      y=200.0 + 20 * index)
            for index in range(6)
        ]
        imprint = [
            make_line('doi', ':', '10.0000/0000-00000000000', y=700.0),
            make_line('Financial', 'support', ':', 'The', 'marsh', 'trust', '.',
                      y=712.0),
            make_line('+', 'Corresponding', 'author', ':', 'a@example.org',
                      y=724.0),
            make_line('Received', '24', 'February', '2017', y=736.0),
            make_line('Accepted', '6', 'June', '2017', y=748.0),
        ]
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=body), LayoutBlock(lines=imprint)],
                meta=make_page_meta(1),
            ),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(line): JatsFieldNames.BODY_SECTION_PARAGRAPH for line in body
        }
        fields[lines.index(imprint[1])] = JatsFieldNames.FUNDING
        fields[lines.index(imprint[2])] = JatsFieldNames.CORRESPONDENCE
        for line in imprint[3:]:
            fields[lines.index(line)] = JatsFieldNames.HISTORY_DATE
        return doc, annotate(doc, fields), imprint

    def test_the_rest_of_the_block_is_front_matter_too(self):
        doc, annotated, imprint = self._make_doc()
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in imprint] == [SEG_FRONT] * 5
