from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutToken,
)
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames
from sciencebeam_parser.training.jats.segmentation import (
    SEG_ANNEX,
    SEG_COVER,
    SEG_BODY,
    SEG_FOOTNOTE,
    SEG_REVIEW,
    SEG_FRONT,
    SEG_HEADNOTE,
    SEG_PAGE,
    SEG_REFERENCES,
)
from tests.training.jats.segmentation_fixtures import (
    annotate,
    derive_labels,
    make_doc_with_page,
    make_line,
    make_page_meta,
)


class TestMajorityVoteLabeling:
    def test_title_tokens_give_header_label(self):
        line = make_line('My', 'Title')
        doc = make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = annotate(doc, {0: JatsFieldNames.TITLE})
        labels = derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_FRONT

    def test_body_paragraph_gives_body_label(self):
        line = make_line('Some', 'body', 'text')
        doc = make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = annotate(doc, {0: JatsFieldNames.BODY_SECTION_PARAGRAPH})
        labels = derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_BODY

    def test_reference_tokens_give_references_label(self):
        line = make_line('Smith', '2020')
        doc = make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = annotate(doc, {0: JatsFieldNames.REFERENCE})
        labels = derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_REFERENCES

    def test_unannotated_line_defaults_to_body(self):
        line = make_line('Unknown', 'content')
        doc = make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_BODY

    def test_majority_vote_mixed_line(self):
        # 3 tokens labeled as TITLE, 1 as BODY_SECTION_PARAGRAPH → should give FRONT (header)
        line = make_line('My', 'Title', 'Here', 'text')
        doc = make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        tokens = line.tokens
        for t in tokens[:3]:
            annotated.set_token_label(t, JatsFieldNames.TITLE)
        annotated.set_token_label(tokens[3], JatsFieldNames.BODY_SECTION_PARAGRAPH)
        labels = derive_labels(doc, annotated)
        assert labels[id(line)] == SEG_FRONT


class TestPageColumns:
    @staticmethod
    def _page(*specs):
        """The given (text, x, grounded) lines on page two, after a page of body.

        Two pages, because on a one-page document every line is printed on
        every page and so reads as a running text the regions step over.
        """
        def _line(text, x, y, page_number):
            return LayoutLine(tokens=[
                LayoutToken(text=word, coordinates=LayoutPageCoordinates(
                    x=x, y=y, width=200, height=10, page_number=page_number,
                ))
                for word in text.split(' ')
            ])

        first = [
            _line(f'The argument of the article continues {i}.', 56.7, 100.0 + 12 * i, 1)
            for i in range(4)
        ]
        lines, fields = [], {}
        for i, (text, x, grounded) in enumerate(specs):
            lines.append(_line(text, x, 100.0 + 12 * i, 2))
            if grounded:
                fields[len(first) + i] = JatsFieldNames.REFERENCE
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=first)],
                       meta=make_page_meta(1, height=1000.0)),
            LayoutPage(blocks=[LayoutBlock(lines=lines)],
                       meta=make_page_meta(2, height=1000.0)),
        ])
        return doc, lines, annotate(doc, fields)

    def test_a_region_reaches_a_fragment_set_right_on_a_one_column_page(self):
        # PPR458890 ends a reference with its DOI set further along the row;
        # the column is the page's, not wherever the last line happens to start.
        doc, lines, annotated = self._page(
            *[(f'Reference entry number {i} here.', 89.7, True) for i in range(5)],
            ('doi:10.1590/1413-', 416.0, False),
            ('81232020253.14532018', 89.7, False),
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in lines[-2:]] == [SEG_REFERENCES] * 2

    def test_a_region_stops_at_the_other_column_of_a_two_column_page(self):
        doc, lines, annotated = self._page(
            *[(f'Reference entry number {i} here.', 56.7, True) for i in range(5)],
            *[(f'Body text of the article {i}.', 314.3, False) for i in range(5)],
        )
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in lines[5:]] == [SEG_BODY] * 5


class TestFootnoteReach:
    @staticmethod
    def _page(*specs):
        """One block per (text, label-field) pair, plus a page number at the foot."""
        lines, fields = [], {}
        for i, (text, field) in enumerate(specs):
            line = make_line(*text.split(' '), y=600.0 + 12 * i, page_number=1)
            lines.append(line)
            if field is not None:
                fields[i] = field
        number = make_line('6', y=960.0, page_number=1)
        doc = LayoutDocument(pages=[LayoutPage(
            blocks=[LayoutBlock(lines=lines), LayoutBlock(lines=[number])],
            meta=make_page_meta(1),
        )])
        return doc, lines, number, fields

    def test_a_note_reaches_its_last_line_with_only_a_page_number_past_it(self):
        # PPR459453 ends a note one line short of what the page prints, with
        # the page number the only thing under it.
        doc, lines, number, fields = self._page(
            ('se han senalado las relaciones cercanas', JatsFieldNames.BACK_FOOTNOTE),
            ('han senalado que el gobierno no los ha', JatsFieldNames.BACK_FOOTNOTE),
            ('efectos economicos de la pandemia (Chacon, 2020).', None),
        )
        labels = derive_labels(doc, annotate(doc, fields))
        assert [labels[id(line)] for line in lines] == [SEG_FOOTNOTE] * 3
        assert labels[id(number)] == SEG_PAGE

    def test_a_line_before_a_note_is_not_taken_into_it(self):
        doc, lines, _, fields = self._page(
            ('The argument of the article continues here.', None),
            ('1 Marks, Robert B. Exhausting the Earth.', JatsFieldNames.BACK_FOOTNOTE),
        )
        labels = derive_labels(doc, annotate(doc, fields))
        assert labels[id(lines[0])] != SEG_FOOTNOTE


class TestCorrespondingAuthorNote:
    @staticmethod
    def _title_page(*texts):
        """A page whose foot carries the given lines, each a block of its own."""
        lines = [
            make_line(*s.split(' '), y=940.0 + 10 * i, page_number=1)
            for i, s in enumerate(texts)
        ]
        doc = LayoutDocument(pages=[LayoutPage(
            blocks=[LayoutBlock(lines=[line]) for line in lines],
            meta=make_page_meta(page_number=1, height=1000.0),
        )])
        return doc, lines

    def test_the_note_at_the_foot_of_the_title_page_is_front_matter(self):
        # PPR459133 sets "*Autor correspondente" and the e-mail addresses under
        # it at the foot of page one, in two blocks.  GROBID's corpus labels
        # such a line `<front>` 339 times and `<footnote>` never.
        doc, lines = self._title_page(
            '*Autor correspondente',
            'Enderecos de e-mail: a@example.com; b@example.com',
        )
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = [derive_labels(doc, annotated)[id(line)] for line in lines]
        assert labels == [SEG_FRONT, SEG_FRONT]

    def test_a_numbered_note_under_it_is_still_a_footnote(self):
        doc, lines = self._title_page(
            '*Autor correspondente',
            'Enderecos de e-mail: a@example.com',
            '1 This work was funded by a grant from the council.',
        )
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = [derive_labels(doc, annotated)[id(line)] for line in lines]
        assert labels == [SEG_FRONT, SEG_FRONT, SEG_FOOTNOTE]

    def test_an_ordinary_footnote_is_untouched(self):
        doc, lines = self._title_page('1 A note about the method used here.')
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        assert derive_labels(doc, annotated)[id(lines[0])] == SEG_FOOTNOTE


class TestTrailingHeading:
    @staticmethod
    def _two_pages(first_page, second_page):
        """Reference lines then a heading at the foot of page one; tables on two."""
        pages = []
        for page_number, texts in enumerate((first_page, second_page), 1):
            lines = [
                make_line(*s.split(' '), y=100.0 + 60 * i, page_number=page_number)
                for i, s in enumerate(texts)
            ]
            pages.append((LayoutPage(
                blocks=[LayoutBlock(lines=lines)],
                meta=make_page_meta(page_number=page_number, height=1000.0),
            ), lines))
        return LayoutDocument(pages=[pg for pg, _ in pages]), [
            line for _, lines in pages for line in lines
        ]

    def test_a_heading_under_the_last_reference_goes_with_what_follows(self):
        # PPR458963 sets "TABLAS" under its last reference, at the foot of the
        # page, where the region grown to the page end takes it.
        doc, lines = self._two_pages(
            ['Smith J. A study of things. Journal. 2020;1(2):3-4.', 'TABLAS'],
            ['Tabla 1. Los resultados del estudio presentado.'],
        )
        annotated = annotate(doc, {0: JatsFieldNames.REFERENCE})
        labels = [derive_labels(doc, annotated)[id(line)] for line in lines]
        assert labels == [SEG_REFERENCES, SEG_BODY, SEG_BODY]

    def test_a_doi_the_last_reference_ends_with_stays_in_the_list(self):
        # PPR458937's list ends "10.1016/S2213-2600(20)30132-6", which is part
        # of the reference rather than a heading for what follows.
        doc, lines = self._two_pages(
            ['Smith J. A study of things. Journal. 2020;1(2):3-4. doi:',
             '10.1016/S2213-2600(20)30132-6'],
            ['Tabla 1. Los resultados del estudio presentado.'],
        )
        annotated = annotate(doc, {0: JatsFieldNames.REFERENCE})
        labels = [derive_labels(doc, annotated)[id(line)] for line in lines]
        assert labels[1] == SEG_REFERENCES


class TestSupplementaryAnnex:
    def test_a_supplementary_heading_opens_an_annex(self):
        # PPR458963 prints "MATERIAL SUPLEMENTARIO" and 14 pages under it; its
        # JATS carries only a pointer to an attached file, so nothing grounds
        # them and they fall to the body.
        heading = make_line('MATERIAL', 'SUPLEMENTARIO', y=200.0)
        under = make_line('Archivo', 'S1.', 'Descripcion', 'del', 'proceso', y=300.0)
        doc = make_doc_with_page(LayoutBlock(lines=[heading, under]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated)
        assert [labels[id(heading)], labels[id(under)]] == [SEG_ANNEX, SEG_ANNEX]

    def test_a_sentence_mentioning_supplementary_material_does_not(self):
        # The heading has to be a heading: a paragraph the JATS carries is
        # already labelled, and a longer line is not one either way.
        line = make_line(
            'Supplementary', 'material', 'is', 'available', 'online', 'for',
            'this', 'article', 'and', 'may', 'be', 'requested', 'from', 'the',
            'corresponding', 'author', 'at', 'any', 'time', y=300.0,
        )
        doc = make_doc_with_page(LayoutBlock(lines=[line]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        assert derive_labels(doc, annotated)[id(line)] != SEG_ANNEX


class TestDepositCoverPage:
    @staticmethod
    def _pages(*pages_of_text) -> LayoutDocument:
        """One page per argument, each a list of line texts."""
        return LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[
                    make_line(
                        *text.split(' '), y=100.0 + 20 * i, page_number=page_number
                    )
                    for i, text in enumerate(texts)
                ])],
                meta=make_page_meta(page_number=page_number, height=1000.0),
            )
            for page_number, texts in enumerate(pages_of_text, 1)
        ])

    def _labels(self, doc: LayoutDocument):
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated)
        return [labels[id(line)] for line in doc.iter_all_lines()]

    def test_a_status_label_with_its_value_elsewhere_still_opens_a_cover(self):
        # PPR459581 and PPR459643 set "Estado:" on a line of its own, so the
        # word "preprint" the notice is recognised by is not on it.
        doc = self._pages([
            'Estado:',
            'Actitudes y percepciones de la poblacion',
            'https://doi.org/10.1590/SciELOPreprints.3158',
            '(AAAA-MM-DD)',
        ])
        assert self._labels(doc) == [SEG_COVER] * 4

    def test_a_status_label_alone_on_a_page_does_not(self):
        doc = self._pages(['Estado:', 'Some ordinary sentence of the article.'])
        assert SEG_COVER not in self._labels(doc)

    def test_a_submission_sheet_is_a_cover_whatever_it_opens_with(self):
        # PPR458890 carries the review system's sheet on its second page.  The
        # "For Review Only" watermark is set a letter to a line, so the page
        # does not open with anything that names it.
        doc = self._pages(
            ['Estado: El preprint no ha sido enviado', 'Titulo del articulo'],
            ['F', 'o', 'r', 'Economic impact of new recommendations',
             'Manuscript ID PBOCI-2020-0133',
             'https://mc04.manuscriptcentral.com/pboci-scielo'],
        )
        assert self._labels(doc) == [SEG_COVER] * 8

    def test_an_ordinary_page_is_not(self):
        doc = self._pages(
            ['Estado: El preprint no ha sido enviado', 'Titulo del articulo'],
            ['The argument of the article continues over the page here.'],
        )
        assert self._labels(doc)[2:] == [SEG_BODY]

    def test_the_dates_a_cover_runs_over_to_are_part_of_it(self):
        # The form overflows to a page holding only its date legend and the
        # generator's footer.
        doc = self._pages(
            ['Estado: El preprint no ha sido enviado', 'Titulo del articulo'],
            ['Envi ado en (AAAA-MM-DD): 2020-09-23', 'Postado en (AAAA-MM-DD): 2020-09-24'],
        )
        assert self._labels(doc) == [SEG_COVER] * 4


class TestCoordinateBasedDetection:
    @staticmethod
    def _two_pages(*top_lines: LayoutLine) -> LayoutDocument:
        """A page each, with the given line at the top of it."""
        return LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[line])],
                meta=make_page_meta(page_number=page_number, height=1000.0),
            )
            for page_number, line in enumerate(top_lines, 1)
        ])

    def test_a_line_repeated_at_the_top_of_pages_becomes_headnote(self):
        # 20/1000 = 2% < 8%, and a running header is what runs across pages.
        lines = [
            make_line('Running', 'header', y=20.0, page_number=page_number)
            for page_number in (1, 2)
        ]
        doc = self._two_pages(*lines)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated, headnote_y_ratio=0.08)
        assert [labels[id(line)] for line in lines] == [SEG_HEADNOTE, SEG_HEADNOTE]

    def test_a_line_printed_once_at_the_top_is_not_a_headnote(self):
        caption = make_line('Table', '6:', 'results', y=20.0, page_number=1)
        other = make_line('Running', 'header', y=20.0, page_number=2)
        doc = self._two_pages(caption, other)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated, headnote_y_ratio=0.08)
        assert labels[id(caption)] != SEG_HEADNOTE

    def test_a_page_number_at_the_top_becomes_page(self):
        numbers = [
            make_line(str(page_number), y=20.0, page_number=page_number)
            for page_number in (1, 2)
        ]
        doc = self._two_pages(*numbers)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated, headnote_y_ratio=0.08)
        assert [labels[id(line)] for line in numbers] == [SEG_PAGE, SEG_PAGE]

    def test_a_page_number_the_jats_vote_claimed_still_becomes_page(self):
        # PPR458837 prints its number in the same corner of all 23 pages; the
        # aligner matched 7 of them into the article's text, which is where the
        # number stopped being furniture.
        numbers = [
            make_line(str(page_number), y=20.0, page_number=page_number)
            for page_number in (1, 2, 3)
        ]
        doc = self._two_pages(*numbers)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        annotated.set_token_label(
            numbers[1].tokens[0], JatsFieldNames.BODY_SECTION_PARAGRAPH
        )
        labels = derive_labels(doc, annotated, headnote_y_ratio=0.08)
        assert [labels[id(line)] for line in numbers] == [
            SEG_PAGE, SEG_PAGE, SEG_PAGE
        ]

    def test_line_in_middle_of_page_is_not_headnote(self):
        line = make_line('Normal', 'content', y=500.0)  # 50% of page
        doc = make_doc_with_page(LayoutBlock(lines=[line]), page_height=1000.0)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated)
        assert labels[id(line)] != SEG_HEADNOTE

    def test_numeric_line_at_bottom_becomes_page(self):
        line = make_line('42', y=950.0)  # 95% of page > 92%
        doc = make_doc_with_page(LayoutBlock(lines=[line]), page_height=1000.0)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated, footnote_y_ratio=0.92)
        assert labels[id(line)] == SEG_PAGE

    def test_spelled_out_page_marker_at_bottom_becomes_page(self):
        line = make_line('Page', '3', 'of', '12', y=950.0)
        doc = make_doc_with_page(LayoutBlock(lines=[line]), page_height=1000.0)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(doc, annotated, footnote_y_ratio=0.92)
        assert labels[id(line)] == SEG_PAGE

    def test_numeric_line_away_from_the_footer_is_not_a_page_number(self):
        page_marker = make_line('Page', '3', 'of', '12', y=950.0)
        row_number = make_line('2', y=500.0)
        doc = make_doc_with_page(LayoutBlock(lines=[row_number, page_marker]))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        assert derive_labels(doc, annotated)[id(row_number)] != SEG_PAGE


class TestGapMerge:
    def test_untagged_line_between_front_lines_gets_front(self):
        line_front1 = make_line('Title', 'text', y=200.0)
        line_gap = make_line('Some', 'untagged', 'stuff', y=220.0)
        line_front2 = make_line('More', 'header', y=240.0)
        block = LayoutBlock(lines=[line_front1, line_gap, line_front2])
        doc = make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        for t in line_front1.tokens:
            annotated.set_token_label(t, JatsFieldNames.TITLE)
        for t in line_front2.tokens:
            annotated.set_token_label(t, JatsFieldNames.AUTHOR)
        labels = derive_labels(doc, annotated)
        assert labels[id(line_front1)] == SEG_FRONT
        assert labels[id(line_front2)] == SEG_FRONT
        assert labels[id(line_gap)] == SEG_FRONT

    def test_untagged_line_after_body_stays_body(self):
        line_body = make_line('Body', 'paragraph', y=400.0)
        line_gap = make_line('More', 'stuff', y=420.0)
        block = LayoutBlock(lines=[line_body, line_gap])
        doc = make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        for t in line_body.tokens:
            annotated.set_token_label(t, JatsFieldNames.BODY_SECTION_PARAGRAPH)
        labels = derive_labels(doc, annotated)
        assert labels[id(line_body)] == SEG_BODY
        # gap after body → body (default)
        assert labels[id(line_gap)] == SEG_BODY


class TestGapMergeAcrossPageFurniture:
    """A page break interrupts a region without ending it, footer and all."""

    def _make_doc(self, middle_text: str):
        first = make_line('Smith,', 'J.', '(2020).', 'A', 'title', page_number=1)
        middle = make_line(*middle_text.split(), y=950.0, page_number=1)
        gap = make_line('and', 'the', 'rest', 'of', 'it', page_number=2)
        last = make_line('Jones,', 'K.', '(2021).', 'Another', page_number=2)
        # The foot of the second page too: a running foot is what repeats, and
        # that is what tells it apart from a line the margin rules took by
        # accident.
        trailing = make_line(*middle_text.split(), y=950.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[first]), LayoutBlock(lines=[middle])],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=[gap, last]), LayoutBlock(lines=[trailing])],
                meta=make_page_meta(2),
            ),
        ])
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            lines.index(first): JatsFieldNames.REFERENCE,
            lines.index(last): JatsFieldNames.REFERENCE,
        })
        return doc, annotated, middle, gap

    def test_a_page_number_between_two_reference_lines_is_stepped_over(self):
        doc, annotated, middle, gap = self._make_doc('Page 3 of 12')
        labels = derive_labels(doc, annotated)
        assert labels.get(id(middle)) == SEG_PAGE
        assert labels.get(id(gap)) == SEG_REFERENCES

    def test_a_footer_between_two_reference_lines_is_stepped_over(self):
        doc, annotated, middle, gap = self._make_doc('Powered by TCPDF')
        labels = derive_labels(doc, annotated)
        assert labels.get(id(middle)) == SEG_FOOTNOTE
        assert labels.get(id(gap)) == SEG_REFERENCES


class TestReclaimRepeatedHeadnote:
    def test_a_running_header_a_match_absorbed_is_taken_back(self):
        header_lines = [
            make_line('Journal', 'of', 'Things', y=20.0, page_number=page)
            for page in (1, 2, 3)
        ]
        absorbed = header_lines[2]
        body = make_line('a', 'paragraph', 'spanning', 'the', 'break', page_number=3)
        pages = [
            LayoutPage(blocks=[LayoutBlock(lines=[header_lines[0]])], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[header_lines[1]])], meta=make_page_meta(2)),
            LayoutPage(blocks=[LayoutBlock(lines=[absorbed, body])], meta=make_page_meta(3)),
        ]
        doc = LayoutDocument(pages=pages)
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            lines.index(absorbed): JatsFieldNames.BODY_SECTION_PARAGRAPH,
            lines.index(body): JatsFieldNames.BODY_SECTION_PARAGRAPH,
        })
        assert derive_labels(doc, annotated)[id(absorbed)] == SEG_HEADNOTE


class TestRegionStartsAtThePageTop:
    """A section's heading is above the first line the JATS can evidence."""

    def _make_doc(self, heading_is_grounded: bool):
        body = make_line('The', 'article', 'body', 'ends', 'here', page_number=1)
        heading = make_line('Open', 'Peer', 'Review', y=80.0, page_number=2)
        report = make_line('Thank', 'you', 'for', 'addressing', 'the', 'comments',
                           y=300.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=[body])], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=[heading, report])], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        fields = {
            lines.index(body): JatsFieldNames.BODY_SECTION_PARAGRAPH,
            lines.index(report): JatsFieldNames.SUB_ARTICLE,
        }
        if heading_is_grounded:
            fields[lines.index(heading)] = JatsFieldNames.BODY_SECTION_PARAGRAPH
        return doc, annotate(doc, fields), heading

    def test_the_region_takes_the_unevidenced_heading_above_it(self):
        doc, annotated, heading = self._make_doc(heading_is_grounded=False)
        assert derive_labels(doc, annotated)[id(heading)] == SEG_REVIEW

    def test_an_evidenced_line_above_keeps_the_region_where_it_was(self):
        doc, annotated, heading = self._make_doc(heading_is_grounded=True)
        assert derive_labels(doc, annotated)[id(heading)] == SEG_BODY

    def test_the_region_reaches_back_over_an_entry_the_jats_does_not_carry(self):
        # PPR459453's list starts with an entry the JATS has no record of, so
        # the region begins four lines and a page break after its heading.
        heading = make_line('References', y=700.0, page_number=1)
        first = make_line('Alfaro,', 'J.', '(2020).', 'BCCR:', 'actividades', y=720.0,
                          page_number=1)
        second = make_line('transporte', 'y', 'construccion', y=740.0, page_number=1)
        number = make_line('18', y=960.0, page_number=1)
        third = make_line('demia.', 'AmeliaRueda.com,', y=100.0, page_number=2)
        entry = make_line('Almeida,', 'P.', '(2020).', 'Movimientos', 'sociales.',
                          y=120.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[heading, first, second]),
                        LayoutBlock(lines=[number])],
                meta=make_page_meta(1),
            ),
            LayoutPage(
                blocks=[LayoutBlock(lines=[third, entry])], meta=make_page_meta(2),
            ),
        ])
        annotated = annotate(doc, {5: JatsFieldNames.REFERENCE})
        labels = derive_labels(doc, annotated)
        assert [labels[id(line)] for line in (heading, first, second, third, entry)] == (
            [SEG_REFERENCES] * 5
        )
        assert labels[id(number)] == SEG_PAGE


class TestPageFootNotes:
    """A numbered note says what it is; where it prints does not decide it."""

    def _make_doc(self, field_name: str, note_y: float):
        body = make_line('The', 'argument', 'continues', y=300.0, page_number=1)
        note = make_line('1', 'Marks,', 'Robert', 'B.', 'Exhausting', 'the', 'Earth',
                         y=note_y, page_number=1)
        later = make_line('More', 'of', 'the', 'argument', y=300.0, page_number=2)
        doc = LayoutDocument(pages=[
            LayoutPage(
                blocks=[LayoutBlock(lines=[body]), LayoutBlock(lines=[note])],
                meta=make_page_meta(1),
            ),
            LayoutPage(blocks=[LayoutBlock(lines=[later])], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            lines.index(body): JatsFieldNames.BODY_SECTION_PARAGRAPH,
            lines.index(note): field_name,
            lines.index(later): JatsFieldNames.BODY_SECTION_PARAGRAPH,
        })
        return doc, annotated, note

    def test_a_numbered_note_is_a_footnote_low_on_the_page(self):
        doc, annotated, note = self._make_doc(JatsFieldNames.BACK_FOOTNOTE, note_y=780.0)
        assert derive_labels(doc, annotated)[id(note)] == SEG_FOOTNOTE

    def test_a_numbered_note_is_a_footnote_higher_up_too(self):
        doc, annotated, note = self._make_doc(JatsFieldNames.BACK_FOOTNOTE, note_y=420.0)
        assert derive_labels(doc, annotated)[id(note)] == SEG_FOOTNOTE

    def test_a_named_back_section_stays_an_annex_low_on_the_page(self):
        # An ethics statement prints wherever the article ends, which can be the
        # foot of a page, and it is not a note.
        doc, annotated, note = self._make_doc(
            JatsFieldNames.BACK_SECTION_PARAGRAPH, note_y=780.0
        )
        assert derive_labels(doc, annotated)[id(note)] == SEG_ANNEX


class TestFrontThreshold:
    def test_front_block_starting_within_threshold_is_kept(self):
        # A front block starting at line 0 should not be cleared
        lines = [make_line(f'word{i}') for i in range(5)]
        block = LayoutBlock(lines=lines)
        doc = make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        for t in lines[0].tokens:
            annotated.set_token_label(t, JatsFieldNames.TITLE)
        for t in lines[4].tokens:
            annotated.set_token_label(t, JatsFieldNames.AUTHOR)
        labels = derive_labels(doc, annotated, front_max_start_line_index=80)
        assert labels[id(lines[0])] == SEG_FRONT
        assert labels[id(lines[4])] == SEG_FRONT

    def test_front_block_starting_beyond_threshold_is_cleared(self):
        # A front block starting at line 100 (> 80) should be cleared → defaults to body
        lines = [make_line(f'word{i}') for i in range(3)]
        block = LayoutBlock(lines=lines)
        doc = make_doc_with_page(block)
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        # Annotate only the last line as front — its block starts at index 2 which is
        # not cleared.  Use a config with a very low threshold to test the clearing logic.
        # Copyright is what the rule is written against: a licence sentence
        # matching late is a field that landed somewhere it does not belong.
        for t in lines[2].tokens:
            annotated.set_token_label(t, JatsFieldNames.COPYRIGHT)
        labels = derive_labels(doc, annotated, front_max_start_line_index=1)
        # line 2 starts its own front block at index 2 > threshold 1 → cleared → body
        assert labels[id(lines[2])] == SEG_BODY

    def test_an_author_note_beyond_the_threshold_is_kept(self):
        # PPR458834 and PPR522900 print the corresponding author's address
        # after their reference lists; that block is front matter wherever a
        # preprint server sets it.
        lines = [make_line(f'word{i}') for i in range(3)]
        doc = make_doc_with_page(LayoutBlock(lines=lines))
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        for t in lines[2].tokens:
            annotated.set_token_label(t, JatsFieldNames.AUTHOR_NOTES)
        labels = derive_labels(doc, annotated, front_max_start_line_index=1)
        assert labels[id(lines[2])] == SEG_FRONT


class TestTextRepetitionHeadnote:
    def test_repeated_line_near_top_becomes_headnote(self):
        # No coordinates → will fall through to text-repetition detection
        lines_no_coords = [LayoutLine(tokens=[LayoutToken(text=t)]) for t in ['Journal Name'] * 3]
        block2 = LayoutBlock(lines=lines_no_coords)
        doc = LayoutDocument(pages=[LayoutPage(blocks=[block2])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=doc)
        labels = derive_labels(
            doc, annotated, page_header_max_first_line_index=10
        )
        for line in lines_no_coords:
            assert labels.get(id(line)) == SEG_HEADNOTE


class TestFloatPlacement:
    """A float is body wherever it prints, as GROBID's own corpus has it."""

    def test_float_after_the_references_is_still_body(self):
        reference, table = make_line('Smith', '2020'), make_line('Table', '1.', 'Measures')
        doc = make_doc_with_page(LayoutBlock(lines=[reference, table]))
        annotated = annotate(doc, {
            0: JatsFieldNames.REFERENCE,
            1: JatsFieldNames.FLOAT_TABLE,
        })
        assert derive_labels(doc, annotated)[id(table)] == SEG_BODY

    def test_float_before_the_references_is_body(self):
        figure, reference = make_line('Figure', '1.', 'Model'), make_line('Smith', '2020')
        doc = make_doc_with_page(LayoutBlock(lines=[figure, reference]))
        annotated = annotate(doc, {
            0: JatsFieldNames.FLOAT_FIGURE,
            1: JatsFieldNames.REFERENCE,
        })
        assert derive_labels(doc, annotated)[id(figure)] == SEG_BODY


class TestFurnitureInsideReferences:
    """A reference wrapping over a page break prints in the header zone."""

    def _make_doc(self, *header_zone_texts: str, banner_on_first_page: bool = False):
        first_page_lines = [make_line('Smith,', 'J.', '(2020).', 'Journal', 'of')]
        if banner_on_first_page:
            first_page_lines.insert(0, make_line(*header_zone_texts[0].split(), y=50.0))
        second_page_lines = [
            make_line(*text.split(), y=50.0, page_number=2) for text in header_zone_texts
        ] + [make_line('38(2),', '279-294.', page_number=2)]
        doc = LayoutDocument(pages=[
            LayoutPage(blocks=[LayoutBlock(lines=first_page_lines)], meta=make_page_meta(1)),
            LayoutPage(blocks=[LayoutBlock(lines=second_page_lines)], meta=make_page_meta(2)),
        ])
        lines = list(doc.iter_all_lines())
        annotated = annotate(doc, {
            lines.index(first_page_lines[-1]): JatsFieldNames.REFERENCE,
            lines.index(second_page_lines[-1]): JatsFieldNames.REFERENCE,
        })
        return doc, annotated, second_page_lines

    def test_every_line_of_a_wrapped_reference_is_released(self):
        doc, annotated, second_page_lines = self._make_doc(
            'Development,', 'Infancia', 'y', 'Aprendizaje,'
        )
        labels = derive_labels(doc, annotated)
        for line in second_page_lines:
            assert labels.get(id(line)) == SEG_REFERENCES

    def test_a_repeated_running_header_stays_a_headnote(self):
        doc, annotated, second_page_lines = self._make_doc(
            'Preprints - this document is a preprint', banner_on_first_page=True
        )
        labels = derive_labels(doc, annotated)
        assert labels.get(id(second_page_lines[0])) == SEG_HEADNOTE


class TestFrontMatterBoundary:
    """The reference list bounds the front matter; a line index stands in for it."""

    def _make_doc(self, front_after_references: bool):
        texts = (
            [('A', 'title', 'of', 'the', 'paper')]
            + [('body', f'line{index}') for index in range(100)]
            + ([('Smith,', 'J.', '(2020).', 'A', 'reference'),
                ('Grant', 'information:', 'funded', 'by', 'a', 'grant')]
               if front_after_references
               else [('Grant', 'information:', 'funded', 'by', 'a', 'grant'),
                     ('Smith,', 'J.', '(2020).', 'A', 'reference')])
        )
        # One row per line, down the text area of the page, as a page sets them.
        lines = [
            make_line(*text, y=100.0 + 5.0 * row) for row, text in enumerate(texts)
        ]
        title = lines[0]
        grant = lines[-1] if front_after_references else lines[-2]
        reference = lines[-2] if front_after_references else lines[-1]
        doc = make_doc_with_page(LayoutBlock(lines=lines))
        index_of = {id(line): i for i, line in enumerate(lines)}
        annotated = annotate(doc, {
            index_of[id(title)]: JatsFieldNames.TITLE,
            index_of[id(grant)]: JatsFieldNames.FUNDING,
            index_of[id(reference)]: JatsFieldNames.REFERENCE,
        })
        return doc, annotated, grant

    def test_front_matter_past_the_index_is_kept_before_the_references(self):
        doc, annotated, grant = self._make_doc(front_after_references=False)
        labels = derive_labels(doc, annotated, front_max_start_line_index=80)
        assert labels[id(grant)] == SEG_FRONT

    def test_a_front_match_after_the_references_is_dropped(self):
        doc, annotated, grant = self._make_doc(front_after_references=True)
        labels = derive_labels(doc, annotated, front_max_start_line_index=80)
        assert labels[id(grant)] != SEG_FRONT
