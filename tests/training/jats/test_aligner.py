# pylint: disable=too-many-lines
from typing import Optional, Sequence, Set, Tuple

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutToken,
)
from sciencebeam_parser.training.jats.aligner import (
    AlignmentConfig,
    LayoutDocumentJatsAligner,
    extend_partial_reference_matches,
    _search_range,
    narrow_search_range_to_window,
    get_search_stride,
    get_search_window_size,
)
from sciencebeam_parser.training.jats.text_normalizer import normalize_for_alignment
from sciencebeam_parser.training.jats.field_extractor import JatsFieldValue
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames, JatsSubFieldNames


def _make_doc(*line_texts: str) -> LayoutDocument:
    lines = [LayoutLine.for_text(t) for t in line_texts]
    return LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=lines)])])


def _fv(text: str, field: str = JatsFieldNames.BODY_SECTION_TITLE, sub: Optional[str] = None):
    return JatsFieldValue(text=text, field_name=field, sub_field_name=sub)


class TestSearchRangeForARepeatedAnchor:
    """Where a field's next value is looked for, which is the whole of the change.

    An anchor field's values repeat in a document written in several languages,
    and they interleave with another field's: a title, an abstract, then the next
    language's title. One cursor cannot serve both, so each anchor field follows
    where it last matched and the document's own cursor is left to the rest.
    """
    def _start(self, last_match_end: int, anchor_match_end_by_field=None) -> int:
        start, _end = _search_range(
            _fv('a title', JatsFieldNames.TITLE),
            last_match_end=last_match_end,
            body_floor=0,
            body_content_end=0,
            front_matter_end=0,
            keywords_floor=0,
            reference_floor=0,
            parent_match_by_field={},
            anchor_match_end_by_field=anchor_match_end_by_field,
        )
        return start

    def test_should_follow_the_document_where_the_field_has_not_matched_yet(self):
        assert self._start(5000) == 4800

    def test_should_follow_its_own_field_once_it_has(self):
        assert self._start(
            5000, {JatsFieldNames.TITLE: 1000}
        ) == 800

    def test_should_not_follow_another_anchor_field(self):
        # The abstract between two titles moved the document's cursor; the next
        # title is printed after the first one and not after that abstract.
        assert self._start(
            9000, {JatsFieldNames.TITLE: 1000}
        ) == 800


class TestLayoutDocumentJatsAligner:  # pylint: disable=too-many-public-methods
    def _align(self, doc, field_values, **kwargs):
        config = AlignmentConfig(**kwargs) if kwargs else None
        return LayoutDocumentJatsAligner(config).align(doc, field_values)

    def test_empty_field_values_returns_unannotated(self):
        doc = _make_doc('Hello world')
        annotated = self._align(doc, [])
        assert annotated.coverage_ratio() == 0.0

    def test_exact_match_labels_all_tokens(self):
        doc = _make_doc('Introduction')
        annotated = self._align(doc, [_fv('Introduction')])
        tokens = list(doc.iter_all_tokens())
        assert all(
            annotated.get_token_field(t) == JatsFieldNames.BODY_SECTION_TITLE
            for t in tokens
        )

    def test_multi_token_match(self):
        doc = _make_doc('The role of autophagy')
        annotated = self._align(doc, [_fv('The role of autophagy')])
        tokens = list(doc.iter_all_tokens())
        assert all(
            annotated.get_token_field(t) == JatsFieldNames.BODY_SECTION_TITLE
            for t in tokens
        )

    def test_unmatched_tokens_have_no_label(self):
        doc = _make_doc('Introduction', 'Some unrelated text here')
        annotated = self._align(doc, [_fv('Introduction')])
        unrelated_tokens = list(doc.iter_all_lines())[1].tokens
        assert all(annotated.get_token_field(t) is None for t in unrelated_tokens)

    def test_sub_field_overrides_parent_field(self):
        doc = _make_doc('Smith J 2020')
        fvs = [
            _fv('Smith J 2020', JatsFieldNames.REFERENCE),
            _fv('Smith J', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        # 'Smith' and 'J' tokens should have sub_field set
        smith_token = next(t for t in tokens if t.text == 'Smith')
        assert annotated.get_token_field(smith_token) == JatsFieldNames.REFERENCE
        assert annotated.get_token_sub_field(smith_token) == JatsSubFieldNames.REFERENCE_AUTHOR

    def test_fuzzy_match_with_minor_variant(self):
        # em-dash variant in haystack
        doc = _make_doc('foo—bar baz')
        annotated = self._align(doc, [_fv('foo-bar baz')])
        tokens = list(doc.iter_all_tokens())
        assert any(annotated.get_token_field(t) is not None for t in tokens)

    def test_coverage_ratio_partial(self):
        doc = _make_doc('Title text', 'Other text')
        annotated = self._align(doc, [_fv('Title text', JatsFieldNames.TITLE)])
        ratio = annotated.coverage_ratio()
        assert 0 < ratio < 1.0

    def test_coverage_ratio_full(self):
        doc = _make_doc('only this')
        annotated = self._align(doc, [_fv('only this')])
        assert annotated.coverage_ratio() == 1.0

    def test_multiple_fields_labeled_correctly(self):
        doc = _make_doc('My Title', 'John Smith')
        fvs = [
            _fv('My Title', JatsFieldNames.TITLE),
            _fv('John Smith', JatsFieldNames.AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        title_tokens = list(doc.iter_all_lines())[0].tokens
        author_tokens = list(doc.iter_all_lines())[1].tokens
        assert all(annotated.get_token_field(t) == JatsFieldNames.TITLE for t in title_tokens)
        assert all(annotated.get_token_field(t) == JatsFieldNames.AUTHOR for t in author_tokens)

    def test_front_matter_author_found_before_abstract_in_haystack(self):
        # Simulate the common PDF layout: author appears BEFORE the abstract in the
        # document, but JATS XML lists the abstract first.  With a long abstract the
        # old last_match_end-based search_start would skip past the author's position;
        # the front-matter region constraint fixes this.
        abstract_text = ' '.join(['word'] * 60)  # long enough that lme-200 > author pos
        doc = _make_doc('Author Name', abstract_text, 'Introduction')
        fvs = [
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv('Author Name', JatsFieldNames.AUTHOR),
            _fv('Introduction', JatsFieldNames.BODY_SECTION_TITLE),
        ]
        annotated = self._align(doc, fvs)
        author_tokens = list(doc.iter_all_lines())[0].tokens
        intro_tokens = list(doc.iter_all_lines())[2].tokens
        assert all(annotated.get_token_field(t) == JatsFieldNames.AUTHOR for t in author_tokens)
        assert all(
            annotated.get_token_field(t) == JatsFieldNames.BODY_SECTION_TITLE
            for t in intro_tokens
        )

    def test_abstract_is_reached_behind_a_later_title(self):
        """A document in two languages prints a title and an abstract for each.

        The JATS declares both titles before either abstract, and the second
        title prints after the first abstract. Advancing one cursor through both
        titles puts the first abstract behind it.
        """
        first_abstract = ' '.join(['alpha'] * 60)
        second_abstract = ' '.join(['beta'] * 60)
        doc = _make_doc(
            'The first title', first_abstract,
            'The translated title', second_abstract,
        )
        fvs = [
            _fv('The first title', JatsFieldNames.TITLE),
            _fv('The translated title', JatsFieldNames.TITLE),
            _fv(first_abstract, JatsFieldNames.ABSTRACT),
            _fv(second_abstract, JatsFieldNames.ABSTRACT),
        ]
        annotated = self._align(doc, fvs)
        lines = list(doc.iter_all_lines())
        for index in (1, 3):
            assert all(
                annotated.get_token_field(token) == JatsFieldNames.ABSTRACT
                for token in lines[index].tokens
            ), f'abstract on line {index} was not reached'

    def test_declaration_is_reached_where_the_front_matter_prints_it(self):
        """A preprint server prints the declarations before the abstract.

        The JATS files them in the back matter, so they are searched from the
        body floor, which the page has already passed by the time it sets them.
        """
        abstract_text = ' '.join(['word'] * 300)
        body_text = ' '.join(['sentence'] * 300)
        statement = 'The authors own no shares in the ferry company'
        doc = _make_doc(
            'Author Name', statement, abstract_text, 'Introduction', body_text,
        )
        fvs = [
            _fv('Author Name', JatsFieldNames.AUTHOR),
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv('Introduction', JatsFieldNames.BODY_SECTION_TITLE),
            _fv(body_text, JatsFieldNames.BODY_SECTION_PARAGRAPH),
            _fv(statement, JatsFieldNames.CONFLICT_SECTION_PARAGRAPH),
        ]
        annotated = self._align(doc, fvs)
        statement_tokens = list(doc.iter_all_lines())[1].tokens
        assert all(
            annotated.get_token_field(token)
            == JatsFieldNames.CONFLICT_SECTION_PARAGRAPH
            for token in statement_tokens
        )

    def test_declaration_naming_only_the_authors_leaves_the_byline_alone(self):
        """One task per paragraph makes a declaration a list of names.

        Printed after the body or not printed at all, such a paragraph matches
        the byline as well as it matches itself, and the byline is the authors
        rather than a statement about them.
        """
        abstract_text = ' '.join(['word'] * 60)
        byline = 'Petra Vance, Kaya Oduya, Lars Petit'
        doc = _make_doc(byline, abstract_text, 'Introduction Body')
        fvs = [
            _fv(byline, JatsFieldNames.AUTHOR),
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv(byline, JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH),
        ]
        annotated = self._align(doc, fvs)
        byline_tokens = list(doc.iter_all_lines())[0].tokens
        assert all(
            annotated.get_token_field(token) == JatsFieldNames.AUTHOR
            for token in byline_tokens
        )

    def test_identifier_is_labelled_wherever_the_front_matter_prints_it(self):
        """A preprint server prints the article's identifier more than once."""
        doi = 'https://doi.org/10.1234/preprints.567'
        abstract_text = ' '.join(['word'] * 60)
        doc = _make_doc(
            'Author Name', doi, abstract_text, 'How to cite: ' + doi, 'Introduction',
        )
        fvs = [
            _fv('Author Name', JatsFieldNames.AUTHOR),
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv(doi, JatsFieldNames.DOI),
        ]
        annotated = self._align(doc, fvs)
        lines = list(doc.iter_all_lines())
        for index in (1, 3):
            assert any(
                annotated.get_token_field(token) == JatsFieldNames.DOI
                for token in lines[index].tokens
            ), f'the identifier on line {index} was not labelled'

    def test_a_neighbouring_identifier_is_left_alone(self):
        """The versions a preprint server has published differ in one character."""
        doi = 'https://doi.org/10.1234/preprints.567.3'
        other_version = 'https://doi.org/10.1234/preprints.567.1'
        abstract_text = ' '.join(['word'] * 60)
        doc = _make_doc('Author Name', doi, abstract_text, other_version, 'Introduction')
        fvs = [
            _fv('Author Name', JatsFieldNames.AUTHOR),
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv(doi, JatsFieldNames.DOI),
        ]
        annotated = self._align(doc, fvs)
        lines = list(doc.iter_all_lines())
        assert not any(
            annotated.get_token_field(token) == JatsFieldNames.DOI
            for token in lines[3].tokens
        )

    def test_a_value_keeps_the_short_piece_it_opens_with(self):
        """The page sets a space after each dot, and the match dropped the first.

        Smith-Waterman stops when extending would not raise the score, and a
        one-character opening separated by a gap scores net zero.
        """
        doc = _make_doc('J.J. Leahy and Nidal Khalaf', 'Introduction Body')
        fvs = [_fv('J.J. Leahy', JatsFieldNames.AUTHOR)]
        annotated = self._align(doc, fvs)
        labelled = [
            token.text for token in list(doc.iter_all_lines())[0].tokens
            if annotated.get_token_field(token) == JatsFieldNames.AUTHOR
        ]
        assert labelled[:2] == ['J', '.']

    def test_a_name_the_front_matter_prints_again_is_labelled_again(self):
        """A correspondence block names the authors a second time."""
        abstract_text = ' '.join(['word'] * 60)
        doc = _make_doc(
            'Petra Vance 1 , Kaya Oduya 2',
            'Correspondence: Petra Vance (petra@example.org)',
            abstract_text,
            'Introduction Body',
        )
        fvs = [
            _fv('Petra Vance 1', JatsFieldNames.AUTHOR, JatsSubFieldNames.AUTHOR_NAME),
            _fv('Kaya Oduya 2', JatsFieldNames.AUTHOR, JatsSubFieldNames.AUTHOR_NAME),
            _fv('Petra Vance 1 Kaya Oduya 2', JatsFieldNames.AUTHOR),
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
        ]
        fvs[0].fallback_text = 'Petra Vance'
        fvs[1].fallback_text = 'Kaya Oduya'
        annotated = self._align(doc, fvs)
        second_line = list(doc.iter_all_lines())[1]
        labelled = [
            token.text for token in second_line.tokens
            if annotated.get_token_field(token) == JatsFieldNames.AUTHOR
        ]
        assert labelled == ['Petra', 'Vance']

    def test_an_affiliation_two_authors_share_is_labelled_under_each(self):
        """The JATS declares it once per author and the page prints it once per author."""
        abstract_text = ' '.join(['word'] * 60)
        aff_text = 'Institute of Marine Research, Bergen, Norway'
        doc = _make_doc(
            'Petra Vance',
            aff_text,
            'Kaya Oduya',
            aff_text,
            abstract_text,
            'Introduction Body',
        )
        fvs = [
            _fv('Petra Vance Kaya Oduya', JatsFieldNames.AUTHOR),
            _fv(aff_text, JatsFieldNames.AUTHOR_AFF),
            _fv(aff_text, JatsFieldNames.AUTHOR_AFF),
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
        ]
        annotated = self._align(doc, fvs)
        lines = list(doc.iter_all_lines())
        labelled_lines = [
            index for index, line in enumerate(lines)
            if all(
                annotated.get_token_field(token) == JatsFieldNames.AUTHOR_AFF
                for token in line.tokens
            )
        ]
        assert labelled_lines == [1, 3]

    def test_an_orcid_the_page_prints_bare_is_labelled(self):
        abstract_text = ' '.join(['word'] * 60)
        doc = _make_doc(
            'Petra Vance',
            'ORCID: 0000-0002-1672-9686',
            abstract_text,
            'Introduction Body',
        )
        orcid = _fv('https://orcid.org/0000-0002-1672-9686', JatsFieldNames.ORCID)
        orcid.fallback_text = '0000-0002-1672-9686'
        annotated = self._align(doc, [
            _fv('Petra Vance', JatsFieldNames.AUTHOR),
            orcid,
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
        ])
        second_line = list(doc.iter_all_lines())[1]
        labelled = ''.join(
            token.text for token in second_line.tokens
            if annotated.get_token_field(token) == JatsFieldNames.ORCID
        )
        assert labelled == '0000-0002-1672-9686'

    def test_affiliation_found_after_body_when_not_in_front_matter(self):
        # Some journals place full author/affiliation blocks at the END of the paper.
        # The front-matter region constraint is soft: if the aff text is not found in
        # [0, abstract_end + buffer] it should fall back to a global search.
        abstract_text = ' '.join(['word'] * 60)
        aff_text = 'Department of Computer Science University of Toronto Canada'
        doc = _make_doc(abstract_text, 'Introduction Body', aff_text)
        fvs = [
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv(aff_text, JatsFieldNames.AUTHOR_AFF),
        ]
        annotated = self._align(doc, fvs)
        aff_tokens = list(doc.iter_all_lines())[2].tokens
        assert all(annotated.get_token_field(t) == JatsFieldNames.AUTHOR_AFF for t in aff_tokens)

    def test_keywords_anchored_without_keywords_title(self):
        # Even when there is no KEYWORDS_TITLE in the JATS (kwd-group has no <title>),
        # the combined keyword string should not match in the abstract.  The fallback
        # anchors the search to just after front_matter_end (abstract end).
        abstract_text = 'gene regulation and enhancers in limb development'
        doc = _make_doc(abstract_text, 'gene regulation, enhancers')
        fvs = [
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            # no KEYWORDS_TITLE — single combined value per GROBID guidelines
            _fv('gene regulation, enhancers', JatsFieldNames.KEYWORDS),
        ]
        annotated = self._align(doc, fvs)
        kw_tokens = list(doc.iter_all_lines())[1].tokens
        assert all(annotated.get_token_field(t) == JatsFieldNames.KEYWORDS for t in kw_tokens)

    def test_keywords_anchored_to_keywords_section(self):
        # Per GROBID guidelines, all keywords form ONE field value.  The whole list
        # should be tagged on the keywords line, not in the abstract where the same
        # words appear individually.
        abstract_text = 'Bayesian confidence readout in dynamic stimuli tasks'
        doc = _make_doc(abstract_text, 'Keywords confidence, Bayesian, DDM')
        fvs = [
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv('Keywords', JatsFieldNames.KEYWORDS_TITLE),
            _fv('confidence, Bayesian, DDM', JatsFieldNames.KEYWORDS),
        ]
        annotated = self._align(doc, fvs)
        kw_tokens = list(doc.iter_all_lines())[1].tokens
        kw_by_text = {t.text.lower().strip(','): t for t in kw_tokens}
        assert annotated.get_token_field(kw_by_text['keywords']) == JatsFieldNames.KEYWORDS_TITLE
        assert annotated.get_token_field(kw_by_text['confidence']) == JatsFieldNames.KEYWORDS
        assert annotated.get_token_field(kw_by_text['bayesian']) == JatsFieldNames.KEYWORDS
        assert annotated.get_token_field(kw_by_text['ddm']) == JatsFieldNames.KEYWORDS

    def test_sub_field_confined_to_parent_range(self):
        # "Canada" appears in both lines; sub-field containment should pin the
        # AUTHOR_AFF_COUNTRY match to the aff line, not the earlier occurrence.
        aff_text = 'University of Toronto Canada'
        # Put the ambiguous "Canada" earlier in the document than the aff block.
        doc = _make_doc(
            'Some funding from Canada',
            aff_text,
        )
        fvs = [
            JatsFieldValue(aff_text, JatsFieldNames.AUTHOR_AFF),
            JatsFieldValue(
                'Canada', JatsFieldNames.AUTHOR_AFF,
                sub_field_name=JatsSubFieldNames.AUTHOR_AFF_COUNTRY,
            ),
        ]
        annotated = self._align(doc, fvs)
        aff_tokens = list(doc.iter_all_lines())[1].tokens
        canada_in_aff = next(t for t in aff_tokens if 'canada' in t.text.lower())
        assert annotated.get_token_sub_field(canada_in_aff) == JatsSubFieldNames.AUTHOR_AFF_COUNTRY

    def test_consecutive_affiliations_have_distinct_instance_ids(self):
        # Each JATS <aff> must produce a separate TEI <affiliation> element.  The
        # mechanism relies on the aligner assigning a distinct instance_id to each
        # main (sub_field_name=None) AUTHOR_AFF field value so the header label fn
        # emits B- on the first token of every new affiliation — even when no
        # <address> tokens appear between them to force a label change.
        aff1_text = '1 Institut Barcelona Spain'
        aff2_text = '2 Cochrane Iberoamerica Madrid Spain'
        doc = _make_doc(aff1_text, aff2_text)
        fvs = [
            JatsFieldValue(aff1_text, JatsFieldNames.AUTHOR_AFF),
            JatsFieldValue(aff2_text, JatsFieldNames.AUTHOR_AFF),
        ]
        annotated = self._align(doc, fvs)
        aff1_tokens = list(doc.iter_all_lines())[0].tokens
        aff2_tokens = list(doc.iter_all_lines())[1].tokens
        assert all(
            annotated.get_token_field(t) == JatsFieldNames.AUTHOR_AFF
            for t in aff1_tokens + aff2_tokens
        )
        # First aff → instance 1, second aff → instance 2: must differ
        assert annotated.get_token_instance(aff1_tokens[0]) == 1
        assert annotated.get_token_instance(aff2_tokens[0]) == 2
        assert (
            annotated.get_token_instance(aff1_tokens[0])
            != annotated.get_token_instance(aff2_tokens[0])
        )

    def test_an_abstract_is_read_across_what_the_page_prints_inside_it(self):
        """The footnotes of the page it runs onto are matched values of their own."""
        first_half = ' '.join(['alpha'] * 40)
        second_half = ' '.join(['omega'] * 40)
        interruption = 'Instituto de Pesquisa Economica Aplicada, Brasilia, Brasil'
        doc = _make_doc(
            'Petra Vance',
            first_half,
            interruption,
            second_half,
            'Introduction Body',
        )
        abstract_text = first_half + ' ' + second_half
        annotated = self._align(doc, [
            _fv('Petra Vance', JatsFieldNames.AUTHOR),
            _fv(abstract_text, JatsFieldNames.ABSTRACT),
            _fv(interruption, JatsFieldNames.AUTHOR_AFF),
        ])
        lines = list(doc.iter_all_lines())
        labelled = [
            {annotated.get_token_field(token) for token in line.tokens}
            for line in lines
        ]
        assert labelled[1] == {JatsFieldNames.ABSTRACT}
        assert labelled[2] == {JatsFieldNames.AUTHOR_AFF}
        assert labelled[3] == {JatsFieldNames.ABSTRACT}

    def test_abstract_does_not_label_sidebar_content(self):
        # PDFs sometimes have a sidebar (e.g. Open Peer Review box) between the
        # first and second page of an abstract.  The JATS abstract needle contains
        # no sidebar text, so Smith-Waterman creates only tiny (size ≤ 4) scatter
        # blocks while traversing the sidebar.  Those blocks must NOT cause sidebar
        # tokens to be labelled as abstract.
        page1 = (
            'Every day important healthcare decisions are made with incomplete '
            'information about the effects of the healthcare interventions '
            'available. It is necessary to invest in strategies that allow access '
            'to reliable and updated evidence on which to base health decisions.'
        )
        # Sidebar: distinct proper-noun vocabulary with no 5+-char substring in page1/page2
        sidebar = (
            'Open Peer Approval Ingrid Schmitt Lozano Maastricht '
            'Pontificia Universidad Catolica panel assessment'
        )
        page2 = (
            'The project will be developed in three complementary phases. '
            'Expected results include an effective capacity-building strategy '
            'for health system organizations to implement the living evidence model.'
        )
        abstract_text = page1 + ' ' + page2
        # Doc reading order: page1, then sidebar, then page2 (three separate lines)
        doc = _make_doc(page1, sidebar, page2)
        fvs = [_fv(abstract_text, JatsFieldNames.ABSTRACT)]
        annotated = self._align(doc, fvs)

        lines = list(doc.iter_all_lines())
        sidebar_tokens = lines[1].tokens
        labeled_sidebar = [
            t.text for t in sidebar_tokens
            if annotated.get_token_field(t) == JatsFieldNames.ABSTRACT
        ]
        assert labeled_sidebar == [], (
            f'Sidebar tokens incorrectly labelled as abstract: {labeled_sidebar}'
        )

        # Page-2 abstract content (not the very first boundary token) must be labelled
        page2_tokens = lines[2].tokens
        page2_by_text = {t.text.lower(): t for t in page2_tokens}
        for word in ('expected', 'capacity', 'building', 'organizations'):
            if word in page2_by_text:
                assert annotated.get_token_field(page2_by_text[word]) == JatsFieldNames.ABSTRACT, (
                    f"Page-2 abstract token '{word}' was not labelled"
                )

    def test_doi_sub_field_all_tokens_labeled(self):
        # The DOI tokenises into many short sub-tokens (2, 1, 4, 1, 3, 8 chars each).
        # Without special handling the anchor+chain filter only labels the last long
        # segment; all preceding dot/slash/digit segments should also be labeled.
        ref_text = 'Smith J 2020 Some paper J Virol 10.1128/mBio.00524-13'
        doi = '10.1128/mBio.00524-13'
        doc = _make_doc(ref_text)
        fvs = [
            _fv(ref_text, JatsFieldNames.REFERENCE),
            _fv(doi, JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_DOI),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        doi_tokens = [
            t for t in tokens
            if annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_DOI
        ]
        doi_text = ''.join(t.text for t in doi_tokens)
        assert doi_text == doi, (
            f'Expected DOI tokens "{doi}", got "{doi_text}"'
        )

    def test_doi_sub_field_labeled_when_split_across_line_break(self):
        # DOI split at end-of-line hyphen: PDF tokenizer emits a bare '-' as the last
        # token of the first line, which the aligner strips (skip_tokens).  All
        # prefix sub-tokens before the join must still receive the DOI sub-field label.
        ref_text = 'Smith J 2020 Some paper JRS 10.3233/JRS-201017'
        doi = '10.3233/JRS-201017'
        # Two-line doc: first line ends with the hyphen, second line has the suffix.
        doc = _make_doc('Smith J 2020 Some paper JRS 10.3233/JRS-', '201017')
        fvs = [
            _fv(ref_text, JatsFieldNames.REFERENCE),
            _fv(doi, JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_DOI),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        doi_tokens = [
            t for t in tokens
            if annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_DOI
        ]
        doi_labeled_text = ''.join(t.text for t in doi_tokens)
        # tokens_in_range re-includes the bare '-' skip token because it follows the
        # labeled 'JRS' token, so the full hyphenated form is reconstructed.
        expected_labeled = '10.3233/JRS-201017'
        assert doi_labeled_text == expected_labeled, (
            f'Expected labeled DOI text "{expected_labeled}", got "{doi_labeled_text}"'
        )

    def test_reference_spanning_page_break_does_not_label_headnote(self):
        # A reference whose text spans a PDF page break has a running page header
        # ("Journal Name 2025, 5:251") interleaved between its pre-break and
        # post-break tokens.  The SW blocks for a reference sub-field (e.g. article
        # title) that crosses the break must NOT cause the headnote line to be
        # labeled as a reference field.
        # The anchor+chain filter handles this: the large gap between the last
        # pre-break anchor block and the headnote blocks means they are not
        # within_gap and are not part of pre_anchor, so they are dropped.
        headnote = 'Journal Name 2025, 5:251 Last updated: 13 MAR 2026'
        ref_text = (
            'Smith J 2020 Clogging phenomenon in continuous casting of steel '
            'a review Steel Res Int 10.1002/srin.201800'
        )
        # Doc layout: ref pre-break line, then the running headnote, then ref suffix
        doc = _make_doc(
            'Smith J 2020 Clogging phenomenon in continuous casting of steel',
            headnote,
            'a review Steel Res Int 10.1002/srin.201800',
        )
        fvs = [
            _fv(ref_text, JatsFieldNames.REFERENCE),
            _fv(
                'Clogging phenomenon in continuous casting of steel a review',
                JatsFieldNames.REFERENCE,
                JatsSubFieldNames.REFERENCE_ARTICLE_TITLE,
            ),
        ]
        annotated = self._align(doc, fvs)
        lines = list(doc.iter_all_lines())
        headnote_tokens = lines[1].tokens
        labeled_headnote = [
            t.text for t in headnote_tokens
            if annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_ARTICLE_TITLE
        ]
        assert labeled_headnote == [], (
            f'Headnote tokens incorrectly labeled as reference sub-field: {labeled_headnote}'
        )

    def test_heading_label_not_overwritten_by_paragraph_mid_token_match(self):
        # Regression: SW alignment for a paragraph starting with "O crescimento"
        # finds the last 'o' of "Introdução" (the heading token's tail char) as a
        # spurious 2-char block ('o ').  Without the token-boundary guard in the
        # pre-anchor pass, tokens_in_range on that block returns the "Introdução"
        # heading token and overwrites its BODY_SECTION_TITLE label with
        # BODY_SECTION_PARAGRAPH.
        doc = _make_doc('Introdução', 'O crescimento da pandemia do Covid')
        fvs = [
            _fv('Introdução', JatsFieldNames.BODY_SECTION_TITLE),
            _fv('O crescimento da pandemia do Covid', JatsFieldNames.BODY_SECTION_PARAGRAPH),
        ]
        annotated = self._align(doc, fvs)
        heading_token = list(doc.iter_all_lines())[0].tokens[0]
        assert annotated.get_token_field(heading_token) == JatsFieldNames.BODY_SECTION_TITLE, (
            f'Heading token label was overwritten to {annotated.get_token_field(heading_token)!r}'
        )

    def test_per_name_authors_label_last_initial(self):
        # Regression: when the whole person-group was one needle, a JATS/PDF
        # format mismatch on an earlier name (e.g. "BE" vs "B. F.") could cause
        # the SW to terminate before labelling the last author's given-name initial.
        # Per-name emission gives each name its own independent SW run.
        doc = _make_doc('MAIER, B. F.; BROCKMANN, D. Some title 2020')
        fvs = [
            _fv('MAIER, B. F.; BROCKMANN, D. Some title 2020', JatsFieldNames.REFERENCE),
            _fv('Maier BE', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Brockmann D', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        d_token = next(
            (t for t in tokens if normalize_for_alignment(t.text) == 'd'),
            None,
        )
        assert d_token is not None, 'No token matching "D" found in document'
        assert annotated.get_token_sub_field(d_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            f'"D." token sub-field was {annotated.get_token_sub_field(d_token)!r}, '
            'expected REFERENCE_AUTHOR'
        )

    def test_trailing_period_after_last_initial_is_labeled(self):
        # The PDF tokeniser splits 'D.' into two tokens 'D' and '.'.
        # The period is not in the JATS text, so it must be attached by the
        # trailing-period pass rather than by SW.
        doc = _make_doc('BROCKMANN, D. Some title 2020')
        fvs = [
            _fv('BROCKMANN, D. Some title 2020', JatsFieldNames.REFERENCE),
            _fv('Brockmann D', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        # '.' immediately after 'D' should be REFERENCE_AUTHOR
        d_idx = next(
            i for i, t in enumerate(tokens) if normalize_for_alignment(t.text) == 'd'
        )
        period_token = tokens[d_idx + 1]
        assert period_token.text == '.', 'Expected period token after D'
        assert annotated.get_token_sub_field(period_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            f'Period after D should be REFERENCE_AUTHOR, got '
            f'{annotated.get_token_sub_field(period_token)!r}'
        )

    def test_trailing_period_unlabeled_by_outer_match_is_attached(self):
        # When the gap between the last initial and the next matched word exceeds
        # _MAX_HAYSTACK_GAP_TO_FILL, the abbreviation period is outside the outer
        # SW match range (entry=None). The trailing-period pass must still attach it.
        doc = _make_doc('Zorblax, C. A. (2010). Some title.')
        fvs = [
            _fv('Zorblax CA Some title 2010', JatsFieldNames.REFERENCE),
            _fv('Zorblax CA', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        a_idx = next(i for i, t in enumerate(tokens) if normalize_for_alignment(t.text) == 'a')
        period_token = tokens[a_idx + 1]
        assert period_token.text == '.', 'Expected period token after A'
        assert annotated.get_token_sub_field(period_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            f'Period after A should be REFERENCE_AUTHOR, got '
            f'{annotated.get_token_sub_field(period_token)!r}'
        )

    def test_gap_fill_merges_per_name_author_spans(self):
        # Between two per-name author matches, separator tokens (comma, semicolon,
        # initials with periods) should be filled in as REFERENCE_AUTHOR so that
        # all author tokens form a single <author> element.
        doc = _make_doc('Smith A. B.; Jones C. D. Some title 2020')
        fvs = [
            _fv('Smith A. B.; Jones C. D. Some title 2020', JatsFieldNames.REFERENCE),
            _fv('Smith AB', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Jones CD', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        # Every token up to and including the final 'D.' of Jones should be REFERENCE_AUTHOR
        smith_idx = next(i for i, t in enumerate(tokens) if t.text == 'Smith')
        jones_idx = next(i for i, t in enumerate(tokens) if t.text == 'Jones')
        # Tokens between Smith and Jones (exclusive) are separator/initial tokens
        between = tokens[smith_idx + 1: jones_idx]
        assert len(between) > 0
        assert all(
            annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_AUTHOR
            for t in between
        ), 'Gap tokens between Smith and Jones should all be REFERENCE_AUTHOR'

    def test_surname_fallback_labels_initial_mismatch(self):
        # When JATS given-names ('BE') do not match the PDF representation
        # ('B. F.'), the aligner should fall back to surname-only matching so
        # that at least the surname token is labeled REFERENCE_AUTHOR.  The gap
        # fill then extends to cover the initials between the two surnames.
        doc = _make_doc('MAIER, B. F.; BROCKMANN, D. Some title 2020')
        fvs = [
            _fv('MAIER, B. F.; BROCKMANN, D. Some title 2020', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Maier BE',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Maier',
            ),
            _fv('Brockmann D', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        maier_token = next(
            (t for t in tokens if normalize_for_alignment(t.text) == 'maier'), None
        )
        assert maier_token is not None
        assert annotated.get_token_sub_field(maier_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            'MAIER token should be REFERENCE_AUTHOR via surname fallback'
        )
        # Gap fill should also label the tokens between MAIER and BROCKMANN
        brockmann_idx = next(
            i for i, t in enumerate(tokens)
            if normalize_for_alignment(t.text) == 'brockmann'
        )
        maier_idx = next(
            i for i, t in enumerate(tokens)
            if normalize_for_alignment(t.text) == 'maier'
        )
        gap_tokens = tokens[maier_idx + 1: brockmann_idx]
        assert all(
            annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_AUTHOR
            for t in gap_tokens
        ), 'All tokens between MAIER and BROCKMANN should be REFERENCE_AUTHOR via gap fill'

    def test_mid_token_within_gap_block_not_labeled(self):
        # When SW matches a short block that starts mid-token (e.g. 't' inside 'staff'
        # when searching for author initial 'T'), that token must NOT be labeled.
        # Regression: GUARDIAN T matched second 'guardian' in 'guardian staff'
        # because 't' inside 'staff' was closer (gap 1) than the real 'T' (gap 2).
        doc = _make_doc('GUARDIAN, T. Guardian staff 2020')
        fvs = [
            _fv('GUARDIAN, T. Guardian staff 2020', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Guardian T',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Guardian',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        staff_token = next((t for t in tokens if t.text == 'staff'), None)
        assert staff_token is not None
        assert annotated.get_token_sub_field(staff_token) != JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"staff" token should not be labeled REFERENCE_AUTHOR (mid-word "t" matched by SW)'
        )
        # The correct first occurrence 'GUARDIAN' should be labeled
        guardian_upper = next((t for t in tokens if t.text == 'GUARDIAN'), None)
        assert guardian_upper is not None
        assert annotated.get_token_sub_field(guardian_upper) == (
            JatsSubFieldNames.REFERENCE_AUTHOR
        ), '"GUARDIAN" (first occurrence) should be labeled via surname fallback'

    def test_mid_token_fallback_extends_with_given_names(self):
        # After the mid-token fallback selects the surname-only earlier match,
        # the given-names initial should be found within the gap and also labeled.
        # Regression: GUARDIAN, T. — only "GUARDIAN" was labeled, not ", T."
        doc = _make_doc('GUARDIAN, T. Guardian staff 2020')
        fvs = [
            _fv('GUARDIAN, T. Guardian staff 2020', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Guardian T',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Guardian',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        t_token = next((t for t in tokens if t.text == 'T'), None)
        assert t_token is not None
        assert annotated.get_token_sub_field(t_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"T" initial should be labeled REFERENCE_AUTHOR after surname fallback extension'
        )

    def test_author_gap_fill_bridges_unlabeled_period_before_et_al(self):
        # When "Chowell G" and "et al." are separate needles, the "." between "G" and
        # "et al." has no label from SW (parent bibl text skips "et al."). The gap fill
        # must bridge it so all tokens become one <author> element.
        doc = _make_doc('CHOWELL, G. et al. Phenomenological models 2016')
        fvs = [
            _fv('CHOWELL G Phenomenological models 2016', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Chowell G',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Chowell',
            ),
            JatsFieldValue(
                text='et al.',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        period_after_g = tokens[3]  # 'CHOWELL'=0, ','=1, 'G'=2, '.'=3
        assert period_after_g.text == '.', f'Expected "." got {period_after_g.text!r}'
        sub = annotated.get_token_sub_field(period_after_g)
        assert sub == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"." after G should be labeled via gap fill'
        )

    def test_author_gap_fill_bridges_middle_initial_before_et_al(self):
        # "Vasconcelos GL" with "et al." as a separate needle — the ". L." between
        # "G" and "et al." in the PDF are not in the JATS author text and must be
        # bridged by gap fill so all tokens merge into one <author> element.
        doc = _make_doc('VASCONCELOS, G. L. et al. Modelling fatality 2020')
        fvs = [
            _fv('VASCONCELOS GL Modelling fatality 2020', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Vasconcelos GL',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Vasconcelos',
            ),
            JatsFieldValue(
                text='et al.',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        l_token = next((t for t in tokens if t.text == 'L'), None)
        assert l_token is not None
        assert annotated.get_token_sub_field(l_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"L" middle initial should be labeled REFERENCE_AUTHOR via gap fill'
        )

    def test_reference_label_does_not_block_year(self):
        # REFERENCE_LABEL "1" can match the "1" in "1987" if the citation number is
        # outside the parent bibl match range.  The masking that REFERENCE_LABEL adds
        # must not block REFERENCE_YEAR from labeling "1987".
        doc = _make_doc('Lewis RW Roberts PM Applied Scientific Research 1987')
        fvs = [
            _fv('1 Lewis RW Roberts PM Applied Scientific Research 1987',
                JatsFieldNames.REFERENCE),
            _fv('1', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('Applied Scientific Research', JatsFieldNames.REFERENCE,
                JatsSubFieldNames.REFERENCE_SOURCE),
            _fv('1987', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        year_token = next((t for t in tokens if t.text == '1987'), None)
        assert year_token is not None
        sub = annotated.get_token_sub_field(year_token)
        assert sub == JatsSubFieldNames.REFERENCE_YEAR, (
            f'"1987" should be reference-year, got {sub!r}'
        )

    def test_regular_fallback_extends_with_given_names(self):
        # When the full JATS name (e.g. "Hsieh Y-H") fails SW quality against the PDF
        # (e.g. "HSIEH, Y.-H.") and falls back to surname only ("Hsieh"), the
        # given-names initial from the original needle ("Y") should still be labeled.
        doc = _make_doc('HSIEH, Y.-H. Richards model 2009')
        fvs = [
            _fv('HSIEH, Y.-H. Richards model 2009', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Hsieh Y-H',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Hsieh',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        y_token = next((t for t in tokens if t.text == 'Y'), None)
        assert y_token is not None, 'Expected "Y" token in doc'
        sub = annotated.get_token_sub_field(y_token)
        assert sub == JatsSubFieldNames.REFERENCE_AUTHOR, (
            f'"Y" initial should be labeled REFERENCE_AUTHOR after fallback, got {sub!r}'
        )

    def test_bracket_label_tokens_are_labeled_reference_label(self):
        # Reference labels like "[1]" tokenize as three separate tokens "[", "1", "]".
        # The haystack has "[ 1 ]" with spaces; SW cannot match "[1]" contiguously.
        # _try_bracket_label_match strips the brackets, finds "1" via exact match,
        # then extends the range to include the adjacent bracket tokens.
        doc = _make_doc(
            '[ 1 ] Richards FA A flexible growth function 1959',
            '[ 2 ] Jones B Another title 2020',
        )
        fvs = [
            _fv('[1] Richards FA A flexible growth function 1959', JatsFieldNames.REFERENCE),
            _fv('[1]', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('Richards FA', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('1959', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
            _fv('[2] Jones B Another title 2020', JatsFieldNames.REFERENCE),
            _fv('[2]', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('Jones B', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('2020', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        label_tokens = [
            t for t in tokens
            if annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_LABEL
        ]
        label_texts = [t.text for t in label_tokens]
        assert label_texts == ['[', '1', ']', '[', '2', ']'], (
            f'Expected 6 bracket label tokens, got {label_texts!r}'
        )

    def test_bracket_label_tokens_labeled_when_parent_starts_at_closing_bracket(self):
        # The parent SW match for "[1] Richards..." often starts at "]" (position 4 in
        # "[ 1 ] ...") because the SW can't match "[1]" across spaces and skips to "]".
        # The REFERENCE_LABEL sub-field search must extend backward to find "[" and "1".
        doc = _make_doc('[ 1 ] Richards FA long title here Journal 1999')
        fvs = [
            _fv(
                '[1] Richards FA long title here Journal 1999',
                JatsFieldNames.REFERENCE,
            ),
            _fv('[1]', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('long title here', JatsFieldNames.REFERENCE,
                JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            _fv('Journal', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_SOURCE),
            _fv('1999', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        bracket_open = next((t for t in tokens if t.text == '['), None)
        assert bracket_open is not None
        assert annotated.get_token_sub_field(bracket_open) == JatsSubFieldNames.REFERENCE_LABEL, (
            '"[" should be labeled REFERENCE_LABEL, not left unlabeled'
        )
        num_token = next((t for t in tokens if t.text == '1'), None)
        assert num_token is not None
        assert annotated.get_token_sub_field(num_token) == JatsSubFieldNames.REFERENCE_LABEL, (
            '"1" should be labeled REFERENCE_LABEL'
        )

    def test_masked_sub_field_prevents_duplicate_match(self):
        # When the same author name appears twice in a reference, masking ensures
        # the second sub-field value matches the second occurrence rather than
        # re-matching the first (already-masked) position.
        doc = _make_doc('Jones A and Jones A 2020')
        fvs = [
            _fv('Jones A and Jones A 2020', JatsFieldNames.REFERENCE),
            _fv('Jones A', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Jones A', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        jones_tokens = [t for t in tokens if t.text == 'Jones']
        assert len(jones_tokens) == 2, f'Expected 2 "Jones" tokens, got {len(jones_tokens)}'
        assert all(
            annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_AUTHOR
            for t in jones_tokens
        ), 'Both "Jones" tokens should be labeled REFERENCE_AUTHOR'

    # ── Comprehensive reference sub-field tests ──────────────────────────────

    def _ref_fv(self, text: str, sub: str, fallback: str = '') -> JatsFieldValue:
        return JatsFieldValue(
            text=text,
            field_name=JatsFieldNames.REFERENCE,
            sub_field_name=sub,
            fallback_text=fallback or None,
        )

    def test_year_within_doi_not_relabeled_as_year(self):
        # When the year digits only appear inside the DOI span, REFERENCE_YEAR must
        # not claim them.  The DOI (longer needle) is processed first via longest-first
        # ordering, its range is masked, and REFERENCE_YEAR finds nothing.
        doc = _make_doc(
            'Smith A article title A 10.12345/test.2001.56'
        )
        doi = '10.12345/test.2001.56'
        fvs = [
            _fv('Smith A article title A ' + doi, JatsFieldNames.REFERENCE),
            self._ref_fv('Smith A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('article title A', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_YEAR),
            self._ref_fv(doi, JatsSubFieldNames.REFERENCE_DOI),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        sub_fields = [annotated.get_token_sub_field(t) for t in tokens]
        assert JatsSubFieldNames.REFERENCE_YEAR not in sub_fields, (
            'Year digits inside DOI should not be labeled as REFERENCE_YEAR'
        )
        doi_tokens = [
            t.text for t in tokens
            if annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_DOI
        ]
        assert doi_tokens, 'DOI should be labeled'

    def test_all_sub_fields_labeled_in_full_reference(self):
        # End-to-end: all standard sub-fields of a single reference are labeled.
        # Year appears standalone; DOI does not contain the year digits.
        doc = _make_doc(
            'Smith A 1999 article title A source A 11 7 101 10.12345/test.paper.56'
        )
        fvs = [
            _fv('Smith A 1999 article title A source A 11 7 101 10.12345/test.paper.56',
                JatsFieldNames.REFERENCE),
            self._ref_fv('Smith A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('article title A', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            self._ref_fv('source A', JatsSubFieldNames.REFERENCE_SOURCE),
            self._ref_fv('1999', JatsSubFieldNames.REFERENCE_YEAR),
            self._ref_fv('11', JatsSubFieldNames.REFERENCE_VOLUME),
            self._ref_fv('7', JatsSubFieldNames.REFERENCE_ISSUE),
            self._ref_fv('101', JatsSubFieldNames.REFERENCE_FPAGE),
            self._ref_fv('10.12345/test.paper.56', JatsSubFieldNames.REFERENCE_DOI),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        by_text = {t.text: annotated.get_token_sub_field(t) for t in tokens}
        assert by_text.get('Smith') == JatsSubFieldNames.REFERENCE_AUTHOR
        assert by_text.get('1999') == JatsSubFieldNames.REFERENCE_YEAR
        assert by_text.get('source') == JatsSubFieldNames.REFERENCE_SOURCE
        assert by_text.get('11') == JatsSubFieldNames.REFERENCE_VOLUME
        assert by_text.get('7') == JatsSubFieldNames.REFERENCE_ISSUE
        assert by_text.get('101') == JatsSubFieldNames.REFERENCE_FPAGE

    def test_page_number_not_matched_inside_year(self):
        # "200" (lpage) must match the exact token "200", not the "200" inside "2020".
        doc = _make_doc('Author A 181-200, acesso em 2020')
        fvs = [
            _fv('Author A 181-200, acesso em 2020', JatsFieldNames.REFERENCE),
            self._ref_fv('Author A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Author'),
            self._ref_fv('181', JatsSubFieldNames.REFERENCE_FPAGE),
            self._ref_fv('200', JatsSubFieldNames.REFERENCE_LPAGE),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        by_text = {t.text: annotated.get_token_sub_field(t) for t in tokens}
        assert by_text.get('200') == JatsSubFieldNames.REFERENCE_LPAGE
        assert by_text.get('2020') != JatsSubFieldNames.REFERENCE_LPAGE

    def test_exact_sw_match_preferred_over_earlier_gap_match(self):
        # Needle "2020b" gap-matches "2020" (quality 0.8) before the exact "2020b".
        # The exact contiguous match must win even though it appears later.
        doc = _make_doc('Author A 2020 title source 2020b')
        fvs = [
            _fv('Author A 2020 title source 2020b', JatsFieldNames.REFERENCE),
            self._ref_fv('Author A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Author'),
            self._ref_fv('2020b', JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        by_text = {t.text: annotated.get_token_sub_field(t) for t in tokens}
        assert by_text.get('2020b') == JatsSubFieldNames.REFERENCE_YEAR
        assert by_text.get('2020') != JatsSubFieldNames.REFERENCE_YEAR

    def test_multiple_authors_separated_by_comma_all_labeled(self):
        # Both "Smith, A" and "Johnson, B" merged into one <author> span.
        doc = _make_doc('Smith A, Johnson B 2001 article title A')
        fvs = [
            _fv('Smith A, Johnson B 2001 article title A', JatsFieldNames.REFERENCE),
            self._ref_fv('Smith A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('Johnson B', JatsSubFieldNames.REFERENCE_AUTHOR, 'Johnson'),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_YEAR),
            self._ref_fv('article title A', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        by_text = {t.text: annotated.get_token_sub_field(t) for t in tokens}
        assert by_text.get('Smith') == JatsSubFieldNames.REFERENCE_AUTHOR
        assert by_text.get('Johnson') == JatsSubFieldNames.REFERENCE_AUTHOR
        # year and title must not bleed into author span
        assert by_text.get('2001') == JatsSubFieldNames.REFERENCE_YEAR

    def test_dot_after_initial_included_in_author(self):
        # "Smith, A." — the trailing dot after "A" should be labeled as author.
        doc = _make_doc('Smith, A. 2001 article title A')
        fvs = [
            _fv('Smith, A. 2001 article title A', JatsFieldNames.REFERENCE),
            self._ref_fv('Smith A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_YEAR),
            self._ref_fv('article title A', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        dot_after_a = next(
            (t for i, t in enumerate(tokens)
             if t.text == '.' and i > 0 and tokens[i - 1].text == 'A'),
            None,
        )
        assert dot_after_a is not None, 'Expected "." after "A" token'
        sub = annotated.get_token_sub_field(dot_after_a)
        assert sub == JatsSubFieldNames.REFERENCE_AUTHOR, (
            f'"." after initial should be REFERENCE_AUTHOR, got {sub!r}'
        )

    def test_et_al_included_in_author_span(self):
        # "Smith A et al." — "et al." as separate needle merges with prior author.
        doc = _make_doc('Smith A et al. article title A 2001')
        fvs = [
            _fv('Smith A et al. article title A 2001', JatsFieldNames.REFERENCE),
            self._ref_fv('Smith A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('et al.', JatsSubFieldNames.REFERENCE_AUTHOR),
            self._ref_fv('article title A', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        et_token = next((t for t in tokens if t.text == 'et'), None)
        al_token = next((t for t in tokens if t.text == 'al'), None)
        assert et_token is not None and al_token is not None
        assert annotated.get_token_sub_field(et_token) == JatsSubFieldNames.REFERENCE_AUTHOR
        assert annotated.get_token_sub_field(al_token) == JatsSubFieldNames.REFERENCE_AUTHOR

    def test_multiple_references_no_cross_contamination(self):
        # Sub-fields of reference 1 must not bleed into reference 2's tokens.
        doc = _make_doc(
            '1 Smith A title one source one 2001',
            '2 Jones B title two source two 2002',
        )
        ref1_parent = '1 Smith A title one source one 2001'
        ref2_parent = '2 Jones B title two source two 2002'
        fvs = [
            _fv(ref1_parent, JatsFieldNames.REFERENCE),
            self._ref_fv('Smith A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('title one', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            self._ref_fv('source one', JatsSubFieldNames.REFERENCE_SOURCE),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_YEAR),
            _fv(ref2_parent, JatsFieldNames.REFERENCE),
            self._ref_fv('Jones B', JatsSubFieldNames.REFERENCE_AUTHOR, 'Jones'),
            self._ref_fv('title two', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            self._ref_fv('source two', JatsSubFieldNames.REFERENCE_SOURCE),
            self._ref_fv('2002', JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        by_text = {t.text: annotated.get_token_sub_field(t) for t in tokens}
        assert by_text.get('Smith') == JatsSubFieldNames.REFERENCE_AUTHOR
        assert by_text.get('Jones') == JatsSubFieldNames.REFERENCE_AUTHOR
        assert by_text.get('2001') == JatsSubFieldNames.REFERENCE_YEAR
        assert by_text.get('2002') == JatsSubFieldNames.REFERENCE_YEAR

    def test_label_with_same_digits_as_year_still_gets_year_labeled(self):
        # If citation label is "2001" (same as year), length-ordering ensures the
        # year sub-field wins its position (same length → original order preserved;
        # both appear in different positions in the doc so no conflict).
        doc = _make_doc('2001 Smith A title one 2001')
        fvs = [
            _fv('2001 Smith A title one 2001', JatsFieldNames.REFERENCE),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_LABEL),
            self._ref_fv('Smith A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('title one', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        year_tokens = [t for t in tokens if t.text == '2001']
        assert len(year_tokens) == 2
        # At least one of the two "2001" tokens should be labeled as reference-year
        sub_fields = [annotated.get_token_sub_field(t) for t in year_tokens]
        assert JatsSubFieldNames.REFERENCE_YEAR in sub_fields, (
            f'Expected reference-year in sub-fields, got {sub_fields}'
        )

    def test_varying_spaces_in_author_name_still_matched(self):
        # The old tool's test: "Smith ,J .A ." (PDF format with odd spacing) should
        # match JATS author "Smith J. A".
        doc = _make_doc('Smith ,J .A . 2001 article title A')
        fvs = [
            _fv('Smith ,J .A . 2001 article title A', JatsFieldNames.REFERENCE),
            self._ref_fv('Smith J. A', JatsSubFieldNames.REFERENCE_AUTHOR, 'Smith'),
            self._ref_fv('2001', JatsSubFieldNames.REFERENCE_YEAR),
            self._ref_fv('article title A', JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        smith = next((t for t in tokens if t.text == 'Smith'), None)
        assert smith is not None
        assert annotated.get_token_sub_field(smith) == JatsSubFieldNames.REFERENCE_AUTHOR

    def test_sub_field_longer_than_label_wins_position(self):
        # Structural regression guard: REFERENCE_YEAR (4 chars) must be processed before
        # REFERENCE_LABEL (1 char) regardless of their order in the JATS document.
        # Explicit ordering of fvs puts LABEL before YEAR to verify the sort is applied.
        doc = _make_doc('Smith A source A 1987')
        fvs = [
            _fv('1 Smith A source A 1987', JatsFieldNames.REFERENCE),
            self._ref_fv('1', JatsSubFieldNames.REFERENCE_LABEL),   # 1 char — before year
            self._ref_fv('source A', JatsSubFieldNames.REFERENCE_SOURCE),
            self._ref_fv('1987', JatsSubFieldNames.REFERENCE_YEAR),  # 4 chars
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        year_tok = next((t for t in tokens if t.text == '1987'), None)
        assert year_tok is not None
        assert annotated.get_token_sub_field(year_tok) == JatsSubFieldNames.REFERENCE_YEAR, (
            'REFERENCE_YEAR should win "1987" even when REFERENCE_LABEL is listed first in fvs'
        )

    def test_hyphenated_surname_labeled_when_jats_has_extra_given_name(self):
        # When JATS has an extra given name absent from the PDF, the SW produces
        # overlapping mid-token blocks; the within-gap guard must advance
        # prev_included_end through them to keep the hyphen within gap reach.
        doc = _make_doc('Braun, Lena Silva-Braun, Meier')
        fvs = [
            _fv('Braun, Lena Silva-Braun, Meier', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Braun Lena Kristina Silva-Braun',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Braun',
            ),
            JatsFieldValue(
                text='Meier',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Meier',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        author_tokens = [
            t.text for t in tokens
            if annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_AUTHOR
        ]
        assert '-' in author_tokens, (
            f'Hyphen in "Silva-Braun" must be REFERENCE_AUTHOR; got: {author_tokens}'
        )

    def test_pure_number_label_found_before_parent_match(self):
        # JATS strips the numeric label from the parent reference text, so the SW
        # match for the parent starts after the label token.  The pre-buffer extends the
        # sub-field search just far enough back to reach "1" before "Smith".
        doc = _make_doc('1. Smith J title here 2020')
        fvs = [
            _fv('Smith J title here 2020', JatsFieldNames.REFERENCE),
            _fv('1', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('Smith J', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('2020', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        label_token = next((t for t in tokens if t.text == '1'), None)
        assert label_token is not None, 'Expected "1" token in doc'
        sub = annotated.get_token_sub_field(label_token)
        assert sub == JatsSubFieldNames.REFERENCE_LABEL, (
            f'"1" should be REFERENCE_LABEL; got {sub!r}'
        )

    def test_suffixed_label_not_false_matched_in_preceding_doi(self):
        # Suffixed labels like "20-" are not pure numbers, so no pre-buffer is applied.
        # Without this guard, "20" inside the parent text would also match the "20" inside
        # a preceding DOI fragment like "2020-0248".
        doc = _make_doc(
            'Preceding ref doi 2020-0248',
            '20- Kato T study title Stress Health 2015',
        )
        fvs = [
            _fv('Preceding ref doi 2020-0248', JatsFieldNames.REFERENCE),
            _fv('20- Kato T study title Stress Health 2015', JatsFieldNames.REFERENCE),
            _fv('20-', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('Kato T', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        # "2020" from the first line must not be labeled as REFERENCE_LABEL
        doi_2020 = next((t for t in tokens if t.text == '2020'), None)
        assert doi_2020 is not None
        assert annotated.get_token_sub_field(doi_2020) != JatsSubFieldNames.REFERENCE_LABEL, (
            '"2020" in the preceding DOI must not be labeled REFERENCE_LABEL'
        )
        # "20" from the second line (the actual label start) must be labeled
        label_tokens = [
            t for t in tokens
            if annotated.get_token_sub_field(t) == JatsSubFieldNames.REFERENCE_LABEL
        ]
        label_texts = [t.text for t in label_tokens]
        assert '20' in label_texts, (
            f'"20" on line 2 should be REFERENCE_LABEL; got {label_texts!r}'
        )

    def test_pure_number_label_found_with_extra_text_before_parent(self):
        # When the JATS parent text begins at the publication name (e.g. "Emenda
        # Constitucional..."), the label and the publisher name ("Brasil.") both sit
        # before p_start — the gap can be ~14 chars.  The pre-buffer must cover it.
        doc = _make_doc('17. Brasil. Title about health 2020')
        fvs = [
            _fv('Title about health 2020', JatsFieldNames.REFERENCE),
            _fv('17', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('2020', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        label_token = next((t for t in tokens if t.text == '17'), None)
        assert label_token is not None, 'Expected "17" token in doc'
        sub = annotated.get_token_sub_field(label_token)
        assert sub == JatsSubFieldNames.REFERENCE_LABEL, (
            f'"17" should be REFERENCE_LABEL even when parent starts 14 chars later; got {sub!r}'
        )

    def test_single_digit_suffixed_label_found(self):
        # Labels like "1-" fail SW matching because the PDF tokeniser produces "1 - …"
        # (space between digit and dash), giving quality=0.5 < threshold.  The parent SW
        # match also skips "1", so p_start lands at "-" (position 2).  The small
        # digit-prefix pre-buffer and _try_numeric_prefix_label_match together fix this.
        doc = _make_doc(
            '1- Adebisi YA Sex workers should not be forgotten Am J Trop Med Hyg 2020',
            '2- Li Q Early transmission dynamics N Engl J Med 2020',
        )
        fvs = [
            _fv(
                '1- Adebisi YA Sex workers should not be forgotten Am J Trop Med Hyg 2020',
                JatsFieldNames.REFERENCE,
            ),
            _fv('1-', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('Adebisi YA', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('2020', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
            _fv(
                '2- Li Q Early transmission dynamics N Engl J Med 2020',
                JatsFieldNames.REFERENCE,
            ),
            _fv('2-', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv('Li Q', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('2020', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_YEAR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        ref1_label = next(t for t in tokens if t.text == '1')
        assert annotated.get_token_sub_field(ref1_label) == JatsSubFieldNames.REFERENCE_LABEL, (
            '"1" (label of first ref) should be REFERENCE_LABEL'
        )
        ref2_label = next(t for t in tokens if t.text == '2')
        assert annotated.get_token_sub_field(ref2_label) == JatsSubFieldNames.REFERENCE_LABEL, (
            '"2" (label of second ref) should be REFERENCE_LABEL'
        )

    def test_reference_author_with_dotted_initials_in_pdf_annotated(self):
        # JATS: "Smith AB" (initials without dots); PDF: "Smith, A. B." (dots after each initial)
        doc = _make_doc(
            'Smith, A. B. (2020). Some article title. Some Journal, 5, 7-9.',
        )
        fvs = [
            _fv(
                'Smith AB Some article title Some Journal 2020 5 7 9',
                JatsFieldNames.REFERENCE,
            ),
            _fv('Smith', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        smith_token = next(t for t in tokens if t.text == 'Smith')
        assert annotated.get_token_sub_field(smith_token) == JatsSubFieldNames.REFERENCE_AUTHOR

    def test_reference_author_initials_annotated_when_pdf_has_compacted_token(self):
        # JATS has "Jones AB" (two-letter initials run together); the PDF also renders
        # them as a single compacted token "AB" rather than dotted "A. B.".  The initials
        # token must be labeled even in compacted form — it must not be dropped because a
        # mid-token SW block triggered a surname-only fallback.
        doc = _make_doc(
            'Smith A, Jones AB. Some article title. Some Journal 2020.',
        )
        fvs = [
            _fv(
                'Smith A Jones AB Some article title Some Journal 2020',
                JatsFieldNames.REFERENCE,
            ),
            _fv('Smith A', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Jones AB', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        ab_token = next(t for t in tokens if t.text == 'AB')
        assert annotated.get_token_sub_field(ab_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"AB" is a compacted initials token; it must be labeled as REFERENCE_AUTHOR'
        )

    def test_reference_author_second_initial_labeled_when_primary_sw_drops_break_even_tail(self):
        # "Jones-Smith AB" is the only author; the PDF renders initials as dotted separate
        # tokens "Jones-Smith, A. B.".  SW matches the long surname (meeting the quality
        # threshold without the initials) and drops "B" because adding it would decrease
        # the alignment score.  The tail-bridging in _extend_match_for_needle_tail must
        # bridge the ". " gap to find "B" at its token-start position.
        doc = _make_doc(
            'Jones-Smith, A. B. Some article title. Some Journal 2020.',
        )
        fvs = [
            _fv(
                'Jones-Smith AB Some article title Some Journal 2020',
                JatsFieldNames.REFERENCE,
            ),
            JatsFieldValue(
                text='Jones-Smith AB',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Jones-Smith',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        b_token = next(t for t in tokens if t.text == 'B')
        assert annotated.get_token_sub_field(b_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"B" (second initial, after a ". " gap) must be labeled as REFERENCE_AUTHOR '
            'when SW tail bridging bridges the ". " gap without a fallback match'
        )

    def test_reference_author_given_name_initials_labeled_when_surname_fallback_used(self):
        # JATS has "Smíth AB" (accented í not in the PDF) which causes the primary match
        # quality to fall below the threshold, triggering the surname-only fallback.  The
        # PDF renders the initials as dotted separate tokens "A. B.".  The fallback tail
        # extension must bridge the ". " gap between the two initials so both "A" and "B"
        # are labeled as REFERENCE_AUTHOR — not just the first.
        doc = _make_doc(
            'Smith, A. B. Some article title. Some Journal 2020.',
        )
        fvs = [
            _fv(
                'Smíth AB Some article title Some Journal 2020',
                JatsFieldNames.REFERENCE,
            ),
            JatsFieldValue(
                text='Smíth AB',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Smíth',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        b_token = next(t for t in tokens if t.text == 'B')
        assert annotated.get_token_sub_field(b_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"B" (second initial, after a ". " gap) must be labeled as REFERENCE_AUTHOR '
            'even when the surname-only fallback is used'
        )

    def test_prior_reference_doi_subfield_end_advances_backward_search_floor(self):
        # The next reference's PDF text ("Smith XY 2002") contains no "Doe", so the
        # primary match fails and the 3-char fallback "Doe" is tried.  Without the
        # floor fix, "doe" in the prior reference's URL would be matched.
        doc = _make_doc(
            'Jones AB 2001. https://doe.org/10.1/abc',
            'Smith XY 2002.',
        )
        fvs = [
            _fv('Jones AB 2001', JatsFieldNames.REFERENCE),
            _fv('Jones AB', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('10.1/abc', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_DOI),
            _fv('Smith XY 2002', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Doe XY',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Doe',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        doe_url_token = next(t for t in tokens if t.text == 'doe')
        assert annotated.get_token_sub_field(doe_url_token) != JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"doe" in the prior reference URL must not be labeled as REFERENCE_AUTHOR; '
            'the DOI sub-field end should advance the floor past the URL'
        )

    def test_reference_author_labeled_when_prev_ref_sw_match_overlaps_current_ref_start(self):
        # Ref 1's JATS text ends with "Zorba" — the same word that starts ref 2.
        # The SW match for ref 1's parent consumes "Zorba" from the haystack, making
        # prev_parent_end land PAST ref 2's parent start.  Without the guard, the
        # backward-search floor is set beyond "Zorba" so the token is never labeled.
        doc = _make_doc(
            'Vreeken AB 2001 Source A',
            'Zorba BC 2002 Title',
        )
        fvs = [
            _fv('Vreeken AB 2001 Source A Zorba', JatsFieldNames.REFERENCE),
            _fv('Vreeken AB', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Zorba BC 2002 Title', JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Zorba BC',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text='Zorba',
            ),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        zorba_token = next(t for t in tokens if t.text == 'Zorba')
        assert annotated.get_token_sub_field(zorba_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"Zorba" is ref 2\'s author; the prev_parent_end guard must prevent the '
            'floor from being placed past the current reference\'s start position'
        )

    def test_prior_reference_et_al_not_labeled_by_next_reference_author_search(self):
        # Reference 1's PDF has "et al." but reference 1's JATS does not list it as
        # an author, so it should remain unlabeled.  Reference 2 follows with dotted
        # initials in the PDF ("Smith, B. C.") and "et al." in its JATS.  Reference
        # 2's author search must not reach back and label reference 1's "et al." tokens.
        doc = _make_doc(
            'Jones, A. et al. Title A Source A 2001',
            'Smith, B. C. Title B Source B 2002',
        )
        fvs = [
            _fv('Jones A et al. Title A Source A 2001', JatsFieldNames.REFERENCE),
            _fv('Jones A', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Smith BC Title B Source B 2002', JatsFieldNames.REFERENCE),
            _fv('Smith BC', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('et al.', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        et_token = next(t for t in tokens if t.text == 'et')
        al_token = next(t for t in tokens if t.text == 'al')
        assert annotated.get_token_sub_field(et_token) != JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"et al." belongs to reference 1 whose JATS has no "et al." author — '
            'it must not be labeled by reference 2\'s author search'
        )
        assert annotated.get_token_sub_field(al_token) != JatsSubFieldNames.REFERENCE_AUTHOR

    def test_reference_source_found_when_it_precedes_parent_match_start(self):
        # The JATS parent text for a reference concatenates sub-fields in JATS element
        # order (author → article-title → source).  The PDF text may instead have the
        # source immediately after the author and before the article-title.  SW then
        # latches onto the article-title anchor and produces a parent match start that
        # is AFTER the source in the haystack, causing the source sub-field search
        # (anchored at p_start) to miss the source.
        #
        # Reproduces the GEOCAPES pattern: institutional author whose name also starts
        # the source, appearing in the order author → source → article-title in the PDF.
        doc = _make_doc(
            # preceding reference (establishes pre_parent_ref_floor)
            'Zorblax A 2001 Title of Prev Article Source of Prev',
            # GEOCAPES-like reference: author → source → article-title in PDF text
            'Zorbax. Zorbax Research Group Source. Article Title Here.',
        )
        fvs = [
            # preceding reference
            _fv('Zorblax A 2001 Title of Prev Article Source of Prev',
                JatsFieldNames.REFERENCE),
            _fv('Zorblax A', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Source of Prev', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_SOURCE),
            # GEOCAPES-like reference: JATS order is author → article-title → source
            # but PDF text order is author → source → article-title
            _fv('Zorbax Article Title Here',  # parent: author + article-title (no source)
                JatsFieldNames.REFERENCE),
            _fv('Zorbax', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Article Title Here', JatsFieldNames.REFERENCE,
                JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
            _fv('Zorbax Research Group Source', JatsFieldNames.REFERENCE,
                JatsSubFieldNames.REFERENCE_SOURCE),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        source_token = next(t for t in tokens if t.text == 'Research')
        assert annotated.get_token_sub_field(source_token) == JatsSubFieldNames.REFERENCE_SOURCE, (
            '"Research" is part of the source that precedes the parent SW match start; '
            'the backward pre-buffer must allow the source search to reach it'
        )
        author_token = next(t for t in tokens if t.text == 'Zorbax')
        assert annotated.get_token_sub_field(author_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"Zorbax" is the author; REFERENCE_AUTHOR must retain its backward pre-buffer '
            'so the author is not displaced by the source pre-buffer fix'
        )

    def test_reference_author_found_when_parent_match_starts_after_author(self):
        # When the JATS parent text contains only the article-title (no author), the parent
        # SW match anchors at the title, making p_start land after the author in the haystack.
        # The REFERENCE_AUTHOR backward pre-buffer must extend the search back to reach it.
        # This reproduces the GEOCAPES regression where adding a REFERENCE_SOURCE pre-buffer
        # accidentally zeroed REFERENCE_AUTHOR's pre-buffer by removing its elif branch.
        doc = _make_doc('Zorblax Very Long Article Title Here End.')
        fvs = [
            _fv('Very Long Article Title Here End', JatsFieldNames.REFERENCE),
            _fv('Zorblax', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Very Long Article Title Here End', JatsFieldNames.REFERENCE,
                JatsSubFieldNames.REFERENCE_ARTICLE_TITLE),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        author_token = next(t for t in tokens if t.text == 'Zorblax')
        assert annotated.get_token_sub_field(author_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            'Author precedes the parent SW match start; the backward pre-buffer must allow '
            'the author search to reach back to it'
        )

    def test_reference_author_found_when_prev_ref_source_appears_later_in_haystack(self):
        # Simulates a two-column PDF layout: ref 1's source text physically appears in the
        # token stream AFTER ref 2's author tokens.  Before this fix, reference_floor
        # advanced past ref 2's author position when ref 1's source was matched, causing
        # the author search to miss ref 2's author.  The fix uses the previous reference's
        # parent-match end as the backward floor instead of the accumulated reference_floor.
        doc = _make_doc(
            'Jones A 2001',       # ref 1 parent (short, appears first in reading order)
            'Smith BC 2002',      # ref 2 parent + author (follows ref 1 in reading order)
            'source of ref 1',    # ref 1 source (appears last — two-column reading order)
        )
        fvs = [
            _fv('Jones A 2001', JatsFieldNames.REFERENCE),
            _fv('Jones A', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('source of ref 1', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_SOURCE),
            _fv('Smith BC 2002', JatsFieldNames.REFERENCE),
            _fv('Smith BC', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
        ]
        annotated = self._align(doc, fvs)
        tokens = list(doc.iter_all_tokens())
        smith_token = next(t for t in tokens if t.text == 'Smith')
        assert annotated.get_token_sub_field(smith_token) == JatsSubFieldNames.REFERENCE_AUTHOR, (
            '"Smith" is ref 2\'s first author; it must be annotated even though '
            'ref 1\'s source match appears after it in the token stream'
        )


class TestPostBodyRegionFloor:
    """An author response quotes the paper, so a body match can land in the peer review."""

    def _align(self, doc, field_values):
        return LayoutDocumentJatsAligner().align(doc, field_values)

    def test_sub_article_before_a_quoted_body_paragraph_still_matches(self):
        filler = ' '.join(f'filler{index}' for index in range(60))
        review_comment = 'Reviewer report one comment about the study design'
        quoted_paragraph = 'we changed the wording of the original claim about the outcome'
        doc = _make_doc(
            'Introduction',
            'A revised sentence about the outcome',
            'References',
            'Smith J 2020 A title Journal of Things 1 2',
            review_comment,
            filler,
            f'Response {quoted_paragraph}',
        )
        annotated = self._align(doc, [
            _fv(quoted_paragraph, JatsFieldNames.BODY_SECTION_PARAGRAPH),
            _fv('Smith J 2020 A title Journal of Things 1 2', JatsFieldNames.REFERENCE),
            _fv(review_comment, JatsFieldNames.SUB_ARTICLE),
        ])
        comment_tokens = list(doc.iter_all_lines())[4].tokens
        assert all(
            annotated.get_token_field(t) == JatsFieldNames.SUB_ARTICLE
            for t in comment_tokens
        )

    def test_an_exact_only_value_matches_verbatim_text(self):
        doc = _make_doc('Introduction', 'University Medical Centre Utrecht')
        annotated = self._align(doc, [
            JatsFieldValue(
                text='University Medical Centre Utrecht',
                field_name=JatsFieldNames.SUB_ARTICLE,
                exact_only=True,
            ),
        ])
        tokens = list(doc.iter_all_lines())[1].tokens
        assert all(
            annotated.get_token_field(t) == JatsFieldNames.SUB_ARTICLE for t in tokens
        )

    def test_an_exact_only_value_does_not_match_near_text(self):
        doc = _make_doc('Introduction', 'University Medical Center of Utrecht')
        annotated = self._align(doc, [
            JatsFieldValue(
                text='University Medical Centre Utrecht',
                field_name=JatsFieldNames.SUB_ARTICLE,
                exact_only=True,
            ),
        ])
        assert all(annotated.get_token_field(t) is None for t in doc.iter_all_tokens())


class TestReferenceSubFieldContainment:
    """A reference's sub-field belongs to that reference, whatever the anchors say."""

    ANCHOR_BEFORE = 'A unique opening sentence about autophagy in cultured plant cells'
    ANCHOR_AFTER = 'A unique closing sentence about the limitations of the present study'
    REFERENCE_ONE = (
        'Fleming PS, Koletsi D, Ioannidis JP: The first cited title.'
        ' J Clin Epidemiol. 2016; 78: 34-42.'
    )
    REFERENCE_TWO = (
        'Howick J, Koletsi D, Pandis N: The second cited title.'
        ' J Clin Epidemiol. 2020; 126: 154-159.'
    )

    def _align(self):
        doc = _make_doc(
            'Introduction',
            self.ANCHOR_BEFORE,
            'References',
            f'1. {self.REFERENCE_ONE}',
            f'2. {self.REFERENCE_TWO}',
            self.ANCHOR_AFTER,
        )
        field_values = [
            _fv(self.ANCHOR_BEFORE, JatsFieldNames.BODY_SECTION_PARAGRAPH),
            _fv(self.REFERENCE_ONE, JatsFieldNames.REFERENCE),
            _fv('Fleming PS', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Koletsi D', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('J Clin Epidemiol', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_SOURCE),
            _fv(self.REFERENCE_TWO, JatsFieldNames.REFERENCE),
            _fv('Howick J', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('Koletsi D', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_AUTHOR),
            _fv('J Clin Epidemiol', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_SOURCE),
            _fv(self.ANCHOR_AFTER, JatsFieldNames.BODY_SECTION_PARAGRAPH),
        ]
        return doc, LayoutDocumentJatsAligner().align(doc, field_values)

    def _reference_instances_of_line(self, doc, annotated, line_index: int):
        line = list(doc.iter_all_lines())[line_index]
        return {
            annotated.get_token_instance(token)
            for token in line.tokens
            if annotated.get_token_field(token) == JatsFieldNames.REFERENCE
        }

    def test_each_printed_reference_carries_one_instance(self):
        doc, annotated = self._align()
        first = self._reference_instances_of_line(doc, annotated, 3)
        second = self._reference_instances_of_line(doc, annotated, 4)
        assert len(first) == 1, (
            f'the first reference holds instances {sorted(first)};'
            ' a shared author or journal title has been matched into it'
        )
        assert len(second) == 1, (
            f'the second reference holds instances {sorted(second)};'
            ' a shared author or journal title has been matched into it'
        )
        assert first != second

    def test_the_shared_author_stays_in_its_own_reference(self):
        doc, annotated = self._align()
        lines = list(doc.iter_all_lines())
        shared = [
            (line_index, token)
            for line_index in (3, 4)
            for token in lines[line_index].tokens
            if token.text == 'Koletsi'
        ]
        assert len(shared) == 2
        assert {
            annotated.get_token_instance(token) for _, token in shared
        } == {1, 2}, 'both copies of the shared author were given to one reference'


class TestNarrowSearchRangeToWindow:
    def test_a_window_inside_the_range_narrows_it(self):
        assert narrow_search_range_to_window(100, 900, (200, 400)) == (200, 400)

    def test_a_window_reaching_outside_the_range_does_not_widen_it(self):
        assert narrow_search_range_to_window(100, 900, (0, 5000)) == (100, 900)

    def test_an_open_ended_range_takes_the_window_end(self):
        assert narrow_search_range_to_window(100, None, (0, 5000)) == (100, 5000)

    def test_a_window_that_leaves_nothing_keeps_the_range(self):
        assert narrow_search_range_to_window(100, 900, (0, 100)) == (100, 900)


class TestSearchStride:
    """How far the sliding search advances, which decides how often it reads a character."""

    def test_a_needle_that_fits_the_window_keeps_its_overlap(self):
        window_size = get_search_window_size(100, max_window=8000)
        assert get_search_stride(window_size, 100) == window_size - 100 - 20

    def test_a_needle_longer_than_the_window_still_advances_by_half_of_it(self):
        max_window = 8000
        window_size = get_search_window_size(max_window * 2, max_window=max_window)
        assert window_size == max_window
        assert get_search_stride(window_size, max_window * 2) == max_window // 2

    def test_the_stride_never_falls_below_half_a_window(self):
        max_window = 8000
        window_size = get_search_window_size(max_window, max_window=max_window)
        for needle_length in range(max_window - 100, max_window + 100):
            assert get_search_stride(window_size, needle_length) >= window_size // 2

    def test_a_value_longer_than_the_window_is_still_found(self):
        max_window = 2000
        sentences = [
            f'Sentence {index} of a reviewer report that runs on past the search window.'
            for index in range(40)
        ]
        review = ' '.join(sentences)
        assert len(normalize_for_alignment(review)) > max_window
        doc = _make_doc('Introduction', *sentences)
        annotated = LayoutDocumentJatsAligner(
            AlignmentConfig(max_window=max_window)
        ).align(doc, [_fv(review, JatsFieldNames.SUB_ARTICLE)])
        review_tokens = [
            token for line in list(doc.iter_all_lines())[1:] for token in line.tokens
        ]
        assert all(
            annotated.get_token_field(token) == JatsFieldNames.SUB_ARTICLE
            for token in review_tokens
        )


class TestReferenceLabelPosition:
    """A printed reference number, where the reference before it ends in digits."""

    FIRST = 'Smith J. A first paper on the subject. J Something. 2020; 5: 1-9.'
    FIRST_TAIL = 'DOI: https://doi.org/10.1016/j.example.2021.10.001'
    TENTH = 'Jones A. A tenth paper on another subject. Other J. 2021; 7: 20-30.'

    def _align(self):
        doc = _make_doc(
            'References',
            f'1. {self.FIRST}',
            self.FIRST_TAIL,
            f'10. {self.TENTH}',
        )
        field_values = [
            _fv(self.FIRST, JatsFieldNames.REFERENCE),
            _fv('1', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
            _fv(self.TENTH, JatsFieldNames.REFERENCE),
            _fv('10', JatsFieldNames.REFERENCE, JatsSubFieldNames.REFERENCE_LABEL),
        ]
        return doc, LayoutDocumentJatsAligner().align(doc, field_values)

    def _labelled_tokens(self, doc, annotated, line_index: int):
        line = list(doc.iter_all_lines())[line_index]
        return [
            token.text for token in line.tokens
            if annotated.get_token_sub_field(token) == JatsSubFieldNames.REFERENCE_LABEL
        ]

    def test_the_label_is_the_number_the_page_prints_before_the_reference(self):
        doc, annotated = self._align()
        assert self._labelled_tokens(doc, annotated, 3) == ['10'], (
            'the printed "10." opening the tenth reference should carry the label'
        )

    def test_a_digit_in_the_previous_reference_tail_is_not_the_label(self):
        doc, annotated = self._align()
        assert not self._labelled_tokens(doc, annotated, 2), (
            'the DOI ending "...2021.10.001" is the first reference\'s, '
            'and no part of it is the next reference\'s label'
        )


class TestReferenceStaysInTheList:
    """A reference is placed between the references either side of it.

    Consecutive references by the same authors share their opening words, so
    neither can anchor and both are steered by the window the anchors around
    them give.  Where that window replaces the reading order rather than
    narrowing it, the second reference is matched against the mention of the
    same work earlier in the document, its printed text is left unlabelled, and
    the element before it takes the whole line.
    """

    FIRST = 'Arcos R Gertrudix M Arribas C: survey instrument for evaluating effects. Zenodo. 2001'
    SECOND = (
        'Arcos R Gertrudix M Arribas C: checklist for responses to disinformation.'
        ' Zenodo. 2002'
    )
    THIRD = 'Bail C Guay B: assessing the impact of research agency accounts. Science. 2003'
    FILLER_ONE = (
        'Competing interests: no competing interests were disclosed by any of the'
        ' authors of this article, and the grant reference numbers are listed below'
    )
    FILLER_TWO = (
        'Grant information: this work was supported by a programme award whose'
        ' number appears in the acknowledgements, and the funder had no role here'
    )

    def _align(self):
        doc = _make_doc(
            'A paragraph of body text that is long enough to anchor on its own words',
            'Underlying data ' + self.SECOND,
            self.FILLER_ONE,
            self.FILLER_TWO,
            'References',
            self.FIRST,
            self.SECOND,
            self.THIRD,
        )
        field_values = [
            _fv(
                'A paragraph of body text that is long enough to anchor on its own words',
                JatsFieldNames.BODY_SECTION_PARAGRAPH,
            ),
            _fv(self.FIRST, JatsFieldNames.REFERENCE),
            _fv(self.SECOND, JatsFieldNames.REFERENCE),
            _fv(self.THIRD, JatsFieldNames.REFERENCE),
        ]
        return doc, LayoutDocumentJatsAligner().align(doc, field_values)

    def _instances_on_line(self, doc, annotated, line_index: int):
        line = list(doc.iter_all_lines())[line_index]
        return {
            annotated.get_token_instance(token)
            for token in line.tokens
            if annotated.get_token_field(token) == JatsFieldNames.REFERENCE
        }

    def test_each_printed_reference_carries_its_own_instance(self):
        doc, annotated = self._align()
        assert self._instances_on_line(doc, annotated, 5) == {1}
        assert self._instances_on_line(doc, annotated, 6) == {2}
        assert self._instances_on_line(doc, annotated, 7) == {3}

    def test_the_earlier_mention_is_not_labelled_as_the_reference(self):
        doc, annotated = self._align()
        assert not self._instances_on_line(doc, annotated, 1), (
            'the deposit mentioned in the body is not the reference the list prints'
        )


class TestPostBodyValueLeavesTheReferenceList:
    """A post-body value may not take tokens a reference already holds.

    Post-body values search from a floor that reaches back into the last
    reference and an exact one carries no length floor, so a word the peer
    review and the reference list share is matched inside the reference and
    overwrites the label it had been given.
    """

    REFERENCE = 'Margaros K, Aposkiti C: Living labs in law enforcement training. Zenodo. 2025'

    def _align(self):
        doc = _make_doc(
            'References',
            self.REFERENCE,
            'Reviewer report: the Zenodo deposit should be cited in full',
        )
        field_values = [
            _fv(self.REFERENCE, JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text='Zenodo.',
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_SOURCE,
            ),
            _fv('Zenodo', JatsFieldNames.SUB_ARTICLE),
        ]
        return doc, LayoutDocumentJatsAligner().align(doc, field_values)

    def _source_tokens(self, doc, annotated):
        return [
            token.text for token in doc.iter_all_tokens()
            if annotated.get_token_sub_field(token) == JatsSubFieldNames.REFERENCE_SOURCE
        ]

    def test_the_reference_keeps_its_source(self):
        doc, annotated = self._align()
        assert 'Zenodo' in self._source_tokens(doc, annotated), (
            'the review naming the same deposit does not take the source from the reference'
        )


def _paged_line(text: str, page_number: int, y: float) -> LayoutLine:
    return LayoutLine(tokens=[
        LayoutToken(text=word, coordinates=LayoutPageCoordinates(
            x=10, y=y, width=50, height=12, page_number=page_number
        ))
        for word in text.split()
    ])


def _paged_doc(pages: Sequence[Sequence[Tuple[str, float]]]) -> LayoutDocument:
    """One page per entry, each line given as (text, y) on a page 1000 high."""
    return LayoutDocument(pages=[
        LayoutPage(
            blocks=[LayoutBlock(lines=[
                _paged_line(text, page_number, y) for text, y in lines
            ])],
            meta=LayoutPageMeta(
                page_number=page_number,
                coordinates=LayoutPageCoordinates(
                    x=0, y=0, width=600, height=1000, page_number=page_number
                ),
            ),
        )
        for page_number, lines in enumerate(pages, 1)
    ])


class TestReferenceInterruptedByPageFurniture:
    """A running foot printed inside a reference does not cost it its title.

    Quality counts the page a match spans as well as the text it accounts for,
    and Smith-Waterman is a local alignment, so a foot printed between the two
    halves of a title costs a gap penalty per character and the best local
    answer is to keep the longer half and drop the other.  The foot repeats at
    the same place on every page, which is what says it is furniture rather
    than the reference's own text, and the haystack leaves it out.
    """

    FOOT = 'Page 17 of 24 Open Research Europe 2025 5:251 Last updated 13 Mar 2026'
    TITLE = (
        'Significance of Nonmetallic Inclusions for the clogging phenomenon in'
        ' continuous casting of steel - a review.'
    )
    REFERENCE = 'Michelic S, Bernhard C: ' + TITLE + ' Steel Res Int. 2022; 93(7): 2200086.'

    def _align(self):
        doc = _paged_doc([
            [
                ('A title block and an author list, which the first page carries', 300.0),
                (self.FOOT, 950.0),
            ],
            [
                ('Earlier reference text that fills the page before the one of interest', 300.0),
                ('34. Michelic S, Bernhard C: Significance of Nonmetallic'
                 ' Inclusions for the', 700.0),
                (self.FOOT, 950.0),
            ],
            [
                ('clogging phenomenon in continuous casting of steel - a review.'
                 ' Steel Res Int. 2022; 93(7): 2200086.', 300.0),
                (self.FOOT, 950.0),
            ],
        ])
        field_values = [
            _fv(self.REFERENCE, JatsFieldNames.REFERENCE),
            JatsFieldValue(
                text=self.TITLE,
                field_name=JatsFieldNames.REFERENCE,
                sub_field_name=JatsSubFieldNames.REFERENCE_ARTICLE_TITLE,
            ),
        ]
        return doc, LayoutDocumentJatsAligner().align(doc, field_values)

    def _title_tokens(self, doc, annotated):
        return [
            token.text for token in doc.iter_all_tokens()
            if annotated.get_token_sub_field(token)
            == JatsSubFieldNames.REFERENCE_ARTICLE_TITLE
        ]

    def test_both_halves_of_the_title_are_labelled(self):
        doc, annotated = self._align()
        labelled = self._title_tokens(doc, annotated)
        assert 'Significance' in labelled, 'the half before the page break'
        assert 'clogging' in labelled, 'the half after it'

    def test_the_running_foot_is_not_part_of_the_title(self):
        doc, annotated = self._align()
        labelled = self._title_tokens(doc, annotated)
        assert not {'Page', 'updated', '2026'} & set(labelled), (
            'the foot the title is printed around is not part of it'
        )


INTERRUPTING_BODY_TEXT = ' '.join(
    'experiences that protect rather than diminish user well being and privacy '
    'this is all the more important in light of the great number of questions '
    'raised by the possibility of the metaverse regarding the authenticity of '
    'digital identities personal data shared during use can be subject to misuse '
    'in the form of data theft and be re used for generating synthetic avatars'.split()
)


class TestExtendPartialReferenceMatches:
    """A reference printed around an interruption, looked for again in its region.

    The document reads the two halves with body text between them, so the first
    pass matches the better half and leaves the rest to no reference.  The region
    holds the reference's lines and not what interrupted them.
    """

    # The half below the interruption is the longer one, so it is the half the
    # first pass keeps -- which is the shape the defect takes on the page.
    FIRST_HALF = 'Gonzales AL Hancock JT effects of'
    SECOND_HALF = (
        'exposure to Facebook on self-esteem a journal of some length '
        'Cyberpsychol Behav Soc Netw volume 14 issue 1 pages 79 to 83 2011'
    )
    REFERENCE_TEXT = FIRST_HALF + ' ' + SECOND_HALF

    def _aligned(self):
        document = _make_doc(
            self.FIRST_HALF,
            # As long as the body text the page really prints between the halves:
            # a short gap the matcher can bridge does not reproduce the defect.
            INTERRUPTING_BODY_TEXT,
            self.SECOND_HALF,
        )
        annotated = LayoutDocumentJatsAligner().align(
            document,
            [_fv(self.REFERENCE_TEXT, JatsFieldNames.REFERENCE)],
        )
        return document, annotated

    def _reference_lines(self, document: LayoutDocument) -> Set[int]:
        lines = list(document.iter_all_lines())
        return {id(lines[0]), id(lines[2])}

    def test_should_leave_the_first_half_to_no_reference(self):
        document, annotated = self._aligned()
        first_line = list(document.iter_all_lines())[0]
        assert not any(
            annotated.get_token_field(token) == JatsFieldNames.REFERENCE
            for token in first_line.tokens
        )

    def test_should_place_the_first_half_when_looked_for_in_the_region(self):
        document, annotated = self._aligned()
        extended = extend_partial_reference_matches(
            annotated, document, self._reference_lines(document)
        )
        assert extended
        first_line = list(document.iter_all_lines())[0]
        assert all(
            annotated.get_token_field(token) == JatsFieldNames.REFERENCE
            for token in first_line.tokens
        )

    def test_should_leave_the_interrupting_text_alone(self):
        document, annotated = self._aligned()
        extend_partial_reference_matches(
            annotated, document, self._reference_lines(document)
        )
        interrupting_line = list(document.iter_all_lines())[1]
        assert not any(
            annotated.get_token_field(token) == JatsFieldNames.REFERENCE
            for token in interrupting_line.tokens
        )

    def test_should_do_nothing_where_the_reference_matched_whole(self):
        document = _make_doc(self.REFERENCE_TEXT)
        annotated = LayoutDocumentJatsAligner().align(
            document, [_fv(self.REFERENCE_TEXT, JatsFieldNames.REFERENCE)]
        )
        lines = list(document.iter_all_lines())
        assert not extend_partial_reference_matches(
            annotated, document, {id(lines[0])}
        )
