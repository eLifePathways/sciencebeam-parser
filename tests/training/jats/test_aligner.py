# pylint: disable=too-many-lines
from typing import Optional

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
)
from sciencebeam_parser.training.jats.aligner import AlignmentConfig, LayoutDocumentJatsAligner
from sciencebeam_parser.training.jats.text_normalizer import normalize_for_alignment
from sciencebeam_parser.training.jats.field_extractor import JatsFieldValue
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames, JatsSubFieldNames


def _make_doc(*line_texts: str) -> LayoutDocument:
    lines = [LayoutLine.for_text(t) for t in line_texts]
    return LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=lines)])])


def _fv(text: str, field: str = JatsFieldNames.BODY_SECTION_TITLE, sub: Optional[str] = None):
    return JatsFieldValue(text=text, field_name=field, sub_field_name=sub)


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
