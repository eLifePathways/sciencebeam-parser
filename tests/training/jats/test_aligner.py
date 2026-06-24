from typing import Optional

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
)
from sciencebeam_parser.training.jats.aligner import AlignmentConfig, LayoutDocumentJatsAligner
from sciencebeam_parser.training.jats.field_extractor import JatsFieldValue
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames, JatsSubFieldNames


def _make_doc(*line_texts: str) -> LayoutDocument:
    lines = [LayoutLine.for_text(t) for t in line_texts]
    return LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=lines)])])


def _fv(text: str, field: str = JatsFieldNames.BODY_SECTION_TITLE, sub: Optional[str] = None):
    return JatsFieldValue(text=text, field_name=field, sub_field_name=sub)


class TestLayoutDocumentJatsAligner:
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
