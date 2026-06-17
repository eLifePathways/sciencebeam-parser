from __future__ import annotations

from benchmarks.analyze_gold_failures._modes import assign_failure_modes, _find_best_raw_window
from benchmarks.analyze_gold_failures._types import FailureMode


RAW_TEXT = 'Introduction Methods Results Discussion Conclusion'
SB_FIELD = 'Introduction | Methods | Results'


class TestFindBestRawWindow:
    def test_exact_match_returns_true_and_1(self):
        in_raw, sim = _find_best_raw_window('methods', 'introduction methods results')
        assert in_raw is True
        assert sim == 1.0

    def test_absent_returns_false_and_low_sim(self):
        in_raw, sim = _find_best_raw_window('acknowledgements', 'introduction methods results')
        assert in_raw is False
        assert sim < 0.5

    def test_best_sim_is_high_for_near_match(self):
        # Gold has an extra word compared to the raw window — should still score high
        in_raw, sim = _find_best_raw_window(
            'isaudiovisualmethodbetterthantraditionalstudents',
            'isaudiovisualmethodthantraditionalstudents',
        )
        assert in_raw is False
        assert sim >= 0.8

    def test_short_value_requires_exact_match(self):
        # Values under 8 chars: fuzzy disabled, no sim computed
        in_raw, sim = _find_best_raw_window('abc', 'abx introduction methods')
        assert in_raw is False
        assert sim == 0.0


class TestAssignFailureModes:  # pylint: disable=too-many-public-methods
    def test_correct_when_value_extracted_and_similar(self):
        results = assign_failure_modes(['Introduction'], RAW_TEXT, SB_FIELD)
        assert len(results) == 1
        assert results[0].mode == FailureMode.CORRECT
        assert results[0].in_raw is True
        assert results[0].in_sb_field is True

    def test_extraction_failed_when_value_in_raw_but_not_in_field(self):
        results = assign_failure_modes(['Discussion'], RAW_TEXT, SB_FIELD)
        assert results[0].mode == FailureMode.EXTRACTION_FAILED
        assert results[0].in_raw is True
        assert results[0].in_sb_field is False

    def test_not_in_raw_when_value_absent_from_raw_text(self):
        results = assign_failure_modes(['Acknowledgements'], RAW_TEXT, SB_FIELD)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].in_raw is False

    def test_partial_wrong_when_value_found_but_low_similarity(self):
        # "Results" appears in "ResultsandDiscussion" after normalization of
        # "Results and Discussion", so in_sb_field=True, but edit-sim is low
        sb_with_long = 'Introduction | Results and Discussion'
        results = assign_failure_modes(
            ['Results'], RAW_TEXT, sb_with_long, similarity_threshold=0.8
        )
        assert results[0].mode == FailureMode.PARTIAL_WRONG
        assert results[0].in_sb_field is True
        assert results[0].best_sb_similarity is not None
        assert results[0].best_sb_similarity < 0.8

    def test_multiple_values_assigned_independently(self):
        gold = ['Introduction', 'Discussion', 'Acknowledgements']
        results = assign_failure_modes(gold, RAW_TEXT, SB_FIELD)
        assert len(results) == 3
        modes = {r.value: r.mode for r in results}
        assert modes['Introduction'] == FailureMode.CORRECT
        assert modes['Discussion'] == FailureMode.EXTRACTION_FAILED
        assert modes['Acknowledgements'] == FailureMode.NOT_IN_RAW_TEXT

    def test_empty_gold_values_returns_empty(self):
        results = assign_failure_modes([], RAW_TEXT, SB_FIELD)
        assert not results

    def test_none_raw_text_marks_all_not_in_raw(self):
        results = assign_failure_modes(['Introduction'], None, SB_FIELD)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT

    def test_none_sb_field_text_marks_extraction_failed_when_in_raw(self):
        results = assign_failure_modes(['Introduction'], RAW_TEXT, None)
        assert results[0].mode == FailureMode.EXTRACTION_FAILED

    def test_whitespace_normalization_matches_across_spacing(self):
        raw = 'Results  and\tDiscussion'
        sb = 'Results and Discussion'
        results = assign_failure_modes(['Results and Discussion'], raw, sb)
        assert results[0].mode == FailureMode.CORRECT

    def test_similarity_stored_on_result(self):
        results = assign_failure_modes(['Introduction'], RAW_TEXT, SB_FIELD)
        assert results[0].best_sb_similarity is not None
        assert results[0].best_sb_match is not None

    def test_curly_quote_matches_straight_quote_in_raw(self):
        # JATS gold uses RIGHT SINGLE QUOTATION MARK (U+2019); TEI uses ASCII apostrophe
        raw = "C/EBPβ's binding"   # plain apostrophe
        gold = "C/EBPβ’s binding"  # curly quote
        results = assign_failure_modes([gold], raw, None)
        assert results[0].in_raw is True
        assert results[0].mode == FailureMode.EXTRACTION_FAILED

    def test_html_entity_in_raw_matches_decoded_gold(self):
        # Raw TEI XML is not parsed so &amp; appears literally; gold is already decoded
        raw = 'CUT&amp;RUN validates our predictions'
        gold = 'CUT&RUN validates our predictions'
        results = assign_failure_modes([gold], raw, None)
        assert results[0].in_raw is True
        assert results[0].mode == FailureMode.EXTRACTION_FAILED

    def test_fuzzy_search_matches_single_char_substitution(self):
        # Residual single-char encoding difference not caught by normalization
        # (e.g. ä vs a — NFKC does not convert non-ASCII letters to ASCII)
        raw = 'Regulation of cell death by caspase activation'
        gold = 'Regulation of cell death by casp\xe4se activation'  # 'a' -> 'ä'
        results = assign_failure_modes([gold], raw, None)
        assert results[0].in_raw is True

    def test_fuzzy_search_requires_minimum_length(self):
        # Short strings (< 8 chars) must match exactly; fuzzy fallback disabled
        raw = 'Methods'
        gold = 'Methoxs'  # 2 chars differ in a 7-char string
        results = assign_failure_modes([gold], raw, None)
        assert results[0].in_raw is False

    def test_not_in_raw_shows_similar_extracted_value(self):
        # Gold title differs from the extracted title by one extra word and a
        # missing hyphen — the raw-text fuzzy check can't bridge that gap, so
        # the result is NOT_IN_RAW_TEXT.  But the extracted value should appear
        # as best_sb_match so the report can show what was actually extracted.
        raw = (
            'A Pan-respiratory Antiviral Chemotype Targeting '
            'a Transient Host Multiprotein Complex'
        )
        gold = (
            'A Pan-Respiratory Antiviral Chemotype Targeting '
            'a Host Multi-Protein Complex'
        )
        sb_field = (
            'A Pan-respiratory Antiviral Chemotype Targeting '
            'a Transient Host Multiprotein Complex'
        )
        results = assign_failure_modes([gold], raw, sb_field)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].best_sb_match == sb_field
        assert results[0].best_sb_similarity is not None
        assert results[0].best_sb_similarity >= 0.8

    def test_not_in_raw_stores_extracted_value_even_when_unrelated(self):
        # The extracted field value is always stored so the report can show what
        # was extracted.  The (low) similarity score tells the user it's unrelated.
        raw = 'Some completely different text on this page'
        gold = 'A Pan-Respiratory Antiviral Chemotype'
        sb_field = 'Some completely different title'
        results = assign_failure_modes([gold], raw, sb_field)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].best_sb_match == sb_field
        assert results[0].best_sb_similarity is not None
        assert results[0].best_sb_similarity < 0.5

    def test_not_in_raw_best_sb_match_is_none_when_nothing_extracted(self):
        # When sb_field is None (or empty), there's no extracted value to show.
        raw = 'Some text in the document'
        gold = 'A title never extracted'
        results = assign_failure_modes([gold], raw, None)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].best_sb_match is None

    def test_all_caps_extracted_is_correct_not_partial_wrong(self):
        # When the extracted value is ALL CAPS but gold is mixed-case, case-insensitive
        # similarity should be ~1.0, so the result is CORRECT not PARTIAL_WRONG.
        raw = 'ETHNO-MEDICO BOTANICAL STUDY AMONG THE FOUR INDIGENOUS COMMUNITIES'
        gold = 'Ethno-medico botanical study among the four indigenous communities'
        sb_field = 'ETHNO-MEDICO BOTANICAL STUDY AMONG THE FOUR INDIGENOUS COMMUNITIES'
        results = assign_failure_modes([gold], raw, sb_field)
        assert results[0].mode == FailureMode.CORRECT
        assert results[0].best_sb_similarity is not None
        assert results[0].best_sb_similarity >= 0.99

    def test_not_in_raw_populates_best_raw_similarity(self):
        # best_raw_similarity reflects how closely the gold matches the raw PDF text.
        # A value that is verbatim in the raw text scores 1.0; a truly absent value
        # scores near 0.
        raw = 'some completely unrelated document text about nothing relevant'
        gold = 'A title that does not appear anywhere in this document'
        results = assign_failure_modes([gold], raw, None)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].best_raw_similarity is not None
        assert results[0].best_raw_similarity < 0.5

    def test_not_in_raw_best_raw_similarity_high_for_near_match(self):
        # When gold has a small content difference from the PDF (e.g. an extra word),
        # best_raw_similarity should be high, indicating a gold/PDF form mismatch.
        raw = 'IS AUDIO VISUAL METHOD THAN TRADITIONAL FOR MEDICAL STUDENTS'
        gold = 'is audio visual method better than traditional for medical students'
        results = assign_failure_modes([gold], raw, None)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].best_raw_similarity is not None
        assert results[0].best_raw_similarity >= 0.8

    def test_all_caps_title_is_not_not_in_raw_text(self):
        # PDF rendering may produce ALL-CAPS titles.  Presence detection must be
        # case-insensitive so these are not falsely classified as NOT_IN_RAW_TEXT.
        raw = 'ETHNO-MEDICO BOTANICAL STUDY AMONG THE FOUR INDIGENOUS COMMUNITIES'
        gold = 'Ethno-medico botanical study among the four indigenous communities'
        sb_field = 'ETHNO-MEDICO BOTANICAL STUDY AMONG THE FOUR INDIGENOUS COMMUNITIES'
        results = assign_failure_modes([gold], raw, sb_field)
        assert results[0].in_raw is True
        assert results[0].mode != FailureMode.NOT_IN_RAW_TEXT

    def test_not_in_raw_high_raw_sim_low_extracted_sim(self):
        # Gold has an extra word absent from the PDF title, so in_raw=False.
        # best_raw_similarity should still be high (the bulk of the text matches),
        # while best_sb_similarity is low (model extracted something unrelated).
        # The split uses max(raw_sim, extr_sim) so the high raw_sim is sufficient
        # to place this in "Present in PDF as different form".
        raw = (
            'IS AUDIO VISUAL METHOD THAN TRADITIONAL FOR MEDICAL STUDENTS A SURVEY REPORT'
        )
        gold = (
            'is audio visual method better than traditional for medical students a survey report'
        )
        sb_field = 'Completely unrelated extracted text from a different section'
        results = assign_failure_modes([gold], raw, sb_field)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].best_raw_similarity is not None
        assert results[0].best_raw_similarity >= 0.6
        assert results[0].best_sb_similarity is not None
        assert results[0].best_sb_similarity < 0.6

    def test_not_in_raw_word_order_swap_high_extracted_sim(self):
        # Gold has subtitle first, PDF has it at the end (word-order swap).
        # The fixed-window raw similarity may be low, but the extracted title
        # has high similarity to the gold — evidence the content IS in the PDF.
        # The split must use max(raw_sim, extr_sim) so this lands in
        # "Present in PDF as different form" rather than "Absent from source PDF".
        pdf_title = (
            'Functional imaging of time on task and the involvement of '
            'dopaminergic and cholinergic substrates in cognitive effort and reward'
        )
        gold = (
            'Cognitive effort and reward. Functional imaging of time on task and '
            'the involvement of dopaminergic and cholinergic substrates'
        )
        results = assign_failure_modes([gold], pdf_title, pdf_title)
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        assert results[0].best_sb_similarity is not None
        # Extracted sim should be high enough to rescue this from "Absent"
        assert results[0].best_sb_similarity >= 0.6

    def test_not_in_raw_all_caps_extracted_has_high_similarity(self):
        # When gold is mixed-case and the extracted value is ALL CAPS, the
        # case-insensitive similarity must be high so the entry lands in
        # "Present in PDF as different form" not "Absent from source PDF".
        raw = (
            'IS AUDIO VISUAL METHOD BETTER THAN TRADITIONAL FOR MEDICAL STUDENTS?'
            ' - A SURVEY REPORT'
        )
        gold = (
            'Is Audio Visual Method Better than Traditional for Medical Students?'
            ' - A Better Survey Report'
        )
        sb_field = (
            'IS AUDIO VISUAL METHOD BETTER THAN TRADITIONAL FOR MEDICAL STUDENTS?'
            ' - A SURVEY REPORT'
        )
        results = assign_failure_modes([gold], raw, sb_field)
        # Gold has "A Better Survey Report" but TEI has "A SURVEY REPORT" — genuinely absent.
        assert results[0].mode == FailureMode.NOT_IN_RAW_TEXT
        # But the extracted title is clearly the same document title, so similarity must be high.
        assert results[0].best_sb_similarity is not None
        assert results[0].best_sb_similarity >= 0.6

    def test_inline_markup_in_raw_tei_is_found_in_raw(self):
        # Title text split across <hi> elements must still be found in raw text.
        # Without tag stripping the normalized string would have tag markup
        # between the text fragments, breaking substring search.
        raw = (
            '<title level="a" type="main">'
            '<hi rend="bold">Roles for the long non-coding RNA</hi>'
            ' <hi rend="bold"><hi rend="italic">Pax6os1/PAX6-AS1</hi></hi>'
            ' <hi rend="bold">in pancreatic beta cell identity and function</hi>'
            '</title>'
        )
        gold = (
            'Roles for the long non-coding RNA Pax6os1/PAX6-AS1'
            ' in pancreatic beta cell identity and function'
        )
        results = assign_failure_modes([gold], raw, None)
        assert results[0].in_raw is True
        assert results[0].mode == FailureMode.EXTRACTION_FAILED

    def test_case_difference_in_sb_field_is_correct_not_extraction_failed(self):
        # Gold JATS: "Synthesis of PAV-431 Resin" (capital R)
        # SB field output: "Synthesis of PAV-431 resin" (lowercase r from PDF text)
        # The single case difference must not cause EXTRACTION_FAILED — fuzzy
        # matching in the sb_field check should absorb it.
        raw = 'Synthesis of PAV-431 resin and other methods'
        sb_field = 'Synthesis of PAV-431 resin'
        gold = 'Synthesis of PAV-431 Resin'
        results = assign_failure_modes([gold], raw, sb_field)
        assert results[0].in_sb_field is True
        assert results[0].mode == FailureMode.CORRECT
