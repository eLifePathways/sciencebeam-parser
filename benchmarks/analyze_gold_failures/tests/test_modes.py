from __future__ import annotations

from benchmarks.analyze_gold_failures._modes import assign_failure_modes
from benchmarks.analyze_gold_failures._types import FailureMode


RAW_TEXT = 'Introduction Methods Results Discussion Conclusion'
SB_FIELD = 'Introduction | Methods | Results'


class TestAssignFailureModes:
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
