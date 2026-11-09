from benchmarks.report_legend import column_keys, render_legend


class TestColumnKeys:
    def test_takes_the_first_letter_of_a_single_name(self):
        assert column_keys(["wapiti", "current"]) == ["W", "C"]

    def test_takes_up_to_three_initials(self):
        assert column_keys(["ScienceBeam all retrained extra", "Agentic LLM"]) == [
            "SAR", "AL",
        ]

    def test_splits_words_on_punctuation(self):
        assert column_keys(["grobid_crf", "llm_all"]) == ["GC", "LA"]

    def test_skips_versions_hashes_and_timestamps(self):
        assert column_keys([
            "grobid 0.9.1-crf (default)",
            "sciencebeam-parser:pr-802-f98179dd-20261109.1449 (delft_hub)",
        ]) == ["G", "SPP"]

    def test_leaves_the_profile_out_where_the_names_differ(self):
        assert column_keys(["GROBID 0.9.1 (non-DL)", "ScienceBeam (non-DL)"]) == ["G", "S"]

    def test_adds_the_profile_where_the_names_are_alike(self):
        assert column_keys([
            "sciencebeam-parser:main-4b436dd1-20261109.1013 (grobid_crf)",
            "sciencebeam-parser:manual-9d0a62fa-20261109.0959 on main (delft_hub)",
        ]) == ["SPM+GC", "SPM+DH"]

    def test_numbers_any_still_alike_in_declared_order(self):
        assert column_keys(["SB (PR)", "SB (PR)", "SB 0.1.x"]) == ["S+P", "S+P2", "S"]

    def test_falls_back_to_the_position_where_a_label_has_no_letters(self):
        assert column_keys(["0.1", "grobid"]) == ["V1", "G"]


class TestRenderLegend:
    def test_names_each_key_and_marks_the_reference(self):
        lines = render_legend(["grobid", "current"], primary=-1)
        assert "| G | grobid |  |" in lines
        assert "| C | current | deltas are measured against this |" in lines

    def test_marks_nothing_for_a_single_column(self):
        assert "deltas" not in "\n".join(render_legend(["current"], primary=-1))

    def test_says_what_each_is_where_described(self):
        lines = render_legend(["grobid"], primary=0, described=[("grobid", "`grobid` 0.9")])
        assert "| G | grobid | `grobid` 0.9 |  |" in lines
