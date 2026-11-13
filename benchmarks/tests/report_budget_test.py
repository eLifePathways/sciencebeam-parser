from __future__ import annotations

from benchmarks.report_budget import dropped_corpora_note, fit


def _note(dropped):
    return dropped_corpora_note(dropped, "")


class TestFit:
    def test_keeps_everything_that_fits_and_says_nothing(self):
        text = fit(["head"], [("a", ["aaa"]), ("b", ["bbb"])], 1000, _note)
        assert text == "head\naaa\nbbb"

    def test_drops_whole_sections_from_the_end(self):
        text = fit(["head"], [("a", ["aaa"]), ("b", ["b" * 500])], 200, _note)
        assert "aaa" in text
        assert "b" * 500 not in text

    def test_names_what_it_dropped(self):
        text = fit(["head"], [("a", ["x" * 500]), ("b", ["y" * 500])], 200, _note)
        assert "a, b" in text

    def test_stays_within_the_budget(self):
        sections = [(str(i), ["z" * 300]) for i in range(20)]
        assert len(fit(["head"], sections, 900, _note)) <= 900

    def test_keeps_the_note_at_the_floor(self):
        # Nothing further to drop, so the note is kept rather than the reader being
        # left with a head that silently lost every section.
        text = fit(["head"], [("a", ["x" * 500])], 10, _note)
        assert text.startswith("head")
        assert "a" in text
        assert "not in this comment" in text


class TestDroppedCorporaNote:
    def test_reads_as_one_section(self):
        assert "1 corpus section is not" in dropped_corpora_note(["ore"], "")[0]

    def test_reads_as_several_sections(self):
        note = dropped_corpora_note(["ore", "pkp"], "")[0]
        assert "2 corpus sections are not" in note
        assert "ore, pkp" in note

    def test_names_where_the_full_report_is(self):
        note = dropped_corpora_note(["ore"], "in the [job summary](http://x/1)")[0]
        assert "The full report is in the [job summary](http://x/1)." in note

    def test_falls_back_to_the_artifact(self):
        assert "artifact" in dropped_corpora_note(["ore"], "")[0]
