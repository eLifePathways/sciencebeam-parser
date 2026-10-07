from __future__ import annotations

from pathlib import Path

from benchmarks.training_selection import (
    read_selection,
    select_for_mode,
    write_selection,
)

SEED = 42
SPLIT = "train"
UPSTREAM = [f"doc{i}" for i in range(20)]


class TestSelectionFile:
    def test_round_trips_the_order(self, tmp_path: Path):
        write_selection(tmp_path, SPLIT, "ore", ["b", "a", "c"])
        assert read_selection(tmp_path, SPLIT, "ore") == ["b", "a", "c"]

    def test_a_corpus_with_nothing_recorded_yet_reads_as_empty(self, tmp_path: Path):
        assert read_selection(tmp_path, SPLIT, "ore") == []


class TestSelectForMode:
    def test_tops_an_empty_selection_up_to_the_mode(self):
        result = select_for_mode([], UPSTREAM, 5, SEED)
        assert len(result.present) == 5
        assert result.selection == result.present
        assert list(result.appended) == list(result.present)

    def test_is_deterministic(self):
        first = select_for_mode([], UPSTREAM, 5, SEED)
        second = select_for_mode([], UPSTREAM, 5, SEED)
        assert first.present == second.present

    def test_a_larger_mode_is_a_superset_of_a_smaller_one(self):
        small = select_for_mode([], UPSTREAM, 5, SEED)
        large = select_for_mode(small.selection, UPSTREAM, 10, SEED)
        assert set(small.present).issubset(set(large.present))
        assert list(large.selection[:5]) == list(small.selection)

    def test_raising_a_mode_appends_and_removes_nothing(self):
        recorded = select_for_mode([], UPSTREAM, 5, SEED).selection
        raised = select_for_mode(recorded, UPSTREAM, 8, SEED)
        assert list(raised.selection[:5]) == list(recorded)
        assert len(raised.appended) == 3

    def test_lowering_a_mode_takes_a_shorter_prefix_and_drops_no_id(self):
        """The recorded list is what never shrinks, not each mode's slice of it.

        A smaller mode generates fewer documents; none leaves the corpus, so
        raising it again names the same ones it did before.
        """
        recorded = select_for_mode([], UPSTREAM, 10, SEED).selection
        lowered = select_for_mode(recorded, UPSTREAM, 4, SEED)
        assert list(lowered.selection) == list(recorded)
        assert list(lowered.present) == list(recorded[:4])

        raised = select_for_mode(lowered.selection, UPSTREAM, 10, SEED)
        assert list(raised.present) == list(recorded)

    def test_a_grown_dataset_adds_nothing_to_a_mode_already_met(self):
        recorded = select_for_mode([], UPSTREAM, 5, SEED).selection
        grown = select_for_mode(recorded, UPSTREAM + ["newdoc1", "newdoc2"], 5, SEED)
        assert list(grown.selection) == list(recorded)
        assert not grown.appended

    def test_a_document_that_has_gone_keeps_its_place(self):
        """Dropping it would shift the one after it into the prefix and evict the last.

        That eviction is what the recorded selection exists to prevent, so the
        prefix keeps its length and the corpus keeps every other document.
        """
        recorded = select_for_mode([], UPSTREAM, 5, SEED).selection
        gone = recorded[1]
        remaining = [doc for doc in UPSTREAM if doc != gone]

        result = select_for_mode(recorded, remaining, 5, SEED)

        assert list(result.selection) == list(recorded)
        assert list(result.missing) == [gone]
        assert list(result.present) == [doc for doc in recorded if doc != gone]
        assert not result.appended

    def test_full_mode_takes_everything_and_keeps_the_recorded_order(self):
        recorded = select_for_mode([], UPSTREAM, 5, SEED).selection
        result = select_for_mode(recorded, UPSTREAM, None, SEED)
        assert list(result.selection[:5]) == list(recorded)
        assert len(result.present) == len(UPSTREAM)

    def test_a_mode_larger_than_the_corpus_takes_what_there_is(self):
        result = select_for_mode([], UPSTREAM, 500, SEED)
        assert len(result.present) == len(UPSTREAM)
