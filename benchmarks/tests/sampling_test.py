from __future__ import annotations

import numpy as np
import pytest

from benchmarks.sampling import (
    positional_ids,
    sample_indices,
    stratified_ids,
    stratified_order,
    stratum_order,
)


def _old_positional_selection(all_ids, n, seed):
    """The selection the single-file reader made before ids replaced positions.

    Kept as the reference for the guarantee that recorded numbers depend on: a
    corpus without strata must sample exactly as it used to.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_ids))
    rng.shuffle(indices)
    picked = set(int(i) for i in indices[:n])
    return [record_id for index, record_id in enumerate(all_ids) if index in picked]


class TestSampleIndices:
    def test_returns_correct_size(self):
        assert len(sample_indices(100, 10, seed=42)) == 10

    def test_is_deterministic(self):
        assert sample_indices(100, 10, seed=42) == sample_indices(100, 10, seed=42)

    def test_different_seeds_give_different_results(self):
        assert sample_indices(100, 10, seed=42) != sample_indices(100, 10, seed=99)

    def test_smaller_is_subset_of_larger(self):
        s10 = sample_indices(158, 10, seed=42)
        s25 = sample_indices(158, 25, seed=42)
        s50 = sample_indices(158, 50, seed=42)
        assert s10 <= s25
        assert s25 <= s50
        assert s10 <= s50

    def test_caps_at_n_total(self):
        assert len(sample_indices(30, 100, seed=42)) == 30

    def test_indices_within_range(self):
        n_total = 50
        assert sample_indices(n_total, 20, seed=42) <= set(range(n_total))

    def test_frozen_selection_for_a_known_size(self):
        # Pinned so that a change in numpy's shuffle cannot silently redefine
        # every number already recorded for the unstratified corpora.
        assert sorted(sample_indices(59, 10, seed=42)) == [
            7, 17, 18, 24, 27, 28, 39, 49, 54, 58,
        ]


class TestPositionalIds:
    def test_matches_the_selection_positions_used_to_make(self):
        all_ids = [f"id{i}" for i in range(59)]
        for n in (10, 25, 50):
            assert positional_ids(all_ids, n, seed=42) == _old_positional_selection(
                all_ids, n, seed=42
            )

    def test_keeps_corpus_order_so_a_multi_file_read_needs_no_ordering(self):
        all_ids = [f"id{i}" for i in range(20)]
        picked = positional_ids(all_ids, 5, seed=42)
        assert picked == [i for i in all_ids if i in set(picked)]

    def test_none_selects_everything(self):
        all_ids = [f"id{i}" for i in range(7)]
        assert positional_ids(all_ids, None, seed=42) == all_ids

    def test_rejects_duplicate_ids(self):
        with pytest.raises(ValueError, match="duplicate id"):
            positional_ids(["a", "b", "a"], 2, seed=42)


class TestStratumOrder:
    def test_is_deterministic_and_covers_every_stratum(self):
        strata = ["pone", "pbio", "pcsy"]
        assert stratum_order(strata, 42) == stratum_order(reversed(strata), 42)
        assert sorted(stratum_order(strata, 42)) == sorted(strata)

    def test_adding_a_stratum_leaves_the_others_in_order(self):
        before = stratum_order(["a", "b", "c"], 42)
        after = stratum_order(["a", "b", "c", "d"], 42)
        assert [s for s in after if s != "d"] == before


class TestStratifiedSampling:
    def _corpus(self):
        return {
            "big": [f"big{i}" for i in range(10)],
            "mid": [f"mid{i}" for i in range(5)],
            "tiny": ["tiny0", "tiny1"],
        }

    def test_a_sample_of_the_stratum_count_covers_every_stratum(self):
        picked = stratified_ids(self._corpus(), 3, seed=42)
        assert {record_id[:-1] for record_id in picked} == {"big", "mid", "tiny"}

    def test_draws_evenly_and_gives_the_remainder_to_the_first_served(self):
        picked = stratified_ids(self._corpus(), 4, seed=42)
        counts = {}
        for record_id in picked:
            counts[record_id[:-1]] = counts.get(record_id[:-1], 0) + 1
        assert sorted(counts.values()) == [1, 1, 2]
        first_served = stratum_order(self._corpus(), 42)[0]
        assert counts[first_served] == 2

    def test_takes_each_stratum_in_the_order_given(self):
        picked = stratified_ids(self._corpus(), 6, seed=42)
        assert [p for p in picked if p.startswith("big")] == ["big0", "big1"]

    def test_nests_per_stratum_as_the_size_grows(self):
        corpus = self._corpus()
        samples = {n: stratified_ids(corpus, n, seed=42) for n in (3, 6, 9, 12)}
        for smaller, larger in ((3, 6), (6, 9), (9, 12)):
            for stratum in corpus:
                in_smaller = [i for i in samples[smaller] if i.startswith(stratum)]
                in_larger = [i for i in samples[larger] if i.startswith(stratum)]
                assert in_larger[: len(in_smaller)] == in_smaller

    def test_an_exhausted_stratum_drops_out_and_the_rest_absorb_the_rest(self):
        corpus = self._corpus()
        picked = stratified_ids(corpus, 12, seed=42)
        assert len(picked) == 12
        assert len([i for i in picked if i.startswith("tiny")]) == 2

    def test_none_selects_the_whole_corpus(self):
        assert len(stratified_ids(self._corpus(), None, seed=42)) == 17

    def test_a_size_over_the_corpus_yields_the_corpus(self):
        assert len(stratified_ids(self._corpus(), 100, seed=42)) == 17

    def test_reports_when_a_size_cannot_cover_every_stratum(self, caplog):
        with caplog.at_level("WARNING"):
            picked = stratified_ids(self._corpus(), 2, seed=42)
        assert len(picked) == 2
        assert "smaller than the 3 strata" in caplog.text

    def test_order_is_stable_when_a_stratum_gains_documents(self):
        corpus = self._corpus()
        grown = {**corpus, "mid": corpus["mid"] + ["mid5", "mid6"]}
        before = stratified_order(corpus, 42)
        after = stratified_order(grown, 42)
        # The published rank order only ever appends, so a sample of a given size
        # picks the same documents after the corpus grows.
        assert after[:6] == before[:6]
