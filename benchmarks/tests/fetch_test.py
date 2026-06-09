from __future__ import annotations

from benchmarks.fetch import _sample_indices


class TestSampleIndices:
    def test_returns_correct_size(self):
        assert len(_sample_indices(100, 10, seed=42)) == 10

    def test_is_deterministic(self):
        assert _sample_indices(100, 10, seed=42) == _sample_indices(100, 10, seed=42)

    def test_different_seeds_give_different_results(self):
        assert _sample_indices(100, 10, seed=42) != _sample_indices(100, 10, seed=99)

    def test_smaller_is_subset_of_larger(self):
        s10 = _sample_indices(158, 10, seed=42)
        s25 = _sample_indices(158, 25, seed=42)
        s50 = _sample_indices(158, 50, seed=42)
        assert s10 <= s25
        assert s25 <= s50
        assert s10 <= s50

    def test_caps_at_n_total(self):
        assert len(_sample_indices(30, 100, seed=42)) == 30

    def test_indices_within_range(self):
        n_total = 50
        assert _sample_indices(n_total, 20, seed=42) <= set(range(n_total))
