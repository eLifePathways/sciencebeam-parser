from __future__ import annotations

from benchmarks.analyze_field_regressions._loop import _resolve_concurrency


class TestResolveConcurrency:
    def test_nonzero_returned_as_is(self):
        assert _resolve_concurrency(8) == 8

    def test_zero_returns_at_least_two(self):
        result = _resolve_concurrency(0)
        assert result >= 2
