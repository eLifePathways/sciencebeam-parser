from __future__ import annotations

from benchmarks.analyze_field_regressions._cases import _normalize_for_comparison


class TestNormalizeForComparison:
    def test_removes_spaces(self):
        assert _normalize_for_comparison('10.7554 / eLife.12345') == '10.7554/eLife.12345'

    def test_removes_newlines(self):
        assert _normalize_for_comparison('10.7554\n/eLife.12345') == '10.7554/eLife.12345'

    def test_no_whitespace_unchanged(self):
        assert _normalize_for_comparison('10.7554/eLife.12345') == '10.7554/eLife.12345'
