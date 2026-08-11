from __future__ import annotations

from typing import Optional

import pytest

from benchmarks.report import (
    _get_f1,
    _get_overall_f1,
    _parse_labeled_summary,
    _render_comparison_report,
    _unequal_docs_note,
)


def _agg(scoring_type: str, method: str, by_field: dict) -> dict:
    return {
        "scoring_type": scoring_type,
        "scoring_method": method,
        "summary_scores": {
            "by-field": {field: {"scores": {"f1": f1}} for field, f1 in by_field.items()}
        },
    }


def _summary(fields, field_measures, field_scoring_types, corpora) -> dict:
    return {
        "fields": fields,
        "field_measures": field_measures,
        "field_scoring_types": field_scoring_types,
        "corpora": corpora,
    }


def _corpus(aggregated: list, n: int = 10) -> dict:
    return {"biorxiv": {"n": n, "aggregated": aggregated}}


def _multi_corpus(names: list, aggregated: list, n: int = 10) -> dict:
    return {name: {"n": n, "aggregated": aggregated} for name in names}


def _title_summary(f1: float, method: str = "levenshtein", corpora: Optional[dict] = None) -> dict:
    return _summary(
        fields=["title"],
        field_measures={"title": [method]},
        field_scoring_types={"title": "string"},
        corpora=(
            corpora if corpora is not None
            else _corpus([_agg("string", method, {"title": f1})])
        ),
    )


class TestGetF1:
    def test_returns_f1_for_present_field(self):
        s = _title_summary(0.85)
        assert _get_f1(s, "biorxiv", "title", "levenshtein") == pytest.approx(0.85)

    def test_returns_none_when_field_absent_from_by_field(self):
        s = _summary(
            fields=["title"],
            field_measures={"title": ["levenshtein"]},
            field_scoring_types={"title": "string"},
            corpora=_corpus([_agg("string", "levenshtein", {})]),
        )
        assert _get_f1(s, "biorxiv", "title", "levenshtein") is None

    def test_returns_none_for_missing_corpus(self):
        s = _title_summary(0.85)
        assert _get_f1(s, "missing", "title", "levenshtein") is None

    def test_returns_none_when_scoring_type_mismatches_field(self):
        s = _summary(
            fields=["authors"],
            field_measures={"authors": ["levenshtein"]},
            field_scoring_types={"authors": "ulist"},
            corpora=_corpus([_agg("string", "levenshtein", {"authors": 0.7})]),
        )
        assert _get_f1(s, "biorxiv", "authors", "levenshtein") is None

    def test_returns_none_for_wrong_method(self):
        s = _title_summary(0.85, method="levenshtein")
        assert _get_f1(s, "biorxiv", "title", "exact") is None


class TestGetOverallF1:
    def test_single_corpus_matches_corpus_f1(self):
        s = _title_summary(0.85)
        assert _get_overall_f1(s, "title", "levenshtein") == pytest.approx(0.85)

    def test_weighted_mean_across_corpora(self):
        # biorxiv: n=10, f1=0.8 → weight 8.0
        # ore:     n=20, f1=0.6 → weight 12.0
        # overall: 20/30 ≈ 0.667
        s = _summary(
            fields=["title"],
            field_measures={"title": ["levenshtein"]},
            field_scoring_types={"title": "string"},
            corpora={
                "biorxiv": {"n": 10, "aggregated": [_agg("string", "levenshtein", {"title": 0.8})]},
                "ore": {"n": 20, "aggregated": [_agg("string", "levenshtein", {"title": 0.6})]},
            },
        )
        assert _get_overall_f1(s, "title", "levenshtein") == pytest.approx(20 / 30, abs=1e-6)

    def test_skips_corpus_with_missing_f1(self):
        s = _summary(
            fields=["title"],
            field_measures={"title": ["levenshtein"]},
            field_scoring_types={"title": "string"},
            corpora={
                "biorxiv": {"n": 10, "aggregated": [_agg("string", "levenshtein", {"title": 0.8})]},
                "ore": {"n": 20, "aggregated": [_agg("string", "levenshtein", {})]},
            },
        )
        assert _get_overall_f1(s, "title", "levenshtein") == pytest.approx(0.8)

    def test_returns_none_when_no_corpora_have_f1(self):
        s = _summary(
            fields=["title"],
            field_measures={"title": ["levenshtein"]},
            field_scoring_types={"title": "string"},
            corpora={
                "biorxiv": {"n": 10, "aggregated": [_agg("string", "levenshtein", {})]},
            },
        )
        assert _get_overall_f1(s, "title", "levenshtein") is None


class TestParseLabeledSummary:
    def test_simple_label_and_path(self):
        label, path = _parse_labeled_summary("GROBID=runs/grobid/summary.json")
        assert label == "GROBID"
        assert str(path) == "runs/grobid/summary.json"

    def test_label_with_colons(self):
        label, path = _parse_labeled_summary(
            "sciencebeam-parser:pr-42-abc1234=runs/sb/summary.json"
        )
        assert label == "sciencebeam-parser:pr-42-abc1234"
        assert str(path) == "runs/sb/summary.json"

    def test_label_with_docker_image_tag(self):
        label, path = _parse_labeled_summary(
            "sciencebeam-parser:pr-610-a1a2927c-20260526.1548=benchmarks/runs/baseline/summary.json"
        )
        assert label == "sciencebeam-parser:pr-610-a1a2927c-20260526.1548"
        assert str(path) == "benchmarks/runs/baseline/summary.json"


class TestRenderComparisonReport:  # pylint: disable=too-many-public-methods
    def _two(self, grobid_f1=0.820, sb_f1=0.852):
        return [
            ("GROBID 0.9.0-crf", _title_summary(grobid_f1)),
            ("ScienceBeam (PR)", _title_summary(sb_f1)),
        ]

    def _two_multi_corpus(self, grobid_f1=0.820, sb_f1=0.852):
        agg_g = [_agg("string", "levenshtein", {"title": grobid_f1})]
        agg_s = [_agg("string", "levenshtein", {"title": sb_f1})]
        return [
            ("GROBID", _title_summary(grobid_f1, corpora=_multi_corpus(["biorxiv", "ore"], agg_g))),
            ("SB (PR)", _title_summary(sb_f1, corpora=_multi_corpus(["biorxiv", "ore"], agg_s))),
        ]

    def test_returns_empty_string_for_empty_input(self):
        assert _render_comparison_report([]) == ""

    def test_includes_standard_header(self):
        report = _render_comparison_report(self._two())
        assert "## ScienceBeam Parser Evaluation" in report

    def _header_line(self, report: str) -> str:
        return next(line for line in report.splitlines() if "Field (method)" in line)

    def test_includes_tool_labels_in_table_header(self):
        header = self._header_line(_render_comparison_report(self._two()))
        assert "GROBID 0.9.0-crf" in header
        assert "ScienceBeam (PR)" in header

    def test_delta_column_for_non_primary_label(self):
        header = self._header_line(_render_comparison_report(self._two()))
        assert "Δ GROBID 0.9.0-crf" in header

    def test_no_delta_column_for_primary_label(self):
        header = self._header_line(_render_comparison_report(self._two()))
        assert "Δ ScienceBeam (PR)" not in header

    def test_renders_positive_delta(self):
        report = _render_comparison_report(self._two(grobid_f1=0.820, sb_f1=0.852))
        assert "+0.032" in report

    def test_renders_negative_delta(self):
        report = _render_comparison_report(self._two(grobid_f1=0.860, sb_f1=0.852))
        assert "-0.008" in report

    def test_renders_f1_values(self):
        report = _render_comparison_report(self._two(grobid_f1=0.820, sb_f1=0.852))
        assert "0.820" in report
        assert "0.852" in report

    def test_field_and_method_combined_in_row_key(self):
        report = _render_comparison_report(self._two())
        assert "title (levenshtein)" in report

    def test_type_shown_in_row(self):
        report = _render_comparison_report(self._two())
        title_row = next(line for line in report.splitlines() if "title (levenshtein)" in line)
        assert "string" in title_row

    def test_renders_dash_when_other_f1_absent(self):
        other = _summary(
            fields=["title"],
            field_measures={"title": ["levenshtein"]},
            field_scoring_types={"title": "string"},
            corpora=_corpus([_agg("string", "levenshtein", {})]),
        )
        report = _render_comparison_report([("GROBID", other), ("SB (PR)", _title_summary(0.85))])
        title_row = next(line for line in report.splitlines() if "title (levenshtein)" in line)
        assert title_row.count("— |") == 2  # GROBID F1 and delta both "—"

    def test_renders_dash_when_primary_f1_absent(self):
        primary = _summary(
            fields=["title"],
            field_measures={"title": ["levenshtein"]},
            field_scoring_types={"title": "string"},
            corpora=_corpus([_agg("string", "levenshtein", {})]),
        )
        report = _render_comparison_report([("GROBID", _title_summary(0.82)), ("SB (PR)", primary)])
        title_row = next(line for line in report.splitlines() if "title (levenshtein)" in line)
        assert title_row.count("— |") == 2  # primary F1 and delta both "—"

    def test_doc_counts_shown_for_each_tool(self):
        report = _render_comparison_report(self._two())
        assert "10 docs" in report
        assert "GROBID 0.9.0-crf" in report

    def test_three_summaries_show_two_delta_columns(self):
        summaries = [
            ("GROBID", _title_summary(0.820)),
            ("SB 0.1.x", _title_summary(0.847)),
            ("SB (PR)", _title_summary(0.852)),
        ]
        report = _render_comparison_report(summaries)
        header = next(line for line in report.splitlines() if "Field (method)" in line)
        assert "Δ GROBID" in header
        assert "Δ SB 0.1.x" in header

    def test_single_summary_renders_without_delta_columns(self):
        report = _render_comparison_report([("SB (PR)", _title_summary(0.852))])
        header = next(line for line in report.splitlines() if "Field (method)" in line)
        assert "Δ" not in header
        assert "0.852" in report

    def test_corpus_section_is_collapsible(self):
        report = _render_comparison_report(self._two())
        assert "<details>" in report
        assert "<summary><b>biorxiv</b>" in report
        assert "</details>" in report

    def test_no_overall_section_for_single_corpus(self):
        report = _render_comparison_report(self._two())
        assert "### Overall" not in report

    def test_overall_section_present_for_multiple_corpora(self):
        report = _render_comparison_report(self._two_multi_corpus())
        assert "### Overall" in report
        assert "2 corpora" in report

    def test_overall_section_shows_total_doc_count(self):
        report = _render_comparison_report(self._two_multi_corpus())
        # 2 corpora × 10 docs each = 20 total
        assert "20 docs" in report

    def test_overall_section_weighted_f1(self):
        # both corpora same n=10 → simple mean
        report = _render_comparison_report(self._two_multi_corpus(grobid_f1=0.800, sb_f1=0.840))
        assert "0.840" in report  # sb overall F1
        assert "0.800" in report  # grobid overall F1

    def test_each_corpus_has_collapsible_section(self):
        report = _render_comparison_report(self._two_multi_corpus())
        assert report.count("<details>") == 2
        assert "<summary><b>biorxiv</b>" in report
        assert "<summary><b>ore</b>" in report


class TestUnequalDocsNote:
    def test_silent_when_every_run_covered_the_same_documents(self):
        assert _unequal_docs_note([("grobid", 10), ("local", 10)]) == []

    def test_calls_out_a_difference_with_the_counts(self):
        note = _unequal_docs_note([("grobid", 10), ("local", 14)])
        assert note and "Unequal document sets" in note[0]
        assert "grobid 10" in note[0] and "local 14" in note[0]

    def test_a_corpus_absent_from_one_run_counts_as_unequal(self):
        assert _unequal_docs_note([("grobid", 0), ("local", 14)])
