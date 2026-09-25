from __future__ import annotations

from typing import Optional

import pytest

from benchmarks.report import (
    _common_corpora,
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
        # pylint: disable=use-implicit-booleaness-not-comparison
        # the empty list is the contract; `not ...` would also accept None
        assert _unequal_docs_note([("grobid", 10), ("local", 10)]) == []

    def test_calls_out_a_difference_with_the_counts(self):
        note = _unequal_docs_note([("grobid", 10), ("local", 14)])
        assert note and "Unequal document sets" in note[0]
        assert "grobid 10" in note[0] and "local 14" in note[0]

    def test_a_corpus_absent_from_one_run_counts_as_unequal(self):
        assert _unequal_docs_note([("grobid", 0), ("local", 14)])


class TestCommonCorpora:
    def _summaries(self, *per_run):
        return [(f"run{i}", {"corpora": {c: {"n": n} for c, n in run.items()}})
                for i, run in enumerate(per_run)]

    def test_keeps_corpora_every_run_scored(self):
        summaries = self._summaries({"a": 10, "b": 5}, {"a": 10, "b": 5})
        assert _common_corpora(summaries, ["a", "b"]) == ["a", "b"]

    def test_drops_a_corpus_one_run_did_not_score(self):
        summaries = self._summaries({"a": 10, "b": 0}, {"a": 10, "b": 14})
        assert _common_corpora(summaries, ["a", "b"]) == ["a"]

    def test_drops_a_corpus_absent_from_one_run(self):
        summaries = self._summaries({"a": 10}, {"a": 10, "b": 14})
        assert _common_corpora(summaries, ["a", "b"]) == ["a"]

    def test_preserves_the_given_order(self):
        summaries = self._summaries({"b": 1, "a": 1}, {"b": 1, "a": 1})
        assert _common_corpora(summaries, ["b", "a"]) == ["b", "a"]


class TestOverallRestrictedToCommonCorpora:
    def _summary(self, per_corpus):
        return {
            "fields": ["title"],
            "field_measures": {"title": ["levenshtein"]},
            "field_scoring_types": {"title": "string"},
            "corpora": {
                corpus: {
                    "n": n,
                    "aggregated": [{
                        "scoring_type": "string", "scoring_method": "levenshtein",
                        "summary_scores": {"by-field": {"title": {"scores": {"f1": f1}}}},
                    }],
                }
                for corpus, (n, f1) in per_corpus.items()
            },
        }

    def test_overall_ignores_a_corpus_only_one_run_scored(self):
        # `b` would drag the primary's overall down if it were counted, since the
        # baseline has nothing to compare against there.
        baseline = self._summary({"a": (10, 0.8), "b": (0, 0.0)})
        primary = self._summary({"a": (10, 0.9), "b": (10, 0.1)})
        assert _get_overall_f1(primary, "title", "levenshtein") == pytest.approx(0.5)
        assert _get_overall_f1(primary, "title", "levenshtein", ["a"]) == pytest.approx(0.9)
        report = _render_comparison_report([("base", baseline), ("local", primary)])
        assert "Overall (10 docs across 1 corpora)" in report
        assert "Excludes b, which not every run scored" in report
        # The unequal-columns warning belongs to b's own section, not the overall row.
        overall = report.split("<details>", maxsplit=1)[0]
        assert "Unequal document sets" not in overall


def _usage(
    calls: int = 8,
    input_tokens: int = 800,
    output_tokens: int = 400,
    peak_output_tokens: int = 120,
    cost: Optional[float] = 0.0032,
    n_attempted: int = 10,
    n_with_usage: int = 10,
    by_task: Optional[dict] = None,
) -> dict:
    entry: dict = {
        "n_attempted": n_attempted,
        "n_with_usage": n_with_usage,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "peak_output_tokens": peak_output_tokens,
    }
    if cost is not None:
        entry["cost_credits"] = cost
    if by_task is not None:
        entry["by_task"] = by_task
    return entry


def _with_usage(summary: dict, usage_by_corpus: dict) -> dict:
    return {**summary, "llm_usage": usage_by_corpus}


class TestRenderUsageSection:
    def _labeled(self, crf_usage=None, llm_usage=None, corpora=None):
        corpora = corpora or ["biorxiv"]
        agg = [_agg("string", "levenshtein", {"title": 0.8})]
        crf = _title_summary(0.8, corpora=_multi_corpus(corpora, agg))
        llm = _title_summary(0.85, corpora=_multi_corpus(corpora, agg))
        return [
            ("crf (default)", _with_usage(crf, crf_usage) if crf_usage else crf),
            ("llm (values)", _with_usage(llm, llm_usage) if llm_usage else llm),
        ]

    def test_should_not_render_a_section_without_any_usage(self):
        report = _render_comparison_report(self._labeled())
        assert "LLM usage" not in report

    def test_should_render_totals_for_the_variant_that_spent(self):
        report = _render_comparison_report(
            self._labeled(llm_usage={"biorxiv": _usage()})
        )
        assert "LLM usage" in report
        assert "**llm (values)**" in report
        assert "over 10 of 10 docs" in report
        assert "800 tokens in, 400 tokens out" in report
        assert "* 0.0032 credits" in report

    def test_should_state_that_it_covers_every_document_attempted(self):
        report = _render_comparison_report(
            self._labeled(llm_usage={"biorxiv": _usage()})
        )
        assert "every document attempted" in report

    def test_should_say_when_calls_were_replayed_from_the_cache(self):
        report = _render_comparison_report(self._labeled(llm_usage={
            "biorxiv": {**_usage(calls=2), "replayed": _usage(calls=8, cost=0.5)},
        }))
        assert "8 were replayed" in report
        assert "Of 10 calls" in report
        assert "what these runs spent rather than what a cold run would cost" in report

    def test_should_say_what_the_replayed_responses_cost_when_generated(self):
        report = _render_comparison_report(self._labeled(llm_usage={
            "biorxiv": {**_usage(calls=2), "replayed": _usage(calls=8, cost=0.5)},
        }))
        assert "0.5000 credits when they were generated" in report
        assert "rather than a forecast" in report

    def test_should_say_nothing_about_replays_in_a_cold_report(self):
        report = _render_comparison_report(
            self._labeled(llm_usage={"biorxiv": _usage()})
        )
        assert "replayed" not in report

    def test_should_count_replayed_calls_in_the_calls_total(self):
        report = _render_comparison_report(self._labeled(llm_usage={
            "biorxiv": {**_usage(calls=2), "replayed": _usage(calls=8, cost=0.5)},
        }))
        assert "* 10 calls" in report

    def test_should_still_render_a_run_answered_entirely_from_the_cache(self):
        report = _render_comparison_report(self._labeled(llm_usage={
            "biorxiv": {
                **_usage(calls=0, input_tokens=0, output_tokens=0, cost=None),
                "replayed": _usage(calls=8, cost=0.5),
            },
        }))
        assert "LLM usage" in report
        assert "8 were replayed" in report

    def test_should_leave_out_a_variant_that_spent_nothing(self):
        # The CRF tools would otherwise be a row of dashes in every report.
        report = _render_comparison_report(
            self._labeled(llm_usage={"biorxiv": _usage()})
        )
        bullets = [line for line in report.splitlines() if line.startswith("**")]
        assert "**llm (values)**" in bullets
        assert "**crf (default)**" not in bullets

    def test_should_omit_cost_when_the_backend_stated_none(self):
        report = _render_comparison_report(
            self._labeled(llm_usage={"biorxiv": _usage(cost=None)})
        )
        assert "800 tokens in, 400 tokens out" in report
        assert "credits" not in report

    def test_should_show_partly_recorded_usage_as_a_fraction(self):
        report = _render_comparison_report(
            self._labeled(llm_usage={"biorxiv": _usage(n_attempted=10, n_with_usage=4)})
        )
        assert "over 4 of 10 docs" in report

    def test_should_render_usage_per_corpus(self):
        corpora = ["biorxiv", "ore"]
        report = _render_comparison_report(self._labeled(
            llm_usage={
                "biorxiv": _usage(calls=8),
                "ore": _usage(calls=2),
            },
            corpora=corpora,
        ))
        assert report.count("**LLM usage**, over every document attempted in this corpus.") == 2

    def test_should_sum_corpora_in_the_overall_table(self):
        report = _render_comparison_report(self._labeled(
            llm_usage={
                "biorxiv": _usage(calls=8, n_attempted=10, n_with_usage=10),
                "ore": _usage(calls=2, n_attempted=5, n_with_usage=5),
            },
            corpora=["biorxiv", "ore"],
        ))
        assert "over 15 of 15 docs" in report
        assert "* 10 calls" in report

    def test_should_name_the_tasks_when_more_than_one_ran(self):
        report = _render_comparison_report(self._labeled(llm_usage={
            "biorxiv": _usage(by_task={
                "citation": {"calls": 6, "output_tokens": 300},
                "reference_segmenter": {"calls": 2, "output_tokens": 100},
            })
        }))
        assert "* citation: 6 calls" in report
        assert "* reference_segmenter: 2 calls" in report

    def test_should_not_name_a_single_task(self):
        report = _render_comparison_report(self._labeled(llm_usage={
            "biorxiv": _usage(by_task={"citation": {"calls": 8, "output_tokens": 400}})
        }))
        assert "* citation:" not in report


class TestCachedInputNote:
    def _labeled(self, usage_by_corpus):
        agg = [_agg("string", "levenshtein", {"title": 0.8})]
        crf = _title_summary(0.8, corpora=_multi_corpus(["biorxiv"], agg))
        llm = _title_summary(0.85, corpora=_multi_corpus(["biorxiv"], agg))
        return [
            ("crf", crf),
            ("llm", _with_usage(llm, usage_by_corpus)),
        ]

    def test_should_explain_a_provider_cache_where_it_happened(self):
        usage = {**_usage(), "cached_input_tokens": 18432}
        report = _render_comparison_report(self._labeled({"biorxiv": usage}))
        assert "18,432 served from the provider's own prefix cache" in report

    def test_should_stay_silent_where_nothing_was_cached(self):
        usage = {**_usage(), "cached_input_tokens": 0}
        report = _render_comparison_report(self._labeled({"biorxiv": usage}))
        assert "prefix cache" not in report


class TestCoverageLines:
    def _render(self, records):
        summaries = [(label, {"fields": [], "corpora": {}}) for label, _ in records]
        return _render_comparison_report(summaries, records)

    def test_should_name_the_column_that_lost_documents(self):
        result = self._render([("llm_all", {"n_recovered": 3, "n_errors": 1})])
        assert "**llm_all**: 3 recovered on retry, 1 without a prediction" in result

    def test_should_stay_silent_for_a_column_that_lost_nothing(self):
        result = self._render([("grobid_crf", {"n_recovered": 0, "n_errors": 0})])
        assert "recovered on retry" not in result

    def test_should_stay_silent_for_a_column_with_no_run_record(self):
        result = self._render([("stored baseline", None)])
        assert "recovered on retry" not in result

    def test_should_state_only_the_columns_that_have_something_to_report(self):
        result = self._render([
            ("grobid_crf", {"n_recovered": 0, "n_errors": 0}),
            ("llm_all", {"n_recovered": 0, "n_errors": 2}),
        ])
        assert "**llm_all**: 2 without a prediction" in result
        assert "grobid_crf" not in result

    def test_should_render_without_run_records_at_all(self):
        summaries = [("llm_all", {"fields": [], "corpora": {}})]
        assert "ScienceBeam Parser Evaluation" in _render_comparison_report(summaries)


def _presence(n: int, n_gold: int, no_gold_docs: int = 0, no_gold_values: int = 0) -> dict:
    return {
        "n": n, "n_gold": n_gold,
        "no_gold_docs": no_gold_docs, "no_gold_values": no_gold_values,
    }


def _split_summary(
    f1: float, conditional: float, presence: dict, field: str = "acknowledgement"
) -> dict:
    corpus = {
        "scielo_br": {
            "n": presence["n"],
            "aggregated": [_agg("string", "edit_sim", {field: f1})],
            "aggregated_gold_present": [_agg("string", "edit_sim", {field: conditional})],
            "gold_presence": {field: presence},
        }
    }
    return _summary(
        fields=[field],
        field_measures={field: ["edit_sim"]},
        field_scoring_types={field: "string"},
        corpora=corpus,
    )


class TestGoldSplitSection:
    def test_renders_conditional_f1_with_its_gold_denominator_and_delta(self):
        report = _render_comparison_report([
            ("wapiti", _split_summary(0.545, 0.750, _presence(40, 5, 3, 3))),
            ("current", _split_summary(0.429, 0.590, _presence(40, 5, 3, 3))),
        ])
        split = report.split("### Where the gold does not record the field")[1]
        assert "| acknowledgement (edit_sim) | 5/40 | 0.750 | 0.590 | -0.160 |" in split

    def test_states_what_each_variant_produced_against_no_gold(self):
        report = _render_comparison_report([
            ("wapiti", _split_summary(0.0, 0.0, _presence(40, 0, 13, 13))),
            ("current", _split_summary(0.0, 0.0, _presence(40, 0, 30, 30))),
        ])
        assert (
            "| acknowledgement | 40 | 13 docs, 13 values | 30 docs, 30 values |" in report
        )

    def test_shows_counts_alone_where_the_corpus_records_no_gold(self):
        report = _render_comparison_report([
            ("wapiti", _split_summary(0.0, 0.0, _presence(40, 0, 13, 13))),
            ("current", _split_summary(0.0, 0.0, _presence(40, 0, 30, 30))),
        ])
        split = report.split("### Where the gold does not record the field")[1]
        assert "| Field (method) | Gold |" not in split

    def test_dashes_the_main_table_where_the_corpus_records_no_gold(self):
        report = _render_comparison_report([
            ("wapiti", _split_summary(0.0, 0.0, _presence(40, 0, 13, 13))),
            ("current", _split_summary(0.0, 0.0, _presence(40, 0, 30, 30))),
        ])
        assert "| acknowledgement (edit_sim) | string | — | — | — |" in report

    def test_omits_the_section_where_the_gold_records_every_document(self):
        report = _render_comparison_report([
            ("wapiti", _split_summary(0.8, 0.8, _presence(40, 40))),
            ("current", _split_summary(0.9, 0.9, _presence(40, 40))),
        ])
        assert "Where the gold does not record the field" not in report

    def test_omits_the_section_for_summaries_written_before_the_split(self):
        report = _render_comparison_report([
            ("wapiti", _title_summary(0.80)),
            ("current", _title_summary(0.85)),
        ])
        assert "Where the gold does not record the field" not in report
        assert "0.850" in report

    def test_dashes_a_variant_summarised_before_the_split(self):
        old = _title_summary(0.80, method="edit_sim")
        old["corpora"]["biorxiv"]["n"] = 40
        new = _split_summary(0.9, 0.95, _presence(40, 5, 3, 3), field="title")
        new["corpora"]["biorxiv"] = new["corpora"].pop("scielo_br")
        report = _render_comparison_report([("wapiti", old), ("current", new)])
        split = report.split("### Where the gold does not record the field")[1]
        assert "| title (edit_sim) | 5/40 | — | 0.950 | — |" in split

    def test_warns_where_the_variants_gold_denominators_differ(self):
        report = _render_comparison_report([
            ("wapiti", _split_summary(0.5, 0.600, _presence(20, 2, 2, 2))),
            ("current", _split_summary(0.6, 0.700, _presence(40, 5, 3, 3))),
        ])
        split = report.split("### Where the gold does not record the field")[1]
        assert "**Unequal document sets** (wapiti 2/20, current 5/40)" in split

    def test_does_not_warn_where_the_variants_cover_the_same_documents(self):
        report = _render_comparison_report([
            ("wapiti", _split_summary(0.5, 0.600, _presence(40, 5, 3, 3))),
            ("current", _split_summary(0.6, 0.700, _presence(40, 5, 3, 3))),
        ])
        split = report.split("### Where the gold does not record the field")[1]
        assert "Unequal document sets" not in split

    def test_names_a_variant_summarised_before_the_split_as_unknown(self):
        old = _title_summary(0.80, method="edit_sim")
        old["corpora"]["biorxiv"]["n"] = 40
        new = _split_summary(0.9, 0.95, _presence(40, 5, 3, 3), field="title")
        new["corpora"]["biorxiv"] = new["corpora"].pop("scielo_br")
        report = _render_comparison_report([("wapiti", old), ("current", new)])
        assert "| title | 35 | unknown | 3 docs, 3 values |" in report
