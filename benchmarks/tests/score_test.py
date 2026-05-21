from __future__ import annotations

from benchmarks.score import (
    _build_field_measures,
    _build_field_scoring_types,
    _doc_scores_to_dict,
    _match_to_prf,
    _render_report,
)


class TestMatchToPrf:
    def test_should_return_zeros_when_all_zero(self):
        result = _match_to_prf({"true_positive": 0, "false_positive": 0, "false_negative": 0})
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_should_return_perfect_score_when_tp_only(self):
        result = _match_to_prf({"true_positive": 5, "false_positive": 0, "false_negative": 0})
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_should_compute_precision_recall_f1(self):
        result = _match_to_prf({"true_positive": 2, "false_positive": 2, "false_negative": 2})
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5

    def test_should_handle_missing_keys(self):
        result = _match_to_prf({})
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_should_handle_zero_precision_recall(self):
        result = _match_to_prf({"true_positive": 0, "false_positive": 1, "false_negative": 1})
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


class TestBuildFieldMeasures:
    def test_should_use_default_for_all_fields_when_no_per_field(self):
        result = _build_field_measures(["title", "abstract"], ["levenshtein"], {})
        assert result == {"title": ["levenshtein"], "abstract": ["levenshtein"]}

    def test_should_use_per_field_methods_override_when_present(self):
        result = _build_field_measures(
            ["title", "abstract"],
            ["levenshtein"],
            {"title": {"methods": ["exact", "levenshtein"]}},
        )
        assert result == {"title": ["exact", "levenshtein"], "abstract": ["levenshtein"]}

    def test_should_use_default_when_per_field_has_no_methods_key(self):
        result = _build_field_measures(
            ["title"], ["levenshtein"], {"title": {"type": "ulist"}}
        )
        assert result == {"title": ["levenshtein"]}

    def test_should_ignore_per_field_keys_not_in_field_names(self):
        result = _build_field_measures(
            ["abstract"], ["levenshtein"], {"title": {"methods": ["exact"]}}
        )
        assert result == {"abstract": ["levenshtein"]}


class TestBuildFieldScoringTypes:
    def test_should_use_default_for_all_fields_when_no_per_field(self):
        result = _build_field_scoring_types(["title", "abstract"], "string", {})
        assert result == {"title": "string", "abstract": "string"}

    def test_should_use_per_field_type_when_present(self):
        result = _build_field_scoring_types(
            ["title", "author_full_names"],
            "string",
            {"author_full_names": {"type": "ulist"}},
        )
        assert result == {"title": "string", "author_full_names": "ulist"}

    def test_should_use_default_when_per_field_has_no_type_key(self):
        result = _build_field_scoring_types(
            ["title"], "string", {"title": {"methods": ["exact"]}}
        )
        assert result == {"title": "string"}

    def test_should_ignore_per_field_keys_not_in_field_names(self):
        result = _build_field_scoring_types(
            ["abstract"], "string", {"title": {"type": "ulist"}}
        )
        assert result == {"abstract": "string"}


class TestDocScoresToDict:
    def test_should_reshape_flat_list_to_nested_dict(self):
        doc_scores = [
            {
                "field_name": "title",
                "scoring_method": "exact",
                "match_score": {"true_positive": 1, "false_positive": 0, "false_negative": 0},
            }
        ]
        result = _doc_scores_to_dict(doc_scores)
        assert result == {"title": {"exact": {"precision": 1.0, "recall": 1.0, "f1": 1.0}}}

    def test_should_group_multiple_methods_under_same_field(self):
        doc_scores = [
            {
                "field_name": "title",
                "scoring_method": "exact",
                "match_score": {"true_positive": 1, "false_positive": 0, "false_negative": 0},
            },
            {
                "field_name": "title",
                "scoring_method": "levenshtein",
                "match_score": {"true_positive": 1, "false_positive": 0, "false_negative": 0},
            },
        ]
        result = _doc_scores_to_dict(doc_scores)
        assert set(result["title"].keys()) == {"exact", "levenshtein"}

    def test_should_handle_empty_list(self):
        assert not _doc_scores_to_dict([])


class TestRenderReport:
    def test_should_include_header(self):
        result = _render_report({}, [], {}, None)
        assert "## ScienceBeam Parser Evaluation" in result

    def test_should_show_no_results_for_empty_corpus(self):
        result = _render_report({"biorxiv": {"n": 0}}, ["title"], {"title": "string"}, None)
        assert "biorxiv" in result
        assert "_No results._" in result

    def test_should_render_f1_scores_in_table(self):
        aggregated = [
            {
                "scoring_type": "string",
                "scoring_method": "exact",
                "summary_scores": {
                    "by-field": {"title": {"scores": {"f1": 0.85}}}
                },
            }
        ]
        result = _render_report(
            {"biorxiv": {"n": 5, "aggregated": aggregated}},
            ["title"],
            {"title": "string"},
            None,
        )
        assert "0.850" in result
        assert "title" in result

    def test_should_show_type_column(self):
        aggregated = [
            {
                "scoring_type": "ulist",
                "scoring_method": "levenshtein",
                "summary_scores": {
                    "by-field": {"author_full_names": {"scores": {"f1": 0.75}}}
                },
            }
        ]
        result = _render_report(
            {"biorxiv": {"n": 5, "aggregated": aggregated}},
            ["author_full_names"],
            {"author_full_names": "ulist"},
            None,
        )
        assert "Type" in result
        assert "ulist" in result

    def test_should_deduplicate_method_columns_across_scoring_types(self):
        aggregated = [
            {
                "scoring_type": "string",
                "scoring_method": "levenshtein",
                "summary_scores": {"by-field": {"title": {"scores": {"f1": 0.9}}}},
            },
            {
                "scoring_type": "ulist",
                "scoring_method": "levenshtein",
                "summary_scores": {
                    "by-field": {"author_full_names": {"scores": {"f1": 0.75}}}
                },
            },
        ]
        result = _render_report(
            {"biorxiv": {"n": 5, "aggregated": aggregated}},
            ["title", "author_full_names"],
            {"title": "string", "author_full_names": "ulist"},
            None,
        )
        assert result.count("levenshtein F1") == 1

    def test_should_show_dash_for_method_not_configured_for_field_type(self):
        aggregated = [
            {
                "scoring_type": "string",
                "scoring_method": "exact",
                "summary_scores": {"by-field": {}},
            }
        ]
        result = _render_report(
            {"biorxiv": {"n": 5, "aggregated": aggregated}},
            ["author_full_names"],
            {"author_full_names": "ulist"},
            None,
        )
        assert "— |" in result

    def test_should_include_run_provenance_when_provided(self):
        run_record = {"parser_image": "my-image:v1", "parser_config": "default", "mode": "smoke"}
        result = _render_report({}, [], {}, run_record)
        assert "my-image:v1" in result
        assert "default" in result
