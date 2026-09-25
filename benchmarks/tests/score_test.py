from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from sciencebeam_judge.parsing.xml import parse_xml_mapping
from sciencebeam_judge.parsing.xpath.xpath_functions import register_functions
from sciencebeam_judge.resources import DEFAULT_XML_MAPPING_PATH

import pytest

from benchmarks.score import (
    _doc_scores_from_dict,
    _f1_from_aggregated,
    _summarise_documents,
    _score_pair,
    _build_field_measures,
    _build_field_scoring_types,
    _doc_scores_to_dict,
    _match_to_prf,
    _render_report,
    run_score,
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
    def _make_score(self, field, method, tp, fp=0, fn=0, scoring_type="string"):
        return {
            "field_name": field,
            "scoring_type": scoring_type,
            "scoring_method": method,
            "match_score": {"true_positive": tp, "false_positive": fp, "false_negative": fn},
        }

    def test_should_reshape_flat_list_to_nested_dict(self):
        result = _doc_scores_to_dict([self._make_score("title", "exact", tp=1)])
        assert result == {
            "title": {
                "scoring_type": "string",
                "exact": {
                    "true_positive": 1,
                    "false_positive": 0,
                    "false_negative": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                },
            }
        }

    def test_should_group_multiple_methods_under_same_field(self):
        doc_scores = [
            self._make_score("title", "exact", tp=1),
            self._make_score("title", "levenshtein", tp=1),
        ]
        result = _doc_scores_to_dict(doc_scores)
        assert set(result["title"].keys()) == {"scoring_type", "exact", "levenshtein"}

    def test_should_include_scoring_type_from_entry(self):
        result = _doc_scores_to_dict(
            [self._make_score("author_full_names", "levenshtein", tp=3, scoring_type="ulist")]
        )
        assert result["author_full_names"]["scoring_type"] == "ulist"

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


class TestRunScoreSplitDetermination:
    _CONFIG = {
        "dataset": {
            "splits": {
                "train": {"train_corpus": {}},
                "validation": {"validation_corpus": {}},
            }
        },
        "fields": ["title"],
        "scoring": {
            "default_methods": ["levenshtein"],
            "default_type": "string",
            "per_field": {},
        },
    }

    def _scored_corpora(self, mock_score_corpus) -> list:
        return [call.args[0] for call in mock_score_corpus.call_args_list]

    def test_split_override_takes_precedence_over_run_json(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"split": "train"}))

        with patch("benchmarks.score.register_functions"), \
             patch("benchmarks.score.parse_xml_mapping"), \
             patch("benchmarks.score._score_corpus", return_value={"n": 0}) as mock_score:
            run_score(
                config=self._CONFIG,
                run_dir=run_dir,
                data_dir=tmp_path / "data",
                out_path=tmp_path / "report.md",
                split_override="validation",
            )

        assert self._scored_corpora(mock_score) == ["validation_corpus"]

    def test_uses_split_from_run_json_when_no_override(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"split": "validation"}))

        with patch("benchmarks.score.register_functions"), \
             patch("benchmarks.score.parse_xml_mapping"), \
             patch("benchmarks.score._score_corpus", return_value={"n": 0}) as mock_score:
            run_score(
                config=self._CONFIG,
                run_dir=run_dir,
                data_dir=tmp_path / "data",
                out_path=tmp_path / "report.md",
                split_override=None,
            )

        assert self._scored_corpora(mock_score) == ["validation_corpus"]

    def test_defaults_to_train_when_no_override_and_no_run_json(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with patch("benchmarks.score.register_functions"), \
             patch("benchmarks.score.parse_xml_mapping"), \
             patch("benchmarks.score._score_corpus", return_value={"n": 0}) as mock_score:
            run_score(
                config=self._CONFIG,
                run_dir=run_dir,
                data_dir=tmp_path / "data",
                out_path=tmp_path / "report.md",
                split_override=None,
            )

        assert self._scored_corpora(mock_score) == ["train_corpus"]


class TestRunScoreCorpusSelection:
    """Which corpora get scored once one of them is opt-in.

    The gap these cover is a run whose predictions all came from the store: no
    run.json is written then, so nothing but the caller's opt-in set says the
    corpus was covered, and an unscored corpus vanishes from summary.json and
    from the comparison built out of it without a warning.
    """

    _CONFIG = {
        "dataset": {
            "repo_id": "org/repo",
            "revision": "main",
            "splits": {
                "validation": {
                    "ore": {"file": "ore/val.parquet"},
                    "plos": {
                        "repo_id": "private/plos",
                        "revision": "corpus-v001",
                        "path": "validation/",
                        "manifest": "splits/corpus-v001.csv",
                        "stratum": "journal",
                        "optional": True,
                    },
                }
            },
        },
        "fields": ["title"],
        "scoring": {"default_methods": ["levenshtein"], "default_type": "string",
                    "per_field": {}},
    }

    def _score(
        self, tmp_path: Path, run_json: Optional[dict] = None, include=None
    ) -> list:
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        if run_json is not None:
            (run_dir / "run.json").write_text(json.dumps(run_json))
        with patch("benchmarks.score.register_functions"), \
             patch("benchmarks.score.parse_xml_mapping"), \
             patch("benchmarks.score._score_corpus", return_value={"n": 0}) as mock_score:
            run_score(
                config=self._CONFIG,
                run_dir=run_dir,
                data_dir=tmp_path / "data",
                out_path=tmp_path / "report.md",
                split_override="validation",
                include=include,
            )
        return [call.args[0] for call in mock_score.call_args_list]

    def test_leaves_out_an_opt_in_corpus_nobody_asked_for(self, tmp_path: Path):
        assert self._score(tmp_path) == ["ore"]

    def test_scores_an_opt_in_corpus_the_caller_asked_for(self, tmp_path: Path):
        assert self._score(tmp_path, include=["plos"]) == ["ore", "plos"]

    def test_run_json_is_the_authority_where_it_exists(self, tmp_path: Path):
        scored = self._score(
            tmp_path, run_json={"split": "validation", "corpora": ["ore", "plos"]}
        )
        assert scored == ["ore", "plos"]


class TestRunScoreLlmUsage:
    _CONFIG = {
        "dataset": {"splits": {"train": {"biorxiv": {}}}},
        "fields": ["title"],
        "scoring": {
            "default_methods": ["levenshtein"],
            "default_type": "string",
            "per_field": {},
        },
    }

    def _run(self, tmp_path: Path, manifest_entries: list) -> dict:
        run_dir = tmp_path / "run"
        manifest = run_dir / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            "".join(json.dumps(entry) + "\n" for entry in manifest_entries)
        )
        with patch("benchmarks.score.register_functions"), \
             patch("benchmarks.score.parse_xml_mapping"), \
             patch("benchmarks.score._score_corpus", return_value={"n": 1}):
            run_score(
                config=self._CONFIG,
                run_dir=run_dir,
                data_dir=tmp_path / "data",
                out_path=tmp_path / "report.md",
                split_override="train",
            )
        return json.loads((run_dir / "summary.json").read_text())

    def test_should_aggregate_usage_from_the_manifest(self, tmp_path: Path):
        summary = self._run(tmp_path, [
            {
                "corpus": "biorxiv", "record_id": "doc1", "status": "ok",
                "llm_usage": {"calls": 2, "input_tokens": 200, "output_tokens": 100},
            },
            {
                "corpus": "biorxiv", "record_id": "doc2", "status": "error",
                "llm_usage": {"calls": 1, "input_tokens": 100, "output_tokens": 16000},
            },
        ])
        usage = summary["llm_usage"]["biorxiv"]
        assert usage["calls"] == 3
        assert usage["output_tokens"] == 16100
        assert usage["n_attempted"] == 2
        assert usage["n_with_usage"] == 2

    def test_should_leave_the_summary_unchanged_without_usage(self, tmp_path: Path):
        summary = self._run(tmp_path, [
            {"corpus": "biorxiv", "record_id": "doc1", "status": "ok"},
        ])
        assert "llm_usage" not in summary


GOLD_JATS_1 = b"""<article>
  <front>
    <article-meta>
      <title-group><article-title>The title</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="person">
          <name><surname>Smith</surname><given-names>Jo</given-names></name>
        </contrib>
      </contrib-group>
      <aff>Institute 1</aff>
      <abstract><p>The abstract</p></abstract>
      <kwd-group><kwd>keyword 1</kwd></kwd-group>
    </article-meta>
  </front>
  <body><sec><title>Introduction</title><p>Body text</p></sec></body>
  <back>
    <ack><p>Thanks</p></ack>
    <ref-list><ref><element-citation>
      <article-title>Reference title</article-title>
      <pub-id pub-id-type="doi">10.1234/doi1</pub-id>
    </element-citation></ref></ref-list>
  </back>
</article>"""

PREDICTED_TEI_1 = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title level="a" type="main">The title</title></titleStmt>
      <sourceDesc><biblStruct><analytic>
        <author>
          <persName><forename type="first">Jo</forename><surname>Smith</surname></persName>
          <affiliation key="aff0">
            <note type="raw_affiliation">Institute 1</note>
            <orgName type="institution">Institute 1</orgName>
          </affiliation>
        </author>
      </analytic></biblStruct></sourceDesc>
    </fileDesc>
    {encoding_desc}
    <profileDesc>
      <abstract><p>The abstract</p></abstract>
      <textClass><keywords><term>keyword 1</term></keywords></textClass>
    </profileDesc>
  </teiHeader>
  <text>
    <body><div><head>Introduction</head><p>Body text</p></div></body>
    <back>
      <div type="acknowledgement"><div><p>Thanks</p></div></div>
      <div type="references"><listBibl><biblStruct>
        <note type="raw_reference">Reference title</note>
        <analytic>
          <title level="a">Reference title</title>
          <idno type="DOI">10.1234/doi1</idno>
        </analytic>
      </biblStruct></listBibl></div>
    </back>
  </text>
</TEI>"""

ENCODING_DESC_1 = """<encodingDesc>
      <appInfo>
        <application ident="sciencebeam-parser" version="1.2.3">
          <label type="profile">profile1</label>
          <label type="profile-digest">digest1</label>
        </application>
      </appInfo>
    </encodingDesc>"""


class TestAttributionIsNotScored:
    """The attribution element must sit where no scored field's xpath looks."""

    def test_should_score_a_prediction_the_same_with_and_without_attribution(self):
        register_functions()
        xml_mapping = parse_xml_mapping(DEFAULT_XML_MAPPING_PATH)
        field_names = [
            "title", "abstract", "author_full_names", "affiliation_text", "keywords",
            "body_section_titles", "acknowledgement", "first_reference_text",
            "reference_title", "reference_doi",
        ]
        measures = ["exact", "levenshtein", "edit_sim"]
        without = _score_pair(
            GOLD_JATS_1,
            PREDICTED_TEI_1.format(encoding_desc="").encode("utf-8"),
            field_names, measures, xml_mapping,
        )
        with_attribution = _score_pair(
            GOLD_JATS_1,
            PREDICTED_TEI_1.format(encoding_desc=ENCODING_DESC_1).encode("utf-8"),
            field_names, measures, xml_mapping,
        )
        assert with_attribution == without
        # A prediction the mapping read nothing out of would pass the comparison
        # above without saying anything about where the element sits.
        assert {score["field_name"] for score in without} == set(field_names)
        assert all(
            score["match_score"]["actual_something"]
            for score in without
            if score["scoring_method"] == "exact"
        )


class TestCoverageNote:
    def _report(self, run_record: dict) -> str:
        return _render_report({}, [], {}, run_record)

    def test_should_say_how_many_documents_a_retry_recovered(self):
        result = self._report({"n_recovered": 3, "n_errors": 0, "retry_passes": 2})
        assert "**Coverage:** 3 recovered on retry" in result

    def test_should_say_how_many_documents_were_never_predicted(self):
        result = self._report({"n_recovered": 0, "n_errors": 2, "retry_passes": 1})
        assert "2 without a prediction, and so not scored" in result

    def test_should_state_both_when_a_run_recovered_some_and_lost_others(self):
        result = self._report({"n_recovered": 3, "n_errors": 1, "retry_passes": 2})
        assert "3 recovered on retry, 1 without a prediction" in result

    def test_should_stay_silent_when_every_document_was_predicted_first_time(self):
        result = self._report({"n_recovered": 0, "n_errors": 0, "retry_passes": 2})
        assert "Coverage" not in result

    def test_should_stay_silent_for_a_run_record_that_predates_the_counts(self):
        assert "Coverage" not in self._report({"mode": "smoke"})


def _aggregated(method: str, field: str, f1: float, scoring_type: str = "string") -> list:
    return [{
        "scoring_type": scoring_type,
        "scoring_method": method,
        "summary_scores": {"by-field": {field: {"scores": {"f1": f1}}}},
    }]


def _gold_presence(n: int, n_gold: int, no_gold_docs: int = 0, no_gold_values: int = 0) -> dict:
    return {
        "n": n, "n_gold": n_gold,
        "no_gold_docs": no_gold_docs, "no_gold_values": no_gold_values,
    }


class TestRenderReportGoldSplit:
    def test_should_dash_a_field_the_corpus_records_no_gold_for(self):
        result = _render_report(
            {"pkp": {
                "n": 40,
                "aggregated": _aggregated("edit_sim", "acknowledgement", 0.0),
                "gold_presence": {"acknowledgement": _gold_presence(40, 0, 14, 14)},
            }},
            ["acknowledgement"], {"acknowledgement": "string"}, None,
        )
        assert "| acknowledgement | string | 40 | — |" in result
        assert "0.000" not in result

    def test_should_pair_a_row_over_the_documents_whose_gold_records_the_field(self):
        result = _render_report(
            {"scielo_br": {
                "n": 40,
                "aggregated": _aggregated("edit_sim", "acknowledgement", 0.429),
                "aggregated_gold_present": _aggregated("edit_sim", "acknowledgement", 0.590),
                "gold_presence": {"acknowledgement": _gold_presence(40, 5, 3, 3)},
            }},
            ["acknowledgement"], {"acknowledgement": "string"}, None,
        )
        assert "| acknowledgement | string | all 40 | 0.429 |" in result
        assert "| acknowledgement | string | gold 5 | 0.590 |" in result

    def test_should_not_pair_a_row_over_no_documents(self):
        result = _render_report(
            {"pkp": {
                "n": 40,
                "aggregated": _aggregated("edit_sim", "reference_title", 0.0, "partial_list"),
                "gold_presence": {"reference_title": _gold_presence(40, 0, 37, 650)},
            }},
            ["reference_title"], {"reference_title": "partial_list"}, None,
        )
        assert "gold 0" not in result
        assert "| reference_title | partial_list | 40 | — |" in result

    def test_should_count_what_was_produced_where_the_gold_records_nothing(self):
        result = _render_report(
            {"pkp": {
                "n": 40,
                "aggregated": _aggregated("edit_sim", "reference_title", 0.0, "partial_list"),
                "gold_presence": {"reference_title": _gold_presence(40, 0, 37, 650)},
            }},
            ["reference_title"], {"reference_title": "partial_list"}, None,
        )
        counts = result.split("Produced where the gold records nothing:")[1]
        assert "| reference_title | 40 | 37 docs, 650 values |" in counts

    def test_should_state_that_a_model_produced_nothing(self):
        result = _render_report(
            {"pkp": {
                "n": 40,
                "aggregated": _aggregated("edit_sim", "keywords", 0.0, "partial_ulist"),
                "gold_presence": {"keywords": _gold_presence(40, 0)},
            }},
            ["keywords"], {"keywords": "partial_ulist"}, None,
        )
        assert "| keywords | 40 | none |" in result

    def test_should_omit_the_section_where_the_gold_records_every_document(self):
        result = _render_report(
            {"biorxiv": {
                "n": 31,
                "aggregated": _aggregated("edit_sim", "title", 0.9),
                "aggregated_gold_present": _aggregated("edit_sim", "title", 0.9),
                "gold_presence": {"title": _gold_presence(31, 31)},
            }},
            ["title"], {"title": "string"}, None,
        )
        assert "Produced where the gold records nothing" not in result
        assert "| title | string | 31 | 0.900 |" in result

    def test_should_omit_the_section_for_a_run_scored_before_the_split(self):
        result = _render_report(
            {"biorxiv": {"n": 31, "aggregated": _aggregated("edit_sim", "title", 0.9)}},
            ["title"], {"title": "string"}, None,
        )
        assert "Produced where the gold records nothing" not in result
        assert "0.900" in result


def _scored_document(field: str, expected: int, predicted: int, sim: float = 0.0) -> dict:
    return {
        field: {
            "scoring_type": "string",
            "edit_sim": {
                "sim_sum": sim, "expected_count": expected, "predicted_count": predicted,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
            },
        }
    }


class TestDocScoresFromDict:
    def test_should_round_trip_doc_scores_to_dict(self):
        doc_scores = [{
            "field_name": "title",
            "scoring_type": "string",
            "scoring_method": "edit_sim",
            "match_score": {"sim_sum": 0.5, "expected_count": 1, "predicted_count": 1},
        }]
        result = list(_doc_scores_from_dict(_doc_scores_to_dict(doc_scores)))
        assert len(result) == 1
        assert result[0]["field_name"] == "title"
        assert result[0]["scoring_type"] == "string"
        assert result[0]["scoring_method"] == "edit_sim"
        # `_doc_scores_to_dict` also stores precision/recall/f1; the counts are what the
        # sums read.
        assert result[0]["match_score"]["sim_sum"] == 0.5
        assert result[0]["match_score"]["expected_count"] == 1


class TestSummariseDocuments:
    def test_should_report_no_aggregate_for_no_documents(self):
        assert _summarise_documents([], ["title"], {"title": ["edit_sim"]}) == {"n": 0}

    def test_should_aggregate_over_every_document_and_over_the_gold_present_ones(self):
        documents = [
            _scored_document("acknowledgement", expected=1, predicted=1, sim=1.0),
            _scored_document("acknowledgement", expected=0, predicted=1),
        ]
        result = _summarise_documents(
            documents, ["acknowledgement"], {"acknowledgement": ["edit_sim"]}
        )
        assert result["n"] == 2
        assert result["gold_presence"]["acknowledgement"] == {
            "n": 2, "n_gold": 1, "no_gold_docs": 1, "no_gold_values": 1,
        }
        combined = _f1_from_aggregated(
            result["aggregated"], "acknowledgement", "string", "edit_sim"
        )
        conditional = _f1_from_aggregated(
            result["aggregated_gold_present"], "acknowledgement", "string", "edit_sim"
        )
        # The spurious prediction only enlarges precision's denominator.
        assert combined == pytest.approx(2 / 3)
        assert conditional == pytest.approx(1.0)

    def test_should_ignore_a_method_the_config_no_longer_asks_for(self):
        documents = [_scored_document("title", expected=1, predicted=1, sim=1.0)]
        result = _summarise_documents(documents, ["title"], {"title": ["levenshtein"]})
        assert result == {"n": 1}


class TestRunScoreFromScores:
    def test_should_summarise_existing_score_files_without_scoring_again(self, tmp_path):
        run_dir = tmp_path / "run"
        scores_dir = run_dir / "scores" / "biorxiv"
        scores_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps(
            {"split": "train", "corpora": ["biorxiv"]}
        ))
        for record_id, expected in (("a", 1), ("b", 0)):
            (scores_dir / f"{record_id}.json").write_text(json.dumps({
                "record_id": record_id, "corpus": "biorxiv",
                "fields": _scored_document("acknowledgement", expected, 1, sim=1.0),
            }))

        with patch("benchmarks.score._score_corpus") as score_corpus:
            run_score(
                config={
                    "fields": ["acknowledgement"],
                    "scoring": {"default_methods": ["edit_sim"], "default_type": "string"},
                },
                run_dir=run_dir, data_dir=tmp_path / "data", out_path=None,
                from_scores=True,
            )
        score_corpus.assert_not_called()

        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["corpora"]["biorxiv"]["n"] == 2
        assert summary["corpora"]["biorxiv"]["gold_presence"]["acknowledgement"] == {
            "n": 2, "n_gold": 1, "no_gold_docs": 1, "no_gold_values": 1,
        }

    def test_should_warn_and_report_nothing_where_a_corpus_was_never_scored(
        self, tmp_path, caplog
    ):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps(
            {"split": "train", "corpora": ["biorxiv"]}
        ))
        run_score(
            config={"fields": ["title"], "scoring": {}},
            run_dir=run_dir, data_dir=tmp_path / "data", out_path=None,
            from_scores=True,
        )
        assert "No scores directory" in caplog.text
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["corpora"]["biorxiv"] == {"n": 0}
