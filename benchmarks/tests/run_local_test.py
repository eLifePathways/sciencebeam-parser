from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.run_local import (
    _count_predictions,
    _expected_count,
    _tool_docker_config,
    run_local,
)

_CONFIG = {
    "baselines": [{"tool": "grobid", "version": "0.9.0-crf"}],
    "sampling": {"smoke": {"biorxiv": 10}},
    "dataset": {
        "splits": {
            "train": {"biorxiv": {}},
            "validation": {"biorxiv": {}},
        }
    },
    "fields": ["title"],
    "scoring": {"default_methods": ["levenshtein"], "default_type": "string", "per_field": {}},
}


class TestCountPredictions:
    def test_returns_zero_when_predictions_dir_absent(self, tmp_path: Path):
        assert _count_predictions(tmp_path) == 0

    def test_counts_tei_xml_files_recursively(self, tmp_path: Path):
        pred_dir = tmp_path / "predictions" / "biorxiv"
        pred_dir.mkdir(parents=True)
        (pred_dir / "doc1.tei.xml").write_text("")
        (pred_dir / "doc2.tei.xml").write_text("")
        assert _count_predictions(tmp_path) == 2

    def test_ignores_non_tei_xml_files(self, tmp_path: Path):
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "manifest.jsonl").write_text("")
        (pred_dir / "doc1.tei.xml").write_text("")
        assert _count_predictions(tmp_path) == 1


class TestExpectedCount:
    def test_sums_corpora_for_mode(self):
        config = {"sampling": {"smoke": {"biorxiv": 10, "pmc": 5}}}
        assert _expected_count(config, "smoke") == 15

    def test_returns_zero_for_missing_mode(self):
        config = {"sampling": {"smoke": {"biorxiv": 10}}}
        assert _expected_count(config, "full") == 0


class TestToolDockerConfig:
    def test_grobid_uses_correct_image_and_port(self):
        cfg = _tool_docker_config("grobid", "0.9.0-crf")
        assert cfg["image"] == "grobid/grobid:0.9.0-crf"
        assert cfg["port"] == "8070:8070"
        assert cfg["url"] == "http://localhost:8070"
        assert cfg["health_path"] == "/api/isalive"

    def test_sciencebeam_parser_uses_ghcr_image(self):
        cfg = _tool_docker_config("sciencebeam-parser", "1.2.3")
        assert cfg["image"] == "ghcr.io/elifepathways/sciencebeam-parser:1.2.3"
        assert cfg["port"] == "8080:8070"


class TestRunLocal:
    def _make_summary(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps({
            "fields": ["title"],
            "field_measures": {"title": ["levenshtein"]},
            "field_scoring_types": {"title": "string"},
            "corpora": {},
        }))

    @patch("benchmarks.run_local._docker_stop")
    @patch("benchmarks.run_local._docker_start")
    @patch("benchmarks.run_local._wait_healthy")
    @patch("benchmarks.run_local.run_score")
    @patch("benchmarks.run_local.run_predict")
    def test_generates_baseline_predictions_when_none_exist(
        self, mock_predict, mock_score, _mock_wait, mock_start, _mock_stop, tmp_path: Path
    ):
        runs_dir = tmp_path / "runs"

        def fake_score(_config, run_dir, _data_dir, **_kwargs):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_local(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir,
                  parser_url=None, parser_image=None, parser_profile=None, baseline_only=True)

        mock_start.assert_called_once()
        mock_predict.assert_called_once()

    @patch("benchmarks.run_local._docker_stop")
    @patch("benchmarks.run_local._docker_start")
    @patch("benchmarks.run_local._wait_healthy")
    @patch("benchmarks.run_local.run_score")
    @patch("benchmarks.run_local.run_predict")
    def test_skips_docker_when_predictions_already_present(
        self, mock_predict, mock_score, _mock_wait, mock_start, _mock_stop, tmp_path: Path
    ):
        runs_dir = tmp_path / "runs"
        pred_dir = (
            runs_dir / "baselines" / "grobid" / "0.9.0-crf" / "default" / "train"
            / "predictions" / "biorxiv"
        )
        pred_dir.mkdir(parents=True)
        for i in range(10):
            (pred_dir / f"doc{i}.tei.xml").write_text("")

        def fake_score(_config, run_dir, _data_dir, **_kwargs):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_local(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir,
                  parser_url=None, parser_image=None, parser_profile=None, baseline_only=True)

        mock_start.assert_not_called()
        mock_predict.assert_not_called()

    @patch("benchmarks.run_local._docker_stop")
    @patch("benchmarks.run_local._docker_start")
    @patch("benchmarks.run_local._wait_healthy")
    @patch("benchmarks.run_local.run_compare")
    @patch("benchmarks.run_local.run_score")
    @patch("benchmarks.run_local.run_predict")
    def test_baseline_only_skips_primary_predict_and_report(
        self, mock_predict, mock_score, mock_compare, _mock_wait, _mock_start, _mock_stop,
        tmp_path: Path,
    ):
        runs_dir = tmp_path / "runs"

        def fake_score(_config, run_dir, _data_dir, **_kwargs):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_local(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir,
                  parser_url="http://localhost:8080", parser_image=None,
                  parser_profile=None, baseline_only=True)

        mock_compare.assert_not_called()
        assert mock_predict.call_count <= 1  # only baseline predict, no primary

    @patch("benchmarks.run_local._docker_stop")
    @patch("benchmarks.run_local._docker_start")
    @patch("benchmarks.run_local._wait_healthy")
    @patch("benchmarks.run_local.run_compare")
    @patch("benchmarks.run_local.run_score")
    @patch("benchmarks.run_local.run_predict")
    def test_runs_comparison_report_when_parser_url_provided(
        self, _mock_predict, mock_score, mock_compare, _mock_wait, _mock_start, _mock_stop,
        tmp_path: Path,
    ):
        runs_dir = tmp_path / "runs"

        def fake_score(_config, run_dir, _data_dir, **_kwargs):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_local(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir,
                  parser_url="http://localhost:8080", parser_image="my-image:v1",
                  parser_profile=None, baseline_only=False)

        mock_compare.assert_called_once()
        labeled_paths = mock_compare.call_args.args[0]
        labels = [label for label, _ in labeled_paths]
        assert "grobid 0.9.0-crf" in labels
        assert "my-image:v1" in labels

    @patch("benchmarks.run_local._docker_stop")
    @patch("benchmarks.run_local._docker_start")
    @patch("benchmarks.run_local._wait_healthy")
    @patch("benchmarks.run_local.run_score")
    @patch("benchmarks.run_local.run_predict")
    def test_passes_profile_env_var_to_sbp_container(
        self, _mock_predict, mock_score, _mock_wait, mock_start, _mock_stop, tmp_path: Path
    ):
        config = {
            **_CONFIG,
            "baselines": [{"tool": "sciencebeam-parser", "version": "1.0.0",
                           "profile": "grobid_crf_0_9_0"}],
        }

        def fake_score(_config, run_dir, _data_dir, **_kwargs):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_local(config, "smoke", "train", tmp_path / "data", tmp_path / "runs",
                  parser_url=None, parser_image=None, parser_profile=None, baseline_only=True)

        call_kwargs = mock_start.call_args
        env_vars = (
            call_kwargs.args[3]
            if len(call_kwargs.args) > 3
            else call_kwargs.kwargs.get("env_vars", {})
        )
        assert env_vars.get("SCIENCEBEAM_PARSER__PROFILE") == "grobid_crf_0_9_0"

    @patch("benchmarks.run_local._docker_stop")
    @patch("benchmarks.run_local._docker_start")
    @patch("benchmarks.run_local._wait_healthy")
    @patch("benchmarks.run_local.run_score")
    @patch("benchmarks.run_local.run_predict")
    def test_grobid_baseline_does_not_get_profile_env_var(
        self, _mock_predict, mock_score, _mock_wait, mock_start, _mock_stop, tmp_path: Path
    ):
        config = {
            **_CONFIG,
            "baselines": [{"tool": "grobid", "version": "0.9.0-crf",
                           "profile": "some_profile"}],
        }

        def fake_score(_config, run_dir, _data_dir, **_kwargs):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_local(config, "smoke", "train", tmp_path / "data", tmp_path / "runs",
                  parser_url=None, parser_image=None, parser_profile=None, baseline_only=True)

        call_kwargs = mock_start.call_args
        env_vars = (
            call_kwargs.args[3]
            if len(call_kwargs.args) > 3
            else call_kwargs.kwargs.get("env_vars", {})
        )
        assert "SCIENCEBEAM_PARSER__PROFILE" not in (env_vars or {})

    @patch("benchmarks.run_local._docker_stop")
    @patch("benchmarks.run_local._docker_start")
    @patch("benchmarks.run_local._wait_healthy")
    @patch("benchmarks.run_local.run_score")
    @patch("benchmarks.run_local.run_predict")
    def test_profile_included_in_baseline_label(
        self, mock_predict, mock_score, _mock_wait, _mock_start, _mock_stop, tmp_path: Path
    ):
        config = {
            **_CONFIG,
            "baselines": [{"tool": "grobid", "version": "0.9.0-crf",
                           "profile": "grobid_crf_0_9_0"}],
        }
        runs_dir = tmp_path / "runs"

        def fake_score(_config, run_dir, _data_dir, **_kwargs):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_local(config, "smoke", "train", tmp_path / "data", runs_dir,
                  parser_url="http://localhost:8080", parser_image=None,
                  parser_profile=None, baseline_only=False)

        # run_predict is called with run_dir as the 5th positional arg;
        # the path must contain the profile name segment
        baseline_call = mock_predict.call_args_list[0]
        run_dir = baseline_call.args[4]
        assert "grobid_crf_0_9_0" in str(run_dir)
