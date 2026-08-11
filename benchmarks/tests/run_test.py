from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.predictions_store import LocalPredictionsStore
from benchmarks.run import (
    _baseline_env_vars,
    _coverage,
    _run_baseline,
    _get_corpus_variants,
    _make_label,
    _tool_docker_config,
    run_benchmark,
)

_CONFIG = {
    "baselines": [{"tool": "grobid", "version": "0.9.0-crf", "profile": "default"}],
    "sampling": {"smoke": {"biorxiv": 10}},
    "dataset": {
        "repo_id": "org/repo",
        "revision": "main",
        "splits": {
            "train": {"biorxiv": {"file": "biorxiv/train.parquet", "id_column": "ppr_id",
                                  "variant": "v1"}},
            "validation": {"biorxiv": {"file": "biorxiv/val.parquet", "id_column": "ppr_id",
                                       "variant": "v1"}},
        },
    },
    "seeds": {"sample": 42},
    "fields": ["title"],
    "scoring": {"default_methods": ["levenshtein"], "default_type": "string", "per_field": {}},
}


class TestToolDockerConfig:
    def test_grobid_image_and_port(self):
        cfg = _tool_docker_config("grobid", "0.9.0-crf")
        assert cfg["image"] == "grobid/grobid:0.9.0-crf"
        assert cfg["port"] == "8070:8070"
        assert cfg["url"] == "http://localhost:8070"
        assert cfg["health_path"] == "/api/isalive"

    def test_sciencebeam_parser_image(self):
        cfg = _tool_docker_config("sciencebeam-parser", "1.2.3")
        assert cfg["image"] == "ghcr.io/elifepathways/sciencebeam-parser:1.2.3"
        assert cfg["port"] == "8080:8070"


class TestBaselineEnvVars:
    def test_grobid_returns_empty(self):
        assert not _baseline_env_vars("grobid", "default")

    def test_sbp_includes_preload(self):
        env = _baseline_env_vars("sciencebeam-parser", "grobid_crf")
        assert env["SCIENCEBEAM_PARSER__PRELOAD_ON_STARTUP"] == "true"
        assert env["SCIENCEBEAM_PARSER__PROFILE"] == "grobid_crf"

    def test_sbp_default_profile_omits_profile_var(self):
        env = _baseline_env_vars("sciencebeam-parser", "default")
        assert "SCIENCEBEAM_PARSER__PROFILE" not in env
        assert "SCIENCEBEAM_PARSER__PRELOAD_ON_STARTUP" in env


class TestGetCorpusVariants:
    def test_reads_variant_from_config(self):
        config = {"dataset": {"splits": {"train": {"biorxiv": {"variant": "v2"}}}}}
        assert _get_corpus_variants(config, "train") == {"biorxiv": "v2"}

    def test_defaults_to_v1(self):
        config = {"dataset": {"splits": {"train": {"biorxiv": {}}}}}
        assert _get_corpus_variants(config, "train") == {"biorxiv": "v1"}

    def test_leaves_out_an_opt_in_corpus_nobody_asked_for(self):
        config = {"dataset": {"splits": {"train": {
            "biorxiv": {"variant": "v1"},
            "plos": {"variant": "plos-v002", "optional": True},
        }}}}
        # Predictions for a corpus the run did not cover must be neither looked
        # for in the store nor pushed to it.
        assert _get_corpus_variants(config, "train") == {"biorxiv": "v1"}
        assert _get_corpus_variants(config, "train", ["plos"]) == {
            "biorxiv": "v1", "plos": "plos-v002",
        }


class TestMakeLabel:
    def test_grobid_uses_tool_and_version(self):
        assert _make_label("grobid", "0.9.0-crf", "default", None) == "grobid 0.9.0-crf (default)"

    def test_sbp_uses_version_as_base(self):
        assert _make_label("sciencebeam-parser", "my-image:v1", "grobid_crf", None) == \
               "my-image:v1 (grobid_crf)"

    def test_metadata_image_overrides_version(self):
        meta = {"image": "sciencebeam-parser:main-abc123"}
        assert _make_label("sciencebeam-parser", "main", "grobid_crf", meta) == \
               "sciencebeam-parser:main-abc123 (grobid_crf)"

    def test_no_metadata_image_falls_back(self):
        assert _make_label("sciencebeam-parser", "main", "grobid_crf", {"mode": "medium"}) == \
               "main (grobid_crf)"


class TestRunBenchmark:
    def _make_summary(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps({
            "fields": ["title"],
            "field_measures": {"title": ["levenshtein"]},
            "field_scoring_types": {"title": "string"},
            "corpora": {},
        }))

    def _gold_records(self):
        return [{"corpus": "biorxiv", "record_id": f"r{i}", "xml_path": f"/tmp/r{i}.jats.xml"}
                for i in range(10)]

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_generates_baseline_when_missing(
        self, mock_gold, mock_predict, mock_score, _mock_wait,
        mock_start, _mock_stop, tmp_path: Path,
    ):
        mock_gold.return_value = self._gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_benchmark(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir,
                      store, baseline_only=True)

        mock_start.assert_called_once()
        mock_predict.assert_called_once()

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_takes_what_the_store_has_before_generating_the_rest(
        self, mock_gold, mock_predict, mock_score, _mock_wait,
        _mock_start, _mock_stop, tmp_path: Path,
    ):
        """A partly-populated store is used, not discarded.

        Generating is per document and slow -- tens of seconds each -- so a run
        missing one prediction must not re-predict the ones it already has.
        """
        mock_gold.return_value = self._gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        # pylint: disable-next=protected-access
        run_dir = store._run_dir("grobid", "0.9.0-crf", "default", "train")
        manifest = run_dir / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(
            json.dumps({"corpus": "biorxiv", "record_id": f"r{i}", "status": "ok"}) + "\n"
            for i in range(8)  # two of the ten gold records are absent
        ))

        def fake_score(_cfg, score_run_dir, *_a, **_kw):
            self._make_summary(score_run_dir)

        mock_score.side_effect = fake_score
        with patch.object(store, "fetch", wraps=store.fetch) as mock_fetch:
            run_benchmark(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir,
                          store, baseline_only=True)
            mock_fetch.assert_called_once()
        mock_predict.assert_called_once()

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_fetches_from_store_when_complete(
        self, mock_gold, mock_predict, mock_score, _mock_wait,
        mock_start, _mock_stop, tmp_path: Path,
    ):
        gold = self._gold_records()
        mock_gold.return_value = gold
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)

        # Pre-populate manifest so all gold records are "done"
        # pylint: disable-next=protected-access
        run_dir = store._run_dir("grobid", "0.9.0-crf", "default", "train")
        manifest = run_dir / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "\n".join(
                json.dumps({"corpus": r["corpus"], "record_id": r["record_id"], "status": "ok"})
                for r in gold
            ) + "\n"
        )

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_benchmark(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir,
                      store, baseline_only=True)

        mock_start.assert_not_called()
        mock_predict.assert_not_called()

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_skips_fetch_only_baseline_when_missing(
        self, mock_gold, _mock_predict, mock_score, _mock_compare,
        _mock_wait, mock_start, _mock_stop, tmp_path: Path,
    ):
        mock_gold.return_value = self._gold_records()
        config = {
            **_CONFIG,
            "baselines": [
                {"tool": "grobid", "version": "0.9.0-crf", "profile": "default"},
                {"tool": "sciencebeam-parser", "version": "main",
                 "profile": "grobid_crf", "generate": False},
            ],
        }
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_benchmark(config, "smoke", "train", tmp_path / "data", runs_dir,
                      store, baseline_only=True)

        # Only GROBID docker should start, not SBP main (which is fetch-only)
        assert mock_start.call_count == 1
        started_image = mock_start.call_args[0][1]
        assert "grobid" in started_image

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_runs_comparison_with_parser_url(
        self, mock_gold, _mock_predict, mock_score, mock_compare,
        _mock_wait, _mock_start, _mock_stop, tmp_path: Path,
    ):
        mock_gold.return_value = self._gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_benchmark(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
                      parser_url="http://localhost:8080", parser_image="my-image:v1",
                      parser_profile="grobid_crf")

        mock_compare.assert_called_once()
        labels = [label for label, _ in mock_compare.call_args.args[0]]
        assert any("grobid" in lbl for lbl in labels)
        assert any("my-image:v1" in lbl for lbl in labels)

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_push_current_calls_store_push(
        self, mock_gold, _mock_predict, mock_score, _mock_wait,
        _mock_start, _mock_stop, tmp_path: Path,
    ):
        mock_gold.return_value = self._gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        push_calls: list = []

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self._make_summary(run_dir)

        mock_score.side_effect = fake_score

        with patch.object(store, "push", side_effect=lambda *a, **_kw: push_calls.append(a)):
            run_benchmark(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
                          parser_url="http://localhost:8080", parser_image="my-image:v1",
                          parser_profile="grobid_crf", push_current=True)

        # push called once: for the current predictions
        sbp_push = [c for c in push_calls if c[0] == "sciencebeam-parser"]
        assert len(sbp_push) == 1
        assert sbp_push[0][2] == "grobid_crf"  # profile


class TestCoverage:
    def test_reports_stored_and_expected_per_corpus(self):
        expected = {("a", "1"), ("a", "2"), ("b", "1")}
        assert _coverage(expected, {("a", "1"), ("a", "2")}) == {"a": (2, 2), "b": (0, 1)}

    def test_partial_coverage_is_reported_not_discarded(self):
        assert _coverage({("a", "1"), ("a", "2")}, {("a", "1")}) == {"a": (1, 2)}

    def test_nothing_stored(self):
        assert _coverage({("a", "1")}, set()) == {"a": (0, 1)}


class TestBaselineThatDoesNotGenerate:
    """A stored baseline contributes the corpora it has, not all or nothing.

    An opt-in corpus cannot be in a stored baseline unless a run that asked for it
    pushed one, so requiring every corpus drops the baseline from the comparison
    the moment such a corpus is included.
    """

    def _store_with(self, runs_dir: Path, records) -> LocalPredictionsStore:
        store = LocalPredictionsStore(runs_dir)
        # pylint: disable-next=protected-access
        run_dir = store._run_dir("sciencebeam-parser", "main", "grobid_crf", "validation")
        manifest = run_dir / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(
            json.dumps({"corpus": c, "record_id": r, "status": "ok"}) + "\n"
            for c, r in records
        ))
        return store

    def _run(self, store, runs_dir: Path, expected_ids, variants, include=None):
        with patch("benchmarks.run.run_score") as mock_score, \
             patch.object(store, "fetch", wraps=store.fetch) as mock_fetch:
            mock_score.side_effect = lambda _c, run_dir, *_a, **_kw: (
                run_dir.mkdir(parents=True, exist_ok=True),
                (run_dir / "summary.json").write_text("{}"),
            )
            result = _run_baseline(
                _CONFIG, "smoke", "validation", runs_dir / "data", runs_dir,
                "sciencebeam-parser", "main", "grobid_crf", False,
                expected_ids, variants, store, 0, include,
            )
        return result, mock_fetch

    def test_compares_the_corpora_it_has(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        store = self._store_with(runs_dir, [("biorxiv", "r1"), ("biorxiv", "r2")])
        expected = {("biorxiv", "r1"), ("biorxiv", "r2"), ("plos-manuscripts", "p1")}
        result, mock_fetch = self._run(
            store, runs_dir, expected,
            {"biorxiv": "v1", "plos-manuscripts": "v002"}, ["plos-manuscripts"],
        )
        assert result is not None, "the baseline should still contribute biorxiv"
        # The corpus it cannot cover is left out of what it fetches and scores.
        assert mock_fetch.call_args.args[5] == {"biorxiv": "v1"}

    def test_contributes_a_corpus_it_covers_only_partly(self, tmp_path: Path):
        """Scored over what it has; the report is what flags the shorter column."""
        runs_dir = tmp_path / "runs"
        store = self._store_with(runs_dir, [("biorxiv", "r1")])
        result, mock_fetch = self._run(
            store, runs_dir, {("biorxiv", "r1"), ("biorxiv", "r2")}, {"biorxiv": "v1"},
        )
        assert result is not None
        assert mock_fetch.call_args.args[5] == {"biorxiv": "v1"}

    def test_skips_only_when_it_covers_nothing(self, tmp_path: Path):
        runs_dir = tmp_path / "runs"
        store = self._store_with(runs_dir, [])
        result, _ = self._run(
            store, runs_dir, {("biorxiv", "r1")}, {"biorxiv": "v1"},
        )
        assert result is None
