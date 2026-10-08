from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import yaml

from benchmarks.comparison_config import parse_comparison
from benchmarks.fetch import get_corpus_variants
from benchmarks.predictions_store import LocalPredictionsStore
from benchmarks.run import (
    _baseline_env_vars,
    _comparison_only_variants,
    _coverage,
    _run_baseline,
    _make_label,
    _tool_docker_config,
    run_benchmark,
    run_stored_comparison,
)

_COMPARISON_YAML = """
variants:
  - {label: crf, tool: sciencebeam-parser, version: main, profile: grobid_crf}
  - {label: head, current: true}
"""

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
        assert get_corpus_variants(config, "train") == {"biorxiv": "v2"}

    def test_defaults_to_v1(self):
        config: dict = {"dataset": {"splits": {"train": {"biorxiv": {}}}}}
        assert get_corpus_variants(config, "train") == {"biorxiv": "v1"}

    def test_leaves_out_an_opt_in_corpus_nobody_asked_for(self):
        config = {"dataset": {"splits": {"train": {
            "biorxiv": {"variant": "v1"},
            "plos": {"variant": "plos-v002", "optional": True},
        }}}}
        # Predictions for a corpus the run did not cover must be neither looked
        # for in the store nor pushed to it.
        assert get_corpus_variants(config, "train") == {"biorxiv": "v1"}
        assert get_corpus_variants(config, "train", ["plos"]) == {
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
    def make_summary(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps({
            "fields": ["title"],
            "field_measures": {"title": ["levenshtein"]},
            "field_scoring_types": {"title": "string"},
            "corpora": {},
        }))

    def seed_store(self, store, tool: str, version: str, profile: str) -> None:
        """A stored, complete set of predictions, so a non-generating baseline has
        something to be scored from."""
        # pylint: disable-next=protected-access
        run_dir = store._run_dir(tool, version, profile, "train")
        manifest = run_dir / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(
            json.dumps({
                "corpus": record["corpus"], "record_id": record["record_id"],
                "status": "ok",
            })
            for record in self.gold_records()
        ) + "\n")

    def gold_records(self):
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
        mock_gold.return_value = self.gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self.make_summary(run_dir)

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
        mock_gold.return_value = self.gold_records()
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
            self.make_summary(score_run_dir)

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
        gold = self.gold_records()
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
            self.make_summary(run_dir)

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
        mock_gold.return_value = self.gold_records()
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
            self.make_summary(run_dir)

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
        mock_gold.return_value = self.gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self.make_summary(run_dir)

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
        mock_gold.return_value = self.gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        push_calls: list = []

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self.make_summary(run_dir)

        mock_score.side_effect = fake_score

        with patch.object(store, "push", side_effect=lambda *a, **_kw: push_calls.append(a)):
            run_benchmark(_CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
                          parser_url="http://localhost:8080", parser_image="my-image:v1",
                          parser_profile="grobid_crf", push_current=True)

        # push called once: for the current predictions
        sbp_push = [c for c in push_calls if c[0] == "sciencebeam-parser"]
        assert len(sbp_push) == 1
        assert sbp_push[0][2] == "grobid_crf"  # profile

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_scores_a_comparison_variant_eval_yml_does_not_declare(
        self, mock_gold, _mock_predict, mock_score, _mock_compare,
        _mock_wait, _mock_start, _mock_stop, tmp_path: Path,
    ):
        mock_gold.return_value = self.gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        self.seed_store(store, "sciencebeam-parser", "main", "grobid_crf")
        comparison = tmp_path / "extra.yml"
        comparison.write_text(_COMPARISON_YAML)
        scored = []

        def fake_score(_cfg, run_dir, *_a, **_kw):
            scored.append(Path(run_dir))
            self.make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_benchmark(
            {**_CONFIG, "baselines": [
                {"tool": "grobid", "version": "0.9.0-crf", "profile": "default"},
            ]},
            "smoke", "train", tmp_path / "data", runs_dir, store,
            baseline_only=True, comparison=str(comparison),
        )

        assert any(
            "sciencebeam-parser/main/grobid_crf" in str(path) for path in scored
        ), scored

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_keeps_a_comparison_variant_out_of_the_default_report(
        self, mock_gold, _mock_predict, mock_score, mock_compare,
        _mock_wait, _mock_start, _mock_stop, tmp_path: Path,
    ):
        mock_gold.return_value = self.gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        self.seed_store(store, "sciencebeam-parser", "main", "grobid_crf")
        comparison = tmp_path / "extra.yml"
        comparison.write_text(_COMPARISON_YAML)

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self.make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_benchmark(
            {**_CONFIG, "baselines": [
                {"tool": "grobid", "version": "0.9.0-crf", "profile": "default"},
            ]},
            "smoke", "train", tmp_path / "data", runs_dir, store,
            parser_url="http://parser", comparison=str(comparison),
        )

        # The first run_compare call is the report CI always posts.
        default_labels = [label for label, _ in mock_compare.call_args_list[0][0][0]]
        assert not any("grobid_crf" in label for label in default_labels), default_labels

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_warns_when_a_comparison_does_not_name_the_run_under_test(
        self, mock_gold, _mock_predict, mock_score, _mock_compare,
        _mock_wait, _mock_start, _mock_stop, tmp_path: Path, caplog,
    ):
        mock_gold.return_value = self.gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        self.seed_store(store, "sciencebeam-parser", "main", "grobid_crf")
        comparison = tmp_path / "stored.yml"
        comparison.write_text(
            "variants:\n"
            "  - {label: crf, tool: sciencebeam-parser, version: main,"
            " profile: grobid_crf}\n"
            "  - {label: grobid, tool: grobid, version: 0.9.0-crf, profile: default}\n"
        )

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self.make_summary(run_dir)

        mock_score.side_effect = fake_score

        with caplog.at_level(logging.WARNING, logger="benchmarks.run"):
            run_benchmark(
                _CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
                parser_url="http://parser", comparison=str(comparison),
            )

        assert any("no `current: true` variant" in r.message for r in caplog.records), (
            [r.message for r in caplog.records]
        )

    @patch("benchmarks.run._docker_stop")
    @patch("benchmarks.run._docker_start")
    @patch("benchmarks.run._wait_healthy")
    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_stays_quiet_when_the_comparison_names_the_run_under_test(
        self, mock_gold, _mock_predict, mock_score, _mock_compare,
        _mock_wait, _mock_start, _mock_stop, tmp_path: Path, caplog,
    ):
        mock_gold.return_value = self.gold_records()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        self.seed_store(store, "sciencebeam-parser", "main", "grobid_crf")
        comparison = tmp_path / "extra.yml"
        comparison.write_text(_COMPARISON_YAML)

        def fake_score(_cfg, run_dir, *_a, **_kw):
            self.make_summary(run_dir)

        mock_score.side_effect = fake_score

        with caplog.at_level(logging.WARNING, logger="benchmarks.run"):
            run_benchmark(
                _CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
                parser_url="http://parser", comparison=str(comparison),
            )

        assert not any("`current: true`" in r.message for r in caplog.records), (
            [r.message for r in caplog.records]
        )


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


class TestComparisonOnlyVariants:
    """A comparison names runs of its own; `eval.yml` need not already declare them."""

    def _config(self):
        return {"baselines": [
            {"tool": "grobid", "version": "0.9.1-crf", "profile": "default"},
            {"tool": "sciencebeam-parser", "version": "main", "profile": "grobid_crf"},
        ]}

    def _comparison(self, body: str):
        return parse_comparison(yaml.safe_load(body))

    def test_should_be_empty_without_a_comparison(self):
        assert _comparison_only_variants(self._config(), None) == []

    def test_should_skip_a_variant_eval_yml_already_runs(self):
        comparison = self._comparison("""
variants:
  - {label: a, tool: grobid, version: 0.9.1-crf, profile: default}
  - {label: b, current: true}
""")
        assert _comparison_only_variants(self._config(), comparison) == []

    def test_should_name_a_variant_eval_yml_does_not_run(self):
        comparison = self._comparison("""
variants:
  - {label: llm, tool: sciencebeam-parser, version: main, profile: llm_all}
  - {label: b, current: true}
""")
        variants = _comparison_only_variants(self._config(), comparison)
        assert [(v.tool, v.version, v.profile) for v in variants] == [
            ("sciencebeam-parser", "main", "llm_all"),
        ]

    def test_should_tell_two_profiles_of_one_version_apart(self):
        comparison = self._comparison("""
variants:
  - {label: crf, tool: sciencebeam-parser, version: main, profile: grobid_crf}
  - {label: llm, tool: sciencebeam-parser, version: main, profile: llm_all}
""")
        variants = _comparison_only_variants(self._config(), comparison)
        assert [v.profile for v in variants] == ["llm_all"]

    def test_should_leave_out_the_run_under_test(self):
        comparison = self._comparison("""
variants:
  - {label: a, tool: grobid, version: 0.9.1-crf, profile: default}
  - {label: head, current: true}
""")
        assert all(
            not v.current for v in _comparison_only_variants(self._config(), comparison)
        )

    def test_should_leave_out_a_variant_given_by_path(self):
        comparison = self._comparison("""
variants:
  - {label: a, summary: some/summary.json}
  - {label: b, summary: other/summary.json}
""")
        assert _comparison_only_variants(self._config(), comparison) == []


class TestRunStoredComparison:
    """The whole run for a question asked after the fact: no parser, nothing generated."""

    def _seeded(self, tmp_path: Path):
        helper = TestRunBenchmark()
        runs_dir = tmp_path / "runs"
        store = LocalPredictionsStore(runs_dir)
        for tool, version, profile in (
            ("grobid", "0.9.0-crf", "default"),
            ("sciencebeam-parser", "main", "grobid_crf"),
        ):
            helper.seed_store(store, tool, version, profile)
        path = tmp_path / "stored.yml"
        path.write_text(
            "variants:\n"
            "  - {label: grobid, tool: grobid, version: 0.9.0-crf, profile: default}\n"
            "  - {label: crf, tool: sciencebeam-parser, version: main,"
            " profile: grobid_crf}\n"
        )
        return helper, runs_dir, store, path

    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_scores_every_named_variant_from_the_store(
        self, mock_gold, _mock_predict, mock_score, _mock_compare, tmp_path: Path,
    ):
        helper, runs_dir, store, comparison = self._seeded(tmp_path)
        mock_gold.return_value = helper.gold_records()
        scored = []

        def fake_score(_cfg, run_dir, *_a, **_kw):
            scored.append(str(run_dir))
            helper.make_summary(run_dir)

        mock_score.side_effect = fake_score

        run_stored_comparison(
            _CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
            comparison=str(comparison),
        )

        assert any("grobid/0.9.0-crf/default" in path for path in scored)
        assert any("sciencebeam-parser/main/grobid_crf" in path for path in scored)

    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_never_predicts_anything(
        self, mock_gold, mock_predict, mock_score, _mock_compare, tmp_path: Path,
    ):
        helper, runs_dir, store, comparison = self._seeded(tmp_path)
        mock_gold.return_value = helper.gold_records()
        mock_score.side_effect = lambda _cfg, run_dir, *_a, **_kw: helper.make_summary(run_dir)

        run_stored_comparison(
            _CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
            comparison=str(comparison),
        )

        mock_predict.assert_not_called()

    @patch("benchmarks.run.run_compare")
    @patch("benchmarks.run.run_score")
    @patch("benchmarks.run.run_predict")
    @patch("benchmarks.run.fetch_gold")
    def test_writes_the_comparison_under_the_split(
        self, mock_gold, _mock_predict, mock_score, mock_compare, tmp_path: Path,
    ):
        helper, runs_dir, store, comparison = self._seeded(tmp_path)
        mock_gold.return_value = helper.gold_records()
        mock_score.side_effect = lambda _cfg, run_dir, *_a, **_kw: helper.make_summary(run_dir)

        run_stored_comparison(
            _CONFIG, "smoke", "train", tmp_path / "data", runs_dir, store,
            comparison=str(comparison),
        )

        assert mock_compare.call_args[0][1] == runs_dir / "train/comparison-stored.md"
