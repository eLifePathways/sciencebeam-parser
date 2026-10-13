from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from benchmarks.fetch_training_source_cli import corpora_by_mode, main

_CONFIG: Dict[str, Any] = {
    "dataset": {
        "repo_id": "org/repo",
        "revision": "main",
        "splits": {"train": {"ore": "ore/train.parquet", "scielo": "scielo/train.parquet"}},
    },
    "cc_by_corpora": ["ore", "scielo"],
    "sampling": {
        "smoke": {"ore": 2, "scielo": 2},
        "medium": {"ore": 5, "scielo": 5},
        "large": {"ore": 9, "scielo": 9},
    },
    "seeds": {"sample": 42},
    # scielo is declared at smoke only, which is the case a restriction has to carry.
    "generate": {
        "ore": {"segmentation": "medium", "header": "smoke", "citation": "none"},
        "scielo": {"header": "smoke"},
    },
}


class TestWhatTheDeclarationAsksFor:
    def test_every_declared_mode_with_the_corpora_declared_at_it(self):
        assert corpora_by_mode(_CONFIG) == {"medium": ["ore"], "smoke": ["ore", "scielo"]}

    def test_a_pair_declared_none_asks_for_nothing(self):
        assert "citation" not in str(corpora_by_mode(_CONFIG))

    def test_narrowing_to_one_mode(self):
        assert corpora_by_mode(_CONFIG, modes=["smoke"]) == {"smoke": ["ore", "scielo"]}

    def test_narrowing_to_one_corpus_drops_a_mode_it_is_not_declared_at(self):
        assert corpora_by_mode(_CONFIG, corpora=["scielo"]) == {"smoke": ["scielo"]}

    def test_a_mode_nothing_declares_still_fetches_everything_allowed(self, caplog):
        """Naming a mode by hand is how a one-off tree gets made."""
        assert corpora_by_mode(_CONFIG, modes=["large"]) == {"large": None}
        assert "Nothing declares mode 'large'" in caplog.text

    def test_a_mode_nothing_declares_still_honours_a_named_corpus(self):
        assert corpora_by_mode(_CONFIG, modes=["large"], corpora=["ore"]) == {
            "large": ["ore"]
        }

    def test_a_config_declaring_nothing_asks_for_nothing(self):
        assert not corpora_by_mode({**_CONFIG, "generate": {}})


class TestTheCommand:
    @staticmethod
    def _run(tmp_path: Path, *extra: str):
        config = tmp_path / "training-source.yml"
        config.write_text("{}", encoding="utf-8")
        with patch(
            "benchmarks.fetch_training_source_cli.yaml.safe_load", return_value=_CONFIG
        ), patch(
            "benchmarks.fetch_training_source_cli.fetch_training_source"
        ) as mock_fetch:
            main([
                "--config", str(config),
                "--source-root", str(tmp_path / "source"),
                *extra,
            ])
        return mock_fetch

    def test_writes_one_tree_per_mode(self, tmp_path: Path):
        mock_fetch = self._run(tmp_path)
        by_mode = {
            call.args[1]: (call.args[3], call.kwargs["include"])
            for call in mock_fetch.call_args_list
        }
        assert by_mode["medium"] == (tmp_path / "source" / "medium", ["ore"])
        assert by_mode["smoke"] == (tmp_path / "source" / "smoke", ["ore", "scielo"])

    def test_reads_the_dataset_every_time(self, tmp_path: Path):
        """It is the command that asks, so it never takes the recorded path."""
        mock_fetch = self._run(tmp_path)
        for call in mock_fetch.call_args_list:
            assert "recheck_upstream" not in call.kwargs

    def test_narrowed_to_one_mode_fetches_only_that(self, tmp_path: Path):
        mock_fetch = self._run(tmp_path, "--mode", "smoke")
        assert [call.args[1] for call in mock_fetch.call_args_list] == ["smoke"]

    def test_a_config_declaring_nothing_fetches_nothing(self, tmp_path: Path):
        config = tmp_path / "training-source.yml"
        config.write_text("{}", encoding="utf-8")
        with patch(
            "benchmarks.fetch_training_source_cli.yaml.safe_load",
            return_value={**_CONFIG, "generate": {}},
        ), patch(
            "benchmarks.fetch_training_source_cli.fetch_training_source"
        ) as mock_fetch:
            main(["--config", str(config), "--source-root", str(tmp_path / "source")])
        mock_fetch.assert_not_called()

    def test_the_selection_defaults_to_beside_the_config(self, tmp_path: Path):
        mock_fetch = self._run(tmp_path)
        for call in mock_fetch.call_args_list:
            assert call.kwargs["selection_dir"] == tmp_path / "selection"
