from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.generate_training_data_from_tree_cli import main

GENERATE_DATA = "benchmarks.generate_training_data_from_tree_cli.generate_data_main"


def _write_config(path: Path, corpora: list) -> Path:
    config_file = path / "training-source.yml"
    config_file.write_text(
        textwrap.dedent(f"""\
            cc_by_corpora: {corpora!r}
        """),
        encoding="utf-8",
    )
    return config_file


def _make_corpus_dir(source_root: Path, split: str, corpus: str) -> Path:
    d = source_root / split / corpus
    d.mkdir(parents=True)
    return d


class TestGenerateTrainingDataCli:
    def test_calls_generate_data_once_per_corpus(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore", "scielo"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")
        _make_corpus_dir(source, "train", "scielo")

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
            ])

        assert mock_gen.call_count == 2
        called_corpora = [c.args[0][c.args[0].index("--output-path") + 1]
                          for c in mock_gen.call_args_list]
        assert any("ore" in p for p in called_corpora)
        assert any("scielo" in p for p in called_corpora)

    def test_source_and_output_paths_contain_split_and_corpus(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
                "--split", "train",
            ])

        argv = mock_gen.call_args.args[0]
        source_path = argv[argv.index("--source-path") + 1]
        xml_path = argv[argv.index("--source-xml-path") + 1]
        out_path = argv[argv.index("--output-path") + 1]

        assert source_path == str(source / "train" / "ore" / "*.pdf")
        assert xml_path == str(source / "train" / "ore" / "*.jats.xml")
        assert out_path == str(output / "train" / "ore")

    def test_use_directory_structure_always_forwarded(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
            ])

        argv = mock_gen.call_args.args[0]
        assert "--use-directory-structure" in argv

    def test_extra_args_forwarded_to_generate_data(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
                "--num-workers", "4",
                "--document-timeout", "60",
                "--debug",
            ])

        argv = mock_gen.call_args.args[0]
        assert "--num-workers" in argv
        assert "4" in argv
        assert "--document-timeout" in argv
        assert "60" in argv
        assert "--debug" in argv

    def test_skips_corpus_with_missing_source_directory(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore", "missing"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")
        # "missing" corpus directory is intentionally not created

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
            ])

        assert mock_gen.call_count == 1
        argv = mock_gen.call_args.args[0]
        assert "ore" in argv[argv.index("--output-path") + 1]

    def test_empty_cc_by_corpora_exits_cleanly(self, tmp_path: Path):
        config = _write_config(tmp_path, [])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()

        with patch(GENERATE_DATA) as mock_gen:
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--config", str(config),
                    "--source-data", str(source),
                    "--output-path", str(output),
                ])

        assert exc_info.value.code == 0
        mock_gen.assert_not_called()

    def test_corpus_failure_causes_exit_1(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")

        with patch(
            GENERATE_DATA,
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--config", str(config),
                    "--source-data", str(source),
                    "--output-path", str(output),
                ])

        assert exc_info.value.code == 1

    def test_second_corpus_still_runs_after_first_fails(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore", "scielo"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")
        _make_corpus_dir(source, "train", "scielo")

        call_count = 0

        def _side_effect(argv):
            nonlocal call_count
            call_count += 1
            if "ore" in argv[argv.index("--output-path") + 1]:
                raise RuntimeError("ore failed")

        with patch(
            GENERATE_DATA,
            side_effect=_side_effect,
        ):
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--config", str(config),
                    "--source-data", str(source),
                    "--output-path", str(output),
                ])

        assert call_count == 2
        assert exc_info.value.code == 1


def _write_manifest(corpus_dir: Path, mode: str = "medium") -> None:
    (corpus_dir / "source.json").write_text(
        json.dumps({
            "corpus": "ore",
            "split": "train",
            "mode": mode,
            "seed": 42,
            "dataset": {
                "repo_id": "org/repo",
                "revision": "main",
                "commit": "abc123",
                "location": "ore/train.parquet",
            },
            "requested_document_count": 50,
            "selected_document_ids": ["a", "b"],
        }),
        encoding="utf-8",
    )


class TestPairRecord:
    def test_writes_a_record_per_named_model(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _write_manifest(_make_corpus_dir(source, "train", "ore"))

        with patch(GENERATE_DATA):
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
                "--models", "segmentation", "header",
            ])

        pair_dir = output / "train" / "ore"
        assert sorted(p.name for p in pair_dir.iterdir()) == ["header", "segmentation"]
        record = json.loads(
            (pair_dir / "segmentation" / "provenance.json").read_text(encoding="utf-8")
        )
        assert record["mode"] == "medium"
        assert record["model"] == "segmentation"
        assert record["corpus"] == "ore"
        assert record["seed"] == 42
        assert record["dataset"]["commit"] == "abc123"

    def test_forwards_the_named_models_to_generate_data(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _write_manifest(_make_corpus_dir(source, "train", "ore"))

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
                "--models", "segmentation",
            ])

        argv = mock_gen.call_args.args[0]
        assert argv[argv.index("--models") + 1] == "segmentation"

    def test_writes_no_record_without_a_manifest(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")

        with patch(GENERATE_DATA):
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
                "--models", "segmentation",
            ])

        assert not (output / "train" / "ore" / "segmentation").exists()

    def test_writes_no_record_for_a_corpus_that_failed(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _write_manifest(_make_corpus_dir(source, "train", "ore"))

        with patch(
            GENERATE_DATA,
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(SystemExit):
                main([
                    "--config", str(config),
                    "--source-data", str(source),
                    "--output-path", str(output),
                    "--models", "segmentation",
                ])

        assert not (output / "train" / "ore" / "segmentation").exists()


class TestCorpusNarrowing:
    def test_generates_only_the_named_corpus(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore", "scielo"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")
        _make_corpus_dir(source, "train", "scielo")

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(config),
                "--source-data", str(source),
                "--output-path", str(output),
                "--corpus", "scielo",
            ])

        assert mock_gen.call_count == 1
        argv = mock_gen.call_args.args[0]
        assert "scielo" in argv[argv.index("--output-path") + 1]

    def test_rejects_a_corpus_outside_the_allow_list(self, tmp_path: Path):
        config = _write_config(tmp_path, ["ore"])
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _make_corpus_dir(source, "train", "ore")

        with patch(GENERATE_DATA) as mock_gen:
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--config", str(config),
                    "--source-data", str(source),
                    "--output-path", str(output),
                    "--corpus", "plos",
                ])

        assert exc_info.value.code == 1
        mock_gen.assert_not_called()


DECLARING_CONFIG = """\
cc_by_corpora: ['ore', 'scielo']
generate:
  ore:
    segmentation: smoke
    citation: medium
    affiliation-address: none
"""


class TestDefaultModels:
    def _config(self, path: Path) -> Path:
        config_file = path / "training-source.yml"
        config_file.write_text(DECLARING_CONFIG, encoding="utf-8")
        return config_file

    def test_generates_what_the_config_declares_when_no_models_are_named(
        self, tmp_path: Path
    ):
        """Without this, the only default left is every model generate_data has.

        A corpus that declares three models would get figure, fulltext, table and
        the name models too -- data nothing asked for, in a repo that is reviewed
        by reading its diffs.
        """
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _write_manifest(_make_corpus_dir(source, "train", "ore"))

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(self._config(tmp_path)),
                "--source-data", str(source),
                "--output-path", str(output),
            ])

        argv = mock_gen.call_args.args[0]
        assert argv[argv.index("--models") + 1:] == ["segmentation", "citation"]

    def test_named_models_still_win(self, tmp_path: Path):
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _write_manifest(_make_corpus_dir(source, "train", "ore"))

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(self._config(tmp_path)),
                "--source-data", str(source),
                "--output-path", str(output),
                "--models", "header",
            ])

        argv = mock_gen.call_args.args[0]
        assert argv[argv.index("--models") + 1:] == ["header"]

    def test_a_corpus_declaring_nothing_falls_back_to_every_model(self, tmp_path: Path):
        source = tmp_path / "source"
        output = tmp_path / "output"
        output.mkdir()
        _write_manifest(_make_corpus_dir(source, "train", "scielo"))

        with patch(GENERATE_DATA) as mock_gen:
            main([
                "--config", str(self._config(tmp_path)),
                "--source-data", str(source),
                "--output-path", str(output),
                "--corpus", "scielo",
            ])

        assert "--models" not in mock_gen.call_args.args[0]
