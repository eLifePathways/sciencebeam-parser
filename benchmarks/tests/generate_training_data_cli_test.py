from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.generate_training_data_cli import main


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

        with patch("benchmarks.generate_training_data_cli.generate_data_main") as mock_gen:
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

        with patch("benchmarks.generate_training_data_cli.generate_data_main") as mock_gen:
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

        with patch("benchmarks.generate_training_data_cli.generate_data_main") as mock_gen:
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

        with patch("benchmarks.generate_training_data_cli.generate_data_main") as mock_gen:
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

        with patch("benchmarks.generate_training_data_cli.generate_data_main") as mock_gen:
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

        with patch("benchmarks.generate_training_data_cli.generate_data_main") as mock_gen:
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
            "benchmarks.generate_training_data_cli.generate_data_main",
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
            "benchmarks.generate_training_data_cli.generate_data_main",
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
