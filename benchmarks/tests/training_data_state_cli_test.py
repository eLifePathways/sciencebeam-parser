from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Dict, Optional

import pytest
import yaml

from benchmarks.training_data_state_cli import (
    DRIFTED,
    MISSING,
    OK,
    UNDECLARED,
    UNKNOWN,
    get_missing_documents,
    get_pair_states,
    main,
)

CONFIG = """\
dataset:
  splits:
    train:
      ore:
        file: ore/train.parquet
cc_by_corpora:
  - ore
sampling:
  smoke:
    ore: 10
  medium:
    ore: 50
generate:
  ore:
    segmentation: smoke
    citation: medium
    affiliation-address: none
"""

SPLIT = "train"


@pytest.fixture(name="cfg")
def _cfg() -> dict:
    return yaml.safe_load(textwrap.dedent(CONFIG))


def _write_pair(
    root: Path, corpus: str, model: str, mode: Optional[str], data_file: str = "a.tei.xml"
) -> Path:
    pair_dir = root / SPLIT / corpus / model
    (pair_dir / "corpus").mkdir(parents=True, exist_ok=True)
    (pair_dir / "corpus" / data_file).write_text("<x/>", encoding="utf-8")
    if mode is not None:
        (pair_dir / "provenance.json").write_text(
            json.dumps({"corpus": corpus, "model": model, "mode": mode}),
            encoding="utf-8",
        )
    return pair_dir


def _status_by_pair(cfg: dict, root: Path) -> Dict[str, str]:
    return {
        f"{state.corpus}/{state.model}": state.status
        for state in get_pair_states(cfg, root, SPLIT)
    }


class TestGetPairStates:
    def test_a_pair_at_its_declared_mode_is_ok(self, cfg: dict, tmp_path: Path):
        _write_pair(tmp_path, "ore", "segmentation", "smoke")
        assert _status_by_pair(cfg, tmp_path)["ore/segmentation"] == OK

    def test_a_pair_generated_at_another_mode_has_drifted(self, cfg: dict, tmp_path: Path):
        _write_pair(tmp_path, "ore", "segmentation", "medium")
        assert _status_by_pair(cfg, tmp_path)["ore/segmentation"] == DRIFTED

    def test_a_pair_with_data_and_no_record_is_unknown(self, cfg: dict, tmp_path: Path):
        _write_pair(tmp_path, "ore", "segmentation", None)
        assert _status_by_pair(cfg, tmp_path)["ore/segmentation"] == UNKNOWN

    def test_a_declared_pair_with_no_data_is_missing(self, cfg: dict, tmp_path: Path):
        assert _status_by_pair(cfg, tmp_path)["ore/citation"] == MISSING

    def test_a_pair_holding_data_that_nothing_declares_is_listed(
        self, cfg: dict, tmp_path: Path
    ):
        _write_pair(tmp_path, "ore", "affiliation-address", None)
        assert _status_by_pair(cfg, tmp_path)["ore/affiliation-address"] == UNDECLARED

    def test_a_model_declared_none_and_absent_is_not_listed(
        self, cfg: dict, tmp_path: Path
    ):
        assert "ore/affiliation-address" not in _status_by_pair(cfg, tmp_path)

    def test_a_pair_is_recognised_whatever_layout_its_model_uses(
        self, cfg: dict, tmp_path: Path
    ):
        """`citation` writes into `corpus/` where `segmentation` writes `corpus/tei/`.

        Counting `tei/` is what reported a 38-document citation corpus as empty.
        """
        _write_pair(tmp_path, "ore", "citation", "medium", data_file="a.references.tei.xml")
        assert _status_by_pair(cfg, tmp_path)["ore/citation"] == OK

    def test_an_empty_corpus_directory_does_not_count_as_data(
        self, cfg: dict, tmp_path: Path
    ):
        (tmp_path / SPLIT / "ore" / "segmentation" / "corpus").mkdir(parents=True)
        assert _status_by_pair(cfg, tmp_path)["ore/segmentation"] == MISSING


class TestMain:
    def _config_file(self, tmp_path: Path) -> Path:
        config_file = tmp_path / "training-source.yml"
        config_file.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
        return config_file

    def test_prints_a_row_per_pair(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        data = tmp_path / "data"
        _write_pair(data, "ore", "segmentation", "smoke")

        main([
            "--config", str(self._config_file(tmp_path)),
            "--training-data", str(data),
        ])

        out = capsys.readouterr().out
        assert "segmentation" in out
        assert "citation" in out
        assert OK in out

    def test_check_exits_non_zero_when_a_pair_is_not_at_its_declared_mode(
        self, tmp_path: Path
    ):
        data = tmp_path / "data"
        _write_pair(data, "ore", "segmentation", "medium")
        _write_pair(data, "ore", "citation", "medium")

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--config", str(self._config_file(tmp_path)),
                "--training-data", str(data),
                "--check",
            ])
        assert exc_info.value.code == 1

    def test_check_is_quiet_when_everything_is_where_it_was_declared(
        self, tmp_path: Path
    ):
        data = tmp_path / "data"
        _write_pair(data, "ore", "segmentation", "smoke")
        _write_pair(data, "ore", "citation", "medium")

        main([
            "--config", str(self._config_file(tmp_path)),
            "--training-data", str(data),
            "--check",
        ])


class TestMissingDocuments:
    def test_lists_documents_whose_source_has_gone(self, tmp_path: Path):
        pair_dir = _write_pair(tmp_path, "ore", "segmentation", "smoke")
        record = json.loads((pair_dir / "provenance.json").read_text(encoding="utf-8"))
        record["missing_document_ids"] = ["gone1", "gone2"]
        (pair_dir / "provenance.json").write_text(json.dumps(record), encoding="utf-8")

        assert get_missing_documents(tmp_path, SPLIT) == {
            "ore/segmentation": ["gone1", "gone2"]
        }

    def test_says_nothing_when_every_selected_document_is_there(self, tmp_path: Path):
        _write_pair(tmp_path, "ore", "segmentation", "smoke")
        assert not get_missing_documents(tmp_path, SPLIT)

    def test_main_prints_them_under_the_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        data = tmp_path / "data"
        pair_dir = _write_pair(data, "ore", "segmentation", "smoke")
        record = json.loads((pair_dir / "provenance.json").read_text(encoding="utf-8"))
        record["missing_document_ids"] = ["gone1"]
        (pair_dir / "provenance.json").write_text(json.dumps(record), encoding="utf-8")

        config_file = tmp_path / "training-source.yml"
        config_file.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
        main(["--config", str(config_file), "--training-data", str(data)])

        out = capsys.readouterr().out
        assert "whose source has gone" in out
        assert "gone1" in out
