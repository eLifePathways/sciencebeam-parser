from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarks.fetch import fetch_training_source


_BASE_CONFIG = {
    "dataset": {
        "repo_id": "org/repo",
        "revision": "main",
        "splits": {
            "train": {
                "ore": {"file": "ore/train.parquet", "id_column": "ppr_id"},
                "biorxiv": {"file": "biorxiv/train.parquet", "id_column": "ppr_id"},
            }
        },
    },
    "cc_by_corpora": ["ore"],
    "sampling": {
        "smoke": {"ore": 2, "biorxiv": 2},
        "full": {"ore": None, "biorxiv": None},
    },
    "seeds": {"sample": 42},
}


def _make_parquet_mock(ids: list) -> MagicMock:
    pf = MagicMock()
    id_col = MagicMock()
    id_col.to_pylist.return_value = ids
    pf.read.return_value.column.return_value = id_col

    batch = MagicMock()
    batch.num_rows = len(ids)

    def _col(name):
        col = MagicMock()
        col.__getitem__ = lambda self, i: _cell(ids[i] if name == "ppr_id" else b"pdf")
        return col

    def _cell(val):
        m = MagicMock()
        m.as_py.return_value = val
        return m

    batch.column = _col
    pf.iter_batches.return_value = [batch]
    return pf


class TestFetchTrainingSource:
    def test_only_fetches_cc_by_corpora(self, tmp_path: Path):
        ore_pf = _make_parquet_mock(["ore1", "ore2", "ore3"])
        biorxiv_pf = _make_parquet_mock(["bx1", "bx2", "bx3"])

        def _parquet_file(path):
            return ore_pf if "ore" in path else biorxiv_pf

        with patch("benchmarks.fetch.hf_hub_download", return_value="fake.parquet"), \
             patch("benchmarks.fetch.pq.ParquetFile", side_effect=_parquet_file):
            records = fetch_training_source(_BASE_CONFIG, "smoke", "train", tmp_path)

        corpora = {r["corpus"] for r in records}
        assert "ore" in corpora
        assert "biorxiv" not in corpora

    def test_full_mode_none_fetches_all_records(self, tmp_path: Path):
        ids = [f"ore{i}" for i in range(5)]
        pf = _make_parquet_mock(ids)

        with patch("benchmarks.fetch.hf_hub_download", return_value="fake.parquet"), \
             patch("benchmarks.fetch.pq.ParquetFile", return_value=pf):
            records = fetch_training_source(_BASE_CONFIG, "full", "train", tmp_path)

        assert len(records) == len(ids)

    def test_empty_cc_by_corpora_fetches_nothing(self, tmp_path: Path):
        cfg = {**_BASE_CONFIG, "cc_by_corpora": []}
        pf = _make_parquet_mock(["id1", "id2"])
        with patch("benchmarks.fetch.hf_hub_download", return_value="fake.parquet"), \
             patch("benchmarks.fetch.pq.ParquetFile", return_value=pf):
            records = fetch_training_source(cfg, "smoke", "train", tmp_path)
        assert not records

    def test_writes_pdf_and_xml(self, tmp_path: Path):
        pf = _make_parquet_mock(["ore1", "ore2"])
        # Provide xml column too
        batch = MagicMock()
        batch.num_rows = 2

        def _col(name):
            col = MagicMock()
            if name == "ppr_id":
                col.__getitem__ = lambda self, i: _cell(["ore1", "ore2"][i])
            elif name == "pdf":
                col.__getitem__ = lambda self, i: _cell(b"pdfcontent")
            else:
                col.__getitem__ = lambda self, i: _cell("<xml/>")
            return col

        def _cell(val):
            m = MagicMock()
            m.as_py.return_value = val
            return m

        batch.column = _col
        pf.iter_batches.return_value = [batch]

        with patch("benchmarks.fetch.hf_hub_download", return_value="fake.parquet"), \
             patch("benchmarks.fetch.pq.ParquetFile", return_value=pf):
            records = fetch_training_source(_BASE_CONFIG, "smoke", "train", tmp_path)

        assert len(records) == 2
        for r in records:
            assert Path(r["pdf_path"]).exists()
            assert Path(r["xml_path"]).exists()
