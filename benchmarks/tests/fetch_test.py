from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarks.fetch import _sample_indices, fetch_gold


class TestSampleIndices:
    def test_returns_correct_size(self):
        assert len(_sample_indices(100, 10, seed=42)) == 10

    def test_is_deterministic(self):
        assert _sample_indices(100, 10, seed=42) == _sample_indices(100, 10, seed=42)

    def test_different_seeds_give_different_results(self):
        assert _sample_indices(100, 10, seed=42) != _sample_indices(100, 10, seed=99)

    def test_smaller_is_subset_of_larger(self):
        s10 = _sample_indices(158, 10, seed=42)
        s25 = _sample_indices(158, 25, seed=42)
        s50 = _sample_indices(158, 50, seed=42)
        assert s10 <= s25
        assert s25 <= s50
        assert s10 <= s50

    def test_caps_at_n_total(self):
        assert len(_sample_indices(30, 100, seed=42)) == 30

    def test_indices_within_range(self):
        n_total = 50
        assert _sample_indices(n_total, 20, seed=42) <= set(range(n_total))


_FETCH_CONFIG = {
    "dataset": {
        "repo_id": "org/repo",
        "revision": "main",
        "splits": {
            "train": {
                "biorxiv": {"file": "biorxiv/train.parquet", "id_column": "ppr_id"},
            }
        },
    },
    "sampling": {"smoke": {"biorxiv": 2}},
    "seeds": {"sample": 42},
}


def _make_parquet_mock(ids: list, xml_values: list) -> MagicMock:
    pf = MagicMock()
    id_col = MagicMock()
    id_col.to_pylist.return_value = ids
    pf.read.return_value.column.return_value = id_col

    batch = MagicMock()
    batch.num_rows = len(ids)

    def id_col_val(i):
        m = MagicMock()
        m.as_py.return_value = ids[i]
        return m

    def xml_col_val(i):
        m = MagicMock()
        m.as_py.return_value = xml_values[i]
        return m

    def column(name):
        if name == "ppr_id":
            col = MagicMock()
            col.__getitem__ = lambda self, i: id_col_val(i)
            return col
        col = MagicMock()
        col.__getitem__ = lambda self, i: xml_col_val(i)
        return col

    batch.column = column
    pf.iter_batches.return_value = [batch]
    return pf


class TestFetchGold:
    def test_materialises_gold_xml_no_pdf(self, tmp_path: Path):
        pf = _make_parquet_mock(["id1", "id2", "id3"], ["<xml1/>", "<xml2/>", "<xml3/>"])
        with patch("benchmarks.fetch.hf_hub_download", return_value="fake.parquet"), \
             patch("benchmarks.fetch.pq.ParquetFile", return_value=pf):
            records = fetch_gold(_FETCH_CONFIG, "smoke", "train", tmp_path)

        assert len(records) == 2
        for r in records:
            assert "corpus" in r
            assert "record_id" in r
            assert "xml_path" in r
            assert "pdf_path" not in r
            assert Path(r["xml_path"]).exists()

    def test_skips_existing_xml(self, tmp_path: Path):
        corpus_dir = tmp_path / "train" / "biorxiv"
        corpus_dir.mkdir(parents=True)
        pf = _make_parquet_mock(["id1"], ["<xml/>"])
        id_col = MagicMock()
        id_col.to_pylist.return_value = ["id1"]
        pf.read.return_value.column.return_value = id_col
        existing = corpus_dir / "id1.jats.xml"
        existing.write_text("<existing/>")
        with patch("benchmarks.fetch.hf_hub_download", return_value="fake.parquet"), \
             patch("benchmarks.fetch.pq.ParquetFile", return_value=pf):
            fetch_gold(
                {**_FETCH_CONFIG, "sampling": {"smoke": {"biorxiv": 1}}},
                "smoke", "train", tmp_path,
            )
        assert existing.read_text() == "<existing/>"

    def test_returns_subset_consistent_with_fetch_data(self, tmp_path: Path):
        ids = [f"id{i}" for i in range(10)]
        xml_vals = [f"<xml{i}/>" for i in range(10)]

        pf_gold = _make_parquet_mock(ids, xml_vals)
        id_col = MagicMock()
        id_col.to_pylist.return_value = ids
        pf_gold.read.return_value.column.return_value = id_col

        config = {
            **_FETCH_CONFIG,
            "sampling": {"smoke": {"biorxiv": 3}},
        }

        with patch("benchmarks.fetch.hf_hub_download", return_value="fake.parquet"), \
             patch("benchmarks.fetch.pq.ParquetFile", return_value=pf_gold):
            gold_records = fetch_gold(config, "smoke", "train", tmp_path)

        gold_ids = {(r["corpus"], r["record_id"]) for r in gold_records}
        assert len(gold_ids) == 3
