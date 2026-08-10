from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def _write_corpus(path: Path, ids: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "ppr_id": ids,
            "xml": [f"<article id='{record_id}'/>" for record_id in ids],
            "pdf": [f"%PDF-{record_id}".encode() for record_id in ids],
        }),
        path,
    )


@pytest.fixture(name="repo")
def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    _write_corpus(root / "ore" / "train.parquet", [f"ore{i}" for i in range(5)])
    _write_corpus(root / "biorxiv" / "train.parquet", [f"bx{i}" for i in range(5)])
    monkeypatch.setenv("BENCH_LOCAL_PARQUET_DIR", str(root))
    return root


class TestFetchTrainingSource:
    def test_only_fetches_cc_by_corpora(self, repo: Path):
        records = fetch_training_source(_BASE_CONFIG, "smoke", "train", repo / "out")
        corpora = {record["corpus"] for record in records}
        assert "ore" in corpora
        assert "biorxiv" not in corpora

    def test_full_mode_none_fetches_all_records(self, repo: Path):
        records = fetch_training_source(_BASE_CONFIG, "full", "train", repo / "out")
        assert len(records) == 5

    def test_empty_cc_by_corpora_fetches_nothing(self, repo: Path):
        cfg = {**_BASE_CONFIG, "cc_by_corpora": []}
        assert not fetch_training_source(cfg, "smoke", "train", repo / "out")

    def test_a_corpus_outside_the_allow_list_is_never_read(self, repo: Path):
        """The gate has to hold even when the corpus is configured and sized.

        `biorxiv` stands in for PLOS here: present in the split, present in
        `sampling`, absent from `cc_by_corpora`, and therefore never fetched.
        """
        fetch_training_source(_BASE_CONFIG, "smoke", "train", repo / "out")
        assert not (repo / "out" / "train" / "biorxiv").exists()

    def test_writes_pdf_and_xml(self, repo: Path):
        records = fetch_training_source(_BASE_CONFIG, "smoke", "train", repo / "out")
        assert len(records) == 2
        for record in records:
            assert Path(record["pdf_path"]).read_bytes().startswith(b"%PDF-")
            assert Path(record["xml_path"]).read_text(encoding="utf-8").startswith("<article")
