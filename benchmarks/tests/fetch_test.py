from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.corpus_source import CorpusConfigError, resolve_source
from benchmarks.fetch import (
    fetch_data,
    fetch_gold,
    fetch_training_source,
    iter_corpus_sources,
    resolved_sources,
)
from benchmarks.sampling import positional_ids

# Three strata of different sizes, so that capping and remainders are exercised:
# the smallest cannot fill a large sample and has to drop out of later rounds.
STRATA = {"aaa": 6, "bbb": 4, "ccc": 2}


def _write_parquet(path: Path, ids: Sequence[str], row_group_size: int = 2) -> None:
    """A corpus file with the payload columns, plus a `doc` no reader should want."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "id": list(ids),
        "doc": [f"DOC-{record_id}".encode() * 64 for record_id in ids],
        "doc_ext": ["docx"] * len(ids),
        "xml": [f"<article id='{record_id}'/>" for record_id in ids],
        "pdf": [f"%PDF-{record_id}".encode() for record_id in ids],
    })
    pq.write_table(table, path, row_group_size=row_group_size)


def _stratified_repo(root: Path, split: str = "validation") -> None:
    """A Hive tree with the stratum in the path and not in the files, plus a manifest.

    Chunked into more than one file per stratum, since a corpus that grows by adding
    files is the case a single `file:` entry cannot express.
    """
    rows = []
    for stratum, count in STRATA.items():
        ids = [f"{stratum}-{index:02d}" for index in range(count)]
        for chunk_start in range(0, count, 3):
            chunk = ids[chunk_start:chunk_start + 3]
            _write_parquet(
                root / split / f"journal={stratum}" / f"v001-{chunk_start:05d}.parquet",
                chunk,
            )
        for rank, record_id in enumerate(ids):
            rows.append({"id": record_id, "journal": stratum, "rank": rank, "split": split})
        # A row in another split, to prove the manifest is read per split.
        rows.append({
            "id": f"{stratum}-other", "journal": stratum, "rank": 99, "split": "test",
        })

    manifest = root / "splits" / "corpus-v001.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["id", "journal", "rank", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _single_file_repo(root: Path, n: int = 12) -> List[str]:
    ids = [f"single-{index:02d}" for index in range(n)]
    _write_parquet(root / "single" / "validation-00000-of-00001.parquet", ids)
    return ids


def _config(
    corpora: Dict[str, dict], sizes: Dict[str, Optional[int]], split: str = "validation"
) -> dict:
    return {
        "dataset": {
            "repo_id": "shared/repo",
            "revision": "main",
            "splits": {split: corpora},
        },
        "sampling": {"smoke": sizes},
        "seeds": {"sample": 42},
    }


STRATIFIED_ENTRY = {
    "repo_id": "private/plos",
    "revision": "corpus-v001",
    "path": "validation/",
    "manifest": "splits/corpus-v001.csv",
    "stratum": "journal",
    "variant": "corpus-v001",
}

SINGLE_ENTRY = {"file": "single/validation-00000-of-00001.parquet", "id_column": "id"}


@pytest.fixture(name="repo")
def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    _stratified_repo(root)
    _single_file_repo(root)
    monkeypatch.setenv("BENCH_LOCAL_PARQUET_DIR", str(root))
    return root


class TestResolveSource:
    def test_a_corpus_may_name_its_own_repo_and_revision(self):
        source = resolve_source(
            _config({"plos": STRATIFIED_ENTRY}, {"plos": 3}),
            "validation", "plos", STRATIFIED_ENTRY,
        )
        assert (source.repo_id, source.revision) == ("private/plos", "corpus-v001")

    def test_falls_back_to_the_dataset_level_repo_and_revision(self):
        source = resolve_source(
            _config({"ore": SINGLE_ENTRY}, {"ore": 3}), "validation", "ore", SINGLE_ENTRY
        )
        assert (source.repo_id, source.revision) == ("shared/repo", "main")

    def test_a_bare_string_entry_still_names_a_file(self):
        source = resolve_source({"dataset": {"repo_id": "a/b", "revision": "main"}},
                                "validation", "ore", "ore/train.parquet")
        assert source.file == "ore/train.parquet"
        assert source.id_column == "id"

    def test_rejects_a_stratum_without_a_manifest(self):
        entry = {**STRATIFIED_ENTRY}
        del entry["manifest"]
        with pytest.raises(CorpusConfigError, match="no `manifest`"):
            resolve_source(_config({"plos": entry}, {}), "validation", "plos", entry)

    def test_rejects_both_a_file_and_a_path(self):
        entry = {**STRATIFIED_ENTRY, "file": "somewhere.parquet"}
        with pytest.raises(CorpusConfigError, match="either one `file` or a `path`"):
            resolve_source(_config({"plos": entry}, {}), "validation", "plos", entry)


class TestResolvedSources:
    def test_reports_what_each_corpus_resolved_to(self):
        cfg = _config(
            {"plos": STRATIFIED_ENTRY, "ore": SINGLE_ENTRY}, {"plos": 3, "ore": 3}
        )
        resolved = resolved_sources(cfg, "validation")
        assert resolved["plos"]["revision"] == "corpus-v001"
        assert resolved["ore"]["revision"] == "main"
        assert resolved["plos"]["repo_id"] != resolved["ore"]["repo_id"]


class TestOptInCorpora:
    def test_an_optional_corpus_is_left_out_by_default(self, repo: Path):
        cfg = _config(
            {"plos": {**STRATIFIED_ENTRY, "optional": True}, "ore": SINGLE_ENTRY},
            {"plos": 3, "ore": 3},
        )
        assert [s.corpus for s in iter_corpus_sources(cfg, "validation")] == ["ore"]
        records = fetch_gold(cfg, "smoke", "validation", repo / "out")
        assert {record["corpus"] for record in records} == {"ore"}

    def test_naming_it_includes_it_alongside_the_defaults(self, repo: Path):
        cfg = _config(
            {"plos": {**STRATIFIED_ENTRY, "optional": True}, "ore": SINGLE_ENTRY},
            {"plos": 3, "ore": 3},
        )
        records = fetch_gold(cfg, "smoke", "validation", repo / "out", include=["plos"])
        assert {record["corpus"] for record in records} == {"ore", "plos"}

    def test_rejects_an_unknown_corpus_name(self):
        cfg = _config({"ore": SINGLE_ENTRY}, {"ore": 3})
        with pytest.raises(ValueError, match="not in split"):
            iter_corpus_sources(cfg, "validation", include=["nope"])


class TestFetchStratified:
    def _fetch(self, repo: Path, n: Optional[int], with_pdf: bool = False):
        cfg = _config({"plos": STRATIFIED_ENTRY}, {"plos": n})
        fetch = fetch_data if with_pdf else fetch_gold
        return fetch(cfg, "smoke", "validation", repo / "out")

    def test_a_sample_of_the_stratum_count_covers_every_stratum(self, repo: Path):
        records = self._fetch(repo, 3)
        assert len(records) == 3
        assert {r["record_id"].split("-")[0] for r in records} == set(STRATA)

    def test_reads_many_files_per_stratum(self, repo: Path):
        records = self._fetch(repo, None)
        assert len(records) == sum(STRATA.values())
        # `aaa` is spread over two files; a single-file read would find only three.
        assert len([r for r in records if r["record_id"].startswith("aaa")]) == 6

    def test_ignores_manifest_rows_of_another_split(self, repo: Path):
        records = self._fetch(repo, None)
        assert not [r for r in records if r["record_id"].endswith("other")]

    def test_nests_per_stratum(self, repo: Path):
        smaller = {r["record_id"] for r in self._fetch(repo, 3)}
        larger = {r["record_id"] for r in self._fetch(repo, 6)}
        assert smaller <= larger
        for stratum in STRATA:
            in_smaller = {i for i in smaller if i.startswith(stratum)}
            in_larger = {i for i in larger if i.startswith(stratum)}
            assert in_smaller <= in_larger

    def test_materialises_gold_xml_without_pdfs(self, repo: Path):
        records = self._fetch(repo, 3)
        for record in records:
            assert Path(record["xml_path"]).exists()
            assert "pdf_path" not in record

    def test_materialises_pdf_and_xml(self, repo: Path):
        records = self._fetch(repo, 3, with_pdf=True)
        for record in records:
            assert Path(record["pdf_path"]).read_bytes().startswith(b"%PDF-")
            assert Path(record["xml_path"]).read_text(encoding="utf-8").startswith("<article")

    def test_skips_documents_already_on_disk(self, repo: Path, monkeypatch):
        self._fetch(repo, 3)
        existing = sorted((repo / "out" / "validation" / "plos").glob("*.jats.xml"))[0]
        existing.write_text("<kept/>", encoding="utf-8")

        def _fail(*_args, **_kwargs):
            raise AssertionError("a document already on disk was read again")

        monkeypatch.setattr("benchmarks.fetch.iter_partitioned_rows", _fail)
        records = self._fetch(repo, 3)
        assert len(records) == 3
        assert existing.read_text(encoding="utf-8") == "<kept/>"

    def test_reads_only_the_columns_it_needs(self, repo: Path, monkeypatch):
        seen: List[Sequence[str]] = []
        original = pq.ParquetFile.read_row_group

        def _spy(self, index, columns=None, **kwargs):
            seen.append(list(columns or []))
            return original(self, index, columns=columns, **kwargs)

        monkeypatch.setattr(pq.ParquetFile, "read_row_group", _spy)
        self._fetch(repo, 3, with_pdf=True)
        assert seen
        for columns in seen:
            assert "doc" not in columns
            assert set(columns) == {"id", "pdf", "xml"}

    def test_gold_and_pdfs_come_from_the_same_sample_and_revision(self, repo: Path):
        cfg = _config({"plos": STRATIFIED_ENTRY, "ore": SINGLE_ENTRY},
                      {"plos": 3, "ore": 3})
        gold = fetch_gold(cfg, "smoke", "validation", repo / "out")
        data = fetch_data(cfg, "smoke", "validation", repo / "out")
        assert {(r["corpus"], r["record_id"]) for r in gold} == {
            (r["corpus"], r["record_id"]) for r in data
        }
        # And both read the corpus from the one place its entry resolves to.
        assert resolved_sources(cfg, "validation")["plos"]["revision"] == "corpus-v001"

    def test_reports_a_manifest_that_does_not_match_the_data(self, repo: Path):
        entry = {**STRATIFIED_ENTRY, "manifest_split": "test"}
        cfg = _config({"plos": entry}, {"plos": 3})
        with pytest.raises(CorpusConfigError, match="absent from"):
            fetch_gold(cfg, "smoke", "validation", repo / "out")

    def test_reports_a_manifest_whose_columns_are_named_differently(self, repo: Path):
        entry = {**STRATIFIED_ENTRY, "stratum": "publisher"}
        cfg = _config({"plos": entry}, {"plos": 3})
        with pytest.raises(CorpusConfigError, match="missing column"):
            fetch_gold(cfg, "smoke", "validation", repo / "out")


class TestFetchSingleFile:
    def test_samples_a_one_file_corpus_as_before(self, repo: Path):
        cfg = _config({"ore": SINGLE_ENTRY}, {"ore": 5})
        records = fetch_gold(cfg, "smoke", "validation", repo / "out")
        assert len(records) == 5
        # The same ids the positional sampling picks, in corpus order.
        expected = positional_ids([f"single-{i:02d}" for i in range(12)], 5, 42)
        assert [r["record_id"] for r in records] == expected

    def test_full_takes_everything(self, repo: Path):
        cfg = _config({"ore": SINGLE_ENTRY}, {"ore": None})
        assert len(fetch_gold(cfg, "smoke", "validation", repo / "out")) == 12


class TestFetchTrainingSource:
    def test_fetches_only_cc_by_corpora(self, repo: Path):
        cfg = {
            **_config(
                {"ore": SINGLE_ENTRY, "plos": STRATIFIED_ENTRY},
                {"ore": 3, "plos": 3},
            ),
            "cc_by_corpora": ["ore"],
        }
        records = fetch_training_source(cfg, "smoke", "validation", repo / "out")
        assert {record["corpus"] for record in records} == {"ore"}

    def test_refuses_a_corpus_outside_the_allow_list_even_if_opt_in(self, repo: Path):
        cfg = {
            **_config(
                {"plos": {**STRATIFIED_ENTRY, "optional": True}}, {"plos": 3}
            ),
            "cc_by_corpora": [],
        }
        assert not fetch_training_source(cfg, "smoke", "validation", repo / "out")
