from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from benchmarks.corpus_source import (
    CorpusConfigError,
    ReleasedCorpus,
    RepoReader,
    resolve_source,
)
from benchmarks.fetch import (
    RELEASE_CACHE_FILE,
    fetch_data,
    fetch_gold,
    fetch_training_source,
    resolved_sources,
)
from benchmarks.sampling import release_ids

PUBLIC = "org/dataset-public"
RESTRICTED = "org/dataset-restricted"

PUBLIC_FIRST = [f"pub-{i:02d}" for i in range(12)]
RESTRICTED_FIRST = [f"res-{i:02d}" for i in range(8)]


def _write_release(root: Path, repo_id: str, release: str, ids: Sequence[str]) -> None:
    path = root / repo_id / "releases" / f"{release}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as release_file:
        writer = csv.writer(release_file)
        writer.writerow(["corpus", "id", "split"])
        writer.writerows(["corpus", record_id, "validation"] for record_id in ids)
        # Another split's row, to prove a release is read per split.
        writer.writerow(["corpus", f"{repo_id}-train", "train"])


def _write_data(
    root: Path, repo_id: str, ids: Sequence[str], gold: Optional[Dict[str, str]] = None
) -> None:
    """The split in two shards, so a single-file read would miss some."""
    folder = root / repo_id / "pdf-jats" / "validation"
    for old in folder.glob("*.parquet"):
        old.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    half = (len(ids) + 1) // 2
    for shard, chunk in enumerate((ids[:half], ids[half:])):
        if not chunk:
            continue
        pq.write_table(pa.table({
            "id": list(chunk),
            "xml": [(gold or {}).get(i, f"<article id='{i}'/>") for i in chunk],
            "pdf": [f"%PDF-{i}".encode() for i in chunk],
        }), folder / f"validation-{shard:05d}.parquet", row_group_size=2)


def _source(repo_id: str, revision: str, commit: str) -> dict:
    return {"repo_id": repo_id, "revision": revision, "commit": commit}


def _config(
    sources: List[dict], n: Optional[int] = 6, sizes: Optional[dict] = None
) -> dict:
    return {
        "dataset": {"splits": {"validation": {
            "corpus": {"sources": sources, "variant": "v1"},
        }}},
        "sampling": sizes or {"smoke": {"corpus": n}},
        "seeds": {"sample": 42},
    }


BOTH = [_source(PUBLIC, "v1.0.0", "c-pub-1"), _source(RESTRICTED, "v1.0.0", "c-res-1")]


@pytest.fixture(name="root")
def _root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repos"
    _write_release(root, PUBLIC, "v1.0.0", PUBLIC_FIRST)
    _write_data(root, PUBLIC, PUBLIC_FIRST)
    _write_release(root, RESTRICTED, "v1.0.0", RESTRICTED_FIRST)
    _write_data(root, RESTRICTED, RESTRICTED_FIRST)
    monkeypatch.setenv("BENCH_LOCAL_PARQUET_DIR", str(root))
    return root


def _ids(records: Sequence[Dict[str, str]]) -> List[str]:
    return [record["record_id"] for record in records]


class TestResolveReleased:
    def test_reads_each_source_in_order(self):
        source = resolve_source({}, "validation", "corpus", {"sources": BOTH})
        assert isinstance(source, ReleasedCorpus)
        assert [repo.repo_id for repo in source.repos] == [PUBLIC, RESTRICTED]

    def test_requires_a_commit(self):
        entry = {"sources": [{"repo_id": PUBLIC, "revision": "v1.0.0"}]}
        with pytest.raises(CorpusConfigError, match="missing commit"):
            resolve_source({}, "validation", "corpus", entry)

    def test_rejects_sources_mixed_with_a_file(self):
        entry = {"sources": BOTH, "file": "x.parquet"}
        with pytest.raises(CorpusConfigError, match="sources only"):
            resolve_source({}, "validation", "corpus", entry)

    def test_records_every_source_with_its_commit(self):
        resolved = resolved_sources(_config(BOTH), "validation")["corpus"]
        assert resolved["sources"] == BOTH


class TestSelection:
    def test_public_and_restricted_take_turns(self, root: Path):
        records = fetch_gold(_config(BOTH), "smoke", "validation", root / "out")
        first = [PUBLIC_FIRST, RESTRICTED_FIRST]
        assert _ids(records) == release_ids(first, first, 6, 42)
        assert [r["repo_id"] for r in records].count(RESTRICTED) == 3

    def test_is_identical_across_runs(self, root: Path, tmp_path: Path):
        first = fetch_data(_config(BOTH), "smoke", "validation", root / "out")
        second = fetch_data(_config(BOTH), "smoke", "validation", tmp_path / "again")
        assert [
            (r["record_id"], r["repo_id"], r["xml_sha256"], r["pdf_sha256"]) for r in first
        ] == [
            (r["record_id"], r["repo_id"], r["xml_sha256"], r["pdf_sha256"]) for r in second
        ]

    def test_smaller_modes_are_prefixes_of_larger_ones(self, root: Path):
        sizes = {"smoke": {"corpus": 5}, "small": {"corpus": 10}, "full": {"corpus": None}}
        cfg = _config(BOTH, sizes=sizes)
        smoke, small, full = (
            _ids(fetch_gold(cfg, mode, "validation", root / "out"))
            for mode in ("smoke", "small", "full")
        )
        assert small[:5] == smoke and full[:10] == small
        assert len(full) == len(PUBLIC_FIRST) + len(RESTRICTED_FIRST)

    def test_reads_every_shard(self, root: Path):
        records = fetch_gold(_config(BOTH, None), "smoke", "validation", root / "out")
        assert set(_ids(records)) == set(PUBLIC_FIRST + RESTRICTED_FIRST)

    def test_a_removed_document_lets_the_next_one_in(self, root: Path):
        before = fetch_gold(_config(BOTH), "smoke", "validation", root / "out")
        removed = before[0]["record_id"]
        _write_release(root, PUBLIC, "v2.0.0", [i for i in PUBLIC_FIRST if i != removed])
        sources = [_source(PUBLIC, "v2.0.0", "c-pub-2"), BOTH[1]]
        after = fetch_gold(_config(sources), "smoke", "validation", root / "out")
        assert removed not in _ids(after)
        assert len(after) == len(before)

    def test_a_moved_document_keeps_its_place_and_is_read_from_its_new_repo(
        self, root: Path
    ):
        before = fetch_gold(_config(BOTH), "smoke", "validation", root / "out")
        moved = next(r["record_id"] for r in before if r["repo_id"] == PUBLIC)
        _write_release(root, PUBLIC, "v2.0.0", [i for i in PUBLIC_FIRST if i != moved])
        _write_release(root, RESTRICTED, "v1.1.0", RESTRICTED_FIRST + [moved])
        _write_data(root, RESTRICTED, RESTRICTED_FIRST + [moved])
        sources = [
            _source(PUBLIC, "v2.0.0", "c-pub-2"), _source(RESTRICTED, "v1.1.0", "c-res-2"),
        ]
        after = fetch_gold(_config(sources), "smoke", "validation", root / "out")
        assert _ids(after) == _ids(before)
        assert next(r for r in after if r["record_id"] == moved)["repo_id"] == RESTRICTED

    def test_rejects_an_unknown_mode(self, root: Path):
        with pytest.raises(ValueError, match="Unknown mode"):
            fetch_gold(_config(BOTH), "huge", "validation", root / "out")


class TestCache:
    def test_a_new_commit_replaces_corrected_gold(self, root: Path):
        records = fetch_gold(_config(BOTH, None), "smoke", "validation", root / "out")
        record_id = PUBLIC_FIRST[0]
        _write_release(root, PUBLIC, "v2.0.0", PUBLIC_FIRST)
        _write_data(root, PUBLIC, PUBLIC_FIRST, gold={record_id: "<corrected/>"})
        sources = [_source(PUBLIC, "v2.0.0", "c-pub-2"), BOTH[1]]
        records = fetch_gold(_config(sources, None), "smoke", "validation", root / "out")
        record = next(r for r in records if r["record_id"] == record_id)
        assert Path(record["xml_path"]).read_text(encoding="utf-8") == "<corrected/>"

    def test_a_changed_file_on_disk_is_read_again(self, root: Path):
        record = fetch_gold(_config(BOTH), "smoke", "validation", root / "out")[0]
        Path(record["xml_path"]).write_text("<tampered/>", encoding="utf-8")
        again = fetch_gold(_config(BOTH), "smoke", "validation", root / "out")[0]
        assert Path(again["xml_path"]).read_text(encoding="utf-8") != "<tampered/>"
        assert again["xml_sha256"] == record["xml_sha256"]

    def test_a_cached_document_is_not_read_again(self, root: Path, monkeypatch):
        fetch_data(_config(BOTH), "smoke", "validation", root / "out")

        def _fail(*_args, **_kwargs):
            raise AssertionError("a cached document was read again")

        monkeypatch.setattr("benchmarks.fetch.iter_release_rows", _fail)
        assert len(fetch_data(_config(BOTH), "smoke", "validation", root / "out")) == 6
        assert (root / "out" / "validation" / "corpus" / RELEASE_CACHE_FILE).exists()


class TestErrors:
    def test_a_moved_tag_fails(self, root: Path):
        with patch.object(RepoReader, "resolve_commit", return_value="other"):
            with pytest.raises(CorpusConfigError, match="tag was moved"):
                fetch_gold(_config(BOTH), "smoke", "validation", root / "out")

    def test_no_access_to_a_restricted_repo(self, monkeypatch):
        monkeypatch.delenv("BENCH_LOCAL_PARQUET_DIR", raising=False)
        # Built without a response: its HTTP client type varies by huggingface_hub version.
        error = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        with patch("benchmarks.corpus_source.HfApi") as api:
            api.return_value.repo_info.side_effect = error
            with pytest.raises(CorpusConfigError, match="no access"):
                fetch_gold(_config(BOTH), "smoke", "validation", Path("unused"))

    def test_a_missing_release_list_fails(self, root: Path):
        sources = [_source(PUBLIC, "v9.0.0", "c"), BOTH[1]]
        with pytest.raises(CorpusConfigError, match="has no releases/v9.0.0.csv"):
            fetch_gold(_config(sources), "smoke", "validation", root / "out")

    def test_a_listed_document_missing_from_the_data_fails(self, root: Path):
        _write_data(root, PUBLIC, PUBLIC_FIRST[1:])
        with pytest.raises(CorpusConfigError, match="absent from"):
            fetch_gold(_config(BOTH, None), "smoke", "validation", root / "out")

    def test_a_document_in_two_repos_fails(self, root: Path):
        _write_release(root, RESTRICTED, "v1.0.0", RESTRICTED_FIRST + [PUBLIC_FIRST[0]])
        with pytest.raises(CorpusConfigError, match="listed by both"):
            fetch_gold(_config(BOTH), "smoke", "validation", root / "out")

    def test_training_source_fetch_refuses_it(self, root: Path):
        cfg = {**_config(BOTH), "cc_by_corpora": ["corpus"]}
        with pytest.raises(CorpusConfigError, match="training-source"):
            fetch_training_source(cfg, "smoke", "validation", root / "out")
