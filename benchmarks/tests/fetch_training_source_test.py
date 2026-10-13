from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks import fetch
from benchmarks.fetch import fetch_data, fetch_training_source

_BASE_CONFIG: Dict[str, Any] = {
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


def _write_corpus(path: Path, ids: List[str]) -> None:
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


class TestSourceManifest:
    def test_writes_a_manifest_naming_the_mode_and_selection(self, repo: Path):
        out = repo / "out"
        fetch_training_source(_BASE_CONFIG, "smoke", "train", out)

        manifest = json.loads(
            (out / "train" / "ore" / "source.json").read_text(encoding="utf-8")
        )
        assert manifest["corpus"] == "ore"
        assert manifest["split"] == "train"
        assert manifest["mode"] == "smoke"
        assert manifest["seed"] == 42
        assert manifest["requested_document_count"] == 2
        assert len(manifest["selected_document_ids"]) == 2
        assert manifest["dataset"]["repo_id"] == "org/repo"
        assert manifest["dataset"]["revision"] == "main"

    def test_names_the_documents_the_mode_selected_rather_than_the_directory(
        self, repo: Path
    ):
        """A fetch only ever adds, so a smaller mode leaves the larger one's files.

        The manifest has to describe the selection, or `medium` holding 66
        documents reads as what `medium` asked for.
        """
        out = repo / "out"
        fetch_training_source(_BASE_CONFIG, "full", "train", out)
        fetch_training_source(_BASE_CONFIG, "smoke", "train", out)

        manifest = json.loads(
            (out / "train" / "ore" / "source.json").read_text(encoding="utf-8")
        )
        assert manifest["mode"] == "smoke"
        assert len(manifest["selected_document_ids"]) == 2
        assert len(list((out / "train" / "ore").glob("*.pdf"))) == 5

    def test_omits_a_commit_when_there_is_no_revision_to_resolve(self, repo: Path):
        out = repo / "out"
        fetch_training_source(_BASE_CONFIG, "smoke", "train", out)

        manifest = json.loads(
            (out / "train" / "ore" / "source.json").read_text(encoding="utf-8")
        )
        assert "commit" not in manifest["dataset"]


class TestRecordedSelection:
    def test_records_what_the_mode_selected(self, repo: Path):
        out = repo / "out"
        selection = repo / "selection"
        fetch_training_source(
            _BASE_CONFIG, "smoke", "train", out, selection_dir=selection
        )

        recorded = (selection / "train" / "ore.txt").read_text(encoding="utf-8").split()
        assert len(recorded) == 2
        manifest = json.loads(
            (out / "train" / "ore" / "source.json").read_text(encoding="utf-8")
        )
        assert sorted(manifest["selected_document_ids"]) == sorted(recorded)

    def test_a_later_larger_mode_keeps_the_earlier_one(self, repo: Path):
        out = repo / "out"
        selection = repo / "selection"
        fetch_training_source(
            _BASE_CONFIG, "smoke", "train", out, selection_dir=selection
        )
        first = (selection / "train" / "ore.txt").read_text(encoding="utf-8").split()

        fetch_training_source(
            _BASE_CONFIG, "full", "train", out, selection_dir=selection
        )
        second = (selection / "train" / "ore.txt").read_text(encoding="utf-8").split()

        assert second[:2] == first
        assert len(second) == 5

    def test_going_back_to_a_smaller_mode_evicts_nothing(self, repo: Path):
        out = repo / "out"
        selection = repo / "selection"
        fetch_training_source(
            _BASE_CONFIG, "full", "train", out, selection_dir=selection
        )
        full = (selection / "train" / "ore.txt").read_text(encoding="utf-8").split()

        fetch_training_source(
            _BASE_CONFIG, "smoke", "train", out, selection_dir=selection
        )

        assert (selection / "train" / "ore.txt").read_text(encoding="utf-8").split() == full

    def test_without_a_selection_directory_nothing_is_recorded(self, repo: Path):
        out = repo / "out"
        fetch_training_source(_BASE_CONFIG, "smoke", "train", out)
        assert not (repo / "selection").exists()


class _DatasetWasRead(Exception):
    """Raised instead of reading the dataset, so a test can assert it was not."""


class _UnreadableRepo:
    """A reader that refuses every call, standing in for having no credential."""

    @classmethod
    def from_env(cls) -> "_UnreadableRepo":
        return cls()

    def __getattr__(self, name: str):
        raise _DatasetWasRead(name)


def _refuse_to_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any read of the dataset raise, from here on.

    Called in the test body rather than from a fixture, because every one of these
    tests first has to leave behind the record a real fetch would have written.
    """
    monkeypatch.setattr(fetch, "RepoReader", _UnreadableRepo)


@pytest.fixture(name="fetched")
def _fetched(repo: Path) -> Path:
    """A tree, a selection and a manifest, left by a fetch that did read."""
    fetch_training_source(
        _BASE_CONFIG, "smoke", "train", repo / "out", selection_dir=repo / "selection"
    )
    return repo


def _recorded_fetch(
    cfg: Dict[str, Any], repo: Path, mode: str = "smoke"
) -> List[Dict[str, str]]:
    return fetch_training_source(
        cfg, mode, "train", repo / "out",
        selection_dir=repo / "selection", recheck_upstream=False,
    )


def _manifest(repo: Path) -> Dict[str, Any]:
    return json.loads(
        (repo / "out" / "train" / "ore" / "source.json").read_text(encoding="utf-8")
    )


class TestTakingTheRecordRatherThanReadingUpstream:
    def test_a_tree_its_manifest_still_describes_is_taken_as_it_is(
        self, fetched: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _refuse_to_read(monkeypatch)
        records = _recorded_fetch(_BASE_CONFIG, fetched)
        assert len(records) == 2
        for record in records:
            assert Path(record["pdf_path"]).exists()
            assert Path(record["xml_path"]).exists()

    def test_the_manifest_is_left_byte_for_byte(
        self, fetched: Path, monkeypatch: pytest.MonkeyPatch
    ):
        path = fetched / "out" / "train" / "ore" / "source.json"
        before = path.read_bytes()
        _refuse_to_read(monkeypatch)
        _recorded_fetch(_BASE_CONFIG, fetched)
        assert path.read_bytes() == before

    def test_a_mode_asking_more_than_the_corpus_holds_is_still_answered(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """`ore` at `medium` asks 50 of a corpus holding 37, so the recorded list is
        shorter than `n` and nothing about its length rules out appending."""
        cfg = {
            **_BASE_CONFIG,
            "sampling": {**_BASE_CONFIG["sampling"], "wide": {"ore": 9}},
        }
        fetch_training_source(
            cfg, "wide", "train", repo / "out", selection_dir=repo / "selection"
        )
        assert _manifest(repo)["requested_document_count"] == 9
        assert len(_manifest(repo)["selected_document_ids"]) == 5

        _refuse_to_read(monkeypatch)
        assert len(_recorded_fetch(cfg, repo, mode="wide")) == 5

    def test_the_benchmark_fetch_keeps_its_own_rule(self, fetched: Path):
        """`fetch_data` has no recorded selection and no allow-list, and gains
        neither: every recorded benchmark aggregate depends on what it does."""
        records = fetch_data(_BASE_CONFIG, "smoke", "train", fetched / "bench")
        assert {record["corpus"] for record in records} == {"ore", "biorxiv"}


class TestADocumentWhoseSourceRowHasGone:
    @pytest.fixture(name="shrunk")
    def _shrunk(self, repo: Path) -> Path:
        """Five documents fetched, then two of them removed upstream."""
        fetch_training_source(
            _BASE_CONFIG, "full", "train", repo / "out",
            selection_dir=repo / "selection",
        )
        _write_corpus(repo / "ore" / "train.parquet", [f"ore{i}" for i in range(3)])
        fetch_training_source(
            _BASE_CONFIG, "full", "train", repo / "out",
            selection_dir=repo / "selection",
        )
        return repo

    def test_it_is_recorded_as_missing_although_its_files_are_still_there(
        self, shrunk: Path
    ):
        manifest = _manifest(shrunk)
        assert len(manifest["selected_document_ids"]) == 3
        assert sorted(manifest["missing_document_ids"]) == ["ore3", "ore4"]
        for record_id in manifest["missing_document_ids"]:
            assert (shrunk / "out" / "train" / "ore" / f"{record_id}.pdf").exists()

    def test_the_recorded_path_does_not_promote_it(
        self, shrunk: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _refuse_to_read(monkeypatch)
        records = _recorded_fetch(_BASE_CONFIG, shrunk, mode="full")
        assert sorted(record["record_id"] for record in records) == [
            "ore0", "ore1", "ore2",
        ]
        assert sorted(_manifest(shrunk)["missing_document_ids"]) == ["ore3", "ore4"]


class TestWhenTheRecordStopsAnswering:
    @staticmethod
    def _assert_reads(
        cfg: Dict[str, Any], repo: Path, monkeypatch: pytest.MonkeyPatch,
        mode: str = "smoke",
    ) -> None:
        _refuse_to_read(monkeypatch)
        with pytest.raises(_DatasetWasRead):
            _recorded_fetch(cfg, repo, mode=mode)

    def test_a_raised_mode(self, fetched: Path, monkeypatch: pytest.MonkeyPatch):
        sampling = {**_BASE_CONFIG["sampling"], "smoke": {"ore": 4}}
        self._assert_reads({**_BASE_CONFIG, "sampling": sampling}, fetched, monkeypatch)

    def test_a_lowered_mode(self, fetched: Path, monkeypatch: pytest.MonkeyPatch):
        sampling = {**_BASE_CONFIG["sampling"], "smoke": {"ore": 1}}
        self._assert_reads({**_BASE_CONFIG, "sampling": sampling}, fetched, monkeypatch)

    def test_a_bumped_revision(self, fetched: Path, monkeypatch: pytest.MonkeyPatch):
        dataset = {**_BASE_CONFIG["dataset"], "revision": "v2"}
        self._assert_reads({**_BASE_CONFIG, "dataset": dataset}, fetched, monkeypatch)

    def test_a_changed_seed(self, fetched: Path, monkeypatch: pytest.MonkeyPatch):
        self._assert_reads(
            {**_BASE_CONFIG, "seeds": {"sample": 7}}, fetched, monkeypatch
        )

    def test_a_corpus_repointed_at_another_file(
        self, fetched: Path, monkeypatch: pytest.MonkeyPatch
    ):
        dataset = _BASE_CONFIG["dataset"]
        splits = {"train": {
            **dataset["splits"]["train"],
            "ore": {"file": "ore/other.parquet", "id_column": "ppr_id"},
        }}
        self._assert_reads(
            {**_BASE_CONFIG, "dataset": {**dataset, "splits": splits}},
            fetched, monkeypatch,
        )

    def test_a_selected_document_deleted_from_the_tree(
        self, fetched: Path, monkeypatch: pytest.MonkeyPatch
    ):
        record_id = _manifest(fetched)["selected_document_ids"][0]
        (fetched / "out" / "train" / "ore" / f"{record_id}.pdf").unlink()
        self._assert_reads(_BASE_CONFIG, fetched, monkeypatch)

    def test_a_selection_edited_within_the_mode_s_prefix(
        self, fetched: Path, monkeypatch: pytest.MonkeyPatch
    ):
        path = fetched / "selection" / "train" / "ore.txt"
        path.write_text("ore9\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
        self._assert_reads(_BASE_CONFIG, fetched, monkeypatch)

    def test_no_tree_at_all(self, repo: Path, monkeypatch: pytest.MonkeyPatch):
        self._assert_reads(_BASE_CONFIG, repo, monkeypatch)

    def test_appending_beyond_the_prefix_does_not(
        self, fetched: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """It does not change what the mode names, so there is nothing to re-ask."""
        path = fetched / "selection" / "train" / "ore.txt"
        path.write_text(path.read_text(encoding="utf-8") + "ore9\n", encoding="utf-8")
        _refuse_to_read(monkeypatch)
        assert len(_recorded_fetch(_BASE_CONFIG, fetched)) == 2

    def test_the_fetch_command_reads_whatever_the_record_says(
        self, fetched: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """`recheck_upstream` defaults to asking, so fetching stays fetching."""
        _refuse_to_read(monkeypatch)
        with pytest.raises(_DatasetWasRead):
            fetch_training_source(
                _BASE_CONFIG, "smoke", "train", fetched / "out",
                selection_dir=fetched / "selection",
            )
