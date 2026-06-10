from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarks.predictions_store import (
    LocalPredictionsStore,
    RepoPredictionsStore,
    _read_done_ids_from_manifest,
)


def _write_manifest(path: Path, entries: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )


class TestReadDoneIdsFromManifest:
    def test_returns_ok_entries(self):
        text = (
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}) + "\n"
            + json.dumps({"corpus": "biorxiv", "record_id": "r2", "status": "error"}) + "\n"
        )
        assert _read_done_ids_from_manifest(text) == {("biorxiv", "r1")}

    def test_skips_blank_lines(self):
        text = json.dumps({"corpus": "ore", "record_id": "r1", "status": "ok"}) + "\n\n"
        assert _read_done_ids_from_manifest(text) == {("ore", "r1")}

    def test_empty_manifest(self):
        assert _read_done_ids_from_manifest("") == set()


class TestLocalPredictionsStore:
    def _store(self, tmp_path: Path) -> LocalPredictionsStore:
        return LocalPredictionsStore(runs_dir=tmp_path / "runs")

    def test_get_done_ids_returns_empty_when_no_manifest(self, tmp_path: Path):
        assert self._store(tmp_path).get_done_ids("grobid", "0.9", "default", "train") == set()

    def test_get_done_ids_reads_manifest(self, tmp_path: Path):
        store = self._store(tmp_path)
        # pylint: disable-next=protected-access
        run_dir = store._run_dir("grobid", "0.9", "default", "train")
        _write_manifest(run_dir / "predictions" / "manifest.jsonl", [
            {"corpus": "biorxiv", "record_id": "r1", "status": "ok"},
            {"corpus": "biorxiv", "record_id": "r2", "status": "error"},
        ])
        assert store.get_done_ids("grobid", "0.9", "default", "train") == {("biorxiv", "r1")}

    def test_fetch_is_noop(self, tmp_path: Path):
        store = self._store(tmp_path)
        store.fetch("grobid", "0.9", "default", "train", tmp_path / "local", {})

    def test_push_is_noop(self, tmp_path: Path):
        store = self._store(tmp_path)
        store.push("grobid", "0.9", "default", "train", tmp_path / "local", {}, {})

    def test_read_metadata_returns_none_when_absent(self, tmp_path: Path):
        assert self._store(tmp_path).read_metadata("grobid", "0.9", "default", "train") is None

    def test_read_metadata_returns_dict(self, tmp_path: Path):
        store = self._store(tmp_path)
        # pylint: disable-next=protected-access
        run_dir = store._run_dir("grobid", "0.9", "default", "train")
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text(json.dumps({"mode": "smoke"}))
        assert store.read_metadata("grobid", "0.9", "default", "train") == {"mode": "smoke"}


def _make_git_mock(stdout_by_args: dict | None = None) -> MagicMock:
    """Return a mock for subprocess.run that simulates git show responses."""
    def _run(cmd, **_kwargs):
        result = MagicMock()
        key = tuple(cmd[3:]) if len(cmd) > 3 else ()
        if stdout_by_args and key in stdout_by_args:
            result.returncode = 0
            result.stdout = stdout_by_args[key]
        else:
            result.returncode = 128
            result.stdout = ""
        return result
    return MagicMock(side_effect=_run)


class TestRepoPredictionsStore:
    def _store(self, tmp_path: Path) -> RepoPredictionsStore:
        repo = tmp_path / "repo"
        repo.mkdir()
        return RepoPredictionsStore(repo_dir=repo)

    def test_get_done_ids_returns_empty_when_manifest_missing(self, tmp_path: Path):
        store = self._store(tmp_path)
        with patch("subprocess.run", _make_git_mock({})):
            result = store.get_done_ids("grobid", "0.9", "default", "train")
        assert result == set()

    def test_get_done_ids_parses_manifest_from_repo(self, tmp_path: Path):
        store = self._store(tmp_path)
        manifest_text = (
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}) + "\n"
        )
        key = ("show", "HEAD:grobid/0.9/default/train/manifest.jsonl")
        with patch("subprocess.run", _make_git_mock({key: manifest_text})):
            result = store.get_done_ids("grobid", "0.9", "default", "train")
        assert result == {("biorxiv", "r1")}

    def test_fetch_copies_tei_xml_files(self, tmp_path: Path):
        store = self._store(tmp_path)
        src = store.repo_dir / "grobid/0.9/default/biorxiv/v1/train"
        src.mkdir(parents=True)
        (src / "doc1.tei.xml").write_text("<tei/>")
        local_dir = tmp_path / "local"
        with patch.object(store, "_git"):
            store.fetch("grobid", "0.9", "default", "train", local_dir, {"biorxiv": "v1"})
        assert (local_dir / "predictions" / "biorxiv" / "doc1.tei.xml").read_text() == "<tei/>"

    def test_fetch_copies_manifest(self, tmp_path: Path):
        store = self._store(tmp_path)
        manifest_dir = store.repo_dir / "grobid/0.9/default/train"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.jsonl").write_text('{"status":"ok"}\n')
        local_dir = tmp_path / "local"
        with patch.object(store, "_git"):
            store.fetch("grobid", "0.9", "default", "train", local_dir, {})
        assert (local_dir / "predictions" / "manifest.jsonl").exists()

    def test_push_copies_files_and_commits(self, tmp_path: Path):
        store = self._store(tmp_path)
        local_dir = tmp_path / "local"
        corpus_dir = local_dir / "predictions" / "biorxiv"
        corpus_dir.mkdir(parents=True)
        (corpus_dir / "doc1.tei.xml").write_text("<tei/>")

        git_calls = []

        def fake_git(*args, **_kw):
            git_calls.append(args)
            result = MagicMock()
            result.returncode = 1  # simulate diff --cached returning 1 (changes present)
            result.stdout = ""
            return result

        with patch.object(store, "_git", side_effect=fake_git):
            store.push("grobid", "0.9", "default", "train", local_dir,
                       {"biorxiv": "v1"}, {"tool": "grobid", "version": "0.9"})

        dest = store.repo_dir / "grobid/0.9/default/biorxiv/v1/train/doc1.tei.xml"
        assert dest.exists()
        assert any("commit" in str(c) for c in git_calls)
        assert any("push" in str(c) for c in git_calls)

    def test_push_writes_metadata_json(self, tmp_path: Path):
        store = self._store(tmp_path)
        local_dir = tmp_path / "local"
        (local_dir / "predictions").mkdir(parents=True)

        with patch.object(store, "_git", return_value=MagicMock(returncode=0)):
            store.push("grobid", "0.9", "default", "train", local_dir, {},
                       {"tool": "grobid", "version": "0.9", "mode": "smoke"})

        meta_path = store.repo_dir / "grobid/0.9/default/train/metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["tool"] == "grobid"
        assert "generated_at" in meta

    def test_read_metadata_returns_none_when_absent(self, tmp_path: Path):
        store = self._store(tmp_path)
        with patch("subprocess.run", _make_git_mock({})):
            assert store.read_metadata("grobid", "0.9", "default", "train") is None

    def test_read_metadata_parses_json(self, tmp_path: Path):
        store = self._store(tmp_path)
        key = ("show", "HEAD:grobid/0.9/default/train/metadata.json")
        payload = json.dumps({"mode": "medium", "image": "grobid:0.9"})
        with patch("subprocess.run", _make_git_mock({key: payload})):
            result = store.read_metadata("grobid", "0.9", "default", "train")
        assert result == {"mode": "medium", "image": "grobid:0.9"}
