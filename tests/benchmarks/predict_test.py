from __future__ import annotations

import json
from pathlib import Path

from benchmarks.predict import _append_manifest, _load_done


class TestLoadDone:
    def test_should_return_empty_set_when_manifest_absent(self, tmp_path: Path):
        assert _load_done(tmp_path) == set()

    def test_should_include_ok_record(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}) + "\n"
        )
        assert _load_done(tmp_path) == {("biorxiv", "r1")}

    def test_should_exclude_error_record(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "error"}) + "\n"
        )
        assert _load_done(tmp_path) == set()

    def test_should_handle_mixed_ok_and_error(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        lines = [
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}),
            json.dumps({"corpus": "biorxiv", "record_id": "r2", "status": "error"}),
            json.dumps({"corpus": "biorxiv", "record_id": "r3", "status": "ok"}),
        ]
        manifest.write_text("\n".join(lines) + "\n")
        assert _load_done(tmp_path) == {("biorxiv", "r1"), ("biorxiv", "r3")}

    def test_should_ignore_blank_lines(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}) + "\n\n"
        )
        assert _load_done(tmp_path) == {("biorxiv", "r1")}


class TestAppendManifest:
    def test_should_create_parent_dirs_and_write_entry(self, tmp_path: Path):
        entry = {"corpus": "biorxiv", "record_id": "r1", "status": "ok", "elapsed_ms": 500}
        _append_manifest(tmp_path, entry)
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        assert manifest.exists()
        written = json.loads(manifest.read_text().strip())
        assert written == entry

    def test_should_append_successive_entries(self, tmp_path: Path):
        _append_manifest(tmp_path, {"corpus": "biorxiv", "record_id": "r1", "status": "ok"})
        _append_manifest(tmp_path, {"corpus": "biorxiv", "record_id": "r2", "status": "error"})
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        lines = [line for line in manifest.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["record_id"] == "r1"
        assert json.loads(lines[1])["record_id"] == "r2"
