"""What a fetch resolved, and what a generated pair was produced from.

The mode is known at fetch and gone by generation: fetch takes `--mode` and writes
documents, generation takes a directory and processes what it finds. So fetch
leaves a manifest beside the documents, and generation carries it forward into a
record beside the data it produced.

The manifest stays in the source tree, which is a local cache, and names the
documents the mode selected. The pair record is committed with the corpus and
carries only what reading a diff wants: the mode, the commit the revision resolved
to, and the seed. How many documents there are, and what each produced, is already
recorded per document by the quality record.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

LOGGER = logging.getLogger(__name__)

__all__ = [
    "PAIR_RECORD_FILENAME",
    "SOURCE_MANIFEST_FILENAME",
    "PairRecord",
    "SourceManifest",
    "read_source_manifest",
]

SOURCE_MANIFEST_FILENAME = "source.json"
PAIR_RECORD_FILENAME = "provenance.json"


def _dataset_json(
    repo_id: str, revision: str, commit: Optional[str], location: str
) -> Dict[str, Any]:
    dataset: Dict[str, Any] = {"repo_id": repo_id, "revision": revision}
    if commit:
        dataset["commit"] = commit
    dataset["location"] = location
    return dataset


@dataclasses.dataclass(frozen=True)
class SourceManifest:
    """What one fetch of one corpus resolved, written beside the documents."""

    corpus: str
    split: str
    mode: str
    seed: int
    repo_id: str
    revision: str
    location: str
    commit: Optional[str] = None
    requested_document_count: Optional[int] = None
    selected_document_ids: Sequence[str] = ()

    @staticmethod
    def get_file_path(corpus_dir: Path) -> Path:
        return corpus_dir / SOURCE_MANIFEST_FILENAME

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "corpus": self.corpus,
            "split": self.split,
            "mode": self.mode,
            "seed": self.seed,
            "dataset": _dataset_json(
                self.repo_id, self.revision, self.commit, self.location
            ),
            "requested_document_count": self.requested_document_count,
            "selected_document_ids": list(self.selected_document_ids),
        }

    def write(self, corpus_dir: Path) -> Path:
        file_path = self.get_file_path(corpus_dir)
        file_path.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return file_path

    def to_pair_record(self, model: str) -> "PairRecord":
        return PairRecord(
            corpus=self.corpus,
            model=model,
            split=self.split,
            mode=self.mode,
            seed=self.seed,
            repo_id=self.repo_id,
            revision=self.revision,
            location=self.location,
            commit=self.commit,
        )


@dataclasses.dataclass(frozen=True)
class PairRecord:
    """What one corpus and model's data was generated from, committed beside it."""

    corpus: str
    model: str
    split: str
    mode: str
    seed: int
    repo_id: str
    revision: str
    location: str
    commit: Optional[str] = None

    @staticmethod
    def get_file_path(pair_dir: Path) -> Path:
        return pair_dir / PAIR_RECORD_FILENAME

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "corpus": self.corpus,
            "model": self.model,
            "split": self.split,
            "mode": self.mode,
            "seed": self.seed,
            "dataset": _dataset_json(
                self.repo_id, self.revision, self.commit, self.location
            ),
        }

    def write(self, pair_dir: Path) -> Path:
        pair_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.get_file_path(pair_dir)
        file_path.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return file_path


def _manifest_from_json(json_dict: Dict[str, Any]) -> SourceManifest:
    dataset = json_dict.get("dataset") or {}
    return SourceManifest(
        corpus=json_dict["corpus"],
        split=json_dict["split"],
        mode=json_dict["mode"],
        seed=json_dict["seed"],
        repo_id=dataset.get("repo_id", ""),
        revision=dataset.get("revision", ""),
        location=dataset.get("location", ""),
        commit=dataset.get("commit"),
        requested_document_count=json_dict.get("requested_document_count"),
        selected_document_ids=json_dict.get("selected_document_ids") or (),
    )


def read_source_manifest(corpus_dir: Path) -> Optional[SourceManifest]:
    """What the fetch into this directory resolved, or None if it left no manifest.

    None rather than an error: generation runs against a source tree assembled by
    hand often enough — one document, to see what it produces — and refusing that
    would make a debugging run depend on a fetch.
    """
    file_path = SourceManifest.get_file_path(corpus_dir)
    if not file_path.is_file():
        return None
    try:
        return _manifest_from_json(json.loads(file_path.read_text(encoding="utf-8")))
    except (ValueError, KeyError):
        LOGGER.warning("ignoring unreadable source manifest: %s", file_path)
        return None
