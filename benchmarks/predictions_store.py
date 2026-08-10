from __future__ import annotations

import datetime
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def _read_done_ids_from_manifest(manifest_text: str) -> set:
    done = set()
    for line in manifest_text.splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("status") == "ok":
                done.add((entry["corpus"], entry["record_id"]))
    return done


@dataclass
class LocalPredictionsStore:
    """Predictions stored on the local filesystem under runs_dir/baselines/."""

    runs_dir: Path

    def _run_dir(self, tool: str, version: str, profile: str, split: str) -> Path:
        return self.runs_dir / "baselines" / tool / version / profile / split

    def get_done_ids(
        self, tool: str, version: str, profile: str, split: str,
        corpus_variants: Optional[dict] = None,
    ) -> set:
        # Variants do not appear in this layout: the manifest and the predictions
        # it describes are the same directory, so they cannot disagree.
        del corpus_variants
        manifest = self._run_dir(tool, version, profile, split) / "predictions" / "manifest.jsonl"
        if not manifest.exists():
            return set()
        return _read_done_ids_from_manifest(manifest.read_text(encoding="utf-8"))

    def fetch(
        self, tool: str, version: str, profile: str, split: str,
        local_dir: Path, corpus_variants: dict,
    ) -> None:
        pass  # predictions already live at local_dir

    def push(
        self, tool: str, version: str, profile: str, split: str,
        local_dir: Path, corpus_variants: dict, metadata: dict,
    ) -> None:
        pass  # no remote to push to

    def read_metadata(
        self, tool: str, version: str, profile: str, split: str,
    ) -> Optional[dict]:
        path = self._run_dir(tool, version, profile, split) / "metadata.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RepoPredictionsStore:
    """Predictions stored in a checked-out sciencebeam-eval-predictions git repo."""

    repo_dir: Path

    def _prefix(self, tool: str, version: str, profile: str) -> str:
        return f"{tool}/{version}/{profile}"

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repo_dir), *args],
            capture_output=True, text=True, check=check,
        )

    def get_done_ids(
        self, tool: str, version: str, profile: str, split: str,
        corpus_variants: Optional[dict] = None,
    ) -> set:
        """Records already predicted and still retrievable for these variants.

        The manifest is per split and says nothing about variants, while the
        predictions themselves are filed under one. Trusting the manifest alone
        therefore reports a record as done that a variant bump has made
        unreachable — and since a bump means "this is a different corpus now",
        that turns re-prediction into a run with no predictions at all. Passing
        the variants a run will use is what keeps the two answers to "have we got
        this one?" from disagreeing.
        """
        path = f"{self._prefix(tool, version, profile)}/{split}/manifest.jsonl"
        result = self._git("show", f"HEAD:{path}", check=False)
        if result.returncode != 0:
            return set()
        done = _read_done_ids_from_manifest(result.stdout)
        if corpus_variants is None:
            return done
        stored = self._stored_ids(
            self._prefix(tool, version, profile), split, corpus_variants
        )
        return done & stored

    def _stored_ids(self, prefix: str, split: str, corpus_variants: dict) -> set:
        """(corpus, record_id) for every prediction filed under these variants."""
        suffix = ".tei.xml"
        stored = set()
        for corpus, variant in corpus_variants.items():
            listing = self._git(
                "ls-tree", "-r", "--name-only", "HEAD",
                f"{prefix}/{corpus}/{variant}/{split}/", check=False,
            )
            if listing.returncode != 0:
                continue
            for line in listing.stdout.splitlines():
                name = line.strip().rsplit("/", 1)[-1]
                if name.endswith(suffix):
                    stored.add((corpus, name[: -len(suffix)]))
        return stored

    def fetch(
        self, tool: str, version: str, profile: str, split: str,
        local_dir: Path, corpus_variants: dict,
    ) -> None:
        prefix = self._prefix(tool, version, profile)
        self._git("sparse-checkout", "add", prefix)
        self._git("checkout")
        for corpus, variant in corpus_variants.items():
            src = self.repo_dir / prefix / corpus / variant / split
            if not src.is_dir():
                continue
            dest = local_dir / "predictions" / corpus
            dest.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.tei.xml"):
                shutil.copy2(f, dest / f.name)
        manifest_src = self.repo_dir / prefix / split / "manifest.jsonl"
        if manifest_src.exists():
            manifest_dest = local_dir / "predictions" / "manifest.jsonl"
            manifest_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_src, manifest_dest)

    def _copy_predictions_to_repo(
        self, dest_base: Path, split: str, local_dir: Path, corpus_variants: dict,
    ) -> None:
        for corpus, variant in corpus_variants.items():
            src_corpus = local_dir / "predictions" / corpus
            if not src_corpus.is_dir():
                continue
            dest_corpus = dest_base / corpus / variant / split
            dest_corpus.mkdir(parents=True, exist_ok=True)
            for f in src_corpus.glob("*.tei.xml"):
                shutil.copy2(f, dest_corpus / f.name)
        manifest_src = local_dir / "predictions" / "manifest.jsonl"
        if manifest_src.exists():
            dest_split = dest_base / split
            dest_split.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_src, dest_split / "manifest.jsonl")

    def push(
        self, tool: str, version: str, profile: str, split: str,
        local_dir: Path, corpus_variants: dict, metadata: dict,
    ) -> None:
        dest_base = self.repo_dir / self._prefix(tool, version, profile)
        self._copy_predictions_to_repo(dest_base, split, local_dir, corpus_variants)
        dest_split = dest_base / split
        dest_split.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        (dest_split / "metadata.json").write_text(
            json.dumps({**metadata, "generated_at": ts}, indent=2), encoding="utf-8"
        )
        self._git("config", "user.name", "github-actions[bot]")
        self._git("config", "user.email", "github-actions[bot]@users.noreply.github.com")
        self._git("add", "--sparse", ".")
        if self._git("diff", "--cached", "--quiet", check=False).returncode != 0:
            label = metadata.get("image", f"{tool}:{version}")
            self._git("commit", "-m",
                      f"Update {tool}/{version}/{profile} {split} predictions ({label})")
            self._git("push")
        else:
            LOGGER.info("No changes to push for %s/%s/%s", tool, version, profile)

    def read_metadata(
        self, tool: str, version: str, profile: str, split: str,
    ) -> Optional[dict]:
        path = f"{self._prefix(tool, version, profile)}/{split}/metadata.json"
        result = self._git("show", f"HEAD:{path}", check=False)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
