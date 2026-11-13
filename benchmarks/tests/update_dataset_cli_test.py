from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import pytest
import yaml

from benchmarks.corpus_source import ReleaseRepo, RepoReader
from benchmarks.update_dataset_cli import (
    RepoUpdate,
    apply_updates,
    find_updates,
    parse_version,
    pinned_repos,
    report,
)

REPO = "org/dataset"
FIRST = [f"doc-{i:02d}" for i in range(10)]
SECOND = [i for i in FIRST if i != "doc-01"] + ["doc-10"]

CONFIG_TEXT = """\
dataset:
  repos:
    public: &public
      repo_id: org/dataset
      revision: v1.0.0
      commit: aaa
  splits:
    validation:
      corpus:
        sources: [*public]
sampling:
  smoke:
    corpus: 5
  full:
    corpus: null
seeds:
  sample: 42
"""


class _Api:
    def __init__(self, commits_by_tag: Dict[str, str]):
        self.commits_by_tag = commits_by_tag

    def list_repo_refs(self, _repo_id, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.commits_by_tag])

    def repo_info(self, _repo_id, revision, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.commits_by_tag[revision])


def _write_csv(path: Path, header: Sequence[str], rows: List[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture(name="reader")
def _reader(tmp_path: Path) -> RepoReader:
    releases = tmp_path / REPO / "releases"
    for release, ids in (("v1.0.0", FIRST), ("v2.0.0", SECOND)):
        _write_csv(
            releases / f"{release}.csv", ["corpus", "id", "split"],
            [["corpus", record_id, "validation"] for record_id in ids],
        )
    _write_csv(
        releases / "document-changes.csv",
        ["id", "split", "base_release", "change_type"],
        [
            ["doc-02", "validation", "v1.0.0", "content-corrected"],
            ["doc-01", "validation", "v1.0.0", "removed"],
            ["doc-03", "train", "v1.0.0", "content-corrected"],
        ],
    )
    return RepoReader(local_root=str(tmp_path))


def _config() -> dict:
    return yaml.safe_load(CONFIG_TEXT)


PINNED = ReleaseRepo(REPO, "v1.0.0", "aaa")
UPDATE = RepoUpdate(current=PINNED, newest=ReleaseRepo(REPO, "v2.0.0", "bbb"))


def test_versions_compare_numerically():
    newer, older = parse_version("v10.0.0"), parse_version("v9.1.0")
    assert newer is not None and older is not None and newer > older
    assert parse_version("main") is None


def test_reads_each_pinned_repo_once():
    assert pinned_repos(_config()) == [PINNED]


class TestFindUpdates:
    def test_finds_the_newest_release_tag(self):
        api = _Api({"v1.0.0": "aaa", "v2.0.0": "bbb", "v10.0.0": "ccc", "draft": "x"})
        updates = find_updates(api, [PINNED])
        assert [update.newest for update in updates] == [ReleaseRepo(REPO, "v10.0.0", "ccc")]

    def test_nothing_when_pinned_to_the_newest(self):
        assert not find_updates(_Api({"v1.0.0": "aaa"}), [PINNED])

    def test_reports_a_moved_tag(self):
        updates = find_updates(_Api({"v1.0.0": "moved"}), [PINNED])
        assert [update.tag_moved for update in updates] == [True]


def test_reports_counts_per_release_and_mode(reader: RepoReader):
    lines = report(_config(), reader, [UPDATE])
    assert "  validation: 1 added, 1 removed, 1 corrected" in lines
    assert "  full: 9 kept (1 corrected), 1 out, 1 in" in lines
    assert not [line for line in lines if "doc-" in line]


class TestApplyUpdates:
    def test_rewrites_the_pin(self):
        text = apply_updates(CONFIG_TEXT, [UPDATE])
        assert "revision: v2.0.0\n      commit: bbb\n" in text
        assert text.replace("v2.0.0", "v1.0.0").replace("bbb", "aaa") == CONFIG_TEXT

    def test_refuses_a_pin_it_cannot_find(self):
        other = RepoUpdate(
            current=ReleaseRepo("org/other", "v1.0.0", "aaa"),
            newest=ReleaseRepo("org/other", "v2.0.0", "bbb"),
        )
        with pytest.raises(ValueError, match="found 0 pins"):
            apply_updates(CONFIG_TEXT, [other])
