"""Move the dataset pins in eval.yml to each repo's newest release tag.

Shows what each newer release adds, removes and corrects, and how many of each
mode's documents change. Prints counts only, never document ids. `--apply` writes
the new tag and commit into the config.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import yaml
from huggingface_hub import HfApi

from benchmarks.corpus_source import (
    RELEASES_DIR,
    ReleasedCorpus,
    ReleaseRepo,
    RepoReader,
    read_release_ids,
)
from benchmarks.fetch import iter_corpus_sources, select_released_ids

LOGGER = logging.getLogger(__name__)

SPLITS = ("validation", "train")
VERSION_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
CORRECTED = "content-corrected"


def parse_version(tag: str) -> Optional[Tuple[int, int, int]]:
    match = VERSION_TAG.match(tag)
    if not match:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


@dataclasses.dataclass(frozen=True)
class RepoUpdate:
    current: ReleaseRepo
    newest: ReleaseRepo

    @property
    def tag_moved(self) -> bool:
        return self.current.revision == self.newest.revision


def pinned_repos(cfg: Mapping[str, Any]) -> List[ReleaseRepo]:
    """Every repo a released corpus reads, once, in configuration order."""
    seen: Dict[str, ReleaseRepo] = {}
    for split in cfg["dataset"]["splits"]:
        for source in iter_corpus_sources(cfg, split, _all_corpora(cfg, split)):
            if isinstance(source, ReleasedCorpus):
                for repo in source.repos:
                    seen.setdefault(repo.repo_id, repo)
    return list(seen.values())


def _all_corpora(cfg: Mapping[str, Any], split: str) -> List[str]:
    return list(cfg["dataset"]["splits"][split])


def find_updates(api: Any, repos: Iterable[ReleaseRepo]) -> List[RepoUpdate]:
    """A newer release tag for each repo that has one, or its tag if it moved."""
    updates = []
    for repo in repos:
        versions = {
            tag.name: parse_version(tag.name)
            for tag in api.list_repo_refs(repo.repo_id, repo_type="dataset").tags
        }
        released = {tag: version for tag, version in versions.items() if version}
        newest_tag = max(released, key=released.__getitem__, default=repo.revision)
        # repo_info resolves an annotated tag to its commit; list_repo_refs does not.
        commit = api.repo_info(repo.repo_id, revision=newest_tag, repo_type="dataset").sha
        if (newest_tag, commit) != (repo.revision, repo.commit):
            updates.append(RepoUpdate(
                current=repo,
                newest=ReleaseRepo(repo.repo_id, newest_tag, commit),
            ))
    return updates


def corrected_ids(
    reader: RepoReader, update: RepoUpdate, split: str
) -> Set[str]:
    """Documents corrected in a release after the pinned one, up to the newest.

    A change log row names the release it was made against, and ships in the
    next one.
    """
    path = reader.download(update.newest, f"{RELEASES_DIR}/document-changes.csv")
    after = parse_version(update.current.revision)
    upto = parse_version(update.newest.revision)
    if after is None or upto is None:
        return set()
    with open(path, encoding="utf-8", newline="") as changes_file:
        return {
            row["id"]
            for row in csv.DictReader(changes_file)
            if row.get("change_type") == CORRECTED
            and row.get("split") == split
            and _in_range(parse_version(row.get("base_release") or ""), after, upto)
        }


def _in_range(
    version: Optional[Tuple[int, int, int]],
    after: Tuple[int, int, int],
    upto: Tuple[int, int, int],
) -> bool:
    return version is not None and after <= version < upto


@dataclasses.dataclass
class ModeEffect:
    kept: int
    corrected: int
    dropped: int
    entered: int

    def describe(self) -> str:
        return (
            f"{self.kept} kept ({self.corrected} corrected), "
            f"{self.dropped} out, {self.entered} in"
        )


def mode_effect(
    before: Sequence[str], after: Sequence[str], corrected: Set[str]
) -> ModeEffect:
    kept = set(before) & set(after)
    return ModeEffect(
        kept=len(kept),
        corrected=len(kept & corrected),
        dropped=len(set(before) - set(after)),
        entered=len(set(after) - set(before)),
    )


def _with_repos(corpus: ReleasedCorpus, newest: Mapping[str, ReleaseRepo]) -> ReleasedCorpus:
    return dataclasses.replace(corpus, repos=tuple(
        newest.get(repo.repo_id, repo) for repo in corpus.repos
    ))


def report(
    cfg: Mapping[str, Any], reader: RepoReader, updates: Sequence[RepoUpdate]
) -> List[str]:
    """What the updates change, as lines of counts."""
    lines: List[str] = []
    newest = {update.newest.repo_id: update.newest for update in updates}
    corrected: Dict[Tuple[str, str], Set[str]] = {}
    for update in updates:
        note = " (tag moved)" if update.tag_moved else ""
        lines.append(
            f"{update.current.repo_id}: {update.current.revision} -> "
            f"{update.newest.revision}{note}"
        )
        for split in SPLITS:
            before = set(read_release_ids(
                reader, update.current, update.current.revision, split
            ))
            after = set(read_release_ids(
                reader, update.newest, update.newest.revision, split
            ))
            fixed = corrected_ids(reader, update, split)
            corrected[(update.current.repo_id, split)] = fixed
            lines.append(
                f"  {split}: {len(after - before)} added, {len(before - after)} "
                f"removed, {len(fixed & after)} corrected"
            )
    lines.extend(_mode_lines(cfg, reader, newest, corrected))
    return lines


def _mode_lines(
    cfg: Mapping[str, Any],
    reader: RepoReader,
    newest: Mapping[str, ReleaseRepo],
    corrected: Mapping[Tuple[str, str], Set[str]],
) -> List[str]:
    lines: List[str] = []
    seed = cfg["seeds"]["sample"]
    for split in SPLITS:
        if split not in cfg["dataset"]["splits"]:
            continue
        for corpus in iter_corpus_sources(cfg, split, _all_corpora(cfg, split)):
            if not isinstance(corpus, ReleasedCorpus) or not any(
                repo.repo_id in newest for repo in corpus.repos
            ):
                continue
            fixed = set().union(*(
                corrected.get((repo.repo_id, split), set()) for repo in corpus.repos
            ))
            effects = []
            for mode, sizes in cfg["sampling"].items():
                if corpus.corpus not in sizes:
                    continue
                n = sizes[corpus.corpus]
                before, _ = select_released_ids(reader, corpus, n, seed, check_pins=False)
                after, _ = select_released_ids(
                    reader, _with_repos(corpus, newest), n, seed, check_pins=False
                )
                effects.append(f"{mode}: {mode_effect(before, after, fixed).describe()}")
            lines.append(f"{split} {corpus.corpus}")
            lines.extend(f"  {effect}" for effect in effects)
    return lines


def apply_updates(config_text: str, updates: Sequence[RepoUpdate]) -> str:
    """The config with each repo's revision and commit replaced."""
    for update in updates:
        current = update.current
        pattern = re.compile(
            rf"(repo_id: {re.escape(current.repo_id)}\n\s+revision: )"
            rf"{re.escape(current.revision)}(\n\s+commit: ){re.escape(current.commit)}\b"
        )
        config_text, count = pattern.subn(
            rf"\g<1>{update.newest.revision}\g<2>{update.newest.commit}", config_text
        )
        if count != 1:
            raise ValueError(
                f"found {count} pins for {current.repo_id} in the config, expected 1 "
                f"block of repo_id, revision and commit lines"
            )
    return config_text


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the newest tag and commit into the config",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    config_path = Path(args.config)
    config_text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(config_text)
    reader = RepoReader.from_env()
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    updates = find_updates(api, pinned_repos(cfg))
    if not updates:
        print("Every dataset repo is pinned to its newest release.")
        return
    for line in report(cfg, reader, updates):
        print(line)
    if args.apply:
        config_path.write_text(apply_updates(config_text, updates), encoding="utf-8")
        print(f"Updated {config_path}.")
    else:
        print("Run with APPLY=1 to write these pins.")


if __name__ == "__main__":
    main()
