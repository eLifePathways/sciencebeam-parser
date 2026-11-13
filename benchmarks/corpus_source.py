"""Where one corpus's rows come from, and how they are read.

A corpus may live in its own repo at its own revision, spread over many files,
Hive-partitioned by a stratum that is in the path rather than in the data. Resolving
that in one place is the point of this module: `fetch_data`, `fetch_gold` and the
training-source fetch each answering "which repo, which revision" for themselves is
how a run ends up with gold XML and PDFs from different revisions.

Reading a partitioned corpus fetches only the column chunks of the row groups it
needs, with the file object's read-ahead turned off. With read-ahead on, most of
each file arrives anyway and the column projection saves nothing while appearing
to work.

A released corpus (`sources:`) is listed by `releases/<tag>.csv` in one or more
dataset repos, each pinned to a tag and the commit it pointed at.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    IO,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_ID_COLUMN = "id"
DEFAULT_RANK_COLUMN = "rank"
MANIFEST_SPLIT_FIELD = "split"

# Local override for tests and offline work: a directory laid out like the repo,
# or for a released corpus, one directory per repo at `<root>/<repo_id>/`.
LOCAL_ROOT_ENV_VAR = "BENCH_LOCAL_PARQUET_DIR"

# The layout every sciencebeam-dataset-* repo shares.
RELEASE_DATA_DIR = "pdf-jats"
RELEASES_DIR = "releases"
# The release whose list is shuffled to order a released corpus.
FIRST_RELEASE = "v1.0.0"


class CorpusConfigError(ValueError):
    """A corpus entry that cannot be used as written."""


@dataclasses.dataclass(frozen=True)
class CorpusSource:
    """One corpus's location, resolved from the configuration."""

    corpus: str
    repo_id: str
    revision: str
    id_column: str = DEFAULT_ID_COLUMN
    # Exactly one of these: a corpus is one named file, or a prefix holding many.
    file: Optional[str] = None
    path: Optional[str] = None
    # A stratified corpus also names the manifest that carries its strata and ranks.
    manifest: Optional[str] = None
    stratum_column: Optional[str] = None
    rank_column: str = DEFAULT_RANK_COLUMN
    manifest_split: Optional[str] = None
    optional: bool = False

    @property
    def is_stratified(self) -> bool:
        return self.stratum_column is not None

    @property
    def location(self) -> str:
        """The prefix or the file this corpus is read from.

        Named here because a source manifest records it and a later run compares
        what it recorded against what the configuration now resolves.
        """
        return self.path or str(self.file)

    @property
    def read_revision(self) -> str:
        return self.revision

    @property
    def local_dir(self) -> str:
        return ""

    def describe(self) -> str:
        return f"{self.repo_id}@{self.revision}/{self.location}"


@dataclasses.dataclass(frozen=True)
class ReleaseRepo:
    """One dataset repo of a released corpus, pinned to a release tag and its commit."""

    repo_id: str
    revision: str
    commit: str

    @property
    def read_revision(self) -> str:
        # Read at the commit, so a tag moved after the pin check can't change the data.
        return self.commit

    @property
    def local_dir(self) -> str:
        return self.repo_id

    def describe(self) -> str:
        return f"{self.repo_id}@{self.revision}"

    def to_dict(self) -> Dict[str, str]:
        return {"repo_id": self.repo_id, "revision": self.revision, "commit": self.commit}


@dataclasses.dataclass(frozen=True)
class ReleasedCorpus:
    """A corpus listed by the release files of one or more dataset repos.

    `repos` keeps the configured order (public first), which is the order they
    take turns in when a mode is selected.
    """

    corpus: str
    split: str
    repos: Tuple[ReleaseRepo, ...]
    id_column: str = DEFAULT_ID_COLUMN
    optional: bool = False

    @property
    def location(self) -> str:
        return f"{RELEASE_DATA_DIR}/{self.split}/"

    def describe(self) -> str:
        return " + ".join(repo.describe() for repo in self.repos)


AnySource = Union[CorpusSource, ReleasedCorpus]


def resolve_source(
    cfg: Mapping[str, Any], split: str, corpus: str, corpus_cfg: Any
) -> AnySource:
    """Resolve one corpus entry, falling back to the dataset-level repo and revision.

    Both fall back together: pinning matters most for a corpus that grows, and one
    global revision cannot pin two repos independently.
    """
    dataset = cfg.get("dataset", {})
    if not isinstance(corpus_cfg, dict):
        corpus_cfg = {"file": str(corpus_cfg)}
    if "sources" in corpus_cfg:
        return _resolve_released(split, corpus, corpus_cfg)
    repo_id, revision = _resolve_repo_and_revision(corpus, corpus_cfg, dataset)
    source = CorpusSource(
        corpus=corpus,
        repo_id=repo_id,
        revision=revision,
        id_column=corpus_cfg.get("id_column", DEFAULT_ID_COLUMN),
        file=corpus_cfg.get("file"),
        path=corpus_cfg.get("path"),
        manifest=corpus_cfg.get("manifest"),
        stratum_column=corpus_cfg.get("stratum"),
        rank_column=corpus_cfg.get("rank_column", DEFAULT_RANK_COLUMN),
        manifest_split=corpus_cfg.get("manifest_split", split),
        optional=bool(corpus_cfg.get("optional", False)),
    )
    _check_source(source)
    return source


def _resolve_released(
    split: str, corpus: str, corpus_cfg: Mapping[str, Any]
) -> ReleasedCorpus:
    other_keys = {"file", "path", "manifest", "stratum", "repo_id", "revision"}
    mixed = sorted(other_keys.intersection(corpus_cfg))
    if mixed:
        raise CorpusConfigError(
            f"corpus {corpus!r} names `sources` and also {', '.join(mixed)}; a "
            f"released corpus is read from its sources only"
        )
    entries = corpus_cfg.get("sources") or []
    if not entries:
        raise CorpusConfigError(f"corpus {corpus!r} has an empty `sources` list")
    repos = []
    for entry in entries:
        missing = [
            key for key in ("repo_id", "revision", "commit")
            if not (isinstance(entry, dict) and entry.get(key))
        ]
        if missing:
            raise CorpusConfigError(
                f"corpus {corpus!r}: each source needs repo_id, revision and commit; "
                f"{entry!r} is missing {', '.join(missing)}"
            )
        repos.append(ReleaseRepo(
            repo_id=str(entry["repo_id"]),
            revision=str(entry["revision"]),
            commit=str(entry["commit"]),
        ))
    if len({repo.repo_id for repo in repos}) != len(repos):
        raise CorpusConfigError(f"corpus {corpus!r} names the same repo twice")
    return ReleasedCorpus(
        corpus=corpus,
        split=split,
        repos=tuple(repos),
        id_column=corpus_cfg.get("id_column", DEFAULT_ID_COLUMN),
        optional=bool(corpus_cfg.get("optional", False)),
    )


def _resolve_repo_and_revision(
    corpus: str, corpus_cfg: Mapping[str, Any], dataset: Mapping[str, Any]
) -> Tuple[str, str]:
    own_repo_id = corpus_cfg.get("repo_id")
    own_revision = corpus_cfg.get("revision")
    if own_repo_id and own_revision:
        return own_repo_id, own_revision
    if own_repo_id or own_revision:
        named, missing = (
            ("repo_id", "revision") if own_repo_id else ("revision", "repo_id")
        )
        raise CorpusConfigError(
            f"corpus {corpus!r} names its own {named} but no {missing}, so it would "
            f"take the other from the dataset level. A revision names a commit in "
            f"one repo: declare both or neither"
        )
    repo_id = dataset.get("repo_id")
    revision = dataset.get("revision")
    if not repo_id or not revision:
        raise CorpusConfigError(
            f"corpus {corpus!r} has no repo_id or revision, and the dataset-level "
            f"values do not supply one"
        )
    return repo_id, revision


def _check_source(source: CorpusSource) -> None:
    corpus = source.corpus
    if bool(source.file) == bool(source.path):
        raise CorpusConfigError(
            f"corpus {corpus!r} must name either one `file` or a `path` prefix, "
            f"not both and not neither"
        )
    if source.is_stratified and not source.path:
        raise CorpusConfigError(
            f"corpus {corpus!r} declares a stratum, so it is read as a partitioned "
            f"`path` rather than one `file`"
        )
    if source.is_stratified and not source.manifest:
        raise CorpusConfigError(
            f"corpus {corpus!r} declares stratum {source.stratum_column!r} but no "
            f"`manifest`. The stratum is in the partition path and the rank exists "
            f"only in the manifest, so a stratified sample needs it"
        )


@dataclasses.dataclass
class RepoReader:
    """Listing, downloading and opening a repo's files, or a local stand-in.

    The local stand-in is a directory laid out like the repo, which is what lets the
    tests exercise a real partitioned layout instead of a mocked Parquet reader.
    """

    token: Optional[str] = None
    local_root: Optional[str] = None
    _fs: Optional[HfFileSystem] = None
    _commit_by_revision: Dict[Tuple[str, str], Optional[str]] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_env(cls) -> "RepoReader":
        return cls(
            token=os.environ.get("HF_TOKEN"),
            local_root=os.environ.get(LOCAL_ROOT_ENV_VAR),
        )

    def _filesystem(self) -> HfFileSystem:
        if self._fs is None:
            self._fs = HfFileSystem(token=self.token)
        return self._fs

    def resolve_commit(self, source: Union[CorpusSource, ReleaseRepo]) -> Optional[str]:
        """The commit a revision points at, so that `main` names a specific one.

        None for a local stand-in, which has no revision to resolve. Cached per
        repo and revision because a run resolves the same one for every corpus.
        """
        if self.local_root:
            return None
        key = (source.repo_id, source.revision)
        if key not in self._commit_by_revision:
            try:
                info = HfApi(token=self.token).repo_info(
                    repo_id=source.repo_id,
                    revision=source.revision,
                    repo_type="dataset",
                )
            except RevisionNotFoundError as error:
                raise CorpusConfigError(
                    f"{source.repo_id} has no revision {source.revision!r}"
                ) from error
            except (RepositoryNotFoundError, GatedRepoError) as error:
                raise CorpusConfigError(
                    f"cannot read {source.repo_id}: it does not exist, or HF_TOKEN "
                    f"has no access to it"
                ) from error
            self._commit_by_revision[key] = info.sha
        return self._commit_by_revision[key]

    def check_pinned(self, repo: ReleaseRepo) -> None:
        """Fail if the tag no longer points at the pinned commit."""
        resolved = self.resolve_commit(repo)
        if resolved is not None and resolved != repo.commit:
            raise CorpusConfigError(
                f"{repo.describe()} now points at {resolved}, not the pinned "
                f"{repo.commit}. The tag was moved; check what changed, then update "
                f"the pin (make docker-benchmark-update-dataset)"
            )

    def download(self, source: Union[CorpusSource, ReleaseRepo], filename: str) -> str:
        """A whole small file — a manifest, a release list, or a one-file corpus."""
        if self.local_root:
            local_path = Path(self.local_root) / source.local_dir / filename
            if not local_path.exists():
                raise CorpusConfigError(f"{source.describe()} has no {filename}")
            return str(local_path)
        try:
            return hf_hub_download(
                repo_id=source.repo_id,
                filename=filename,
                revision=source.read_revision,
                repo_type="dataset",
                token=self.token,
            )
        except EntryNotFoundError as error:
            raise CorpusConfigError(f"{source.describe()} has no {filename}") from error

    def list_parquet(self, source: Union[CorpusSource, ReleaseRepo], prefix: str) -> List[str]:
        if self.local_root:
            root = Path(self.local_root) / source.local_dir / prefix
            return [str(path) for path in sorted(root.glob("*.parquet"))]
        remote = f"datasets/{source.repo_id}@{source.read_revision}/{prefix}"
        listed = [str(entry) for entry in self._filesystem().ls(remote, detail=False)]
        return sorted(path for path in listed if path.endswith(".parquet"))

    @contextlib.contextmanager
    def open_parquet(self, path: str) -> Iterator[IO[bytes]]:
        if self.local_root:
            with open(path, "rb") as local_file:
                yield local_file
            return
        # cache_type="none": anything else reads ahead, and the projection that
        # keeps this affordable stops being a projection.
        with self._filesystem().open(path, "rb", cache_type="none") as remote_file:
            yield remote_file


def read_all_ids(reader: RepoReader, source: CorpusSource) -> List[str]:
    """Every id in a one-file corpus, in the order the file stores them."""
    parquet_path = reader.download(source, str(source.file))
    return [
        str(value)
        for value in pq.ParquetFile(parquet_path)
        .read(columns=[source.id_column])
        .column(source.id_column)
        .to_pylist()
    ]


def read_release_ids(
    reader: RepoReader, repo: ReleaseRepo, release: str, split: str
) -> List[str]:
    """One split's ids in `releases/<release>.csv`, in the order the file lists them."""
    filename = f"{RELEASES_DIR}/{release}.csv"
    release_path = reader.download(repo, filename)
    with open(release_path, encoding="utf-8", newline="") as release_file:
        rows = csv.DictReader(release_file)
        missing = [
            column for column in (DEFAULT_ID_COLUMN, MANIFEST_SPLIT_FIELD)
            if column not in (rows.fieldnames or [])
        ]
        if missing:
            raise CorpusConfigError(
                f"{repo.describe()}/{filename} is missing column(s) {', '.join(missing)}"
            )
        ids = [
            (row.get(DEFAULT_ID_COLUMN) or "").strip()
            for row in rows
            if (row.get(MANIFEST_SPLIT_FIELD) or "").strip() == split
        ]
    if len(ids) != len(set(ids)):
        raise CorpusConfigError(
            f"{repo.describe()}/{filename} lists an id twice in split {split!r}"
        )
    return ids


def read_manifest_ids_by_stratum(
    reader: RepoReader, source: CorpusSource
) -> Dict[str, List[str]]:
    """A stratified corpus's ids per stratum, in rank order, for one split.

    The manifest names a version, so a sample is limited to that version's
    membership even where the revision is a moving branch — and it is what makes
    the stratum available without reading a partition column the files do not
    carry.
    """
    manifest_path = reader.download(source, str(source.manifest))
    stratum_column = str(source.stratum_column)
    wanted_split = source.manifest_split
    by_stratum: Dict[str, List[Tuple[int, str]]] = {}
    with open(manifest_path, encoding="utf-8", newline="") as manifest_file:
        rows = csv.DictReader(manifest_file)
        _check_manifest_columns(source, rows.fieldnames, manifest_path)
        for row in rows:
            if wanted_split and (row.get(MANIFEST_SPLIT_FIELD) or "").strip() != wanted_split:
                continue
            record_id = (row.get(source.id_column) or "").strip()
            stratum = (row.get(stratum_column) or "").strip()
            rank = (row.get(source.rank_column) or "").strip()
            try:
                parsed_rank = int(rank)
            except ValueError as exc:
                raise CorpusConfigError(
                    f"{manifest_path}: {source.rank_column} {rank!r} for "
                    f"{record_id!r} is not an integer"
                ) from exc
            by_stratum.setdefault(stratum, []).append((parsed_rank, record_id))
    if not by_stratum:
        raise CorpusConfigError(
            f"corpus {source.corpus!r}: no rows for split {wanted_split!r} in "
            f"{source.manifest}"
        )
    ids_by_stratum = {
        stratum: [record_id for _, record_id in sorted(ranked)]
        for stratum, ranked in by_stratum.items()
    }
    _check_ids_unique(ids_by_stratum, manifest_path, wanted_split)
    return ids_by_stratum


def _check_ids_unique(
    ids_by_stratum: Mapping[str, Sequence[str]], manifest_path: str, split: Optional[str]
) -> None:
    """Sampling identifies a document by id, so a repeated one is a corrupt manifest.

    Checked across strata rather than within one, since the same id under two strata
    would be selected twice and materialised as two records of one document.
    """
    seen: Dict[str, str] = {}
    for stratum, ids in sorted(ids_by_stratum.items()):
        for record_id in ids:
            if record_id in seen:
                raise CorpusConfigError(
                    f"{manifest_path}: id {record_id!r} appears twice in split "
                    f"{split!r}, under {seen[record_id]!r} and {stratum!r}. Sampling "
                    f"identifies a document by its id, so ids have to be unique"
                )
            seen[record_id] = stratum


def _check_manifest_columns(
    source: CorpusSource, present: Optional[Sequence[str]], manifest_path: str
) -> None:
    expected = [
        source.id_column,
        str(source.stratum_column),
        source.rank_column,
        MANIFEST_SPLIT_FIELD,
    ]
    missing = [column for column in expected if column not in (present or [])]
    if missing:
        raise CorpusConfigError(
            f"{manifest_path} is missing column(s) {', '.join(missing)}; it has "
            f"{', '.join(present or [])}. A manifest names its columns as its corpus "
            f"does, so check `id_column`, `stratum` and `rank_column`"
        )


def iter_single_file_rows(
    reader: RepoReader,
    source: CorpusSource,
    wanted_ids: Sequence[str],
    columns: Sequence[str],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Rows of a one-file corpus whose id was selected."""
    parquet_path = reader.download(source, str(source.file))
    wanted = set(wanted_ids)
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(
        batch_size=64, columns=[source.id_column, *columns]
    ):
        ids = batch.column(source.id_column).to_pylist()
        for row, record_id in enumerate(ids):
            if str(record_id) in wanted:
                yield str(record_id), {
                    column: batch.column(column)[row].as_py() for column in columns
                }


def iter_partitioned_rows(
    reader: RepoReader,
    source: CorpusSource,
    wanted_by_stratum: Mapping[str, Sequence[str]],
    columns: Sequence[str],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Rows of a partitioned corpus whose id was selected.

    Only the partitions holding a selected document are listed, and within a file
    only the row groups holding one are read.
    """
    for stratum in sorted(wanted_by_stratum):
        wanted = set(wanted_by_stratum[stratum])
        if not wanted:
            continue
        prefix = f"{str(source.path).rstrip('/')}/{source.stratum_column}={stratum}/"
        for parquet_path in reader.list_parquet(source, prefix):
            if not wanted:
                break
            with reader.open_parquet(parquet_path) as parquet_io:
                yield from _iter_file_rows(
                    pq.ParquetFile(parquet_io), source.id_column, wanted, columns
                )
        if wanted:
            raise CorpusConfigError(
                f"corpus {source.corpus!r}: {len(wanted)} selected document(s) are "
                f"absent from {prefix} under {source.describe()}, e.g. "
                f"{sorted(wanted)[0]!r}. The manifest and the data disagree — check "
                f"`path` and `manifest_split` name the same split"
            )


def iter_release_rows(
    reader: RepoReader,
    repo: ReleaseRepo,
    split: str,
    wanted_ids: Sequence[str],
    columns: Sequence[str],
    id_column: str = DEFAULT_ID_COLUMN,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Rows of one repo's split whose id was selected, across its shards."""
    wanted = set(wanted_ids)
    if not wanted:
        return
    prefix = f"{RELEASE_DATA_DIR}/{split}/"
    for parquet_path in reader.list_parquet(repo, prefix):
        if not wanted:
            break
        with reader.open_parquet(parquet_path) as parquet_io:
            yield from _iter_file_rows(
                pq.ParquetFile(parquet_io), id_column, wanted, columns
            )
    if wanted:
        raise CorpusConfigError(
            f"{len(wanted)} document(s) listed by {repo.describe()} are absent from "
            f"its {prefix} files"
        )


def _iter_file_rows(
    parquet_file: pq.ParquetFile,
    id_column: str,
    wanted: Set[str],
    columns: Sequence[str],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """The wanted rows of one file, taking each found id out of `wanted`.

    Reading the id column first costs almost nothing and says which row groups are
    worth reading, so the payload columns of the rest are never requested.
    """
    ids = [
        str(value)
        for value in parquet_file.read(columns=[id_column])
        .column(id_column)
        .to_pylist()
    ]
    for row_group in _row_groups_holding(parquet_file, ids, wanted):
        table = parquet_file.read_row_group(
            row_group, columns=[id_column, *columns]
        )
        for row, value in enumerate(table.column(id_column).to_pylist()):
            record_id = str(value)
            if record_id in wanted:
                wanted.discard(record_id)
                yield record_id, {
                    column: table.column(column)[row].as_py() for column in columns
                }


def _row_groups_holding(
    parquet_file: pq.ParquetFile, ids: Sequence[str], wanted: Iterable[str]
) -> List[int]:
    """Indices of the row groups holding at least one wanted id.

    The ids arrive in file order, so walking each row group's row count says which
    group a position falls in without reading any of them.
    """
    wanted_set = set(wanted)
    metadata = parquet_file.metadata
    holding: List[int] = []
    offset = 0
    for row_group in range(metadata.num_row_groups):
        num_rows = metadata.row_group(row_group).num_rows
        if wanted_set.intersection(ids[offset:offset + num_rows]):
            holding.append(row_group)
        offset += num_rows
    return holding
