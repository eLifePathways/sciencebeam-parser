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
)

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

LOGGER = logging.getLogger(__name__)

DEFAULT_ID_COLUMN = "id"
DEFAULT_RANK_COLUMN = "rank"
MANIFEST_SPLIT_FIELD = "split"

# Local override for tests and offline work: a directory laid out like the repo.
LOCAL_ROOT_ENV_VAR = "BENCH_LOCAL_PARQUET_DIR"


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

    def describe(self) -> str:
        return f"{self.repo_id}@{self.revision}/{self.path or self.file}"


def resolve_source(
    cfg: Mapping[str, Any], split: str, corpus: str, corpus_cfg: Any
) -> CorpusSource:
    """Resolve one corpus entry, falling back to the dataset-level repo and revision.

    Both fall back together: pinning matters most for a corpus that grows, and one
    global revision cannot pin two repos independently.
    """
    dataset = cfg.get("dataset", {})
    if not isinstance(corpus_cfg, dict):
        corpus_cfg = {"file": str(corpus_cfg)}
    source = CorpusSource(
        corpus=corpus,
        repo_id=corpus_cfg.get("repo_id") or dataset.get("repo_id"),
        revision=corpus_cfg.get("revision") or dataset.get("revision"),
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


def _check_source(source: CorpusSource) -> None:
    corpus = source.corpus
    if not source.repo_id or not source.revision:
        raise CorpusConfigError(
            f"corpus {corpus!r} has no repo_id or revision, and the dataset-level "
            f"values do not supply one"
        )
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

    def download(self, source: CorpusSource, filename: str) -> str:
        """A whole small file — a manifest, or a one-file corpus."""
        if self.local_root:
            return str(Path(self.local_root) / filename)
        return hf_hub_download(
            repo_id=source.repo_id,
            filename=filename,
            revision=source.revision,
            repo_type="dataset",
            token=self.token,
        )

    def list_parquet(self, source: CorpusSource, prefix: str) -> List[str]:
        if self.local_root:
            root = Path(self.local_root) / prefix
            return [str(path) for path in sorted(root.glob("*.parquet"))]
        remote = f"datasets/{source.repo_id}@{source.revision}/{prefix}"
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
                    pq.ParquetFile(parquet_io), source, wanted, columns
                )
        if wanted:
            raise CorpusConfigError(
                f"corpus {source.corpus!r}: {len(wanted)} selected document(s) are "
                f"absent from {prefix} under {source.describe()}, e.g. "
                f"{sorted(wanted)[0]!r}. The manifest and the data disagree — check "
                f"`path` and `manifest_split` name the same split"
            )


def _iter_file_rows(
    parquet_file: pq.ParquetFile,
    source: CorpusSource,
    wanted: Set[str],
    columns: Sequence[str],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """The wanted rows of one file, taking each found id out of `wanted`.

    Reading the id column first costs almost nothing and says which row groups are
    worth reading, so the payload columns of the rest are never requested.
    """
    ids = [
        str(value)
        for value in parquet_file.read(columns=[source.id_column])
        .column(source.id_column)
        .to_pylist()
    ]
    for row_group in _row_groups_holding(parquet_file, ids, wanted):
        table = parquet_file.read_row_group(
            row_group, columns=[source.id_column, *columns]
        )
        for row, value in enumerate(table.column(source.id_column).to_pylist()):
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
