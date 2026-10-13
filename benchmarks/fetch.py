from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from benchmarks.corpus_source import (
    CorpusConfigError,
    CorpusSource,
    RepoReader,
    iter_partitioned_rows,
    iter_single_file_rows,
    read_all_ids,
    read_manifest_ids_by_stratum,
    resolve_source,
)
from benchmarks.sampling import positional_ids, stratified_ids
from benchmarks.training_records import SourceManifest
from benchmarks.training_selection import (
    ModeSelection,
    read_selection,
    select_for_mode,
    write_selection,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "fetch_data",
    "fetch_gold",
    "fetch_training_source",
    "included_corpora",
    "iter_corpus_sources",
    "resolved_sources",
]


def _split_corpora(cfg: Mapping[str, Any], split: str) -> Dict[str, Any]:
    split_corpora = cfg["dataset"]["splits"].get(split)
    if not split_corpora:
        raise ValueError(
            f"Unknown split {split!r}. Available: {list(cfg['dataset']['splits'])}"
        )
    return split_corpora


def included_corpora(
    cfg: Mapping[str, Any], split: str, include: Optional[Iterable[str]] = None
) -> List[str]:
    """The corpora a run covers, in configuration order.

    A corpus marked `optional` is left out unless it is named: the PLOS corpus is
    private and non-redistributable, so reaching for it is something a caller asks
    for rather than something a default does.

    Deciding this is deliberately separate from resolving where a corpus lives, so
    that everything choosing a corpus set — fetching, prediction variants, scoring —
    agrees without each of them having to resolve a source it may never read.
    """
    split_corpora = _split_corpora(cfg, split)
    requested = set(include or ())
    unknown = requested - set(split_corpora)
    if unknown:
        raise ValueError(
            f"Corpora {sorted(unknown)} are not in split {split!r}. Available: "
            f"{list(split_corpora)}"
        )
    covered = []
    for corpus, corpus_cfg in split_corpora.items():
        optional = isinstance(corpus_cfg, dict) and bool(corpus_cfg.get("optional"))
        if optional and corpus not in requested:
            LOGGER.info("Corpus %r is opt-in and was not requested, skipping", corpus)
            continue
        covered.append(corpus)
    return covered


def iter_corpus_sources(
    cfg: Mapping[str, Any], split: str, include: Optional[Iterable[str]] = None
) -> List[CorpusSource]:
    """Where each covered corpus lives, resolved once for every reader."""
    split_corpora = _split_corpora(cfg, split)
    return [
        resolve_source(cfg, split, corpus, split_corpora[corpus])
        for corpus in included_corpora(cfg, split, include)
    ]


def resolved_sources(
    cfg: Mapping[str, Any], split: str, include: Optional[Iterable[str]] = None
) -> Dict[str, Dict[str, str]]:
    """What each corpus resolved to, for recording alongside a run's output.

    This is what makes "the PDFs and the gold XML came from the same revision"
    checkable after the fact rather than argued from the code.
    """
    return {
        source.corpus: {
            "repo_id": source.repo_id,
            "revision": source.revision,
            "location": source.path or str(source.file),
            **({"manifest": source.manifest} if source.manifest else {}),
        }
        for source in iter_corpus_sources(cfg, split, include)
    }


def _select_ids(
    reader: RepoReader, source: CorpusSource, raw_n: Optional[int], seed: int
) -> Tuple[List[str], Optional[Dict[str, List[str]]]]:
    """The ids this mode selects, and for a stratified corpus, grouped by stratum."""
    if not source.is_stratified:
        return positional_ids(read_all_ids(reader, source), raw_n, seed), None
    by_stratum = read_manifest_ids_by_stratum(reader, source)
    picked = stratified_ids(by_stratum, raw_n, seed)
    selected = set(picked)
    picked_by_stratum = {
        stratum: [record_id for record_id in ids if record_id in selected]
        for stratum, ids in by_stratum.items()
    }
    return picked, {
        stratum: ids for stratum, ids in picked_by_stratum.items() if ids
    }


def _record_id_of(raw_id: str) -> str:
    return raw_id.replace("/", "_")


def _select_recorded_ids(
    reader: RepoReader,
    source: CorpusSource,
    raw_n: Optional[int],
    seed: int,
    split: str,
    selection_dir: Path,
) -> ModeSelection:
    """The documents this mode names, from the list the corpus carries.

    Recorded against the id the corpus stores, so that what is written here is the
    same string the dataset uses rather than the filename it is materialised under.
    """
    if source.is_stratified:
        raise CorpusConfigError(
            f"corpus {source.corpus!r} is stratified, and a recorded selection "
            f"keeps one order per corpus rather than one per stratum. Generating "
            f"from a stratified corpus needs that settled first"
        )
    recorded = read_selection(selection_dir, split, source.corpus)
    result = select_for_mode(recorded, read_all_ids(reader, source), raw_n, seed)
    if list(result.selection) != recorded:
        file_path = write_selection(
            selection_dir, split, source.corpus, result.selection
        )
        LOGGER.info(
            "Appended %d document(s) to the selection: %s",
            len(result.appended),
            file_path,
        )
    return result


def _materialise(
    reader: RepoReader,
    source: CorpusSource,
    picked: Sequence[str],
    by_stratum: Optional[Mapping[str, Sequence[str]]],
    corpus_dir: Path,
    with_pdf: bool,
) -> List[Dict[str, str]]:
    """Write the selected documents that are not on disk yet, and return them all.

    Only the missing ones are read: everything else has already been materialised by
    an earlier run, and re-reading them would fetch their bytes again for nothing.
    """
    columns = ["pdf", "xml"] if with_pdf else ["xml"]
    paths_by_raw_id = {
        raw_id: _record_paths(corpus_dir, _record_id_of(raw_id), with_pdf)
        for raw_id in picked
    }
    missing = [
        raw_id
        for raw_id, paths in paths_by_raw_id.items()
        if not all(path.exists() for path in paths.values())
    ]

    for raw_id, values in _iter_missing_rows(
        reader, source, missing, by_stratum, columns
    ):
        paths = paths_by_raw_id[raw_id]
        if "pdf_path" in paths and not paths["pdf_path"].exists():
            paths["pdf_path"].write_bytes(bytes(values["pdf"]))
        if not paths["xml_path"].exists():
            paths["xml_path"].write_text(str(values["xml"]), encoding="utf-8")

    return [
        _record(source.corpus, _record_id_of(raw_id), paths_by_raw_id[raw_id])
        for raw_id in picked
        if all(path.exists() for path in paths_by_raw_id[raw_id].values())
    ]


def _iter_missing_rows(
    reader: RepoReader,
    source: CorpusSource,
    missing: Sequence[str],
    by_stratum: Optional[Mapping[str, Sequence[str]]],
    columns: Sequence[str],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if not missing:
        return iter(())
    if by_stratum is None:
        return iter_single_file_rows(reader, source, missing, columns)
    still_wanted = set(missing)
    wanted_by_stratum = {
        stratum: [record_id for record_id in ids if record_id in still_wanted]
        for stratum, ids in by_stratum.items()
    }
    return iter_partitioned_rows(reader, source, wanted_by_stratum, columns)


def _record_paths(corpus_dir: Path, record_id: str, with_pdf: bool) -> Dict[str, Path]:
    paths = {"xml_path": corpus_dir / f"{record_id}.jats.xml"}
    if with_pdf:
        paths["pdf_path"] = corpus_dir / f"{record_id}.pdf"
    return paths


def _record(corpus: str, record_id: str, paths: Mapping[str, Path]) -> Dict[str, str]:
    return {
        "corpus": corpus,
        "record_id": record_id,
        **{key: str(path) for key, path in paths.items()},
    }


def _fetch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: Mapping[str, Any],
    mode: str,
    split: str,
    data_dir: Path,
    with_pdf: bool,
    include: Optional[Iterable[str]] = None,
    write_manifest: bool = False,
    selection_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Materialise gold XML, and PDFs when asked, for each corpus of one split.

    Idempotent: a document already on disk is neither re-read nor rewritten. Data is
    stored under data_dir/<split>/<corpus>/ so train and validation records never
    mix in the cache.
    """
    sample_sizes = cfg["sampling"][mode]
    seed = cfg["seeds"]["sample"]
    reader = RepoReader.from_env()

    records: List[Dict[str, str]] = []
    for source in iter_corpus_sources(cfg, split, include):
        if source.corpus not in sample_sizes:
            LOGGER.warning(
                "No sample size configured for corpus %r in mode %r, skipping",
                source.corpus,
                mode,
            )
            continue
        records.extend(
            _fetch_corpus(
                reader,
                source,
                raw_n=sample_sizes[source.corpus],
                seed=seed,
                mode=mode,
                split=split,
                corpus_dir=data_dir / split / source.corpus,
                with_pdf=with_pdf,
                write_manifest=write_manifest,
                selection_dir=selection_dir,
            )
        )
    return records


def _fetch_corpus(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    reader: RepoReader,
    source: CorpusSource,
    raw_n: Optional[int],
    seed: int,
    mode: str,
    corpus_dir: Path,
    with_pdf: bool,
    split: str = "train",
    write_manifest: bool = False,
    selection_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    LOGGER.info(
        "Fetching corpus %r from %s (mode=%s, n=%s)",
        source.corpus,
        source.describe(),
        mode,
        raw_n if raw_n is not None else "all",
    )
    mode_selection: Optional[ModeSelection] = None
    if selection_dir is not None:
        mode_selection = _select_recorded_ids(
            reader, source, raw_n, seed, split, selection_dir
        )
        picked, by_stratum = list(mode_selection.present), None
    else:
        picked, by_stratum = _select_ids(reader, source, raw_n, seed)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records = _materialise(reader, source, picked, by_stratum, corpus_dir, with_pdf)
    if write_manifest:
        _write_source_manifest(
            reader, source, mode, split, seed, raw_n, picked, corpus_dir,
            mode_selection,
        )
    LOGGER.info(
        "Corpus %r: %d record(s) available in %s",
        source.corpus,
        len(records),
        corpus_dir,
    )
    return records


def _write_source_manifest(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    reader: RepoReader,
    source: CorpusSource,
    mode: str,
    split: str,
    seed: int,
    raw_n: Optional[int],
    picked: Sequence[str],
    corpus_dir: Path,
    mode_selection: Optional[ModeSelection] = None,
) -> None:
    """Leave what this fetch resolved beside the documents it wrote.

    The directory accumulates: it is never pruned, and a mode asking for fewer
    documents than are already there leaves the extra ones in place. Naming the
    selection is what keeps "which documents is this mode" answerable from the
    tree rather than inferred from its size.
    """
    manifest = SourceManifest(
        corpus=source.corpus,
        split=split,
        mode=mode,
        seed=seed,
        repo_id=source.repo_id,
        revision=source.revision,
        location=source.path or str(source.file),
        commit=reader.resolve_commit(source),
        requested_document_count=raw_n,
        selected_document_ids=[_record_id_of(raw_id) for raw_id in picked],
        missing_document_ids=[
            _record_id_of(raw_id) for raw_id in (mode_selection.missing if mode_selection else ())
        ],
    )
    file_path = manifest.write(corpus_dir)
    LOGGER.info("Wrote source manifest: %s", file_path)


def fetch_data(
    cfg: Dict[str, Any],
    mode: str,
    split: str,
    data_dir: Path,
    include: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """Download and materialise PDF + gold XML for each corpus.

    Returns a list of records: {corpus, record_id, pdf_path, xml_path}.
    """
    return _fetch(cfg, mode, split, data_dir, with_pdf=True, include=include)


def fetch_gold(
    cfg: Dict[str, Any],
    mode: str,
    split: str,
    data_dir: Path,
    include: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """Download gold XML only (no PDFs) for each corpus.

    Returns records: {corpus, record_id, xml_path}. The sample is the same one
    `fetch_data` selects, since both resolve the corpus the same way.
    """
    return _fetch(cfg, mode, split, data_dir, with_pdf=False, include=include)


def fetch_training_source(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    cfg: Dict[str, Any],
    mode: str,
    split: str,
    data_dir: Path,
    include: Optional[Iterable[str]] = None,
    selection_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Fetch PDF + JATS XML for CC-BY corpora only.

    Reads ``cc_by_corpora`` from the config to determine which corpora are
    permitted.  Corpora absent from that list are silently skipped so that
    the allow-list can be extended without changing call sites.

    `include` narrows the fetch to corpora that are already allowed; it cannot
    widen it. A corpus outside `cc_by_corpora` is not opt-in here, it is refused,
    because this path generates training data that is published elsewhere.
    """
    allowed: Set[str] = set(cfg.get("cc_by_corpora", []))
    filtered_sampling = {
        m: {corpus: n for corpus, n in sizes.items() if corpus in allowed}
        for m, sizes in cfg.get("sampling", {}).items()
    }
    filtered_cfg = {**cfg, "sampling": filtered_sampling}
    in_split = allowed.intersection(cfg["dataset"]["splits"].get(split, {}))
    if include is not None:
        in_split = in_split.intersection(include)
    return _fetch(
        filtered_cfg,
        mode,
        split,
        data_dir,
        with_pdf=True,
        include=in_split,
        write_manifest=True,
        selection_dir=selection_dir,
    )


def get_corpus_variants(
    config: dict, split: str, include: Optional[Iterable[str]] = None
) -> dict:
    """Each covered corpus's prediction variant.

    Limited to the corpora the run covers, so predictions for a corpus that was not
    run are neither looked for nor stored. A versioned corpus names its version here,
    which is what keeps predictions against two versions of it apart.
    """
    split_cfg = config["dataset"]["splits"].get(split, {})
    result = {}
    for corpus in included_corpora(config, split, include):
        corpus_cfg = split_cfg[corpus]
        if isinstance(corpus_cfg, dict):
            result[corpus] = corpus_cfg.get("variant", "v1")
        else:
            result[corpus] = "v1"
    return result
