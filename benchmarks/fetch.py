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
    CorpusSource,
    RepoReader,
    iter_partitioned_rows,
    iter_single_file_rows,
    read_all_ids,
    read_manifest_ids_by_stratum,
    resolve_source,
)
from benchmarks.sampling import positional_ids, stratified_ids

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


def _fetch(
    cfg: Mapping[str, Any],
    mode: str,
    split: str,
    data_dir: Path,
    with_pdf: bool,
    include: Optional[Iterable[str]] = None,
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
                corpus_dir=data_dir / split / source.corpus,
                with_pdf=with_pdf,
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
) -> List[Dict[str, str]]:
    LOGGER.info(
        "Fetching corpus %r from %s (mode=%s, n=%s)",
        source.corpus,
        source.describe(),
        mode,
        raw_n if raw_n is not None else "all",
    )
    picked, by_stratum = _select_ids(reader, source, raw_n, seed)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    records = _materialise(reader, source, picked, by_stratum, corpus_dir, with_pdf)
    LOGGER.info(
        "Corpus %r: %d record(s) available in %s",
        source.corpus,
        len(records),
        corpus_dir,
    )
    return records


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


def fetch_training_source(
    cfg: Dict[str, Any], mode: str, split: str, data_dir: Path
) -> List[Dict[str, str]]:
    """Fetch PDF + JATS XML for CC-BY corpora only.

    Reads ``cc_by_corpora`` from the config to determine which corpora are
    permitted.  Corpora absent from that list are silently skipped so that
    the allow-list can be extended without changing call sites.

    Takes no include list: a corpus outside `cc_by_corpora` is not opt-in here, it
    is refused, because this path generates training data that is published
    elsewhere.
    """
    allowed: Set[str] = set(cfg.get("cc_by_corpora", []))
    filtered_sampling = {
        m: {corpus: n for corpus, n in sizes.items() if corpus in allowed}
        for m, sizes in cfg.get("sampling", {}).items()
    }
    filtered_cfg = {**cfg, "sampling": filtered_sampling}
    in_split = allowed.intersection(cfg["dataset"]["splits"].get(split, {}))
    return fetch_data(filtered_cfg, mode, split, data_dir, include=in_split)
