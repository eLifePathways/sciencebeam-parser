from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)


def _sample_indices(n_total: int, n: int, seed: int) -> set:
    """Return a deterministic set of n indices from [0, n_total).

    Smaller n always produces a subset of larger n with the same seed,
    so smoke ⊂ small ⊂ medium ⊂ large ⊂ full for each corpus.
    """
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)
    return set(int(i) for i in all_indices[:n])


def _get_corpus_filename_and_id_column(entry: Any) -> Tuple[str, str]:
    if isinstance(entry, dict):
        return entry["file"], entry.get("id_column", "id")
    return str(entry), "id"


def fetch_data(  # pylint: disable=too-many-locals
    cfg: Dict[str, Any], mode: str, split: str, data_dir: Path
) -> List[Dict[str, str]]:
    """Download and materialise PDF + gold XML for each corpus.

    Idempotent: skips records whose files already exist.
    Returns a list of records: {corpus, record_id, pdf_path, xml_path}.
    Data is stored under data_dir/<split>/<corpus>/ so train and validation
    records never mix in the cache.
    """
    sample_sizes = cfg["sampling"][mode]
    seed = cfg["seeds"]["sample"]
    token = os.environ.get("HF_TOKEN")
    local_root = os.environ.get("BENCH_LOCAL_PARQUET_DIR")

    split_corpora = cfg["dataset"]["splits"].get(split)
    if not split_corpora:
        raise ValueError(f"Unknown split {split!r}. Available: {list(cfg['dataset']['splits'])}")

    records: List[Dict[str, str]] = []

    for corpus, corpus_cfg in split_corpora.items():
        if corpus not in sample_sizes:
            LOGGER.warning(
                "No sample size configured for corpus %r in mode %r, skipping", corpus, mode
            )
            continue

        filename, id_column = _get_corpus_filename_and_id_column(corpus_cfg)
        LOGGER.info("Fetching corpus %r (mode=%s, n=%d)", corpus, mode, sample_sizes[corpus])

        if local_root:
            parquet_path = str(Path(local_root) / filename)
        else:
            parquet_path = hf_hub_download(
                repo_id=cfg["dataset"]["repo_id"],
                filename=filename,
                revision=cfg["dataset"]["revision"],
                repo_type="dataset",
                token=token,
            )

        # Two-pass read: first load only IDs to pick the sample, then materialise
        # only the selected rows to avoid loading all PDFs into memory.
        pf = pq.ParquetFile(parquet_path)
        all_ids = pf.read(columns=[id_column]).column(id_column).to_pylist()
        n = min(sample_sizes[corpus], len(all_ids))
        picked = _sample_indices(len(all_ids), n, seed)

        corpus_dir = data_dir / split / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)

        global_i = 0
        for batch in pf.iter_batches(batch_size=64, columns=[id_column, "pdf", "xml"]):
            for row in range(batch.num_rows):
                if global_i in picked:
                    record_id = str(batch.column(id_column)[row].as_py()).replace("/", "_")
                    pdf_path = corpus_dir / f"{record_id}.pdf"
                    xml_path = corpus_dir / f"{record_id}.jats.xml"

                    if not pdf_path.exists():
                        pdf_path.write_bytes(bytes(batch.column("pdf")[row].as_py()))
                    if not xml_path.exists():
                        xml_path.write_text(str(batch.column("xml")[row].as_py()), encoding="utf-8")

                    records.append({
                        "corpus": corpus,
                        "record_id": record_id,
                        "pdf_path": str(pdf_path),
                        "xml_path": str(xml_path),
                    })
                global_i += 1

        LOGGER.info("Corpus %r: materialised %d records to %s", corpus, n, corpus_dir)

    return records


def fetch_gold(  # pylint: disable=too-many-locals
    cfg: Dict[str, Any], mode: str, split: str, data_dir: Path
) -> List[Dict[str, str]]:
    """Download gold XML only (no PDFs) for each corpus.

    Idempotent: skips files that already exist.
    Returns records: {corpus, record_id, xml_path}.
    """
    sample_sizes = cfg["sampling"][mode]
    seed = cfg["seeds"]["sample"]
    token = os.environ.get("HF_TOKEN")
    local_root = os.environ.get("BENCH_LOCAL_PARQUET_DIR")

    split_corpora = cfg["dataset"]["splits"].get(split)
    if not split_corpora:
        raise ValueError(f"Unknown split {split!r}. Available: {list(cfg['dataset']['splits'])}")

    records: List[Dict[str, str]] = []

    for corpus, corpus_cfg in split_corpora.items():
        if corpus not in sample_sizes:
            LOGGER.warning(
                "No sample size configured for corpus %r in mode %r, skipping", corpus, mode
            )
            continue

        filename, id_column = _get_corpus_filename_and_id_column(corpus_cfg)
        LOGGER.info(
            "Fetching gold for corpus %r (mode=%s, n=%d)", corpus, mode, sample_sizes[corpus]
        )

        if local_root:
            parquet_path = str(Path(local_root) / filename)
        else:
            parquet_path = hf_hub_download(
                repo_id=cfg["dataset"]["repo_id"],
                filename=filename,
                revision=cfg["dataset"]["revision"],
                repo_type="dataset",
                token=token,
            )

        pf = pq.ParquetFile(parquet_path)
        all_ids = pf.read(columns=[id_column]).column(id_column).to_pylist()
        n = min(sample_sizes[corpus], len(all_ids))
        picked = _sample_indices(len(all_ids), n, seed)

        corpus_dir = data_dir / split / corpus
        corpus_dir.mkdir(parents=True, exist_ok=True)

        global_i = 0
        for batch in pf.iter_batches(batch_size=64, columns=[id_column, "xml"]):
            for row in range(batch.num_rows):
                if global_i in picked:
                    record_id = str(batch.column(id_column)[row].as_py()).replace("/", "_")
                    xml_path = corpus_dir / f"{record_id}.jats.xml"

                    if not xml_path.exists():
                        xml_path.write_text(
                            str(batch.column("xml")[row].as_py()), encoding="utf-8"
                        )

                    records.append({
                        "corpus": corpus,
                        "record_id": record_id,
                        "xml_path": str(xml_path),
                    })
                global_i += 1

        LOGGER.info("Corpus %r: materialised %d gold records to %s", corpus, n, corpus_dir)

    return records
