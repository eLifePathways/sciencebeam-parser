"""CLI: build the generated corpus from what the config declares.

`generate` in the training-source config says which mode each corpus and model is
meant to be at. This reads it, fetches each mode it needs and generates each
corpus for the models declared at that mode, so that there is no second list of
models anywhere and no mode held only in whoever ran it last.

A corpus whose models sit at two modes is two runs: generation reads one source
tree and takes a model list, so the mode is a property of the run rather than of
each model within it.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from sciencebeam_parser.training.cli.generate_data import get_enabled_model_names
from sciencebeam_parser.training.quality.record import QUALITY_RECORD_DIRECTORY_NAME

from benchmarks.fetch import fetch_training_source
from benchmarks.generate_training_data_from_tree_cli import main as generate_from_tree_main
from benchmarks.training_records import PAIR_RECORD_FILENAME, read_source_manifest
from benchmarks.training_intent import (
    PairIntent,
    get_declared_pairs,
    group_by_corpus_and_mode,
    validate_intent,
)
from benchmarks.training_source_config import (
    DEFAULT_CONFIG,
    get_default_selection_path,
)

LOGGER = logging.getLogger(__name__)

# What a rebuild of a pair replaces: the training data, what measures it, and the
# record of where it came from. A record a person wrote about the data -- whether
# it was reviewed, what is wrong with it -- outlives the data it judged and is not
# something a machine step may remove.
MACHINE_WRITTEN_ENTRIES = (
    "corpus",
    QUALITY_RECORD_DIRECTORY_NAME,
    PAIR_RECORD_FILENAME,
)

# Where a rebuild holds a pair while it replaces it. Untracked, so a run that
# dies leaves it in the working tree to be seen rather than losing the pair.
REBUILD_STASH_NAME = ".rebuild-stash"


class SourceMissingError(FileNotFoundError):
    """A corpus to rebuild whose source documents are not on disk.

    Rebuilding clears a pair before it fills it, so a missing source has to stop
    the pair being cleared rather than leave it empty and report a skip.
    """


def select_pairs(
    pairs: Sequence[PairIntent],
    corpora: Optional[Sequence[str]],
    models: Optional[Sequence[str]],
) -> List[PairIntent]:
    """The declared pairs a run covers, narrowed by what was asked for."""
    wanted_corpora = set(corpora or ())
    wanted_models = set(models or ())
    return [
        pair
        for pair in pairs
        if (not wanted_corpora or pair.corpus in wanted_corpora)
        and (not wanted_models or pair.model in wanted_models)
    ]


def stash_pair(pair_dir: Path) -> Optional[Path]:
    """Move what the rebuild is about to replace aside, rather than deleting it.

    Clearing first is what makes a lowered declaration take effect, but a document
    the run then fails to produce -- a timeout, an error, a source row that has
    gone -- would be deleted by it and never written again. Moving instead means
    the rebuild can put back whatever it turns out not to have produced.

    What a person wrote beside the pair is not touched, here or anywhere.
    """
    stash = pair_dir / REBUILD_STASH_NAME
    if stash.exists():
        shutil.rmtree(stash)
    stashed = False
    for name in MACHINE_WRITTEN_ENTRIES:
        entry = pair_dir / name
        if not entry.exists():
            continue
        stash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(stash / name))
        stashed = True
    return stash if stashed else None


def _document_files(root: Path, document_id: str) -> List[Path]:
    prefix = f"{document_id}."
    return [
        path
        for path in root.rglob(f"{document_id}.*")
        if path.is_file() and path.name.startswith(prefix)
    ]


def restore_unproduced_documents(
    pair_dir: Path, stash: Optional[Path], document_ids: Sequence[str]
) -> List[str]:
    """Put back every document the rebuild was asked for and did not produce.

    A document outside the mode's selection is not offered here, so lowering a
    declaration still drops what it no longer names. What comes back is only what
    was asked for and could not be made again.
    """
    if stash is None or not stash.is_dir():
        return []
    restored = []
    for document_id in document_ids:
        if _document_files(pair_dir / "corpus", document_id):
            continue
        files = _document_files(stash, document_id)
        if not files:
            continue
        for path in files:
            target = pair_dir / path.relative_to(stash)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
        restored.append(document_id)
    return restored


def discard_stash(stash: Optional[Path]) -> None:
    if stash is not None and stash.is_dir():
        shutil.rmtree(stash)


def _fetch_modes(
    cfg: dict,
    split: str,
    source_root: Path,
    pairs: Sequence[PairIntent],
    selection_dir: Path,
) -> None:
    corpora_by_mode: Dict[str, List[str]] = {}
    for pair in pairs:
        corpora = corpora_by_mode.setdefault(pair.mode, [])
        if pair.corpus not in corpora:
            corpora.append(pair.corpus)
    for mode, corpora in corpora_by_mode.items():
        LOGGER.info("Fetching %s at mode %r", ", ".join(corpora), mode)
        fetch_training_source(
            cfg, mode, split, source_root / mode,
            include=corpora, selection_dir=selection_dir,
        )


def _generate_group(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config_path: str,
    corpus: str,
    mode: str,
    models: Sequence[str],
    split: str,
    source_root: Path,
    output_path: Path,
    extra_argv: Sequence[str],
) -> None:
    corpus_source = source_root / mode / split / corpus
    if not corpus_source.is_dir():
        raise SourceMissingError(
            f"no source data for {corpus!r} at mode {mode!r}: {corpus_source}."
            f" Fetch it, or run without --skip-fetch"
        )
    asked_for = _documents_asked_for(corpus_source)
    pair_dirs = [output_path / split / corpus / model for model in models]
    stashes = [stash_pair(pair_dir) for pair_dir in pair_dirs]
    generate_from_tree_main([
        "--config", config_path,
        "--source-data", str(source_root / mode),
        "--output-path", str(output_path),
        "--split", split,
        "--corpus", corpus,
        "--models", *models,
        *extra_argv,
    ])
    for pair_dir, stash in zip(pair_dirs, stashes):
        restored = restore_unproduced_documents(pair_dir, stash, asked_for)
        if restored:
            LOGGER.warning(
                "%s: kept %d document(s) this run did not produce: %s",
                pair_dir.name,
                len(restored),
                ", ".join(restored),
            )
        discard_stash(stash)


def _documents_asked_for(corpus_source: Path) -> List[str]:
    """Every document the mode names, whether or not its source can still be read."""
    manifest = read_source_manifest(corpus_source)
    if manifest is None:
        return []
    return list(manifest.selected_document_ids) + list(manifest.missing_document_ids)


def _parse_args(argv: Optional[Sequence[str]]) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Rebuild the generated corpus from the declared modes.",
        epilog="Any additional arguments are forwarded to generate_data.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory holding one source tree per mode (e.g. data/source-training-data)",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Root directory of the output repo",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--corpus",
        nargs="+",
        help="Rebuild only these corpora (default: every declared one)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        help="Rebuild only these models (default: every declared one)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Generate from the source trees already on disk",
    )
    parser.add_argument(
        "--selection-path",
        help=(
            "Directory holding each corpus's recorded selection"
            " (default: a selection directory beside the config)"
        ),
    )
    return parser.parse_known_args(argv)


def main(argv=None):
    args, extra_argv = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_intent(cfg, args.split, known_model_names=get_enabled_model_names(None))

    pairs = select_pairs(get_declared_pairs(cfg), args.corpus, args.model)
    if not pairs:
        LOGGER.warning("Nothing declared to rebuild in %s", args.config)
        sys.exit(0)

    source_root = Path(args.source_root)
    output_path = Path(args.output_path)
    if not args.skip_fetch:
        _fetch_modes(
            cfg, args.split, source_root, pairs,
            Path(args.selection_path) if args.selection_path
            else get_default_selection_path(args.config),
        )

    errors = []
    for (corpus, mode), models in group_by_corpus_and_mode(pairs).items():
        LOGGER.info("Rebuilding %s at mode %r for: %s", corpus, mode, ", ".join(models))
        try:
            _generate_group(
                args.config,
                corpus,
                mode,
                models,
                args.split,
                source_root,
                output_path,
                extra_argv,
            )
        except SystemExit as exc:
            if exc.code not in (None, 0):
                errors.append(f"{corpus}@{mode}")
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Failed to rebuild %s at mode %r", corpus, mode)
            errors.append(f"{corpus}@{mode}")

    if errors:
        LOGGER.error("Rebuild failed for: %s", ", ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
