"""CLI: rebuild the generated corpus from what the config declares.

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

from benchmarks.fetch import fetch_training_source
from benchmarks.generate_training_data_cli import main as generate_training_data_main
from benchmarks.training_intent import (
    PairIntent,
    get_declared_pairs,
    group_by_corpus_and_mode,
    validate_intent,
)
from benchmarks.training_source_config import DEFAULT_CONFIG

LOGGER = logging.getLogger(__name__)

# What a rebuild of a pair replaces. A record a person wrote about the data --
# whether it was reviewed, what is wrong with it -- outlives the data it judged
# and is not something a machine step may remove.
MACHINE_WRITTEN_ENTRIES = ("corpus", "quality", "quality.jsonl", "provenance.json")


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


def clear_pair(pair_dir: Path) -> None:
    """Remove what a previous run generated, leaving anything a person wrote.

    Clearing is what makes a declaration that was lowered take effect: generation
    overwrites the documents it produces and has no opinion about the ones a
    larger mode left behind.
    """
    for name in MACHINE_WRITTEN_ENTRIES:
        entry = pair_dir / name
        if entry.is_dir():
            shutil.rmtree(entry)
        elif entry.exists():
            entry.unlink()


def _fetch_modes(
    cfg: dict,
    split: str,
    source_root: Path,
    pairs: Sequence[PairIntent],
) -> None:
    corpora_by_mode: Dict[str, List[str]] = {}
    for pair in pairs:
        corpora = corpora_by_mode.setdefault(pair.mode, [])
        if pair.corpus not in corpora:
            corpora.append(pair.corpus)
    for mode, corpora in corpora_by_mode.items():
        LOGGER.info("Fetching %s at mode %r", ", ".join(corpora), mode)
        fetch_training_source(cfg, mode, split, source_root / mode, include=corpora)


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
    for model in models:
        clear_pair(output_path / split / corpus / model)
    generate_training_data_main([
        "--config", config_path,
        "--source-data", str(source_root / mode),
        "--output-path", str(output_path),
        "--split", split,
        "--corpus", corpus,
        "--models", *models,
        *extra_argv,
    ])


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
        _fetch_modes(cfg, args.split, source_root, pairs)

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
