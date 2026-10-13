"""CLI: fetch the source documents the training-source config declares.

`generate` says which mode each corpus and model is meant to be at, and a fetch
needs the mode and the corpora at it. So this reads the same declaration the
rebuild does and fetches every mode it names, into the per-mode layout generation
reads from -- which is what makes "fetch what the config declares" and "build from
the trees" two commands rather than one doing both.

This is the one that reads the dataset. The rebuild takes what the source trees
and their manifests already record, so growing a selection happens here.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from benchmarks.fetch import fetch_training_source
from benchmarks.training_intent import get_declared_pairs, group_corpora_by_mode
from benchmarks.training_source_config import (
    DEFAULT_CONFIG,
    get_default_selection_path,
)

LOGGER = logging.getLogger(__name__)


def corpora_by_mode(
    cfg: dict,
    modes: Optional[Sequence[str]] = None,
    corpora: Optional[Sequence[str]] = None,
) -> Dict[str, Optional[List[str]]]:
    """Which corpora to fetch at each mode, as the declaration and the arguments say.

    A value of None means every corpus the allow-list covers: a mode named on the
    command line that nothing declares has no corpus list to take, and fetching a
    one-off tree is what naming a mode is for.
    """
    declared = group_corpora_by_mode(get_declared_pairs(cfg))
    wanted = set(corpora or ())
    if wanted:
        declared = {
            mode: [corpus for corpus in at_mode if corpus in wanted]
            for mode, at_mode in declared.items()
        }
        declared = {mode: at_mode for mode, at_mode in declared.items() if at_mode}
    if not modes:
        return dict(declared)

    chosen: Dict[str, Optional[List[str]]] = {}
    for mode in modes:
        if mode in declared:
            chosen[mode] = declared[mode]
            continue
        LOGGER.warning(
            "Nothing declares mode %r, so fetching every allowed corpus at it", mode
        )
        chosen[mode] = list(corpora) if corpora else None
    return chosen


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the CC-BY PDF + JATS XML pairs the training-source config"
            " declares, one tree per mode."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=(
            "Path to the training-source config in the generated repo"
            f" (default: {DEFAULT_CONFIG})"
        ),
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory to hold one source tree per mode (e.g. data/source-training-data)",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        help=(
            "Fetch only these modes (default: every mode the config declares)."
            " A mode nothing declares fetches every allowed corpus at it"
        ),
    )
    parser.add_argument(
        "--corpus",
        nargs="+",
        help="Fetch only these corpora (default: every declared one)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to fetch (default: train)",
    )
    parser.add_argument(
        "--selection-path",
        help=(
            "Directory holding each corpus's recorded selection"
            " (default: a selection directory beside the config)"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selection_dir = (
        Path(args.selection_path) if args.selection_path
        else get_default_selection_path(args.config)
    )
    source_root = Path(args.source_root)

    wanted = corpora_by_mode(cfg, args.mode, args.corpus)
    if not wanted:
        LOGGER.warning("Nothing declared to fetch in %s", args.config)
        return

    for mode, corpora in wanted.items():
        LOGGER.info(
            "Fetching %s at mode %r",
            ", ".join(corpora) if corpora else "every allowed corpus",
            mode,
        )
        records = fetch_training_source(
            cfg, mode, args.split, source_root / mode,
            include=corpora, selection_dir=selection_dir,
        )
        LOGGER.info(
            "Fetched %d record(s) to %s", len(records), source_root / mode
        )


if __name__ == "__main__":
    main()
