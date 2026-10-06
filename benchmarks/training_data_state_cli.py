"""CLI: what the generated corpus holds, against what the config declares.

The declaration says which mode each corpus and model is meant to be at; the
record beside each pair says which one it was generated at. Where they disagree is
what prompts a rebuild, and it is a comparison of two strings rather than an
inference from how many files a directory holds -- which is what counting got
wrong, since `citation` writes no `tei/` directory and reads as empty while
holding its whole corpus.

This deliberately does not import the generators: learning the real model names
costs seconds, and a listing that is slow is a listing nobody runs.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from benchmarks.training_intent import get_declared_pairs
from benchmarks.training_records import PAIR_RECORD_FILENAME

LOGGER = logging.getLogger(__name__)

OK = "ok"
DRIFTED = "drifted"
UNKNOWN = "unknown"
MISSING = "missing"
UNDECLARED = "undeclared"

# A pair at its declared mode is the only state a rebuild has nothing to do about.
NEEDS_ATTENTION = (DRIFTED, UNKNOWN, MISSING, UNDECLARED)

NOT_DECLARED = "-"


@dataclasses.dataclass(frozen=True)
class PairState:
    corpus: str
    model: str
    declared: str
    recorded: str
    status: str


def _pair_dir(root: Path, split: str, corpus: str, model: str) -> Path:
    return root / split / corpus / model


def _has_data(pair_dir: Path) -> bool:
    """Whether a pair holds generated data, without counting it.

    Each model lays its files out differently -- `citation` writes into `corpus/`
    directly where `segmentation` writes `corpus/tei/` -- so the question is
    whether anything is there, not how much.
    """
    corpus_dir = pair_dir / "corpus"
    return corpus_dir.is_dir() and any(corpus_dir.iterdir())


def _recorded_mode(pair_dir: Path) -> Optional[str]:
    record_path = pair_dir / PAIR_RECORD_FILENAME
    if not record_path.is_file():
        return None
    try:
        record: Dict[str, Any] = json.loads(record_path.read_text(encoding="utf-8"))
    except ValueError:
        LOGGER.warning("ignoring unreadable pair record: %s", record_path)
        return None
    return record.get("mode")


def _declared_state(
    root: Path, split: str, corpus: str, model: str, declared: str
) -> PairState:
    pair_dir = _pair_dir(root, split, corpus, model)
    recorded = _recorded_mode(pair_dir)
    if not _has_data(pair_dir):
        status = MISSING
    elif recorded is None:
        status = UNKNOWN
    elif recorded == declared:
        status = OK
    else:
        status = DRIFTED
    return PairState(
        corpus=corpus,
        model=model,
        declared=declared,
        recorded=recorded or NOT_DECLARED,
        status=status,
    )


def get_pair_states(cfg: dict, root: Path, split: str) -> List[PairState]:
    """Every declared pair, and every pair holding data that nothing declares."""
    states = [
        _declared_state(root, split, pair.corpus, pair.model, pair.mode)
        for pair in get_declared_pairs(cfg)
    ]
    declared = {(state.corpus, state.model) for state in states}

    split_dir = root / split
    if split_dir.is_dir():
        for corpus_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for pair_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
                key = (corpus_dir.name, pair_dir.name)
                if key in declared or not _has_data(pair_dir):
                    continue
                states.append(PairState(
                    corpus=key[0],
                    model=key[1],
                    declared=NOT_DECLARED,
                    recorded=_recorded_mode(pair_dir) or NOT_DECLARED,
                    status=UNDECLARED,
                ))
    return states


def format_states(states: Sequence[PairState]) -> str:
    rows = [("corpus", "model", "declared", "recorded", "status")] + [
        (state.corpus, state.model, state.declared, state.recorded, state.status)
        for state in states
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
        for row in rows
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="List the generated corpus against the modes the config declares."
    )
    parser.add_argument("--config", default="benchmarks/training-source.yml")
    parser.add_argument(
        "--training-data",
        required=True,
        help="Root directory of the generated data repo",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when any pair is not at its declared mode",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    states = get_pair_states(cfg, Path(args.training_data), args.split)
    print(format_states(states))

    if args.check and any(state.status in NEEDS_ATTENTION for state in states):
        sys.exit(1)


if __name__ == "__main__":
    main()
