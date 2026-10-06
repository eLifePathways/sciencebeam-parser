"""What each corpus and model is meant to be generated at.

`sampling` says how many documents a mode draws per corpus; it cannot say that one
corpus wants its references at `medium` and its segmentation at `smoke`. The
`generate` block states that, per corpus and model, and is what a rebuild reads to
decide what to fetch and what to generate.

A pair declared `none`, and a pair left out, mean the same thing: no data is wanted
for it. Writing `none` is how that decision is recorded rather than left to an
omission that reads like an oversight.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from benchmarks.corpus_source import CorpusConfigError

LOGGER = logging.getLogger(__name__)

__all__ = [
    "NOT_WANTED",
    "PairIntent",
    "get_declared_pairs",
    "group_by_corpus_and_mode",
    "validate_intent",
]

NOT_WANTED = "none"

INTENT_KEY = "generate"


@dataclasses.dataclass(frozen=True)
class PairIntent:
    """One corpus and model, and the mode its data is meant to be generated at."""

    corpus: str
    model: str
    mode: str


def _intent_block(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    block = cfg.get(INTENT_KEY) or {}
    if not isinstance(block, Mapping):
        raise CorpusConfigError(
            f"`{INTENT_KEY}` must map each corpus to its models, not {type(block).__name__}"
        )
    return block


def get_declared_pairs(cfg: Mapping[str, Any]) -> List[PairIntent]:
    """The pairs data is wanted for, in configuration order.

    A pair declared `none` is left out, because the rebuild has nothing to do for
    it. What is present and undeclared is a question for the listing, which reads
    the corpus rather than this.
    """
    pairs: List[PairIntent] = []
    for corpus, models in _intent_block(cfg).items():
        if not isinstance(models, Mapping):
            raise CorpusConfigError(
                f"`{INTENT_KEY}.{corpus}` must map each model to a mode, not"
                f" {type(models).__name__}"
            )
        for model, mode in models.items():
            if mode is None or str(mode) == NOT_WANTED:
                continue
            pairs.append(PairIntent(corpus=corpus, model=model, mode=str(mode)))
    return pairs


def group_by_corpus_and_mode(
    pairs: Iterable[PairIntent],
) -> Dict[Tuple[str, str], List[str]]:
    """The models to generate for each corpus and mode, which is what a run covers.

    Generation reads one source tree and takes a model list, so a corpus whose
    models sit at two modes is two runs rather than one.
    """
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for pair in pairs:
        grouped.setdefault((pair.corpus, pair.mode), []).append(pair.model)
    return grouped


def validate_intent(
    cfg: Mapping[str, Any],
    split: str,
    known_model_names: Optional[Sequence[str]] = None,
) -> None:
    """Refuse a declaration that names something the configuration does not define.

    Fetching is the expensive step and a typo in a mode or a corpus is cheap to
    catch, so this runs before it. `known_model_names` is optional because
    learning it means importing the generators, which costs seconds a listing
    should not pay.
    """
    allowed_corpora = set(cfg.get("cc_by_corpora") or ())
    in_split = set((cfg.get("dataset", {}).get("splits", {}) or {}).get(split) or ())
    sampling = cfg.get("sampling") or {}
    known_models: Optional[Set[str]] = (
        set(known_model_names) if known_model_names is not None else None
    )

    for pair in get_declared_pairs(cfg):
        where = f"{INTENT_KEY}.{pair.corpus}.{pair.model}"
        if pair.corpus not in allowed_corpora:
            raise CorpusConfigError(
                f"{where} declares corpus {pair.corpus!r}, which is not in"
                f" `cc_by_corpora`. Generated data is published, so a corpus"
                f" generates only once it is listed there"
            )
        if pair.corpus not in in_split:
            raise CorpusConfigError(
                f"{where} declares corpus {pair.corpus!r}, which split {split!r}"
                f" does not have. Available: {sorted(in_split)}"
            )
        if pair.mode not in sampling:
            raise CorpusConfigError(
                f"{where} declares mode {pair.mode!r}, which `sampling` does not"
                f" define. Available: {sorted(sampling)}"
            )
        if pair.corpus not in (sampling[pair.mode] or {}):
            raise CorpusConfigError(
                f"{where} declares mode {pair.mode!r}, which gives no sample size"
                f" for corpus {pair.corpus!r}"
            )
        if known_models is not None and pair.model not in known_models:
            raise CorpusConfigError(
                f"{where} declares model {pair.model!r}, which generation does not"
                f" produce. Available: {sorted(known_models)}"
            )
