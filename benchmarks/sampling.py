"""Choosing which documents a mode's sample contains.

Two strategies, because two kinds of corpus want different things. An unstratified
corpus samples positions out of a shuffle, which is what every number already
recorded for those corpora depends on. A stratified corpus samples across its
strata, so that a small mode exercises every stratum rather than whichever ones the
draw happened to favour.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

LOGGER = logging.getLogger(__name__)


def sample_indices(n_total: int, n: int, seed: int) -> Set[int]:
    """Return a deterministic set of n indices from [0, n_total).

    Smaller n always produces a subset of larger n with the same seed,
    so smoke ⊂ small ⊂ medium ⊂ large ⊂ full for each corpus.
    """
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)
    return set(int(i) for i in all_indices[:n])


def stratum_order(strata: Iterable[str], seed: int) -> List[str]:
    """The order strata are served in, which is what decides who gets a remainder.

    Keyed on each stratum's own name, so adding one leaves the rest in the order
    they were already in. Shuffling the observed list instead would reorder
    everything, and a sample taken after a corpus gained a stratum would no longer
    contain the sample taken before it.
    """
    return sorted(
        strata, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode("utf-8")).hexdigest()
    )


def stratified_order(
    ids_by_stratum: Mapping[str, Sequence[str]], seed: int
) -> List[str]:
    """Every document, ordered round-robin across strata.

    Each stratum's first document, then each stratum's second, and so on, with a
    stratum dropping out once it is exhausted. Sampling is then a prefix of this
    one sequence, which is what makes balance, the remainder rule, exhausted
    strata and nesting per stratum the same property rather than four rules that
    have to agree with each other.

    Each stratum's own documents must arrive already ordered — for a corpus cut
    from an archive that is its published rank, so that a later version of the
    corpus, which only ever adds higher ranks, extends this sequence rather than
    reshuffling it.
    """
    order = stratum_order(ids_by_stratum, seed)
    longest = max((len(ids_by_stratum[stratum]) for stratum in order), default=0)
    return [
        ids_by_stratum[stratum][position]
        for position in range(longest)
        for stratum in order
        if position < len(ids_by_stratum[stratum])
    ]


def stratified_ids(
    ids_by_stratum: Mapping[str, Sequence[str]], n: Optional[int], seed: int
) -> List[str]:
    """The first n of the round-robin order, or all of it when n is None."""
    total = stratified_order(ids_by_stratum, seed)
    if n is None:
        return total
    if n < len(ids_by_stratum):
        LOGGER.warning(
            "Sample size %d is smaller than the %d strata available, so %d of them "
            "contribute nothing to this sample",
            n,
            len(ids_by_stratum),
            len(ids_by_stratum) - n,
        )
    return total[:n]


def positional_ids(all_ids: Sequence[str], n: Optional[int], seed: int) -> List[str]:
    """The ids at the sampled positions, in the order the corpus stores them.

    Selecting by id rather than by position is what lets a corpus be read from
    more than one file without depending on the order the files are discovered
    in. It is the same sample as selecting positions directly, which duplicate ids
    would quietly break, so those are rejected rather than tolerated.
    """
    duplicates = len(all_ids) - len(set(all_ids))
    if duplicates:
        raise ValueError(
            f"corpus has {duplicates} duplicate id(s); sampling identifies documents "
            f"by id, so ids have to be unique"
        )
    picked = sample_indices(
        len(all_ids), len(all_ids) if n is None else min(n, len(all_ids)), seed
    )
    return [record_id for index, record_id in enumerate(all_ids) if index in picked]
