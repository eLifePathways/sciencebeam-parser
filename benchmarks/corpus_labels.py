"""What a corpus is called when someone outside the project reads it.

The identifiers are what `eval.yml` declares and the predictions store files things
under, so they stay as they are everywhere a name has to be matched. A chart is read
rather than matched, and `scielo_preprints-jats` says less there than SciELO Preprints
does.
"""
from __future__ import annotations

from typing import Dict

CORPUS_LABELS: Dict[str, str] = {
    "biorxiv": "bioRxiv",
    "ore": "Open Research Europe",
    "pkp": "PKP",
    "plos-manuscripts": "PLOS",
    "scielo_br": "SciELO Brazil",
    "scielo_mx": "SciELO Mexico",
    "scielo_preprints-jats": "SciELO Preprints",
}


def corpus_label(corpus: str) -> str:
    """The identifier itself where nothing friendlier is known, so a corpus added to
    `eval.yml` charts under its own name rather than failing."""
    return CORPUS_LABELS.get(corpus, corpus)
