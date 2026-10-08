"""What a corpus or a field is called when someone outside the project reads it.

The identifiers are what `eval.yml` declares and the predictions store files things
under, so they stay as they are everywhere a name has to be matched. A chart is read
rather than matched, and `scielo_preprints-jats` says less there than SciELO Preprints
does.
"""
from __future__ import annotations

from typing import Dict

CORPUS_LABELS: Dict[str, str] = {
    "biorxiv": "bioRxiv",
    "ore": "ORE",
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


FIELD_LABELS: Dict[str, str] = {
    "title": "Title",
    "abstract": "Abstract",
    "abstract_anywhere": "Abstract (anywhere)",
    "abstract_any_language": "Abstract (any language)",
    "abstract_all_languages": "Abstract (all languages)",
    "abstract_legacy": "Abstract (legacy)",
    "abstract_language": "Abstract language",
    "author_full_names": "Authors",
    "affiliation_text": "Affiliations",
    "affiliation_linked": "Affiliations (linked)",
    "keywords": "Keywords",
    "body_section_titles": "Section titles",
    "acknowledgement": "Acknowledgement",
    "first_reference_text": "First reference",
    "reference_title": "Reference titles",
    "reference_doi": "Reference DOIs",
}


def field_label(field: str) -> str:
    """As for a corpus: the identifier where nothing friendlier is known."""
    return FIELD_LABELS.get(field, field)


# What each scoring method does, rather than what the library calls it. `levenshtein` and
# `edit_sim` run the same normalised edit distance; `edit_sim` strips punctuation and
# whitespace from both sides first.
METHOD_LABELS: Dict[str, str] = {
    "exact": "exact match",
    "levenshtein": "edit similarity",
    "edit_sim": "edit similarity, ignoring punctuation",
    "ratcliff_obershelp": "Ratcliff-Obershelp similarity",
    "soft": "soft match",
}


def method_label(method: str) -> str:
    """As for a corpus or a field: the identifier where nothing friendlier is known."""
    return METHOD_LABELS.get(method, method)
