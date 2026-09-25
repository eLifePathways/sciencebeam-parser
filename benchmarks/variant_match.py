"""How often the credited abstract was a translation rather than the article's own.

Scoring a prediction against whichever variant it matches best makes a parser that
systematically extracts the translation look like one that extracts the article's own
abstract. This counts the difference, per field and per corpus, from the same per-document
score files the scores are summarised from.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from benchmarks.variant_scoring import MATCHED_VARIANT_INDEX, VARIANT_COUNT

VARIANT_MATCH_KEY = "variant_match"


def _variant_scores(field_entry: dict) -> Optional[dict]:
    for method, scores in field_entry.items():
        if method == "scoring_type" or not isinstance(scores, dict):
            continue
        if VARIANT_COUNT in scores:
            return scores
    return None


def matched_variant_index(field_entry: dict) -> int:
    """Which gold variant the score was taken against, as the run recorded it."""
    scores = _variant_scores(field_entry)
    return scores.get(MATCHED_VARIANT_INDEX, 0) if scores else 0


def summarise_variant_matches(
    documents: Iterable[Dict[str, dict]],
    field_names: Sequence[str],
) -> Dict[str, dict]:
    """Per field, over the per-document score files of one corpus."""
    stats = {field: {"n_variants": 0, "n_translation": 0} for field in field_names}
    for fields in documents:
        for field in field_names:
            scores = _variant_scores(fields.get(field) or {})
            if not scores or scores.get(VARIANT_COUNT, 0) < 2:
                continue
            stats[field]["n_variants"] += 1
            if scores.get(MATCHED_VARIANT_INDEX):
                stats[field]["n_translation"] += 1
    return {field: entry for field, entry in stats.items() if entry["n_variants"]}


def merge_variant_matches(entries: Iterable[Optional[dict]]) -> Optional[dict]:
    """One count over several corpora, for a figure that spans them."""
    known = [entry for entry in entries if entry]
    if not known:
        return None
    return {
        key: sum(entry[key] for entry in known)
        for key in ("n_variants", "n_translation")
    }


def translation_counts(entry: Optional[dict]) -> str:
    """What one run credited, over the documents it scored.

    Both numbers are stated because runs covering different documents have different
    denominators, and a shared one would be wrong for all but the run it came from.
    """
    if entry is None:
        return "unknown"
    return f"{entry['n_translation']} of {entry['n_variants']}"


def variant_match_row(field: str, entries: Sequence[Optional[dict]]) -> Optional[List[str]]:
    """One table row: the field, and what each run credited. None where no document carries
    the field in more than one language."""
    if not any(entry and entry["n_variants"] for entry in entries):
        return None
    return [field] + [translation_counts(entry) for entry in entries]


def render_variant_match_table(
    variant_match: Dict[str, dict],
    field_names: Sequence[str],
) -> List[str]:
    rows = [
        variant_match_row(field, [variant_match.get(field)])
        for field in field_names
    ]
    present = [row for row in rows if row]
    if not present:
        return []
    return [
        "Where the gold carries a field in more than one language:",
        "",
        "| Field | Credited a translation |",
        "|---|---|",
        *["| " + " | ".join(row) + " |" for row in present],
        "",
    ]
