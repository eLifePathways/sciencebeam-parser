"""Which documents the gold records a field for, and what was produced where it does not.

A field a publisher may simply not record carries two questions in one score: how well it
is extracted, and how often a model abstains. Nothing in the PDF separates them, so they
are reported apart rather than averaged together.

A document whose gold records nothing contributes 0 to both `sim_sum` and `expected_count`,
so it can only enlarge precision's denominator. The conditional and combined figures
therefore differ exactly where a model produced a value the gold has none of, which is also
what decides whether a field is worth reporting separately at all.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

GOLD_PRESENCE_KEY = "gold_presence"
GOLD_PRESENT_AGGREGATED_KEY = "aggregated_gold_present"


def _continuous_scores(field_entry: dict) -> Iterable[dict]:
    for method, scores in field_entry.items():
        if method == "scoring_type" or not isinstance(scores, dict):
            continue
        if "expected_count" in scores:
            yield scores


def gold_records_field(field_entry: dict) -> Optional[bool]:
    """Whether the gold records a value for this field, or None where no method can say.

    `expected_count` counts non-empty gold values. `expected_something` counts the field
    existing at all, which on a padded list is true of a document whose values are all
    empty: forty references without a DOI give a forty item gold list and no gold DOI.
    Only a continuous method distinguishes them.
    """
    for scores in _continuous_scores(field_entry):
        return scores["expected_count"] > 0
    return None


def predicted_value_count(field_entry: dict) -> int:
    for scores in _continuous_scores(field_entry):
        return scores.get("predicted_count", 0)
    return 0


def summarise_gold_presence(
    documents: Iterable[Dict[str, dict]],
    field_names: Sequence[str],
) -> Dict[str, dict]:
    """Per field, over the per-document score files of one corpus."""
    stats = {
        field: {"n": 0, "n_gold": 0, "no_gold_docs": 0, "no_gold_values": 0}
        for field in field_names
    }
    for fields in documents:
        for field in field_names:
            entry = fields.get(field)
            if not entry:
                continue
            records = gold_records_field(entry)
            if records is None:
                continue
            field_stats = stats[field]
            field_stats["n"] += 1
            if records:
                field_stats["n_gold"] += 1
                continue
            predicted = predicted_value_count(entry)
            if predicted:
                field_stats["no_gold_docs"] += 1
                field_stats["no_gold_values"] += predicted
    return {field: s for field, s in stats.items() if s["n"]}


def has_gold(presence: Optional[dict]) -> bool:
    """False only where the corpus is known to record nothing, so an absent figure
    reads as unknown rather than as zero."""
    return not presence or bool(presence["n_gold"])


def is_split_worth_reporting(presence_by_variant: Iterable[Optional[dict]]) -> bool:
    """Some variant produced a value the gold has none of, or the corpus records none at
    all. The first is exactly when the conditional and combined figures differ; the second
    is where both are computed over nothing and an f1 of 0.000 says nothing."""
    return any(
        presence["no_gold_docs"] or not presence["n_gold"]
        for presence in presence_by_variant
        if presence
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def merge_presence(presences: Iterable[Optional[dict]]) -> Optional[dict]:
    """One presence over several corpora, for a figure that spans them."""
    known = [presence for presence in presences if presence]
    if not known:
        return None
    return {
        key: sum(presence[key] for presence in known)
        for key in ("n", "n_gold", "no_gold_docs", "no_gold_values")
    }


def produced_counts(presence: Optional[dict]) -> str:
    """What one variant produced on the documents whose gold records nothing.

    Both counts are stated because on a list field the document count understates it:
    twenty-two documents carried a hundred and eighty-three section titles. `unknown` and
    `none` are opposite conclusions, so a summary written before the split never reads as
    a model that abstained.
    """
    if presence is None:
        return "unknown"
    if not presence["no_gold_docs"]:
        return "none"
    return (
        f"{_plural(presence['no_gold_docs'], 'doc')}"
        f", {_plural(presence['no_gold_values'], 'value')}"
    )


def produced_row(field: str, presences: Sequence[Optional[dict]]) -> Optional[List[str]]:
    """One table row: the field, how many documents its gold records nothing for, and what
    each variant produced on them. None where the gold records every document."""
    no_gold = max(
        (presence["n"] - presence["n_gold"] for presence in presences if presence),
        default=0,
    )
    if not no_gold:
        return None
    return [field, str(no_gold)] + [produced_counts(presence) for presence in presences]
