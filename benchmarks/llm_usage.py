from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SUM_KEYS = (
    "calls", "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"
)
MAX_KEYS = ("peak_output_tokens",)
UNION_KEYS = ("models", "providers")
COUNT_KEYS = ("n_attempted", "n_with_usage")
# Same shape as an entry, so it combines the same way.
NESTED_KEYS = ("replayed",)


def _combine_cost(entries: List[dict]) -> Dict[str, Any]:
    """Absent rather than zero: a self-hosted endpoint reports no cost, and a zero
    would read as free."""
    costs = [
        float(entry["cost_credits"]) for entry in entries
        if entry.get("cost_credits") is not None
    ]
    return {"cost_credits": sum(costs)} if costs else {}


def _combine_unions(entries: List[dict]) -> Dict[str, Any]:
    union: Dict[str, List[str]] = {key: [] for key in UNION_KEYS}
    for entry in entries:
        for key in UNION_KEYS:
            for value in entry.get(key) or []:
                if value not in union[key]:
                    union[key].append(value)
    return {key: values for key, values in union.items() if values}


def _combine_grouped(entries: List[dict], key: str) -> Dict[str, Any]:
    """`by_task` holds one sub-entry per task, combined the same way as the whole."""
    grouped: Dict[str, List[dict]] = {}
    for entry in entries:
        for name, sub_entry in (entry.get(key) or {}).items():
            grouped.setdefault(name, []).append(sub_entry)
    if not grouped:
        return {}
    return {key: {
        name: combine_usage(sub_entries)
        for name, sub_entries in sorted(grouped.items())
    }}


def _combine_nested(entries: List[dict], key: str) -> Dict[str, Any]:
    """`replayed` has the same shape as an entry, so it combines the same way."""
    nested = [entry[key] for entry in entries if entry.get(key)]
    return {key: combine_usage(nested)} if nested else {}


def combine_usage(usage_entries: Iterable[dict]) -> Dict[str, Any]:
    """Sum usage entries, keeping the peak a peak and cost absent where unstated."""
    entries = list(usage_entries)
    combined: Dict[str, Any] = {
        key: sum(entry.get(key) or 0 for entry in entries) for key in SUM_KEYS
    }
    for key in MAX_KEYS:
        combined[key] = max((entry.get(key) or 0 for entry in entries), default=0)
    combined.update(_combine_cost(entries))
    combined.update(_combine_unions(entries))
    for key in NESTED_KEYS:
        combined.update(_combine_nested(entries, key))
    combined.update(_combine_grouped(entries, "by_task"))
    return combined


def read_manifest_entries(run_dir: Path) -> List[dict]:
    manifest_path = run_dir / "predictions" / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate_llm_usage(
    manifest_entries: List[dict], corpora: Iterable[str]
) -> Dict[str, Any]:
    """Per corpus, what the LLM spent producing these predictions.

    Over every document attempted, errors included, which is deliberately not the
    subset the scores cover. Usage is summed per attempt, since a re-run paid
    again, while the document counts are of distinct records.
    """
    if not any(entry.get("llm_usage") for entry in manifest_entries):
        return {}

    usage_by_corpus: Dict[str, Any] = {}
    for corpus in corpora:
        corpus_entries = [
            entry for entry in manifest_entries if entry.get("corpus") == corpus
        ]
        if not corpus_entries:
            continue
        usage_by_corpus[corpus] = {
            "n_attempted": len({entry.get("record_id") for entry in corpus_entries}),
            "n_with_usage": len({
                entry.get("record_id") for entry in corpus_entries
                if entry.get("llm_usage")
            }),
            **combine_usage([
                entry["llm_usage"] for entry in corpus_entries if entry.get("llm_usage")
            ]),
        }
    return usage_by_corpus


def usage_for_corpora(summary: dict, corpora: Iterable[str]) -> Optional[Dict[str, Any]]:
    """One variant's usage over the named corpora, or None where it recorded none."""
    per_corpus = summary.get("llm_usage") or {}
    selected = [per_corpus[corpus] for corpus in corpora if corpus in per_corpus]
    if not selected:
        return None
    return {
        **combine_usage(selected),
        **{
            key: sum(entry.get(key) or 0 for entry in selected)
            for key in COUNT_KEYS
        },
    }
