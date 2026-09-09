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


def combine_usage(usage_entries: Iterable[dict]) -> Dict[str, Any]:
    """Sum usage entries, keeping the peak a peak and cost absent where unstated."""
    combined: Dict[str, Any] = {key: 0 for key in SUM_KEYS + MAX_KEYS}
    cost: Optional[float] = None
    union: Dict[str, List[str]] = {key: [] for key in UNION_KEYS}
    by_task: Dict[str, List[dict]] = {}

    for entry in usage_entries:
        for key in SUM_KEYS:
            combined[key] += entry.get(key) or 0
        for key in MAX_KEYS:
            combined[key] = max(combined[key], entry.get(key) or 0)
        if entry.get("cost_credits") is not None:
            cost = (cost or 0.0) + float(entry["cost_credits"])
        for key in UNION_KEYS:
            for value in entry.get(key) or []:
                if value not in union[key]:
                    union[key].append(value)
        for task, task_entry in (entry.get("by_task") or {}).items():
            by_task.setdefault(task, []).append(task_entry)

    if cost is not None:
        combined["cost_credits"] = cost
    for key in UNION_KEYS:
        if union[key]:
            combined[key] = union[key]
    if by_task:
        combined["by_task"] = {
            task: combine_usage(task_entries)
            for task, task_entries in sorted(by_task.items())
        }
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
