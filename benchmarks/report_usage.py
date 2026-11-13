"""What a comparison's variants spent on tokens."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from benchmarks.llm_usage import usage_for_corpora


def _fmt_count(value: Optional[float]) -> str:
    return f"{round(value):,}" if value else "—"


def _fmt_credits(value: Optional[float], places: int = 4) -> str:
    return f"{value:.{places}f}" if value is not None else "—"


def _all_calls(usage: Optional[Dict[str, Any]]) -> int:
    """Every completion the engine made, replayed ones included. The token and
    credit figures beside it are what was *spent*, which excludes them."""
    if not usage:
        return 0
    return (usage.get("calls") or 0) + ((usage.get("replayed") or {}).get("calls") or 0)


def _usage_bullets(usage: Dict[str, Any]) -> List[str]:
    """Calls, tokens, credits and tasks, a line each.

    Absent figures are left out rather than dashed: most of a line would
    otherwise be dashes for a backend that reports only calls.
    """
    recorded = usage.get("n_with_usage") or 0
    attempted = usage.get("n_attempted") or 0
    bullets = [
        f"{_all_calls(usage):,} calls"
        + (f" over {recorded} of {attempted} docs" if recorded else "")
    ]

    tokens = [
        f"{usage[key]:,} tokens {name}"
        for key, name in (("input_tokens", "in"), ("output_tokens", "out"))
        if usage.get(key)
    ]
    if tokens:
        bullets.append(", ".join(tokens))
    if usage.get("cost_credits") is not None:
        bullets.append(f"{usage['cost_credits']:.4f} credits")

    by_task = usage.get("by_task") or {}
    if len(by_task) > 1:
        for task, entry in sorted(by_task.items()):
            line = f"{task}: {entry.get('calls', 0):,} calls"
            if entry.get("output_tokens"):
                line += f", {entry['output_tokens']:,} tokens out"
            bullets.append(line)
    return bullets


def _render_usage_lines(labeled_usage: List[Tuple[str, Optional[dict]]]) -> List[str]:
    """A block per variant that spent something.

    Only those: a row per variant made most of the old table dashes, since the
    CRF tools spend nothing.
    """
    lines: List[str] = []
    for label, usage in labeled_usage:
        if not usage or not _all_calls(usage):
            continue
        if lines:
            lines.append("")
        lines.append(f"**{label}**")
        lines += [f"* {bullet}" for bullet in _usage_bullets(usage)]
    return lines


def _cached_input_note(labeled_usage: List[Tuple[str, Optional[dict]]]) -> str:
    """Only where it happened, since a cost difference it explains is otherwise
    attributed to the variant."""
    cached = sum((usage or {}).get("cached_input_tokens") or 0 for _, usage in labeled_usage)
    if not cached:
        return ""
    return (
        f" Input tokens include {cached:,} served from the provider's own prefix"
        " cache, which is where a credit difference over the same input comes from."
    )


def _replayed_note(labeled_usage: List[Tuple[str, Optional[dict]]]) -> str:
    """Only where it happened, so a cold report is byte-for-byte unchanged.

    The credits named are what those responses cost when they were generated, not
    a forecast of a cold run: prices and providers move, and generation is not
    reproducible, so a fresh run would not make exactly the same calls.
    """
    replayed = [(usage or {}).get("replayed") or {} for _, usage in labeled_usage]
    replayed_calls = sum(entry.get("calls") or 0 for entry in replayed)
    if not replayed_calls:
        return ""
    total_calls = sum(_all_calls(usage) for _, usage in labeled_usage)
    costs = [
        entry["cost_credits"] for entry in replayed
        if entry.get("cost_credits") is not None
    ]
    when_generated = (
        f" Those responses cost {sum(costs):.4f} credits when they were generated,"
        " at whatever prices and providers applied then, which is a record of one"
        " cold run rather than a forecast of another."
        if costs else ""
    )
    return (
        f" Of {total_calls:,} calls, {replayed_calls:,} were replayed from the"
        " on-disk response cache and spent nothing, so the tokens and credits here"
        f" are what these runs spent rather than what a cold run would cost.{when_generated}"
    )


def render_usage_section(
    labeled_summaries: List[Tuple[str, dict]],
    corpora: List[str],
    note: str,
) -> List[str]:
    """Empty unless some variant recorded usage, so a CRF-only report is unchanged."""
    labeled_usage = [
        (label, usage_for_corpora(summary, corpora))
        for label, summary in labeled_summaries
    ]
    if not any(_all_calls(usage) for _, usage in labeled_usage):
        return []
    return [
        note + _cached_input_note(labeled_usage) + _replayed_note(labeled_usage),
        "",
        *_render_usage_lines(labeled_usage),
    ]
