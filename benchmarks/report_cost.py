"""What a benchmark run spent to produce its predictions, beside what they scored.

Latency, throughput and the machine each run was measured on. Separate from the scores
because a timing delta is partly a property of the measurement rather than of the tool.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from benchmarks.compute_cost import format_duration


def _fmt_ms(value: int) -> str:
    return f"{value / 1000:.1f}s" if value >= 1000 else f"{value}ms"


def _throughput_bullet(cost: dict) -> List[str]:
    """Predictions an hour, over every run that produced any of them.

    Over what the runs produced rather than what they attempted, so that a run
    which reached no parser and failed everything in a tenth of a second does not
    report the fastest throughput in the file. Where the two differ, both are
    stated: the wall clock covers the failures as well.
    """
    n_predicted = cost.get("n_predicted")
    elapsed_s = cost.get("elapsed_s")
    if not n_predicted or not elapsed_s:
        return []
    bullet = f"{n_predicted:,} docs in {format_duration(elapsed_s)}"
    n_attempted = cost.get("n_attempted") or n_predicted
    if n_attempted > n_predicted:
        bullet += f" ({n_attempted:,} attempted)"
    n_runs = cost.get("n_runs") or 1
    if n_runs > 1:
        bullet += f" over {n_runs} runs"
    concurrency = cost.get("concurrency") or []
    if concurrency:
        bullet += " at concurrency " + "/".join(str(value) for value in concurrency)
    return [f"{bullet} — {n_predicted / elapsed_s * 3600:,.0f} docs/hour"]


def _latency_bullet(cost: dict) -> List[str]:
    latency = cost.get("latency_ms") or {}
    if not latency.get("median"):
        return []
    return [
        f"{_fmt_ms(latency['median'])} median latency, {_fmt_ms(latency['p90'])} p90"
        f" over {latency.get('n', 0):,} docs"
    ]


def _machine_label(machine: dict) -> str:
    cpu_count = machine.get("cpu_count")
    return ", ".join(filter(None, [
        machine.get("cpu_model"), f"{cpu_count} cores" if cpu_count else "",
    ]))


def _cpu_bullets(cost: dict) -> List[str]:
    cpu_seconds = cost.get("cpu_seconds")
    cpu_n_predicted = cost.get("cpu_n_predicted") or 0
    n_predicted = cost.get("n_predicted") or 0
    machines = cost.get("machines") or []
    bullets = []
    if cpu_seconds and cpu_n_predicted:
        bullet = f"{cpu_seconds / cpu_n_predicted:.1f} CPU-seconds per document"
        # Only part of a set is measured where another part was generated against a
        # parser on another host, and a rate over all of it would understate it.
        if cpu_n_predicted < n_predicted:
            bullet += f" over the {cpu_n_predicted:,} of {n_predicted:,} measured"
        elif cost.get("elapsed_s") and len(machines) == 1:
            # How much of one machine the run kept busy. Nothing to say of a set
            # measured on two, where the figure would average different machines.
            busy_cores = cpu_seconds / cost["elapsed_s"]
            cpu_count = machines[0].get("cpu_count")
            bullet += f", {busy_cores:.1f}"
            bullet += f" of {cpu_count} cores busy" if cpu_count else " cores busy"
        bullets.append(bullet)
    bullets += [_machine_label(machine) for machine in machines if _machine_label(machine)]
    return bullets


def _cost_bullets(cost: dict) -> List[str]:
    return _throughput_bullet(cost) + _latency_bullet(cost) + _cpu_bullets(cost)


def _render_cost_lines(labeled_bullets: List[Tuple[str, List[str]]]) -> List[str]:
    lines: List[str] = []
    for label, bullets in labeled_bullets:
        if lines:
            lines.append("")
        lines.append(f"**{label}**")
        lines += [f"* {bullet}" for bullet in bullets]
    return lines


def _incomparable_cost_note(labeled_costs: List[Tuple[str, dict]]) -> List[str]:
    """Call out predictions measured on different hardware or at different concurrency.

    Both make a timing delta a property of the measurement rather than of the
    parser, and it is as easy to happen within one column — a set topped up months
    later on another machine — as between two. A column that states neither was
    generated before this was recorded, which is the same problem.
    """
    differing = []
    for name, values_of in (
        ("CPU", lambda cost: [
            machine.get("cpu_model") for machine in (cost.get("machines") or [])
        ]),
        ("concurrency", lambda cost: cost.get("concurrency") or []),
    ):
        stated = [
            [str(value) for value in values_of(cost) if value] or ["unrecorded"]
            for _, cost in labeled_costs
        ]
        if len({value for values in stated for value in values}) > 1:
            differing.append(
                f"{name} ({', '.join('/'.join(values) for values in stated)})"
            )
    if not differing:
        return []
    return [
        "> ⚠️ **Measured differently**: these predictions differ in "
        + " and ".join(differing)
        + ". Timing deltas between them reflect that as well as the parser.",
        "",
    ]


def render_cost_section(
    labeled_summaries: List[Tuple[str, dict]],
    note: str,
) -> List[str]:
    """Empty unless something was recorded, so a report over predictions generated
    before this was is unchanged."""
    labeled_costs = [
        (label, summary.get("cost") or {}) for label, summary in labeled_summaries
    ]
    measured = [
        (label, cost, _cost_bullets(cost)) for label, cost in labeled_costs
    ]
    measured = [entry for entry in measured if entry[2]]
    if not measured:
        return []
    return [
        "<details>",
        "<summary><b>Compute cost</b></summary>",
        "",
        note,
        "",
        *_incomparable_cost_note([(label, cost) for label, cost, _ in measured]),
        *_render_cost_lines([(label, bullets) for label, _, bullets in measured]),
        "",
        "</details>",
    ]


# What a compute chart can draw, and what its axis says. Each reads the same record the
# bullets above are written from, so a chart cannot state a figure the text does not.
# `latency` is what the stored record calls it and what a service usually means by the
# time between a request and its response. What this measures is a whole document being
# converted, which takes seconds to minutes, so the charts say so instead.
COMPUTE_METRICS: Dict[str, Tuple[str, str]] = {
    "cpu_seconds_per_doc": ("Compute per document", "CPU-seconds per document"),
    "latency_median": (
        "Median time per document", "wall-clock seconds per document",
    ),
    "latency_p90": ("p90 time per document", "wall-clock seconds per document"),
    "docs_per_hour": ("Throughput", "documents per hour"),
    # Per thousand rather than per document: rented CPU comes to cents per thousand,
    # and a per-document figure puts that and an LLM's spend on one axis as 0.00002
    # against 0.002.
    "estimated_cost_per_1k": (
        "Estimated cost per 1,000 documents", "US dollars per 1,000 documents",
    ),
}

# A sustained general-purpose on-demand vCPU-hour, checked October 2026 against AWS
# `c7g.medium` at $0.036 for its one vCPU and GCP `e2-standard-4` at $0.134 for four.
# Burstable families are cheaper per vCPU -- `t4g.medium` is about $0.017 -- but they are
# credit-limited, which is not what a converter running flat out would get. A round
# number rather than a quote: rates move, differ by region and fall with commitment, so
# check before quoting any of it, and nothing here is billed at any rate anyway since
# CI's CPU costs us nothing. `cpu_usd_per_hour` on the chart prices it otherwise.
DEFAULT_CPU_USD_PER_HOUR = 0.035


def compute_metric(
    cost: dict,
    metric: str,
    usage: Optional[dict] = None,
    cpu_usd_per_hour: float = DEFAULT_CPU_USD_PER_HOUR,
) -> Optional[float]:
    """One variant's figure, or nothing where the run did not record it.

    Nothing rather than zero: a run that predates this measurement and one that spent
    no time are different claims, and only the second is a number.
    """
    latency = cost.get("latency_ms") or {}
    if metric == "estimated_cost_per_1k":
        per_doc = _estimated_cost_per_doc(cost, usage or {}, cpu_usd_per_hour)
        return per_doc * 1000 if per_doc is not None else None
    if metric == "cpu_seconds_per_doc":
        seconds, documents = cost.get("cpu_seconds"), cost.get("cpu_n_predicted")
        return seconds / documents if seconds and documents else None
    if metric == "latency_median":
        return latency["median"] / 1000 if latency.get("median") else None
    if metric == "latency_p90":
        return latency["p90"] / 1000 if latency.get("p90") else None
    if metric == "docs_per_hour":
        predicted, elapsed = cost.get("n_predicted"), cost.get("elapsed_s")
        return predicted / elapsed * 3600 if predicted and elapsed else None
    raise ValueError(f"Unknown compute metric {metric!r}")


COST_COMPONENTS = ("rented CPU", "LLM provider")


def estimated_cost_components(
    cost: dict, usage: dict, cpu_usd_per_hour: float = DEFAULT_CPU_USD_PER_HOUR
) -> Dict[str, float]:
    """The parts the estimate is made of, per thousand documents.

    Charted apart because they behave differently: rented CPU is cents a thousand and
    barely moves between tools, while an LLM provider's charge is the whole figure where
    there is one. A total alone says which is dearer and not why.
    """
    parts: Dict[str, float] = {}
    cpu_seconds, cpu_documents = cost.get("cpu_seconds"), cost.get("cpu_n_predicted")
    if cpu_seconds and cpu_documents:
        parts[COST_COMPONENTS[0]] = (
            cpu_seconds / cpu_documents * cpu_usd_per_hour / 3600 * 1000
        )
    spent, charged_documents = usage.get("cost_credits"), usage.get("n_with_usage")
    if spent is not None and charged_documents:
        # OpenRouter credits are dollars; `doc/llm_engine.md` says the number is the
        # provider's own, so whatever it discounted is already in it.
        parts[COST_COMPONENTS[1]] = spent / charged_documents * 1000
    return parts


def _estimated_cost_per_doc(
    cost: dict, usage: dict, cpu_usd_per_hour: float
) -> Optional[float]:
    """Rented CPU plus what the provider charged, per document.

    Each part is divided by the documents it was measured over rather than by the run's
    total, so a figure recorded for part of a set states a rate over that part. Nothing
    where neither part is known; a tool whose compute happened elsewhere has no CPU to
    price and no credits reported here either.
    """
    parts = []
    cpu_seconds, cpu_documents = cost.get("cpu_seconds"), cost.get("cpu_n_predicted")
    if cpu_seconds and cpu_documents:
        parts.append(cpu_seconds / cpu_documents * cpu_usd_per_hour / 3600)
    spent, charged_documents = usage.get("cost_credits"), usage.get("n_with_usage")
    if spent is not None and charged_documents:
        # OpenRouter credits are dollars; `doc/llm_engine.md` says the number is the
        # provider's own, so whatever it discounted is already in it.
        parts.append(spent / charged_documents)
    return sum(parts) if parts else None
