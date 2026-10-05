"""What a run cost in compute, and what machine it cost it on.

Machine-wide rather than the parser's own accounting: the work is spread across a
persistent wapiti child process, in-process torch threads and transient
subprocesses, so nothing measured inside the parser sees all of it. CI gives the
job a runner of its own and starts the container with no CPU limit, which makes
the machine the unit and needs nothing from the parser — so a GROBID baseline is
covered on the same terms.
"""
from __future__ import annotations

import ipaddress
import logging
import math
import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

LOGGER = logging.getLogger(__name__)

PROC_STAT = Path("/proc/stat")
PROC_CPUINFO = Path("/proc/cpuinfo")

# user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice.
# Named rather than derived by subtracting the idle ones: `guest` and `guest_nice`
# are already counted inside `user` and `nice`, so a total would count them twice,
# and `steal` is time the hypervisor took rather than work this run did.
BUSY_CPU_STAT_FIELDS = (0, 1, 2, 5, 6)


def get_cpu_model() -> Optional[str]:
    """What `/proc/cpuinfo` calls the CPU, or nothing where it does not say.

    Recorded because nothing fixes the hardware between two CI jobs, and a run
    compared against a baseline from another day has no other way to tell a
    regression from a faster runner.
    """
    try:
        lines = PROC_CPUINFO.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    models = [
        line.split(":", 1)[1].strip()
        for line in lines
        if line.startswith("model name") and ":" in line
    ]
    return models[0] if models else None


def read_busy_cpu_seconds() -> Optional[float]:
    """CPU-seconds this machine has spent busy since boot, across every core."""
    try:
        first_line = PROC_STAT.read_text(encoding="utf-8").split("\n", 1)[0]
    except OSError:
        return None
    values = [int(value) for value in first_line.split()[1:]]
    ticks = sum(values[index] for index in BUSY_CPU_STAT_FIELDS if index < len(values))
    return ticks / os.sysconf("SC_CLK_TCK")


def is_local_parser_url(parser_url: Optional[str]) -> bool:
    if not parser_url:
        return False
    host = (urlsplit(parser_url).hostname or "").lower()
    if host in ("", "localhost"):
        return True
    if host == socket.gethostname().lower():
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def start_cpu_measurement(parser_url: Optional[str]) -> Optional[float]:
    """The reading to measure the run against, or nothing where it would mislead.

    A parser on another host did the work this would attribute to this machine, so
    there is no figure to record rather than a small one.
    """
    if not is_local_parser_url(parser_url):
        LOGGER.info(
            "Parser at %s is not on this machine, so no CPU figure is recorded",
            parser_url
        )
        return None
    return read_busy_cpu_seconds()


def get_machine_record(started_busy_cpu_seconds: Optional[float]) -> Dict[str, Any]:
    """The machine, and what it spent since the measurement started."""
    record: Dict[str, Any] = {}
    cpu_model = get_cpu_model()
    if cpu_model:
        record["cpu_model"] = cpu_model
    cpu_count = os.cpu_count()
    if cpu_count:
        record["cpu_count"] = cpu_count
    if started_busy_cpu_seconds is not None:
        busy = read_busy_cpu_seconds()
        if busy is not None:
            record["cpu_seconds"] = round(busy - started_busy_cpu_seconds, 1)
    return record


def format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"
    return f"{seconds:.0f}s"


def _percentile_ms(sorted_values: List[int], fraction: float) -> int:
    """Nearest-rank, so every figure reported is one a document actually took."""
    rank = max(1, min(len(sorted_values), math.ceil(len(sorted_values) * fraction)))
    return sorted_values[rank - 1]


RUN_ENTRY_TYPE = "run"


def run_manifest_entry(
    run_started_at: str,
    concurrency: int,
    n_processed: int,
    elapsed_s: float,
    machine: Dict[str, Any],
) -> Dict[str, Any]:
    """What one invocation of the predictor cost, as a manifest line.

    In the manifest rather than beside it, because the manifest is what the
    predictions store carries: a set of predictions assembled over several
    invocations then arrives with each one's own measurement, and the whole set
    can be reported on without the run that scores it having generated any of it.
    Every reader of the manifest selects on `status` or on a document's keys, so a
    line with neither is already ignored by all of them.
    """
    return {
        "type": RUN_ENTRY_TYPE,
        "started_at": run_started_at,
        "concurrency": concurrency,
        "n_processed": n_processed,
        "elapsed_s": elapsed_s,
        "machine": machine,
    }


def aggregate_latency_ms(
    manifest_entries: List[dict], corpora: Optional[List[str]] = None
) -> Optional[Dict[str, int]]:
    """How long a document waited, as a distribution.

    Over documents that got a prediction: a request that timed out took the client
    timeout rather than that long to answer, and one outlier moves a mean over
    sixty documents, which is why the median and p90 are reported instead.
    """
    durations = sorted(
        entry["elapsed_ms"] for entry in manifest_entries
        if entry.get("status") == "ok"
        and isinstance(entry.get("elapsed_ms"), int)
        and (corpora is None or entry.get("corpus") in corpora)
    )
    if not durations:
        return None
    return {
        "n": len(durations),
        "median": _percentile_ms(durations, 0.5),
        "p90": _percentile_ms(durations, 0.9),
    }


def _distinct(values: List[Any]) -> List[Any]:
    return sorted({value for value in values if value is not None})


def _distinct_machines(machines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    distinct = {
        (machine.get("cpu_model"), machine.get("cpu_count"))
        for machine in machines
        if machine.get("cpu_model") or machine.get("cpu_count")
    }
    return [
        {
            **({"cpu_model": cpu_model} if cpu_model else {}),
            **({"cpu_count": cpu_count} if cpu_count else {}),
        }
        for cpu_model, cpu_count in sorted(
            distinct, key=lambda pair: (pair[0] or "", pair[1] or 0)
        )
    ]


def aggregate_cost(
    manifest_entries: List[dict], corpora: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """What producing these predictions cost, over every invocation that made any.

    A set of predictions is assembled over as many invocations as it took, each
    recorded where it happened, so the whole set is reported on rather than the
    last top-up. Latency is per document and so covers the corpora asked for;
    throughput and CPU are per invocation and so cover whatever that invocation
    generated, which may reach wider than the corpora being scored.

    The CPU figure is summed only over the invocations that recorded one, with the
    documents those invocations processed, so a set part of which was generated
    against a remote parser states a rate over the part that was measured rather
    than one diluted by the part that was not.
    """
    runs = [entry for entry in manifest_entries if entry.get("type") == RUN_ENTRY_TYPE]
    latency = aggregate_latency_ms(manifest_entries, corpora)
    if not runs and not latency:
        return None

    cost: Dict[str, Any] = {}
    if latency:
        cost["latency_ms"] = latency
    if not runs:
        return cost

    measured = [run for run in runs if (run.get("machine") or {}).get("cpu_seconds")]
    cost.update({
        "n_runs": len(runs),
        "n_processed": sum(run.get("n_processed") or 0 for run in runs),
        "elapsed_s": round(sum(run.get("elapsed_s") or 0.0 for run in runs), 1),
        "concurrency": _distinct([run.get("concurrency") for run in runs]),
        # Kept paired, since a core count belongs to the machine beside it and a
        # set measured on two of them has no single one to report.
        "machines": _distinct_machines([run.get("machine") or {} for run in runs]),
    })
    if measured:
        cost["cpu_seconds"] = round(
            sum(run["machine"]["cpu_seconds"] for run in measured), 1
        )
        cost["cpu_n_processed"] = sum(run.get("n_processed") or 0 for run in measured)
    return cost
