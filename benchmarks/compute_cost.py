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
