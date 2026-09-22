from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml

from sciencebeam_parser.models.llm.usage import USAGE_HEADER_NAME

from benchmarks.fetch import fetch_data, resolved_sources

LOGGER = logging.getLogger(__name__)

CONVERT_ENDPOINT = "/api/processFulltextDocument"
DEFAULT_CONCURRENCY = 0  # 0 = auto: max(2, cpu_count)
# One pass, so a run only retries where it is asked to. A local run usually wants
# the failure in front of it; a long unattended one would rather cover the corpus.
DEFAULT_RETRY_PASSES = 1


def _resolve_concurrency(concurrency: int) -> int:
    if concurrency == 0:
        return max(2, os.cpu_count() or 2)
    return concurrency


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "predictions" / "manifest.jsonl"


def _manifest_entries(run_dir: Path) -> List[Dict[str, Any]]:
    path = _manifest_path(run_dir)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_done(run_dir: Path) -> set:
    """Return set of (corpus, record_id) already in the manifest."""
    return {
        (entry["corpus"], entry["record_id"])
        for entry in _manifest_entries(run_dir)
        if entry.get("status") == "ok"
    }


def _summarise_manifest(
    run_dir: Path, records: List[Dict[str, Any]]
) -> Tuple[int, int, int]:
    """Documents predicted, documents missing, and documents a later pass recovered.

    Counted over the manifest rather than over what this invocation processed, so a
    resumed run reports what the run covers rather than what it did this time. A
    document is recovered when it failed at least once and ended with a prediction:
    the manifest is append-only, so both entries are still there to be counted.
    """
    wanted = {(record["corpus"], record["record_id"]) for record in records}
    predicted, failed = set(), set()
    for entry in _manifest_entries(run_dir):
        key = (entry.get("corpus"), entry.get("record_id"))
        if key not in wanted:
            continue
        if entry.get("status") == "ok":
            predicted.add(key)
        else:
            failed.add(key)
    return len(predicted), len(wanted - predicted), len(predicted & failed)


def _append_manifest(run_dir: Path, entry: Dict[str, Any]) -> None:
    path = _manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _llm_usage_entry(response: Optional[httpx.Response]) -> Dict[str, Any]:
    """The parser's usage header, as given, or nothing.

    Omitted rather than zeroed when the header is absent: a CRF-only run, a
    timeout that produced no response at all, and a document that genuinely
    spent nothing are three different things, and only the last is a zero.
    """
    if response is None:
        return {}
    header_value = response.headers.get(USAGE_HEADER_NAME)
    if not header_value:
        return {}
    try:
        return {"llm_usage": json.loads(header_value)}
    except json.JSONDecodeError:
        LOGGER.warning("Could not parse %s: %r", USAGE_HEADER_NAME, header_value[:200])
        return {}


def _format_eta(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"
    return f"{seconds:.0f}s"


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.n_ok = 0
        self.n_err = 0
        self._t_start = time.monotonic()

    @property
    def completed(self) -> int:
        return self.n_ok + self.n_err

    def record_ok(self, corpus: str, record_id: str, elapsed_ms: int) -> None:
        self.n_ok += 1
        self._log(corpus, record_id, "ok", elapsed_ms)

    def record_err(self, corpus: str, record_id: str) -> None:
        self.n_err += 1
        self._log(corpus, record_id, "err", 0)

    def _log(
        self, corpus: str, record_id: str, status: str, elapsed_ms: int
    ) -> None:
        elapsed = time.monotonic() - self._t_start
        done = self.completed
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = self.total - done
        eta = _format_eta(remaining / rate) if rate > 0 else "?"
        LOGGER.info(
            "[%d/%d] %s/%s %s %dms | %.1f doc/s | ~%s left",
            done, self.total, corpus, record_id, status, elapsed_ms, rate, eta,
        )


async def _run_predict_async(
    records: List[Dict[str, Any]],
    done: set,
    run_dir: Path,
    parser_url: str,
    timeout: int,
    concurrency: int,
    pass_index: int = 1,
) -> Tuple[int, int]:
    to_process = [
        r for r in records if (r["corpus"], r["record_id"]) not in done
    ]
    skipped = len(records) - len(to_process)
    if skipped:
        LOGGER.info("Skipping %d already-cached documents", skipped)
    LOGGER.info(
        "Processing %d documents (concurrency=%d)", len(to_process), concurrency
    )

    progress = _Progress(len(to_process))
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:

        async def _process_one(rec: Dict[str, Any]) -> None:
            corpus = rec["corpus"]
            record_id = rec["record_id"]
            out_path = (
                run_dir / "predictions" / corpus / f"{record_id}.tei.xml"
            )
            async with sem:
                t0 = time.monotonic()
                try:
                    pdf_bytes = Path(rec["pdf_path"]).read_bytes()
                    response = await client.post(
                        f"{parser_url}{CONVERT_ENDPOINT}",
                        data={
                            "includeRawAffiliations": "1",
                            "includeRawCitations": "1",
                        },
                        files={
                            "input": (
                                Path(rec["pdf_path"]).name,
                                pdf_bytes,
                                "application/pdf",
                            )
                        },
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    elapsed_ms = round((time.monotonic() - t0) * 1000)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(response.content)
                    _append_manifest(run_dir, {
                        "corpus": corpus, "record_id": record_id,
                        "status": "ok", "pass": pass_index,
                        "elapsed_ms": elapsed_ms,
                        **_llm_usage_entry(response),
                    })
                    progress.record_ok(corpus, record_id, elapsed_ms)
                except httpx.HTTPStatusError as exc:
                    msg = str(exc)
                    body = exc.response.text[:2000] if exc.response.text else ""
                    if body:
                        LOGGER.error(
                            "err %s/%s  %s\n%s", corpus, record_id, msg, body
                        )
                    else:
                        LOGGER.error("err %s/%s  %s", corpus, record_id, msg)
                    _append_manifest(run_dir, {
                        "corpus": corpus, "record_id": record_id,
                        "status": "error", "pass": pass_index,
                        "error": msg, "error_body": body,
                        **_llm_usage_entry(exc.response),
                    })
                    progress.record_err(corpus, record_id)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    msg = str(exc)
                    _append_manifest(run_dir, {
                        "corpus": corpus, "record_id": record_id,
                        "status": "error", "pass": pass_index, "error": msg,
                    })
                    progress.record_err(corpus, record_id)
                    LOGGER.error("err %s/%s  %s", corpus, record_id, msg)

        await asyncio.gather(*[_process_one(rec) for rec in to_process])

    return progress.n_ok, progress.n_err


def run_predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    run_dir: Path,
    parser_url: str,
    parser_image: Optional[str],
    profile: Optional[str],
    concurrency: int = DEFAULT_CONCURRENCY,
    include: Optional[Iterable[str]] = None,
    retry_passes: int = DEFAULT_RETRY_PASSES,
) -> None:
    records = fetch_data(config, mode, split, data_dir, include=include)
    sources = resolved_sources(config, split, include)

    t_start = time.monotonic()
    timeout = config.get("parser", {}).get("timeout_seconds", 60)
    resolved_concurrency = _resolve_concurrency(concurrency)
    passes = max(1, retry_passes)

    # A later pass asks again for what is still missing, which is a different
    # question from the engine's own retries: those spend their backoff inside one
    # request, where this is separated from the first ask by the rest of the run.
    # A rate-limit window outlasts the former and not the latter.
    for pass_index in range(1, passes + 1):
        done = _load_done(run_dir)
        remaining = [
            record for record in records
            if (record["corpus"], record["record_id"]) not in done
        ]
        if not remaining:
            break
        if pass_index > 1:
            LOGGER.info(
                "Retry pass %d of %d over %d document(s) without a prediction",
                pass_index, passes, len(remaining),
            )
        asyncio.run(
            _run_predict_async(
                records, done, run_dir, parser_url, timeout,
                resolved_concurrency, pass_index,
            )
        )

    n_ok, n_err, n_recovered = _summarise_manifest(run_dir, records)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "parser_image": parser_image,
        "profile": profile,
        # Per corpus rather than one dataset-wide pair, since a corpus may live in
        # its own repo at its own revision. This is what says afterwards which
        # revision a number was measured against.
        "sources": sources,
        "split": split,
        "mode": mode,
        "corpora": list(sources),
        "fields": config["fields"],
        "n_records": n_ok,
        "n_errors": n_err,
        # How many documents only have a prediction because they were asked for
        # twice. Reported beside the scores, since a retried run that states only
        # its scores reads as a clean one. `retry_passes` says whether a zero here
        # means nothing failed or nothing was retried.
        "n_recovered": n_recovered,
        "retry_passes": passes,
        "elapsed_s": round(time.monotonic() - t_start, 1),
    }, indent=2))

    LOGGER.info(
        "done  ok=%d  err=%d  recovered=%d  elapsed=%.1fs",
        n_ok, n_err, n_recovered, time.monotonic() - t_start,
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict: run parser on each PDF and save JATS XML"
    )
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument(
        "--mode", default="smoke",
        help="Sampling mode defined in eval.yml (e.g. smoke, small, medium, large, full)"
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split: train (local) or validation (CI)",
    )
    parser.add_argument(
        "--data", default="benchmarks/data", help="Data cache directory"
    )
    parser.add_argument(
        "--out", required=True, help="Run output dir, e.g. benchmarks/runs/local"
    )
    parser.add_argument("--parser-url", default="http://localhost:8080")
    parser.add_argument(
        "--parser-image", default=None, help="Docker image tag (provenance only)"
    )
    parser.add_argument(
        "--profile", default=None, help="Model configuration profile name"
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help="Concurrent requests to the parser (0 = auto: max(2, cpu_count))",
    )
    parser.add_argument(
        "--retry-passes", type=int, default=DEFAULT_RETRY_PASSES,
        help=(
            "Times to go over the corpus, asking again for documents that have no"
            " prediction yet (1 = no retry). How many a retry recovered is reported"
            " in run.json and the report"
        ),
    )
    parser.add_argument(
        "--include-corpus", action="append", default=None, dest="include_corpus",
        metavar="CORPUS",
        help=(
            "Also run an opt-in corpus, repeatable. Opt-in corpora are left out by"
            " default because they are private"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_predict(
        config=config,
        mode=args.mode,
        split=args.split,
        data_dir=Path(args.data),
        run_dir=Path(args.out),
        parser_url=args.parser_url,
        parser_image=args.parser_image,
        profile=args.profile,
        concurrency=args.concurrency,
        include=args.include_corpus,
        retry_passes=args.retry_passes,
    )


if __name__ == "__main__":
    main()
