from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml

from benchmarks.fetch import fetch_data

LOGGER = logging.getLogger(__name__)

CONVERT_ENDPOINT = "/api/processFulltextDocument"
DEFAULT_CONCURRENCY = 0  # 0 = auto: max(2, cpu_count)


def _resolve_concurrency(concurrency: int) -> int:
    if concurrency == 0:
        return max(2, os.cpu_count() or 2)
    return concurrency


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "predictions" / "manifest.jsonl"


def _load_done(run_dir: Path) -> set:
    """Return set of (corpus, record_id) already in the manifest."""
    path = _manifest_path(run_dir)
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("status") == "ok":
                done.add((entry["corpus"], entry["record_id"]))
    return done


def _append_manifest(run_dir: Path, entry: Dict[str, Any]) -> None:
    path = _manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


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
                        "status": "ok", "elapsed_ms": elapsed_ms,
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
                        "status": "error", "error": msg, "error_body": body,
                    })
                    progress.record_err(corpus, record_id)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    msg = str(exc)
                    _append_manifest(run_dir, {
                        "corpus": corpus, "record_id": record_id,
                        "status": "error", "error": msg,
                    })
                    progress.record_err(corpus, record_id)
                    LOGGER.error("err %s/%s  %s", corpus, record_id, msg)

        await asyncio.gather(*[_process_one(rec) for rec in to_process])

    return progress.n_ok, progress.n_err


def run_predict(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    run_dir: Path,
    parser_url: str,
    parser_image: Optional[str],
    profile: Optional[str],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> None:
    records = fetch_data(config, mode, split, data_dir)
    done = _load_done(run_dir)

    t_start = time.monotonic()
    timeout = config.get("parser", {}).get("timeout_seconds", 60)

    n_ok, n_err = asyncio.run(
        _run_predict_async(
            records, done, run_dir, parser_url, timeout,
            _resolve_concurrency(concurrency),
        )
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "parser_image": parser_image,
        "profile": profile,
        "dataset_repo_id": config["dataset"]["repo_id"],
        "dataset_revision": config["dataset"]["revision"],
        "split": split,
        "mode": mode,
        "corpora": list(config["dataset"]["splits"][split].keys()),
        "fields": config["fields"],
        "n_records": n_ok,
        "n_errors": n_err,
        "elapsed_s": round(time.monotonic() - t_start, 1),
    }, indent=2))

    LOGGER.info(
        "done  ok=%d  err=%d  elapsed=%.1fs",
        n_ok, n_err, time.monotonic() - t_start,
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
    )


if __name__ == "__main__":
    main()
