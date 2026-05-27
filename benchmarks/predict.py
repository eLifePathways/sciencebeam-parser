from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

from benchmarks.fetch import fetch_data

LOGGER = logging.getLogger(__name__)

CONVERT_ENDPOINT = "/api/processFulltextDocument"


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


def _predict_one(
    client: httpx.Client,
    parser_url: str,
    pdf_path: Path,
    out_path: Path,
    timeout: int,
) -> int:
    """POST one PDF to the parser, write the JATS XML response. Returns elapsed ms."""
    t0 = time.monotonic()
    response = client.post(
        f"{parser_url}{CONVERT_ENDPOINT}",
        files={"input": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        timeout=timeout,
    )
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return round((time.monotonic() - t0) * 1000)


def run_predict(  # pylint: disable=too-many-locals
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    run_dir: Path,
    parser_url: str,
    parser_image: Optional[str],
    parser_config: Optional[str],
) -> None:
    records = fetch_data(config, mode, split, data_dir)
    done = _load_done(run_dir)

    n_ok = n_err = 0
    t_start = time.monotonic()
    timeout = config.get("parser", {}).get("timeout_seconds", 60)

    with httpx.Client() as client:
        for rec in records:
            corpus, record_id = rec["corpus"], rec["record_id"]

            if (corpus, record_id) in done:
                LOGGER.info("skip cached  %s/%s", corpus, record_id)
                continue

            out_path = run_dir / "predictions" / corpus / f"{record_id}.tei.xml"
            try:
                elapsed_ms = _predict_one(
                    client, parser_url, Path(rec["pdf_path"]), out_path, timeout
                )
                _append_manifest(run_dir, {
                    "corpus": corpus, "record_id": record_id,
                    "status": "ok", "elapsed_ms": elapsed_ms,
                })
                n_ok += 1
                LOGGER.info("ok  %s/%s  %dms", corpus, record_id, elapsed_ms)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                msg = str(exc)
                _append_manifest(run_dir, {
                    "corpus": corpus, "record_id": record_id,
                    "status": "error", "error": msg,
                })
                n_err += 1
                LOGGER.error("err %s/%s  %s", corpus, record_id, msg)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps({
        "parser_image": parser_image,
        "parser_config": parser_config,
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

    LOGGER.info("done  ok=%d  err=%d  elapsed=%.1fs", n_ok, n_err, time.monotonic() - t_start)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Predict: run parser on each PDF and save JATS XML"
    )
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--split", default="train", help="Dataset split: train (local) or validation (CI)"
    )
    parser.add_argument("--data", default="benchmarks/data", help="Data cache directory")
    parser.add_argument("--out", required=True, help="Run output dir, e.g. benchmarks/runs/local")
    parser.add_argument("--parser-url", default="http://localhost:8080")
    parser.add_argument("--parser-image", default=None, help="Docker image tag (provenance only)")
    parser.add_argument(
        "--parser-config", default=None, help="Parser config override path (provenance only)"
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
        parser_config=args.parser_config,
    )


if __name__ == "__main__":
    main()
