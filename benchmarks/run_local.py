from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List

import httpx
import yaml

from benchmarks.predict import run_predict
from benchmarks.report import run_compare
from benchmarks.score import run_score

LOGGER = logging.getLogger(__name__)


def _tool_docker_config(tool: str, version: str) -> dict:
    if tool == "grobid":
        return {
            "image": f"grobid/grobid:{version}",
            "port": "8070:8070",
            "url": "http://localhost:8070",
            "health_path": "/api/isalive",
        }
    return {
        "image": f"ghcr.io/elifepathways/sciencebeam-parser:{version}",
        "port": "8080:8070",
        "url": "http://localhost:8080",
        "health_path": "/",
    }


def _count_predictions(run_dir: Path) -> int:
    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        return 0
    return sum(1 for _ in pred_dir.rglob("*.tei.xml"))


def _expected_count(config: dict, mode: str) -> int:
    return sum(config.get("sampling", {}).get(mode, {}).values())


def _wait_healthy(url: str, health_path: str, retries: int = 60, interval: int = 5) -> None:
    for i in range(retries):
        try:
            r = httpx.get(f"{url}{health_path}", timeout=5)
            if r.is_success:
                LOGGER.info("Service ready at %s", url)
                return
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        if i < retries - 1:
            time.sleep(interval)
    raise RuntimeError(
        f"Service at {url}{health_path} never became healthy after {retries * interval}s"
    )


def _docker_start(name: str, image: str, port: str, env_vars: Optional[dict] = None) -> None:
    LOGGER.info("Starting container %s (%s)", name, image)
    cmd = ["docker", "run", "-d", "--name", name, "-p", port]
    for k, v in (env_vars or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    subprocess.run(cmd, check=True)


def _docker_stop(name: str) -> None:
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)
    subprocess.run(["docker", "rm", name], check=False, capture_output=True)


def _baseline_env_vars(tool: str, profile: Optional[str]) -> dict:
    if profile and tool != "grobid":
        return {"SCIENCEBEAM_PARSER__PROFILE": profile}
    return {}


def _run_baseline(
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    runs_dir: Path,
    tool: str,
    version: str,
    profile: Optional[str],
    expected: int,
) -> Optional[Tuple[str, Path]]:
    run_dir = runs_dir / "baselines" / tool / version / (profile or "default") / split
    existing = _count_predictions(run_dir)
    LOGGER.info(
        "=== Baseline %s/%s (profile=%s): %d/%d predictions ===",
        tool, version, profile or "default", existing, expected,
    )

    if existing < expected:
        dcfg = _tool_docker_config(tool, version)
        container = f"run-local-{tool}"
        _docker_stop(container)
        _docker_start(container, dcfg["image"], dcfg["port"], _baseline_env_vars(tool, profile))
        try:
            _wait_healthy(dcfg["url"], dcfg["health_path"])
            run_predict(
                config, mode, split, data_dir, run_dir,
                dcfg["url"], f"{tool}:{version}", profile,
            )
        finally:
            _docker_stop(container)
    else:
        LOGGER.info("Predictions already present, skipping predict")

    run_score(config, run_dir, data_dir, out_path=None, split_override=split)

    label = f"{tool} {version}" + (f" ({profile})" if profile else "")
    summary_path = run_dir / "summary.json"
    return (label, summary_path) if summary_path.exists() else None


def run_local(
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    runs_dir: Path,
    parser_url: Optional[str],
    parser_image: Optional[str],
    parser_profile: Optional[str],
    baseline_only: bool,
) -> None:
    expected = _expected_count(config, mode)
    labeled_paths: List[Tuple[str, Path]] = []

    for baseline in config.get("baselines", []):
        entry = _run_baseline(
            config, mode, split, data_dir, runs_dir,
            baseline["tool"], baseline["version"],
            baseline.get("profile"), expected,
        )
        if entry:
            labeled_paths.append(entry)

    if baseline_only:
        return

    if parser_url is None:
        LOGGER.warning("No --parser-url given; skipping primary predict and report")
        return

    primary_run_dir = runs_dir / split
    run_predict(
        config, mode, split, data_dir, primary_run_dir, parser_url, parser_image, parser_profile
    )
    run_score(config, primary_run_dir, data_dir, out_path=None, split_override=split)

    labeled_paths.append((parser_image or "local", primary_run_dir / "summary.json"))

    if len(labeled_paths) >= 2:
        run_compare(labeled_paths, primary_run_dir / "comparison.md")
    else:
        LOGGER.info("Only one summary available; skipping comparison report")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark locally with baseline comparison"
    )
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument(
        "--mode", default="smoke",
        help="Sampling mode defined in eval.yml (e.g. smoke, small, medium, large, full)"
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--data", default="benchmarks/data")
    parser.add_argument(
        "--runs", default="benchmarks/runs", help="Base directory for run outputs"
    )
    parser.add_argument(
        "--parser-url", default=None,
        help="Primary parser URL (e.g. http://localhost:8080)",
    )
    parser.add_argument(
        "--parser-image", default=None, help="Primary parser image tag (provenance label)"
    )
    parser.add_argument(
        "--profile", default=None, help="Model configuration profile for the primary parser"
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only generate and score baselines; skip primary parser",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_local(
        config=config,
        mode=args.mode,
        split=args.split,
        data_dir=Path(args.data),
        runs_dir=Path(args.runs),
        parser_url=args.parser_url,
        parser_image=args.parser_image,
        parser_profile=args.profile,
        baseline_only=args.baseline_only,
    )


if __name__ == "__main__":
    main()
