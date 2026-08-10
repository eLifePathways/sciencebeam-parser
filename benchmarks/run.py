from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import httpx
import yaml

from benchmarks.fetch import fetch_gold, included_corpora
from benchmarks.predict import run_predict
from benchmarks.predictions_store import LocalPredictionsStore, RepoPredictionsStore
from benchmarks.report import run_compare
from benchmarks.score import run_score

LOGGER = logging.getLogger(__name__)

PredictionsStore = Union[LocalPredictionsStore, RepoPredictionsStore]


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
    if tool == "grobid":
        return {}
    env: dict = {"SCIENCEBEAM_PARSER__PRELOAD_ON_STARTUP": "true"}
    if profile and profile != "default":
        env["SCIENCEBEAM_PARSER__PROFILE"] = profile
    return env


def _get_corpus_variants(
    config: dict, split: str, include: Optional[Iterable[str]] = None
) -> dict:
    """Each covered corpus's prediction variant.

    Limited to the corpora the run covers, so predictions for a corpus that was not
    run are neither looked for nor stored. A versioned corpus names its version here,
    which is what keeps predictions against two versions of it apart.
    """
    split_cfg = config["dataset"]["splits"].get(split, {})
    result = {}
    for corpus in included_corpora(config, split, include):
        corpus_cfg = split_cfg[corpus]
        if isinstance(corpus_cfg, dict):
            result[corpus] = corpus_cfg.get("variant", "v1")
        else:
            result[corpus] = "v1"
    return result


def _make_label(tool: str, version: str, profile: str, metadata: Optional[dict]) -> str:
    if metadata and metadata.get("image"):
        base = metadata["image"]
    elif tool == "sciencebeam-parser":
        base = version
    else:
        base = f"{tool} {version}"
    return f"{base} ({profile})"


def _generate_predictions(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: dict, mode: str, split: str, data_dir: Path, run_dir: Path,
    tool: str, version: str, profile: str, concurrency: int,
    include: Optional[Iterable[str]] = None,
) -> None:
    dcfg = _tool_docker_config(tool, version)
    container = f"benchmark-baseline-{tool}"
    _docker_stop(container)
    _docker_start(container, dcfg["image"], dcfg["port"], _baseline_env_vars(tool, profile))
    try:
        _wait_healthy(dcfg["url"], dcfg["health_path"])
        run_predict(config, mode, split, data_dir, run_dir,
                    dcfg["url"], f"{tool}:{version}", profile, concurrency,
                    include=include)
    finally:
        _docker_stop(container)


def _run_baseline(  # pylint: disable=too-many-locals
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    runs_dir: Path,
    tool: str,
    version: str,
    profile: str,
    generate: bool,
    expected_ids: set,
    corpus_variants: dict,
    store: PredictionsStore,
    concurrency: int,
    include: Optional[Iterable[str]] = None,
) -> Optional[Tuple[str, Path]]:
    run_dir = runs_dir / "baselines" / tool / version / profile / split
    done_ids = store.get_done_ids(tool, version, profile, split)
    missing = expected_ids - done_ids
    LOGGER.info(
        "=== Baseline %s/%s (profile=%s): %d/%d predictions ===",
        tool, version, profile, len(done_ids), len(expected_ids),
    )

    if missing:
        if not generate:
            LOGGER.warning(
                "Missing %d predictions for %s/%s (profile=%s) but generate=false, skipping",
                len(missing), tool, version, profile,
            )
            return None
        _generate_predictions(config, mode, split, data_dir, run_dir,
                              tool, version, profile, concurrency, include=include)
        store.push(tool, version, profile, split, run_dir, corpus_variants, {
            "tool": tool, "version": version, "profile": profile,
            "split": split, "mode": mode,
        })
    else:
        LOGGER.info("All predictions present, fetching from store")
        store.fetch(tool, version, profile, split, run_dir, corpus_variants)

    run_score(config, run_dir, data_dir, out_path=None, split_override=split)
    metadata = store.read_metadata(tool, version, profile, split)
    label = _make_label(tool, version, profile, metadata)
    summary_path = run_dir / "summary.json"
    return (label, summary_path) if summary_path.exists() else None


def run_benchmark(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    runs_dir: Path,
    store: PredictionsStore,
    parser_url: Optional[str] = None,
    parser_image: Optional[str] = None,
    parser_profile: Optional[str] = None,
    baseline_only: bool = False,
    push_current: bool = False,
    concurrency: int = 0,
    include: Optional[Iterable[str]] = None,
) -> None:
    # pylint: disable=too-many-locals
    corpus_variants = _get_corpus_variants(config, split, include)
    expected_ids = {
        (r["corpus"], r["record_id"])
        for r in fetch_gold(config, mode, split, data_dir, include=include)
    }

    labeled_paths: List[Tuple[str, Path]] = []
    for baseline in config.get("baselines", []):
        profile = baseline.get("profile", "default")
        entry = _run_baseline(
            config, mode, split, data_dir, runs_dir,
            baseline["tool"], baseline["version"], profile,
            baseline.get("generate", True),
            expected_ids, corpus_variants, store, concurrency, include,
        )
        if entry:
            labeled_paths.append(entry)

    if baseline_only:
        return

    if parser_url is None:
        LOGGER.warning("No --parser-url given; skipping primary predict and report")
        return

    profile = parser_profile or "default"
    primary_run_dir = runs_dir / split
    run_predict(config, mode, split, data_dir, primary_run_dir,
                parser_url, parser_image, parser_profile, concurrency,
                include=include)
    run_score(config, primary_run_dir, data_dir, out_path=None, split_override=split)

    if push_current:
        store.push("sciencebeam-parser", "main", profile, split,
                   primary_run_dir, corpus_variants, {
                       "tool": "sciencebeam-parser", "version": "main",
                       "profile": profile,
                       "image": parser_image or "sciencebeam-parser:main",
                       "split": split, "mode": mode,
                   })

    current_label = _make_label("sciencebeam-parser", parser_image or "local", profile, None)
    labeled_paths.append((current_label, primary_run_dir / "summary.json"))

    if len(labeled_paths) >= 2:
        run_compare(labeled_paths, primary_run_dir / "comparison.md")
    else:
        LOGGER.info("Only one summary available; skipping comparison report")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark: fetch gold, run baselines, predict, score, compare"
    )
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data", default="benchmarks/data")
    parser.add_argument("--runs", default="benchmarks/runs")
    parser.add_argument(
        "--predictions-repo", default=None,
        help="Path to checked-out sciencebeam-eval-predictions repo (enables repo store)",
    )
    parser.add_argument("--parser-url", default=None)
    parser.add_argument("--parser-image", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument(
        "--include-corpus", action="append", default=None, dest="include_corpus",
        metavar="CORPUS",
        help=(
            "Also run an opt-in corpus, repeatable. Opt-in corpora are left out by"
            " default because they are private, so a run reaches for one only when"
            " asked"
        ),
    )
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--push-current", action="store_true",
        help="Push current predictions to store after generation (use on main branch)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.predictions_repo:
        store: PredictionsStore = RepoPredictionsStore(Path(args.predictions_repo))
    else:
        store = LocalPredictionsStore(Path(args.runs))

    run_benchmark(
        config=config,
        mode=args.mode,
        split=args.split,
        data_dir=Path(args.data),
        runs_dir=Path(args.runs),
        store=store,
        parser_url=args.parser_url,
        parser_image=args.parser_image,
        parser_profile=args.profile,
        baseline_only=args.baseline_only,
        push_current=args.push_current,
        concurrency=args.concurrency,
        include=args.include_corpus,
    )


if __name__ == "__main__":
    main()
