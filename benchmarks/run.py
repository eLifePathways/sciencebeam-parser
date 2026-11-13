from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import httpx
import yaml

from benchmarks.fetch import fetch_gold, get_corpus_variants
from benchmarks.predict import DEFAULT_RETRY_PASSES, run_predict
from benchmarks.predict_llm import RESTRICTED_CORPORA_FOR_LLM, run_predict_llm
from benchmarks.predictions_store import LocalPredictionsStore, RepoPredictionsStore
from benchmarks.report import run_compare
from benchmarks.comparison_config import (
    load_comparison,
    primary_index,
    variant_descriptions,
    resolve_variants,
    store_variants,
    to_selection,
)
from benchmarks.report_grid import SelectionError
from benchmarks.score import run_score
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE

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


def _coverage(expected_ids: set, done_ids: set) -> dict:
    """Per corpus, how many of the expected records the store has, and how many
    are expected."""
    counts: dict = {}
    for corpus, record_id in expected_ids:
        entry = counts.setdefault(corpus, [0, 0])
        entry[1] += 1
        if (corpus, record_id) in done_ids:
            entry[0] += 1
    return {corpus: (have, want) for corpus, (have, want) in counts.items()}


def _make_label(tool: str, version: str, profile: str, metadata: Optional[dict]) -> str:
    if metadata and metadata.get("image"):
        image = metadata["image"]
        # The tag is the run's own name, the version the name it is stored under: a
        # dispatch on main was tagged `manual-`, which left its column saying main
        # only through a commit a reader had to recognise.
        tag = image.rsplit(":", 1)[-1]
        base = image if tag.startswith(f"{version}-") else f"{image} on {version}"
    elif tool == "sciencebeam-parser":
        base = version
    else:
        base = f"{tool} {version}"
    return f"{base} ({profile})"


def _generate_predictions(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: dict, mode: str, split: str, data_dir: Path, run_dir: Path,
    tool: str, version: str, profile: str, concurrency: int,
    include: Optional[Iterable[str]] = None,
    endpoint: Optional[str] = None,
) -> None:
    # A served model has no container to start, and its version is the checkpoint.
    if endpoint:
        run_predict_llm(config, mode, split, data_dir, run_dir,
                        endpoint, version, concurrency, include=include)
        return

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
    endpoint: Optional[str] = None,
) -> Optional[Tuple[str, Path]]:
    run_dir = runs_dir / "baselines" / tool / version / profile / split
    done_ids = store.get_done_ids(tool, version, profile, split, corpus_variants)

    if not generate:
        # A baseline that will not generate contributes what it has, rather than
        # dropping out over what it lacks. An opt-in corpus is absent from a stored
        # baseline by construction — only a run that asked for that corpus could
        # have pushed predictions for it — so requiring every corpus would remove
        # this baseline from the comparison whenever one is included. A mode larger
        # than the one it was stored at leaves it short the same way.
        #
        # Where it covers a corpus only partly it still contributes, and the report
        # is what surfaces that: it prints each run's document count per corpus and
        # calls out a difference, since a delta across unequal document sets
        # reflects which documents were measured as well as how the tool behaved.
        coverage = _coverage(expected_ids, done_ids)
        covered = {corpus for corpus, (have, _) in coverage.items() if have}
        if not covered:
            LOGGER.warning(
                "No stored predictions for %s/%s (profile=%s), skipping",
                tool, version, profile,
            )
            return None
        short = {
            corpus: counts for corpus, counts in coverage.items()
            if counts[0] < counts[1]
        }
        if short:
            LOGGER.warning(
                "Baseline %s/%s (profile=%s) does not generate and is short of "
                "stored predictions for %s; comparing what it has",
                tool, version, profile,
                ", ".join(f"{corpus} {have}/{want}"
                          for corpus, (have, want) in sorted(short.items())),
            )
        expected_ids = {entry for entry in expected_ids if entry in done_ids}
        corpus_variants = {
            corpus: variant
            for corpus, variant in corpus_variants.items()
            if corpus in covered
        }
        include = [corpus for corpus in (include or ()) if corpus in covered]

    missing = expected_ids - done_ids
    LOGGER.info(
        "=== Baseline %s/%s (profile=%s): %d/%d predictions ===",
        tool, version, profile, len(done_ids), len(expected_ids),
    )

    # Take whatever the store has before generating the rest. `run_predict` skips
    # records already in the run's manifest, which `fetch` copies, so one absent
    # document no longer costs a re-prediction of every other — which at ~24s per
    # document for the parser and ~20s for a PLOS manuscript is most of an hour.
    if done_ids:
        LOGGER.info("Fetching %d existing prediction(s) from the store", len(done_ids))
        store.fetch(tool, version, profile, split, run_dir, corpus_variants)

    if missing:
        _generate_predictions(config, mode, split, data_dir, run_dir,
                              tool, version, profile, concurrency, include=include,
                              endpoint=endpoint)
        store.push(tool, version, profile, split, run_dir, corpus_variants, {
            "tool": tool, "version": version, "profile": profile,
            "split": split, "mode": mode,
        })

    run_score(config, run_dir, data_dir, out_path=None, split_override=split,
              include=include)
    metadata = store.read_metadata(tool, version, profile, split)
    label = _make_label(tool, version, profile, metadata)
    summary_path = run_dir / "summary.json"
    return (label, summary_path) if summary_path.exists() else None


MAIN_TOOL = "sciencebeam-parser"
MAIN_VERSION = "main"


def _same_profile_main_baseline(config: dict, parser_profile: Optional[str]) -> Optional[dict]:
    """`main` under the profile the run under test uses, where `eval.yml` declares
    `main` under another one only.

    `eval.yml` declares `main` once, under the shipped profile, so a run under any
    other profile was compared with `main` across a profile change as well as a code
    change. Store-only, like the declared one: `main` pushes under whichever profile
    it was run with, and a profile it was never run with has nothing to compare. The
    name is matched as the store files it, so an alias finds nothing under its target.
    """
    if not parser_profile or parser_profile == "default":
        return None
    declared = {
        (baseline["tool"], baseline["version"], baseline.get("profile", "default"))
        for baseline in config.get("baselines", [])
    }
    if (MAIN_TOOL, MAIN_VERSION, parser_profile) in declared:
        return None
    return {
        "tool": MAIN_TOOL, "version": MAIN_VERSION, "profile": parser_profile,
        "generate": False,
    }


def _missing_main_note(profile: str) -> str:
    return (
        f"> `{MAIN_TOOL} {MAIN_VERSION}` has no stored predictions under profile"
        f" `{profile}`, so this run is compared with `{MAIN_VERSION}` only under the"
        f" profile `eval.yml` declares it with. They are stored by running the"
        f" Benchmark workflow on `{MAIN_VERSION}` with that profile:\n"
        f">\n"
        f"> `gh workflow run benchmark.yml --ref {MAIN_VERSION} -f profile={profile}"
        f" -f mode=medium`\n"
    )


def _append_note(path: Path, note: str) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    separator = "" if not text or text.endswith("\n\n") else (
        "\n" if text.endswith("\n") else "\n\n"
    )
    path.write_text(text + separator + note, encoding="utf-8")


def _comment_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}-comment.md")


def _write_posted_report(
    labeled_paths: List[Tuple[str, Path]],
    primary_run_dir: Path,
    missing_main_profile: Optional[str],
    report_location: str = "",
) -> None:
    """The report CI posts: `comparison.md` where there is anything to compare with,
    else the run's own `report.md`, each with the comment-sized one beside it."""
    if len(labeled_paths) >= 2:
        # No charts: this is the report CI has always posted, and a comparison file is
        # how a run asks for one of its own.
        run_compare(
            labeled_paths, primary_run_dir / "comparison.md",
            comment_out=_comment_path(primary_run_dir / "comparison.md"),
            report_location=report_location,
        )
        posted_report = primary_run_dir / "comparison.md"
    else:
        LOGGER.info("Only one summary available; skipping comparison report")
        posted_report = primary_run_dir / "report.md"

    # In the report itself, since a log line on a finished runner is read by no one.
    # The budget leaves room for it rather than the note being fitted into it.
    if missing_main_profile is not None:
        for path in (posted_report, _comment_path(posted_report)):
            _append_note(path, _missing_main_note(missing_main_profile))


def _comparison_only_variants(
    config: dict, comparison_config, extra_baselines: Sequence[dict] = (),
) -> list:
    """A comparison's named variants that `eval.yml` does not already run, nor the run
    adds beside it."""
    if comparison_config is None:
        return []
    declared = {
        (baseline["tool"], baseline["version"], baseline.get("profile", "default"))
        for baseline in [*config.get("baselines", []), *extra_baselines]
    }
    return [
        variant for variant in store_variants(comparison_config)
        if (variant.tool, variant.version, variant.profile) not in declared
    ]


def run_stored_comparison(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    config: dict,
    mode: str,
    split: str,
    data_dir: Path,
    runs_dir: Path,
    store: PredictionsStore,
    comparison: str,
    out_dir: Optional[Path] = None,
    current_run: Optional[Path] = None,
    concurrency: int = 0,
    include: Optional[Iterable[str]] = None,
    chart_prefix: str = "",
    chart_base_url: str = "",
    report_location: str = "",
) -> None:
    """Render a comparison from predictions the store already holds.

    Fetches each named variant's predictions, scores them and compares -- no parser, no
    docker and nothing generated. This is the whole run for a question asked after the
    fact, which is most of them: predictions are the expensive part and they are kept.
    """
    comparison_config = load_comparison(comparison)
    corpus_variants = get_corpus_variants(config, split, include)
    expected_ids = {
        (record["corpus"], record["record_id"])
        for record in fetch_gold(config, mode, split, data_dir, include=include)
    }

    for variant in store_variants(comparison_config):
        _run_baseline(
            config, mode, split, data_dir, runs_dir,
            str(variant.tool), str(variant.version), variant.profile,
            False,
            expected_ids, corpus_variants, store, concurrency, include, None,
        )

    out_dir = out_dir or runs_dir / split
    comparison_path = out_dir / f"comparison-{comparison_config.name}.md"
    run_compare(
        resolve_variants(comparison_config, runs_dir, split, current_run),
        comparison_path,
        to_selection(comparison_config), chart_prefix, chart_base_url,
        primary_index(comparison_config), variant_descriptions(comparison_config),
        comment_out=_comment_path(comparison_path),
        report_location=report_location,
    )


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
    retry_passes: int = DEFAULT_RETRY_PASSES,
    comparison: Optional[str] = None,
    chart_prefix: str = "",
    chart_base_url: str = "",
    report_location: str = "",
) -> None:
    # pylint: disable=too-many-locals
    corpus_variants = get_corpus_variants(config, split, include)
    expected_ids = {
        (r["corpus"], r["record_id"])
        for r in fetch_gold(config, mode, split, data_dir, include=include)
    }

    comparison_config = load_comparison(comparison) if comparison else None
    # A comparison with no `current: true` variant reads the predictions store alone, and
    # the store is already there -- so this run told it nothing, and scoring it alone
    # would have taken minutes. Said once here rather than left to be noticed in a report
    # whose columns all predate the run.
    if comparison_config is not None and not any(
        variant.current for variant in comparison_config.variants
    ):
        LOGGER.warning(
            "Comparison %r names no `current: true` variant, so it compares stored"
            " predictions only and this run adds no column to it. --comparison-only"
            " renders it without a run.",
            comparison_config.name,
        )

    labeled_paths: List[Tuple[str, Path]] = []
    for baseline in config.get("baselines", []):
        profile = baseline.get("profile", "default")
        entry = _run_baseline(
            config, mode, split, data_dir, runs_dir,
            baseline["tool"], baseline["version"], profile,
            baseline.get("generate", True),
            expected_ids, corpus_variants, store, concurrency, include,
            baseline.get("endpoint"),
        )
        if entry:
            labeled_paths.append(entry)

    # Beside the declared `main`, never instead of it: the declared one is the
    # reference every report shares, this one the delta the run's own change makes.
    same_profile_main = (
        None if baseline_only else _same_profile_main_baseline(config, parser_profile)
    )
    missing_main_profile: Optional[str] = None
    if same_profile_main is not None:
        entry = _run_baseline(
            config, mode, split, data_dir, runs_dir,
            same_profile_main["tool"], same_profile_main["version"],
            same_profile_main["profile"], False,
            expected_ids, corpus_variants, store, concurrency, include, None,
        )
        if entry:
            labeled_paths.append(entry)
        else:
            missing_main_profile = same_profile_main["profile"]

    # A comparison names runs of its own, which are rarely all of `eval.yml`'s baselines.
    # They are scored here so the comparison has something to read, and deliberately not
    # added to `labeled_paths`: the report CI always posts keeps the columns it has.
    for variant in _comparison_only_variants(
        config, comparison_config, [same_profile_main] if same_profile_main else [],
    ):
        _run_baseline(
            config, mode, split, data_dir, runs_dir,
            str(variant.tool), str(variant.version), variant.profile,
            False,
            expected_ids, corpus_variants, store, concurrency, include, None,
        )

    if baseline_only:
        return

    if parser_url is None:
        LOGGER.warning("No --parser-url given; skipping primary predict and report")
        return

    profile = parser_profile or "default"
    primary_run_dir = runs_dir / split
    # The retry applies to the run under test rather than to the baselines: those
    # are served from the store or from a CRF that does not fail this way.
    run_predict(config, mode, split, data_dir, primary_run_dir,
                parser_url, parser_image, parser_profile, concurrency,
                include=include, retry_passes=retry_passes)
    run_score(config, primary_run_dir, data_dir, out_path=None, split_override=split,
              report_location=report_location,
              include=include)

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

    _write_posted_report(
        labeled_paths, primary_run_dir, missing_main_profile, report_location
    )

    # Beside the report CI always posts, never instead of it: a named comparison answers
    # a question of its own, and the regression check stays what it was.
    if comparison_config is not None:
        named_comparison = (
            primary_run_dir / f"comparison-{comparison_config.name}.md"
        )
        run_compare(
            resolve_variants(comparison_config, runs_dir, split, primary_run_dir),
            named_comparison,
            to_selection(comparison_config), chart_prefix, chart_base_url,
            primary_index(comparison_config), variant_descriptions(comparison_config),
            comment_out=_comment_path(named_comparison),
            report_location=report_location,
        )


# The restricted set lives with the code that sends documents; see
# `benchmarks.predict_llm`.


def check_llm_profile_corpora(
    parser_config: AppConfig, profile: Optional[str], include: Optional[Sequence[str]]
) -> None:
    """Refuses to send a restricted corpus to a profile that serves a model by llm.

    Read from the parser's own configuration, the one its image is built with, and
    resolved the way the parser resolves it -- through `profile_aliases`, the
    profile's `sequence_models` and their `extends` -- since `eval.yml` says nothing
    about which engine a profile uses. No profile is the parser's default one.
    """
    restricted = sorted(RESTRICTED_CORPORA_FOR_LLM.intersection(include or ()))
    if not restricted:
        return
    try:
        models = parser_config.resolve_profile(profile).props.get("models") or {}
    except ValueError as error:
        # Fails closed: a profile this cannot read is one it cannot vouch for.
        raise SystemExit(
            f"refusing to run profile {profile!r} over {', '.join(restricted)}: it"
            f" does not resolve against the parser config, so whether it sends"
            f" documents to an llm is unknown ({error})"
        ) from error
    llm_models = sorted(
        name for name, model_config in models.items()
        if isinstance(model_config, dict) and model_config.get("engine") == "llm"
    )
    if llm_models:
        raise SystemExit(
            f"refusing to run profile {profile!r}, which serves"
            f" {', '.join(llm_models)} with the llm engine, over"
            f" {', '.join(restricted)}: those manuscripts are not redistributable and"
            " provider zero-retention does not cover the intermediary. Drop the corpus"
            " or point the engine at a self-hosted endpoint."
        )


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
        "--report-location", default="", metavar="TEXT",
        help=(
            "Where the full report can be read, named in the note a shortened comment"
            " carries (e.g. 'in the [job summary](URL)')"
        ),
    )
    parser.add_argument(
        "--predictions-repo", default=None,
        help="Path to checked-out sciencebeam-eval-predictions repo (enables repo store)",
    )
    parser.add_argument("--parser-url", default=None)
    parser.add_argument("--parser-image", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument(
        "--retry-passes", type=int, default=DEFAULT_RETRY_PASSES,
        help=(
            "Times to go over the corpus, asking again for documents that have no"
            " prediction yet (1 = no retry). Applies to the run under test"
        ),
    )
    parser.add_argument(
        "--include-corpus", action="append", default=None, dest="include_corpus",
        metavar="CORPUS",
        help=(
            "Also run an opt-in corpus, repeatable. Opt-in corpora are left out by"
            " default because they are private, so a run reaches for one only when"
            " asked"
        ),
    )
    parser.add_argument(
        "--comparison", default=None, metavar="NAME",
        help=(
            "Also render a named comparison from benchmarks/comparisons/, beside the"
            " report this run already produces"
        ),
    )
    parser.add_argument(
        "--chart-prefix", default="",
        help="Prefix for chart filenames, so runs published together stay apart",
    )
    parser.add_argument(
        "--chart-base-url", default="",
        help="Link charts under this URL rather than by relative path",
    )
    parser.add_argument(
        "--comparison-only", action="store_true",
        help=(
            "Render --comparison from predictions the store already holds: fetch, score"
            " and compare, with no parser and nothing generated"
        ),
    )
    parser.add_argument(
        "--current-run", default=None, metavar="DIR",
        help="The run directory a comparison's `current: true` variant refers to",
    )
    parser.add_argument(
        "--comparison-out", default=None, metavar="DIR",
        help=(
            "Where to write the comparison and its charts. Defaults beside the runs,"
            " which is the wrong place when those are read from somewhere else"
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

    check_llm_profile_corpora(
        AppConfig.load_yaml(DEFAULT_CONFIG_FILE), args.profile, args.include_corpus,
    )

    if args.comparison_only and not args.comparison:
        parser.error("--comparison-only needs --comparison")

    if args.predictions_repo:
        store: PredictionsStore = RepoPredictionsStore(Path(args.predictions_repo))
    else:
        store = LocalPredictionsStore(Path(args.runs))

    if args.comparison_only:
        try:
            run_stored_comparison(
                config=config,
                mode=args.mode,
                split=args.split,
                data_dir=Path(args.data),
                runs_dir=Path(args.runs),
                store=store,
                comparison=args.comparison,
                current_run=Path(args.current_run) if args.current_run else None,
                out_dir=Path(args.comparison_out) if args.comparison_out else None,
                concurrency=args.concurrency,
                include=args.include_corpus,
                chart_prefix=args.chart_prefix,
                chart_base_url=args.chart_base_url,
                report_location=args.report_location,
            )
        except SelectionError as error:
            parser.error(str(error))
        return

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
        retry_passes=args.retry_passes,
        comparison=args.comparison,
        chart_prefix=args.chart_prefix,
        chart_base_url=args.chart_base_url,
        report_location=args.report_location,
    )


if __name__ == "__main__":
    main()
