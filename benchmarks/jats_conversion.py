"""The parser's JATS, converted from the TEI it predicted."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import yaml
from lxml import etree

from benchmarks.prediction_files import (
    SUFFIX_BY_FORMAT,
    iter_scored_prediction_files,
    record_id_from_path,
)
from benchmarks.report import run_compare
from benchmarks.score import run_score
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.transformers.tei_to_jats import TeiToJatsConverter

LOGGER = logging.getLogger(__name__)

STYLESHEET_PATH = "sciencebeam_parser/resources/xslt/tei-to-jats.xsl"
CONFIG_PATH = "sciencebeam_parser/resources/default_config/config.yml"


def checkout_converter() -> TeiToJatsConverter:
    """The conversion the service built from this checkout would apply."""
    return TeiToJatsConverter.from_config(AppConfig.load_yaml(DEFAULT_CONFIG_FILE))


def commit_converter(commit: str) -> TeiToJatsConverter:
    """The stylesheet and config of `commit`, resolved by this checkout's code."""
    return TeiToJatsConverter.from_stylesheet(
        _git_show(commit, STYLESHEET_PATH),
        AppConfig(yaml.safe_load(_git_show(commit, CONFIG_PATH))),
    )


def _git_show(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout


def _jats_path(tei_path: Path) -> Path:
    return tei_path.with_name(record_id_from_path(tei_path) + SUFFIX_BY_FORMAT["jats"])


def convert_predictions_to_jats(
    predictions_dir: Path, converter: TeiToJatsConverter, out_dir: Optional[Path] = None
) -> List[Path]:
    """Writes a `.jats.xml` for each `.tei.xml`, in `out_dir` or beside it; returns the
    TEI files that did not convert, which are left without one."""
    out_dir = out_dir or predictions_dir
    failed = []
    for corpus_dir in sorted(p for p in predictions_dir.iterdir() if p.is_dir()):
        corpus_out_dir = out_dir / corpus_dir.name
        corpus_out_dir.mkdir(parents=True, exist_ok=True)
        for tei_path in iter_scored_prediction_files(corpus_dir, "tei"):
            jats_path = corpus_out_dir / _jats_path(tei_path).name
            try:
                converter.write_jats(etree.parse(str(tei_path)).getroot(), str(jats_path))
            except (etree.XMLSyntaxError, etree.XSLTApplyError) as exc:
                LOGGER.warning("Could not convert %s to JATS: %s", tei_path, exc)
                jats_path.unlink(missing_ok=True)
                failed.append(tei_path)
    if failed:
        LOGGER.warning("%d prediction(s) have no JATS and are not scored on it", len(failed))
    return failed


def _copy_run_records(run_dir: Path, out_run_dir: Path) -> None:
    out_run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run.json").exists():
        shutil.copy2(run_dir / "run.json", out_run_dir / "run.json")
    manifest = run_dir / "predictions" / "manifest.jsonl"
    if manifest.exists():
        (out_run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, out_run_dir / "predictions" / "manifest.jsonl")


def compare_formats(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: dict,
    run_dir: Path,
    out_dir: Path,
    data_dir: Path,
    converter: TeiToJatsConverter,
    split: Optional[str] = None,
) -> Path:
    """A run's TEI predictions scored as they are and converted to JATS, side by side."""
    tei_run_dir = out_dir / "tei"
    jats_run_dir = out_dir / "jats"
    for out_run_dir in (tei_run_dir, jats_run_dir):
        if out_run_dir.exists():
            shutil.rmtree(out_run_dir)
        _copy_run_records(run_dir, out_run_dir)
    for corpus_dir in sorted(p for p in (run_dir / "predictions").iterdir() if p.is_dir()):
        corpus_out_dir = tei_run_dir / "predictions" / corpus_dir.name
        corpus_out_dir.mkdir(parents=True, exist_ok=True)
        for tei_path in iter_scored_prediction_files(corpus_dir, "tei"):
            shutil.copy2(tei_path, corpus_out_dir / tei_path.name)
    convert_predictions_to_jats(
        tei_run_dir / "predictions", converter, jats_run_dir / "predictions"
    )
    for prediction_format, out_run_dir in (("tei", tei_run_dir), ("jats", jats_run_dir)):
        run_score(config, out_run_dir, data_dir, out_path=None, split_override=split,
                  prediction_format=prediction_format)
    comparison = out_dir / "comparison.md"
    run_compare(
        [("TEI", tei_run_dir / "summary.json"), ("JATS", jats_run_dir / "summary.json")],
        comparison, baseline=0,
    )
    return comparison


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score a run's TEI predictions against the JATS converted from them"
    )
    parser.add_argument("--config", default="benchmarks/eval.yml")
    parser.add_argument("--run", required=True, help="Run directory holding TEI predictions")
    parser.add_argument("--out", required=True, help="Directory for both scored runs")
    parser.add_argument("--data", default="benchmarks/data", help="Data cache directory")
    parser.add_argument(
        "--split", default=None, help="Dataset split override (default: read from run.json)"
    )
    parser.add_argument(
        "--commit", default=None,
        help="Convert with this commit's stylesheet and config (default: this checkout's)."
        " Use the commit that produced the TEI.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    compare_formats(
        config, Path(args.run), Path(args.out), Path(args.data),
        commit_converter(args.commit) if args.commit else checkout_converter(),
        split=args.split,
    )


if __name__ == "__main__":
    main()
