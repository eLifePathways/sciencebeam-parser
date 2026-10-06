"""CLI: generate GROBID training data from one source tree.

Reads cc_by_corpora from training-source.yml and calls generate_data once per
corpus, writing output to <output-path>/<split>/<corpus>/.  Any extra arguments
after -- are forwarded verbatim to generate_data.

This takes the tree it is given and has no opinion about which mode each model is
declared at. `generate_training_data_cli` is the one that reads the declaration,
fetches each mode it names and calls this per corpus.

The models are named here rather than forwarded blindly, because each one gets a
record of what its data was generated from, and that record cannot be written for
a model list this does not know.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from sciencebeam_parser.training.cli.generate_data import (
    get_enabled_model_names,
    main as generate_data_main,
)

from benchmarks.training_intent import get_declared_pairs
from benchmarks.training_records import read_source_manifest
from benchmarks.training_source_config import DEFAULT_CONFIG

LOGGER = logging.getLogger(__name__)


def get_declared_models_by_corpus(cfg: dict) -> Dict[str, List[str]]:
    """The models each corpus declares, whatever mode it declares them at.

    This is what generating without `--models` covers: the config is the only
    statement of which models are wanted, so a run that is not told otherwise
    produces those rather than every model there is.
    """
    declared: Dict[str, List[str]] = {}
    for pair in get_declared_pairs(cfg):
        declared.setdefault(pair.corpus, []).append(pair.model)
    return declared


def write_pair_records(
    corpus_source: Path,
    corpus_output: Path,
    model_names: Optional[Sequence[str]],
) -> None:
    """Carry what the fetch resolved into a record beside each model's data.

    Without a manifest there is nothing to carry: the source tree was assembled
    by hand, and a record claiming a mode it cannot know would be worse than none.
    """
    manifest = read_source_manifest(corpus_source)
    if manifest is None:
        LOGGER.warning(
            "No source manifest in %s, so no mode is recorded for what it generated."
            " Fetch writes one; a hand-assembled source tree has none.",
            corpus_source,
        )
        return
    for model_name in model_names or get_enabled_model_names(None):
        file_path = manifest.to_pair_record(model_name).write(corpus_output / model_name)
        LOGGER.info("Wrote pair record: %s", file_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate GROBID training data for all CC-BY corpora in the training-source config."
        ),
        # Allow forwarding unknown flags to generate_data
        epilog="Any additional arguments are forwarded to generate_data.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to training-source config YAML",
    )
    parser.add_argument(
        "--source-data",
        required=True,
        help="Root directory of fetched source PDFs and JATS XML (e.g. data/source-training-data)",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Root directory of the output repo (e.g. data/generated-training-data)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split subdirectory (default: train)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Models to generate for (default: the ones the config declares for each"
            " corpus, or every model generate_data produces if it declares none)"
        ),
    )
    parser.add_argument(
        "--corpus",
        nargs="+",
        help="Generate only these corpora (default: every corpus in cc_by_corpora)",
    )
    args, extra_argv = parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    corpora = cfg.get("cc_by_corpora", [])
    if not corpora:
        LOGGER.warning("No cc_by_corpora defined in %s; nothing to generate.", args.config)
        sys.exit(0)

    if args.corpus:
        unknown = sorted(set(args.corpus) - set(corpora))
        if unknown:
            LOGGER.error("Corpora %s are not in cc_by_corpora", unknown)
            sys.exit(1)
        corpora = [corpus for corpus in corpora if corpus in set(args.corpus)]

    declared_by_corpus = get_declared_models_by_corpus(cfg)

    errors = []
    for corpus in corpora:
        corpus_source = Path(args.source_data) / args.split / corpus
        if not corpus_source.exists():
            LOGGER.warning(
                "Source directory not found for corpus %r, skipping: %s", corpus, corpus_source
            )
            continue

        corpus_output = Path(args.output_path) / args.split / corpus
        LOGGER.info("Generating training data for corpus %r -> %s", corpus, corpus_output)

        model_names = args.models or declared_by_corpus.get(corpus)
        corpus_argv = [
            "--source-path", str(corpus_source / "*.pdf"),
            "--source-xml-path", str(corpus_source / "*.jats.xml"),
            "--output-path", str(corpus_output),
            "--use-directory-structure",
            *(["--models", *model_names] if model_names else []),
            *extra_argv,
        ]
        try:
            generate_data_main(corpus_argv)
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Failed to generate training data for corpus %r", corpus)
            errors.append(corpus)
            continue
        write_pair_records(corpus_source, corpus_output, model_names)

    if errors:
        LOGGER.error("Generation failed for corpora: %s", ", ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
