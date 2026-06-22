"""CLI: generate GROBID training data for each CC-BY source corpus.

Reads cc_by_corpora from training-source.yml and calls generate_data once per
corpus, writing output to <output-path>/<split>/<corpus>/.  Any extra arguments
after -- are forwarded verbatim to generate_data.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

from sciencebeam_parser.training.cli.generate_data import main as generate_data_main

LOGGER = logging.getLogger(__name__)


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
        default="benchmarks/training-source.yml",
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

        corpus_argv = [
            "--source-path", str(corpus_source / "*.pdf"),
            "--source-xml-path", str(corpus_source / "*.jats.xml"),
            "--output-path", str(corpus_output),
            "--use-directory-structure",
            *extra_argv,
        ]
        try:
            generate_data_main(corpus_argv)
        except Exception:  # pylint: disable=broad-except
            LOGGER.exception("Failed to generate training data for corpus %r", corpus)
            errors.append(corpus)

    if errors:
        LOGGER.error("Generation failed for corpora: %s", ", ".join(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()
