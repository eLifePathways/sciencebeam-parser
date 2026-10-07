"""CLI: fetch CC-BY source data (PDF + JATS XML) for GROBID training data generation."""
import argparse
import logging
from pathlib import Path

import yaml

from benchmarks.fetch import fetch_training_source
from benchmarks.training_source_config import (
    DEFAULT_CONFIG,
    get_default_selection_path,
)

LOGGER = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch CC-BY PDF + JATS XML pairs for GROBID training data generation."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=(
            "Path to the training-source config in the generated repo"
            f" (default: {DEFAULT_CONFIG})"
        ),
    )
    parser.add_argument(
        "--mode",
        default="smoke",
        help="Sampling mode defined in the config (e.g. smoke, small, full)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to fetch (default: train)",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory to write PDF and JATS XML files into",
    )
    parser.add_argument(
        "--selection-path",
        help=(
            "Directory holding each corpus's recorded selection"
            " (default: a selection directory beside the config)"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    selection_dir = (
        Path(args.selection_path) if args.selection_path
        else get_default_selection_path(args.config)
    )
    records = fetch_training_source(
        cfg, args.mode, args.split, Path(args.output_path),
        selection_dir=selection_dir,
    )
    LOGGER.info("Fetched %d records to %s", len(records), args.output_path)


if __name__ == "__main__":
    main()
