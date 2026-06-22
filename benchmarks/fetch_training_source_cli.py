"""CLI: fetch CC-BY source data (PDF + JATS XML) for GROBID training data generation."""
import argparse
import logging
from pathlib import Path

import yaml

from benchmarks.fetch import fetch_training_source

LOGGER = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch CC-BY PDF + JATS XML pairs for GROBID training data generation."
    )
    parser.add_argument(
        "--config",
        default="benchmarks/training-source.yml",
        help="Path to training-source config YAML (default: benchmarks/training-source.yml)",
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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    records = fetch_training_source(cfg, args.mode, args.split, Path(args.output_path))
    LOGGER.info("Fetched %d records to %s", len(records), args.output_path)


if __name__ == "__main__":
    main()
