#!/usr/bin/env python3
"""Normalize delft/wapiti model data files to a common tab-separated format.

Handles two input formats:
  GROBID debugMode:               tab-separated, 33 features + label
  ScienceBeam output_format=data: space-separated, 33 features + block_text + label

GROBID multi-model responses may contain '=== model: X ===' headers; these are stripped.
Blank lines (segment separators) are preserved.

Usage:
    python scripts/normalize_model_data.py <input_file> <output_file>
"""

import sys
from pathlib import Path

from sciencebeam_parser.utils.model_data_normalizer import normalize_file


def main() -> None:
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <input_file> <output_file>', file=sys.stderr)
        sys.exit(1)
    normalize_file(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == '__main__':
    main()
