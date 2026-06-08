#!/usr/bin/env python
"""
Diff model predictions before and after applying feature overrides.

Usage:
    python scripts/diff_feature_override.py \\
        --data .temp/compare-with-grobid/by-doc/<doc>/sciencebeam-parser/segmentation.data \\
        --feature-names .temp/compare-with-grobid/by-doc/<doc>/sciencebeam-parser/segmentation.feature_names.json \\
        --model segmentation \\
        --override "Braz:is_repetitive_pattern=0"

Multiple --override flags are applied in order. Each targets every line
whose leading token text matches.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from sciencebeam_trainer_delft.sequence_labelling.reader import (  # noqa: E402
    load_data_crf_lines
)

from sciencebeam_parser.app.parser import ScienceBeamParser  # noqa: E402
from sciencebeam_parser.service.server import get_app_config  # noqa: E402


LOGGER = logging.getLogger(__name__)


def parse_patch(patch_str: str) -> Tuple[str, str, str]:
    """Parse 'TOKEN:feature=value' into (token, feature, value)."""
    token, rest = patch_str.split(':', 1)
    feature, value = rest.split('=', 1)
    return token.strip(), feature.strip(), value.strip()


def apply_patch(
    lines: List[str],
    feature_names: List[str],
    token: str,
    feature: str,
    value: str
) -> List[str]:
    """Replace the feature column on every line whose first token matches."""
    col_idx = feature_names.index(feature)
    result = []
    for line in lines:
        parts = line.split()
        if parts and parts[0] == token:
            parts[col_idx] = value
            result.append(' '.join(parts))
        else:
            result.append(line)
    return result


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Path to .data file')
    parser.add_argument(
        '--feature-names', required=True, dest='feature_names',
        help='Path to feature_names.json sidecar'
    )
    parser.add_argument(
        '--model', default='segmentation',
        help='Model name (segmentation, header, …). Default: segmentation'
    )
    parser.add_argument(
        '--override', action='append', dest='overrides', default=[],
        metavar='TOKEN:feature=value',
        help='Override a feature value: TOKEN:feature=value. May be repeated.'
    )
    args = parser.parse_args()

    feature_names = json.loads(
        Path(args.feature_names).read_text(encoding='utf-8')
    )['feature_names']

    raw_lines = Path(args.data).read_text(encoding='utf-8').splitlines()
    raw_lines = [line for line in raw_lines if line.strip()]

    # Strip label column so load_data_crf_lines sees only features
    stripped = [' '.join(line.split()[:-1]) for line in raw_lines]

    # Apply each override in order
    patched = list(stripped)
    for patch_str in args.overrides:
        token, feature, value = parse_patch(patch_str)
        if feature not in feature_names:
            sys.exit(
                f"Unknown feature {feature!r}. Available: {feature_names}"
            )
        patched = apply_patch(patched, feature_names, token, feature, value)

    # Load model via the real app config path
    config = get_app_config()
    sb_parser = ScienceBeamParser.from_config(config)
    model = sb_parser.fulltext_models.get_sequence_model_by_name(args.model)

    def run_model(lines: List[str]) -> List[str]:
        texts, features = load_data_crf_lines(lines)
        tag_result = model.model_impl.predict_labels(texts, features)
        return [label for _token, label in tag_result[0]]

    # Run inference on baseline (no overrides) and on overridden data
    baseline_labels = run_model(stripped)
    overridden_labels = run_model(patched)

    # Show predictions that changed due to the overrides
    changed = 0
    for i, (orig, new) in enumerate(zip(baseline_labels, overridden_labels)):
        token = raw_lines[i].split()[0]
        if orig != new:
            print(f"* token {i:3d} {token!r:20s}  {orig!r:20s} → {new!r}")
            changed += 1
    print()
    print(f'{changed} label(s) changed out of {len(baseline_labels)} tokens.')


if __name__ == '__main__':
    main()
