#!/usr/bin/env python
"""
Diff model predictions when swapping feature columns from an alternative data file.

Baseline is always the sbeam .data file run through the model.
The alt data file (typically grobid's) is used as the source for swapped values.

Usage:
    # Swap a single feature for ALL tokens and see which labels change:
    python scripts/diff_feature_swap.py \\
        --data .temp/.../sciencebeam-parser/segmentation.data \\
        --alt-data .temp/.../grobid/segmentation.data \\
        --feature-names .../segmentation.feature_names.json \\
        --model segmentation \\
        --swap-features block_relative_line_length

    # Swap ALL differing features (model equivalence check):
    python scripts/diff_feature_swap.py ... --swap-features all

    # Test every differing feature independently:
    python scripts/diff_feature_swap.py ... --find-important

    # Binary search for the minimal feature subset that causes the same label changes
    # as swapping all features (run after --find-important shows nothing individually):
    python scripts/diff_feature_swap.py ... --binary-search
"""

import argparse
import contextlib
import json
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from sciencebeam_trainer_delft.sequence_labelling.reader import (  # noqa: E402
    load_data_crf_lines
)

from sciencebeam_parser.app.parser import ScienceBeamParser  # noqa: E402
from sciencebeam_parser.service.server import get_app_config  # noqa: E402


LOGGER = logging.getLogger(__name__)


def load_parts(data_path: str) -> List[List[str]]:
    """Read a .data file; return list of per-line token lists (label included as last element)."""
    raw = Path(data_path).read_text(encoding='utf-8').splitlines()
    return [line.split() for line in raw if line.strip()]


def parts_to_feature_lines(parts_list: List[List[str]]) -> List[str]:
    """Strip label (last column) from each token row; return as joined strings for the model."""
    return [' '.join(parts[:-1]) for parts in parts_list]


def find_differing_features(
    sbeam: List[List[str]],
    alt: List[List[str]],
    feature_names: List[str],
) -> List[str]:
    """Return the ordered list of feature names that differ at least once across all tokens."""
    differing: Set[str] = set()
    for s_parts, a_parts in zip(sbeam, alt):
        for i, name in enumerate(feature_names):
            if i < len(s_parts) - 1 and i < len(a_parts) - 1:
                if s_parts[i] != a_parts[i]:
                    differing.add(name)
    return [f for f in feature_names if f in differing]


def apply_swap(
    sbeam: List[List[str]],
    alt: List[List[str]],
    feature_names: List[str],
    swap_features: List[str],
) -> List[str]:
    """Return feature-lines with specified feature columns taken from alt instead of sbeam."""
    swap_indices = {feature_names.index(f) for f in swap_features}
    result = []
    for s_parts, a_parts in zip(sbeam, alt):
        merged = list(s_parts)
        for i in swap_indices:
            if i < len(a_parts) - 1:
                merged[i] = a_parts[i]
        result.append(' '.join(merged[:-1]))  # strip label
    return result


def run_model(model, lines: List[str]) -> List[str]:
    texts, features = load_data_crf_lines(lines)
    tag_result = model.model_impl.predict_labels(texts, features)
    return [label for _token, label in tag_result[0]]


def show_diff(
    baseline: List[str],
    updated: List[str],
    sbeam_parts: List[List[str]],
    header: Optional[str] = None,
    baseline_label: str = 'sbeam',
    updated_label: str = 'swapped',
) -> int:
    if header:
        print(f'\n=== {header} ===')
    changed = 0
    for i, (orig, new) in enumerate(zip(baseline, updated)):
        if orig != new:
            token = sbeam_parts[i][0] if sbeam_parts else '?'
            print(
                f'  token {i:3d} {token!r:20s}'
                f'  {baseline_label}:{orig!r:20s}'
                f'  → {updated_label}:{new!r}'
            )
            changed += 1
    print(f'{changed} label(s) changed out of {len(baseline)} tokens.')
    return changed


def binary_search_features(
    sbeam: List[List[str]],
    alt: List[List[str]],
    feature_names: List[str],
    features: List[str],
    target_changes: Set[Tuple[int, str, str]],
    model,
    baseline: List[str],
) -> List[str]:
    """Return the smallest subset of `features` that reproduces `target_changes`."""
    if not features:
        return []
    swapped_lines = apply_swap(sbeam, alt, feature_names, features)
    labels = run_model(model, swapped_lines)
    changes = {
        (i, orig, new)
        for i, (orig, new) in enumerate(zip(baseline, labels))
        if orig != new
    }
    if not (target_changes <= changes):
        return []  # this subset doesn't reproduce the target changes

    if len(features) == 1:
        return features

    mid = len(features) // 2
    left = features[:mid]
    right = features[mid:]

    # Try each half independently
    left_result = binary_search_features(
        sbeam, alt, feature_names, left, target_changes, model, baseline
    )
    if left_result:
        return left_result

    right_result = binary_search_features(
        sbeam, alt, feature_names, right, target_changes, model, baseline
    )
    if right_result:
        return right_result

    # Neither half alone is sufficient — the smallest known set is the current one
    return features


class _Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams: IO[str]):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def tee_stdout(path: str) -> Iterator[None]:
    """Context manager that tees stdout to a file while keeping terminal output."""
    with open(path, 'w', encoding='utf-8') as fh:
        original = sys.stdout
        sys.stdout = _Tee(original, fh)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = original


def main():
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='sbeam .data file (baseline)')
    parser.add_argument('--alt-data', required=True, dest='alt_data',
                        help='Alternative .data file (e.g. grobid) to swap columns from')
    parser.add_argument('--feature-names', required=True, dest='feature_names',
                        help='Path to feature_names.json sidecar')
    parser.add_argument('--model', default='segmentation',
                        help='Model name (segmentation, header, …). Default: segmentation')
    parser.add_argument('--swap-features', dest='swap_features',
                        help=(
                            'Comma-separated feature names to swap from alt-data, '
                            'or "all" to swap every differing feature'
                        ))
    parser.add_argument('--find-important', dest='find_important', action='store_true',
                        help='Test each differing feature independently; report which cause changes')
    parser.add_argument('--binary-search', dest='binary_search', action='store_true',
                        help=(
                            'Binary-search the differing features for the minimal subset '
                            'that reproduces the same label changes as swapping all features'
                        ))
    parser.add_argument('--output', default=None,
                        help='Also write output to this file (tee; terminal output is kept)')
    args = parser.parse_args()

    feature_names = json.loads(
        Path(args.feature_names).read_text(encoding='utf-8')
    )['feature_names']

    ctx = tee_stdout(args.output) if args.output else contextlib.nullcontext()
    with ctx:
        _run(args, feature_names)


def _run(args, feature_names):
    sbeam_parts = load_parts(args.data)
    alt_parts = load_parts(args.alt_data)

    if len(sbeam_parts) != len(alt_parts):
        sys.exit(
            f'Token count mismatch: sbeam has {len(sbeam_parts)} tokens, '
            f'alt has {len(alt_parts)}'
        )

    config = get_app_config()
    sb_parser = ScienceBeamParser.from_config(config)
    model = sb_parser.fulltext_models.get_sequence_model_by_name(args.model)

    baseline_lines = parts_to_feature_lines(sbeam_parts)
    baseline_labels = run_model(model, baseline_lines)

    differing = find_differing_features(sbeam_parts, alt_parts, feature_names)
    print(f'Features that differ between sbeam and alt: {differing}')

    # Column label: what does the baseline represent vs the swapped version?
    baseline_label = 'sbeam'
    updated_label = 'alt(grobid)'

    if args.find_important:
        print(f'\n--- Individual feature importance (label format: {baseline_label} → {updated_label}) ---')
        for feat in differing:
            swapped = apply_swap(sbeam_parts, alt_parts, feature_names, [feat])
            labels = run_model(model, swapped)
            changed = sum(1 for a, b in zip(baseline_labels, labels) if a != b)
            if changed:
                show_diff(
                    baseline_labels, labels, sbeam_parts,
                    header=feat,
                    baseline_label=baseline_label,
                    updated_label=updated_label,
                )
            else:
                print(f'[{feat}]: 0 label(s) changed')
        return

    if args.binary_search:
        print('\n--- Binary search for minimal feature subset ---')
        all_swapped = apply_swap(sbeam_parts, alt_parts, feature_names, differing)
        all_labels = run_model(model, all_swapped)
        target_changes = {
            (i, orig, new)
            for i, (orig, new) in enumerate(zip(baseline_labels, all_labels))
            if orig != new
        }
        if not target_changes:
            print('Swapping all features produces 0 label changes — nothing to search for.')
            return
        print(f'Swapping all {len(differing)} features causes {len(target_changes)} label change(s).')
        minimal = binary_search_features(
            sbeam_parts, alt_parts, feature_names, differing,
            target_changes, model, baseline_labels
        )
        if minimal:
            print(f'Minimal subset found ({len(minimal)} feature(s)): {minimal}')
            swapped = apply_swap(sbeam_parts, alt_parts, feature_names, minimal)
            labels = run_model(model, swapped)
            show_diff(
                baseline_labels, labels, sbeam_parts,
                baseline_label=baseline_label,
                updated_label=updated_label,
            )
        else:
            print('No subset found that reproduces the target changes.')
        return

    # Default: swap specified features and show diff
    if args.swap_features == 'all' or not args.swap_features:
        swap_list = differing
    else:
        swap_list = [f.strip() for f in args.swap_features.split(',')]
        unknown = [f for f in swap_list if f not in feature_names]
        if unknown:
            sys.exit(f'Unknown feature(s): {unknown}. Available: {feature_names}')

    print(f'\nSwapping: {swap_list}  ({baseline_label} → {updated_label})')
    swapped = apply_swap(sbeam_parts, alt_parts, feature_names, swap_list)
    labels = run_model(model, swapped)
    show_diff(
        baseline_labels, labels, sbeam_parts,
        baseline_label=baseline_label,
        updated_label=updated_label,
    )


if __name__ == '__main__':
    main()
