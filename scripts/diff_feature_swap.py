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
import difflib
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


def align_parts(
    sbeam_parts: List[List[str]],
    alt_parts: List[List[str]],
) -> List[Tuple[int, int]]:
    """Align sbeam and alt token lists by token text; return list of (si, gi) index pairs."""
    sbeam_tokens = [p[0] for p in sbeam_parts]
    alt_tokens = [p[0] for p in alt_parts]
    matcher = difflib.SequenceMatcher(None, sbeam_tokens, alt_tokens, autojunk=False)
    pairs = []
    for opcode, s0, s1, g0, g1 in matcher.get_opcodes():
        if opcode == 'equal':
            pairs.extend(zip(range(s0, s1), range(g0, g1)))
    return pairs


def find_differing_features(
    sbeam: List[List[str]],
    alt: List[List[str]],
    feature_names: List[str],
    aligned_pairs: List[Tuple[int, int]],
) -> List[str]:
    """Return the ordered list of feature names that differ at least once across aligned tokens."""
    differing: Set[str] = set()
    for si, gi in aligned_pairs:
        s_parts, a_parts = sbeam[si], alt[gi]
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
    aligned_pairs: List[Tuple[int, int]],
) -> List[str]:
    """Return feature-lines with specified feature columns taken from alt for aligned tokens."""
    swap_indices = {feature_names.index(f) for f in swap_features}
    result = [' '.join(parts[:-1]) for parts in sbeam]
    for si, gi in aligned_pairs:
        merged = list(sbeam[si])
        a_parts = alt[gi]
        for i in swap_indices:
            if i < len(a_parts) - 1:
                merged[i] = a_parts[i]
        result[si] = ' '.join(merged[:-1])
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
    aligned_pairs: List[Tuple[int, int]],
) -> List[str]:
    """Return the smallest subset of `features` that reproduces `target_changes`."""
    if not features:
        return []
    swapped_lines = apply_swap(sbeam, alt, feature_names, features, aligned_pairs)
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
        sbeam, alt, feature_names, left, target_changes, model, baseline, aligned_pairs
    )
    if left_result:
        return left_result

    right_result = binary_search_features(
        sbeam, alt, feature_names, right, target_changes, model, baseline, aligned_pairs
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

    aligned_pairs = align_parts(sbeam_parts, alt_parts)
    sbeam_only = len(sbeam_parts) - len(aligned_pairs)
    alt_only = len(alt_parts) - len(aligned_pairs)

    if len(sbeam_parts) != len(alt_parts):
        print(
            f'Note: token count mismatch — sbeam: {len(sbeam_parts)},'
            f' alt: {len(alt_parts)}.'
            f' Aligned: {len(aligned_pairs)}'
            f' (sbeam-only: {sbeam_only}, alt-only: {alt_only}).',
            file=sys.stderr
        )

    config = get_app_config()
    sb_parser = ScienceBeamParser.from_config(config)
    sb_model_name = args.model.replace('-', '_')
    model = sb_parser.fulltext_models.get_sequence_model_by_name(sb_model_name)

    baseline_lines = parts_to_feature_lines(sbeam_parts)
    baseline_labels = run_model(model, baseline_lines)

    differing = find_differing_features(sbeam_parts, alt_parts, feature_names, aligned_pairs)
    print(f'Features that differ between sbeam and alt: {differing}')

    # Column label: what does the baseline represent vs the swapped version?
    baseline_label = 'sbeam'
    updated_label = 'alt(grobid)'

    if args.find_important:
        print(f'\n--- Individual feature importance (label format: {baseline_label} → {updated_label}) ---')
        for feat in differing:
            swapped = apply_swap(sbeam_parts, alt_parts, feature_names, [feat], aligned_pairs)
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
        all_swapped = apply_swap(sbeam_parts, alt_parts, feature_names, differing, aligned_pairs)
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
            target_changes, model, baseline_labels, aligned_pairs
        )
        if minimal:
            print(f'Minimal subset found ({len(minimal)} feature(s)): {minimal}')
            swapped = apply_swap(sbeam_parts, alt_parts, feature_names, minimal, aligned_pairs)
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
    swapped = apply_swap(sbeam_parts, alt_parts, feature_names, swap_list, aligned_pairs)
    labels = run_model(model, swapped)
    show_diff(
        baseline_labels, labels, sbeam_parts,
        baseline_label=baseline_label,
        updated_label=updated_label,
    )


if __name__ == '__main__':
    main()
