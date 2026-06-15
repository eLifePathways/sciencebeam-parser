"""
Utilities for find-important feature analysis of CRF model data files.

The primary entry point is :func:`find_important_data`, which loads sbeam and
GROBID .data files, aligns them, and for each differing feature tests how many
label predictions change when that feature is swapped from the GROBID values.
"""

import difflib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_crf_lines


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
    pairs: List[Tuple[int, int]] = []
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


def find_important_data(
    sbeam_data_path: str,
    alt_data_path: str,
    feature_names_path: str,
    model,
) -> Dict:
    """
    Load .data files and run find-important analysis.

    For each feature that differs between sbeam and alt (grobid), swap that
    single feature column and record which labels change.  Returns a structured
    dict suitable for JSON serialisation; does not print anything.
    """
    feature_names: List[str] = json.loads(
        Path(feature_names_path).read_text(encoding='utf-8')
    )['feature_names']
    sbeam_parts = load_parts(sbeam_data_path)
    alt_parts = load_parts(alt_data_path)
    aligned_pairs = align_parts(sbeam_parts, alt_parts)
    differing = find_differing_features(sbeam_parts, alt_parts, feature_names, aligned_pairs)
    baseline_labels = run_model(model, parts_to_feature_lines(sbeam_parts))

    per_feature: Dict = {}
    for feat in differing:
        swapped = apply_swap(sbeam_parts, alt_parts, feature_names, [feat], aligned_pairs)
        labels = run_model(model, swapped)
        changed_tokens = [
            {
                'token_idx': i,
                'token_text': sbeam_parts[i][0],
                'sbeam_label': orig,
                'grobid_label': new,
            }
            for i, (orig, new) in enumerate(zip(baseline_labels, labels))
            if orig != new
        ]
        per_feature[feat] = {
            'label_changes': len(changed_tokens),
            'changed_tokens': changed_tokens,
        }

    return {
        'total_tokens': len(baseline_labels),
        'features_with_diffs': differing,
        'per_feature': per_feature,
    }
