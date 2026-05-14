import difflib
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureValueDiff:
    feature_name: str
    grobid_value: Optional[str]
    sbeam_value: Optional[str]


@dataclass
class AlignedTokenDiff:
    token: str
    sbeam_line_number: int
    grobid_line_number: int
    feature_diffs: List[FeatureValueDiff] = field(default_factory=list)


@dataclass
class ModelDataDiffResult:
    sbeam_token_count: int
    grobid_token_count: int
    aligned_count: int
    aligned_with_diffs_count: int
    sbeam_only_count: int
    grobid_only_count: int
    aligned_diffs: List[AlignedTokenDiff] = field(default_factory=list)
    sbeam_only_tokens: List[Tuple[str, int]] = field(default_factory=list)
    grobid_only_tokens: List[Tuple[str, int]] = field(default_factory=list)


def parse_data_lines(text: str) -> List[List[str]]:
    result = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            result.append(stripped.split(' '))
    return result


def _get_token(columns: List[str]) -> str:
    return columns[0] if columns else ''


def compare_token_columns(
    sbeam_cols: List[str],
    grobid_cols: List[str],
    feature_names: Sequence[str]
) -> List[FeatureValueDiff]:
    diffs = []
    n_comparable = min(len(feature_names), len(sbeam_cols) - 1, len(grobid_cols) - 1)
    for i in range(n_comparable):
        if sbeam_cols[i] != grobid_cols[i]:
            diffs.append(FeatureValueDiff(
                feature_name=feature_names[i],
                grobid_value=grobid_cols[i],
                sbeam_value=sbeam_cols[i]
            ))
    sbeam_label = sbeam_cols[-1] if sbeam_cols else None
    grobid_label = grobid_cols[-1] if grobid_cols else None
    if sbeam_label != grobid_label:
        diffs.append(FeatureValueDiff(
            feature_name='label',
            grobid_value=grobid_label,
            sbeam_value=sbeam_label
        ))
    return diffs


def diff_model_data(  # pylint: disable=too-many-locals
    sbeam_text: str,
    grobid_text: str,
    feature_names: Sequence[str]
) -> ModelDataDiffResult:
    sbeam_lines = parse_data_lines(sbeam_text)
    grobid_lines = parse_data_lines(grobid_text)

    sbeam_tokens = [_get_token(line) for line in sbeam_lines]
    grobid_tokens = [_get_token(line) for line in grobid_lines]

    matcher = difflib.SequenceMatcher(None, sbeam_tokens, grobid_tokens, autojunk=False)

    result = ModelDataDiffResult(
        sbeam_token_count=len(sbeam_lines),
        grobid_token_count=len(grobid_lines),
        aligned_count=0,
        aligned_with_diffs_count=0,
        sbeam_only_count=0,
        grobid_only_count=0
    )

    for opcode, s0, s1, g0, g1 in matcher.get_opcodes():
        if opcode == 'equal':
            for si, gi in zip(range(s0, s1), range(g0, g1)):
                result.aligned_count += 1
                feature_diffs = compare_token_columns(
                    sbeam_lines[si], grobid_lines[gi], feature_names
                )
                if feature_diffs:
                    result.aligned_with_diffs_count += 1
                    result.aligned_diffs.append(AlignedTokenDiff(
                        token=sbeam_tokens[si],
                        sbeam_line_number=si + 1,
                        grobid_line_number=gi + 1,
                        feature_diffs=feature_diffs
                    ))
        if opcode in ('delete', 'replace'):
            for si in range(s0, s1):
                result.sbeam_only_count += 1
                result.sbeam_only_tokens.append((sbeam_tokens[si], si + 1))
        if opcode in ('insert', 'replace'):
            for gi in range(g0, g1):
                result.grobid_only_count += 1
                result.grobid_only_tokens.append((grobid_tokens[gi], gi + 1))

    return result


def format_diff_result(diff_result: ModelDataDiffResult) -> str:
    lines = [
        (
            f'sbeam: {diff_result.sbeam_token_count} tokens,'
            f' grobid: {diff_result.grobid_token_count} tokens'
        ),
        (
            f'aligned: {diff_result.aligned_count}'
            f' ({diff_result.aligned_with_diffs_count} with feature diffs),'
            f' sbeam-only: {diff_result.sbeam_only_count},'
            f' grobid-only: {diff_result.grobid_only_count}'
        ),
    ]

    detail_lines = []
    for token_diff in diff_result.aligned_diffs:
        detail_lines.append(
            f'token {token_diff.token!r}'
            f' @ sbeam:{token_diff.sbeam_line_number},'
            f' grobid:{token_diff.grobid_line_number}'
        )
        for fd in token_diff.feature_diffs:
            detail_lines.append(
                f'  {fd.feature_name}:'
                f' {fd.grobid_value!r} (grobid) → {fd.sbeam_value!r} (sbeam)'
            )

    for token, line_no in diff_result.sbeam_only_tokens:
        detail_lines.append(f'< sbeam-only: {token!r} @ sbeam:{line_no}')

    for token, line_no in diff_result.grobid_only_tokens:
        detail_lines.append(f'> grobid-only: {token!r} @ grobid:{line_no}')

    if detail_lines:
        lines.append('')
        lines.extend(detail_lines)

    return '\n'.join(lines)


def format_model_data_diff(
    sbeam_text: str,
    grobid_text: str,
    feature_names: Sequence[str]
) -> str:
    return format_diff_result(diff_model_data(sbeam_text, grobid_text, feature_names))
