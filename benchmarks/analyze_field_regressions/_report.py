from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ._aggregate import _pair_is_relevant
from ._models import MODEL_RELEVANT_LABELS
from ._types import FeatureSummary, ModelSummary, RegressionCase


def _feature_section_label(features: List[FeatureSummary]) -> str:
    n = len(features)
    m = sum(fs.total_label_changes for fs in features)
    return f'{n} feature{"s" if n != 1 else ""}, {m} label change{"s" if m != 1 else ""}'


def _render_feature_summary_table(
    features: List[FeatureSummary],
    focus: Optional[str] = None,
) -> List[str]:
    if focus == 'relevant':
        rows: List[str] = [
            '| Feature | Total docs | Total Δlabels | Relevant docs | Relevant Δlabels |',
            '|---------|------:|------:|------:|------:|',
        ]
        for fs in features:
            rows.append(
                f'| {fs.feature}'
                f' | {fs.docs_affected}'
                f' | {fs.total_label_changes}'
                f' | {fs.relevant_docs_affected or 0}'
                f' | {fs.relevant_label_changes or 0} |'
            )
    elif focus == 'other':
        rows = [
            '| Feature | Total docs | Total Δlabels | Other docs | Other Δlabels |',
            '|---------|------:|------:|------:|------:|',
        ]
        for fs in features:
            rows.append(
                f'| {fs.feature}'
                f' | {fs.docs_affected}'
                f' | {fs.total_label_changes}'
                f' | {fs.other_docs_affected or 0}'
                f' | {fs.other_label_changes or 0} |'
            )
    else:
        rows = [
            '| Feature | Total docs | Total Δlabels |',
            '|---------|------:|------:|',
        ]
        for fs in features:
            rows.append(
                f'| {fs.feature} | {fs.docs_affected} | {fs.total_label_changes} |'
            )
    return rows


def _feature_stat_cols(fs: FeatureSummary, focus: Optional[str]) -> Tuple[str, str]:
    """Return (stat_cols_str, empty_continuation_cells) for a feature row."""
    if focus == 'relevant':
        return (
            f' | {fs.relevant_docs_affected or 0} | {fs.relevant_label_changes or 0}',
            '| | | | |',
        )
    if focus == 'other':
        return (
            f' | {fs.other_docs_affected or 0} | {fs.other_label_changes or 0}',
            '| | | | |',
        )
    return '', '| | | |'


def _filter_transitions(
    all_trans: list,
    focus: Optional[str],
    relevant_labels: Optional[frozenset],
) -> list:
    if focus == 'relevant' and relevant_labels:
        return [((s, g), n) for (s, g), n in all_trans if _pair_is_relevant(s, g, relevant_labels)]
    if focus == 'other' and relevant_labels:
        return [
            ((s, g), n) for (s, g), n in all_trans
            if not _pair_is_relevant(s, g, relevant_labels)
        ]
    return all_trans


def _render_feature_table(
    features: List[FeatureSummary],
    relevant_labels: Optional[frozenset] = None,
    focus: Optional[str] = None,
) -> List[str]:
    if focus == 'relevant':
        rows: List[str] = [
            '| Feature | Total docs | Total Δlabels | Relevant docs | Relevant Δlabels'
            ' | Transition | Count |',
            '|---------|------:|------:|------:|------:|-----------|------:|',
        ]
    elif focus == 'other':
        rows = [
            '| Feature | Total docs | Total Δlabels | Other docs | Other Δlabels'
            ' | Transition | Count |',
            '|---------|------:|------:|------:|------:|-----------|------:|',
        ]
    else:
        rows = [
            '| Feature | Total docs | Total Δlabels | Transition | Count |',
            '|---------|------:|------:|-----------|------:|',
        ]
    for fs in features:
        sorted_trans = _filter_transitions(fs.transitions.most_common(), focus, relevant_labels)
        stat_cols, empty_stat = _feature_stat_cols(fs, focus)
        for i, ((s, g), n) in enumerate(sorted_trans):
            cell = f'`{s} → {g}`'
            if i == 0:
                rows.append(
                    f'| {fs.feature}'
                    f' | {fs.docs_affected}'
                    f' | {fs.total_label_changes}'
                    f'{stat_cols} | {cell} | {n} |'
                )
            else:
                rows.append(f'{empty_stat} {cell} | {n} |')
        if not sorted_trans:
            rows.append(
                f'| {fs.feature}'
                f' | {fs.docs_affected}'
                f' | {fs.total_label_changes}'
                f'{stat_cols} | — | |'
            )
    return rows


def _render_docs_table(
    cases: List[RegressionCase],
) -> List[str]:
    show_presence = any(c.presence is not None for c in cases)
    header = '| Doc | Corpus | GROBID | ScienceBeam | Δ |'
    sep = '|-----|--------|------:|------:|--:|'
    if show_presence:
        header += ' Gold | In GROBID | In raw text | In ScienceBeam |'
        sep += '---:|:---:|:---:|:---:|'
    header += ' Examples |'
    sep += '---------|'
    rows: List[str] = ['## Analyzed documents', '', header, sep]
    for case in cases:
        rel = Path('by-doc') / case.corpus / case.record_id
        cells = [
            case.record_id, case.corpus,
            f'{case.score_b:.2f}', f'{case.score_a:.2f}', f'{case.delta:+.2f}',
        ]
        if show_presence:
            p = case.presence
            cells += (
                [str(p.total), str(p.in_grobid), str(p.in_raw), str(p.in_sb)]
                if p is not None else ['-', '-', '-', '-']
            )
        cells.append(f'[{case.record_id}]({rel}/)')
        rows.append('| ' + ' | '.join(cells) + ' |')
    if show_presence:
        rows.append('')
        rows.append(
            '_Presence counts: how many gold values (out of total) appear in each prediction.'
            ' "In raw text" uses the raw ScienceBeam TEI (no external lookup).'
            ' Values absent from the raw text may have been retrieved'
            ' via external metadata lookup._'
        )
    return rows


def _generate_report(  # pylint: disable=too-many-statements
    analysis_field: str,
    run_a: Path,
    run_b: Path,
    total_regressions: int,
    cases: List[RegressionCase],
    model_summaries: List[ModelSummary],
) -> str:
    field_model_labels = MODEL_RELEVANT_LABELS.get(analysis_field, {})
    regression_note = (
        f'{total_regressions} total, {len(cases)} analyzed'
        if len(cases) < total_regressions
        else str(total_regressions)
    )
    lines: List[str] = []
    lines += [
        f'# Field Analysis: {analysis_field}',
        '',
        f'- **Run A (ScienceBeam)**: `{run_a}`',
        f'- **Run B (GROBID)**: `{run_b}`',
        f'- **Regression documents**: {regression_note}',
        '',
    ]

    lines += _render_docs_table(cases)
    lines.append('')

    for ms in model_summaries:
        lines.append(
            f'## {ms.model} ({ms.docs_analyzed} docs analyzed'
            + (f', {ms.docs_failed} failed' if ms.docs_failed else '')
            + f', {ms.docs_with_feature_diffs} with feature diffs)'
        )
        lines.append('')
        if not ms.features:
            if ms.docs_with_feature_diffs:
                lines += ['_Features differ but caused no meaningful label changes._', '']
            else:
                lines += ['_No feature differences found._', '']
            continue

        relevant_labels = field_model_labels.get(ms.model)
        if relevant_labels:
            labels_str = ', '.join(f'`{lbl}`' for lbl in sorted(relevant_labels))
            rel_feats = [fs for fs in ms.features if fs.relevant_docs_affected]
            other_feats = [fs for fs in ms.features if fs.other_docs_affected]

            if rel_feats:
                lines.append(
                    f'<details><summary>Features affecting {labels_str}'
                    f' — {_feature_section_label(rel_feats)}</summary>'
                )
                lines.append('')
                lines += _render_feature_summary_table(rel_feats, focus='relevant')
                lines.append('')
                lines.append('<details><summary>Transition detail</summary>')
                lines.append('')
                lines += _render_feature_table(rel_feats, relevant_labels, focus='relevant')
                lines.append('')
                lines.append('</details>')
                lines.append('')
                lines.append('</details>')
            else:
                lines.append(
                    f'_No features directly affect {labels_str} in these documents._'
                )
            lines.append('')

            if other_feats:
                lines.append(
                    f'<details><summary>Other feature differences'
                    f' — {_feature_section_label(other_feats)}</summary>'
                )
                lines.append('')
                lines += _render_feature_summary_table(other_feats, focus='other')
                lines.append('')
                lines.append('<details><summary>Transition detail</summary>')
                lines.append('')
                lines += _render_feature_table(other_feats, relevant_labels, focus='other')
                lines.append('')
                lines.append('</details>')
                lines.append('')
                lines.append('</details>')
                lines.append('')
        else:
            lines += _render_feature_table(ms.features)
            lines.append('')

    return '\n'.join(lines)
