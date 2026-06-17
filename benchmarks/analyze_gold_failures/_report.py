from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from ._types import (
    FAILURE_MODE_LABEL,
    DocumentSummary,
    FailureMode,
    FailureModeAggregate,
    FieldFailureSummary,
    GoldValueResult,
    PipelineAttribution,
)

_PRESENCE_FOOTNOTE = (
    '_Presence checks use whitespace-normalised substring matching. '
    '"Not found in raw text" means the gold value was not found in the raw '
    'ScienceBeam TEI output. This may indicate the value is absent from the PDF, '
    'or that the raw text extraction renders it differently '
    '(encoding, ligatures, line-breaks). '
    'Such values are less likely to be recoverable by model training alone, '
    'but this is not guaranteed._'
)

_MODE_DETAIL_FILE: Dict[FailureMode, str] = {
    FailureMode.NOT_IN_RAW_TEXT: 'not_in_raw_text.md',
    FailureMode.EXTRACTION_FAILED: 'extraction_failed.md',
    FailureMode.PARTIAL_WRONG: 'partial_wrong.md',
}
_NEAR_MISS_DETAIL_FILE = 'near_miss.md'
_ANALYZED_DOCS_DETAIL_FILE = 'analyzed_documents.md'
_CONTEXT_WINDOW_DETAIL_FILE = 'context_window.md'

_DASH_NORM_MAP = str.maketrans({
    0x2010: '-', 0x2011: '-', 0x2012: '-', 0x2013: '-', 0x2014: '-',
})


def _classify_near_miss(  # pylint: disable=too-many-return-statements
    gold: str, extracted: Optional[str]
) -> str:
    """Classify the type of difference between a gold value and its extracted near-miss."""
    if not extracted:
        return 'Other'
    if gold.lower() == extracted.lower():
        return 'Case only'
    g_d = gold.translate(_DASH_NORM_MAP)
    e_d = extracted.translate(_DASH_NORM_MAP)
    if g_d == e_d:
        return 'Dash/hyphen variant'
    if g_d.lower() == e_d.lower():
        return 'Case + dash variant'
    g_strip = gold.rstrip('.,;: ')
    e_strip = extracted.rstrip('.,;: ')
    if g_strip == e_strip or g_strip.lower() == e_strip.lower():
        return 'Trailing punctuation'
    if len(gold) == len(extracted):
        diffs = sum(a != b for a, b in zip(gold, extracted))
        if diffs == 1:
            return 'Single-char (encoding)'
    return 'Other'


def _strip_bio(label: str) -> str:
    return re.sub(r'^[BI]-', '', label)


SAMPLE_SIZE = 5


def _find_near_misses(
    doc_summaries: List[DocumentSummary],
) -> List[Tuple[str, str, GoldValueResult]]:
    return [
        (doc.corpus, doc.record_id, result)
        for doc in doc_summaries
        for result in doc.results
        if result.mode == FailureMode.CORRECT
        and result.best_sb_similarity is not None
        and result.best_sb_similarity < 1.0
    ]


def _pct(n: int, total: int) -> str:
    return f'{100 * n / total:.0f}%' if total else '—'


def _doc_link(corpus: str, record_id: str) -> str:
    return f'[{record_id}](by-doc/{corpus}/{record_id}/)'


def _render_summary_table(
    summary: FieldFailureSummary,
    near_miss_count: int = 0,
    near_miss_docs: int = 0,
) -> List[str]:
    lines = [
        '| Failure mode | Docs | Values | % of Gold |',
        '| --- | ---: | ---: | ---: |',
    ]
    for agg in summary.mode_aggregates:
        label = FAILURE_MODE_LABEL[agg.mode]
        pct = _pct(agg.total_values, summary.total_gold)
        lines.append(
            f'| {label} | {agg.docs_affected} | {agg.total_values} | {pct} |'
        )
        if agg.mode == FailureMode.CORRECT and near_miss_count:
            pct_nm = _pct(near_miss_count, summary.total_gold)
            lines.append(
                f'| — of which near-miss | {near_miss_docs} | {near_miss_count} | {pct_nm} |'
            )
    lines.append(
        f'| **Total** | **{summary.total_docs}** | **{summary.total_gold}** | 100% |'
    )
    return lines


_DOC_TABLE_HEADER = [
    '| Doc | Corpus | Gold | NR | Failed | Partial | Correct | ScienceBeam F1 | Examples |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
]


def _render_doc_row(doc: DocumentSummary) -> str:
    counts = doc.mode_counts
    score = f'{doc.score_sb:.2f}' if doc.score_sb is not None else '—'
    link = _doc_link(doc.corpus, doc.record_id)
    return (
        f'| {doc.record_id}'
        f' | {doc.corpus}'
        f' | {doc.total_gold}'
        f' | {counts[FailureMode.NOT_IN_RAW_TEXT]}'
        f' | {counts[FailureMode.EXTRACTION_FAILED]}'
        f' | {counts[FailureMode.PARTIAL_WRONG]}'
        f' | {counts[FailureMode.CORRECT]}'
        f' | {score}'
        f' | {link} |'
    )


def _render_doc_table(doc_summaries: List[DocumentSummary]) -> List[str]:
    lines = list(_DOC_TABLE_HEADER)
    for doc in doc_summaries:
        lines.append(_render_doc_row(doc))
    return lines


def _build_attr_lookup(
    doc_summaries: List[DocumentSummary],
) -> Dict[Tuple[str, str, str], PipelineAttribution]:
    lookup: Dict[Tuple[str, str, str], PipelineAttribution] = {}
    for doc in doc_summaries:
        for value, attr in doc.attributions.items():
            lookup[(doc.corpus, doc.record_id, value)] = attr
    return lookup


def _format_label_info(predicted: Optional[str], expected: Optional[str]) -> str:
    if predicted and expected and predicted != expected:
        return f'`{predicted}` ≠ `{expected}`'
    if predicted:
        return f'`{predicted}`'
    return '—'


def _format_candidate_text(candidate_text: Optional[str]) -> str:
    return candidate_text or '—'


def _format_sim(gold_value: str, candidate_text: Optional[str]) -> str:
    if not candidate_text:
        return '—'
    ratio = SequenceMatcher(None, gold_value, candidate_text).ratio()
    return f'{ratio:.2f}'


def _render_attribution_summary(
    examples: List[Tuple[str, str, GoldValueResult]],
    attr_lookup: Dict[Tuple[str, str, str], PipelineAttribution],
) -> List[str]:
    """Build a combined attribution summary (values + docs) from all extraction-failed examples."""
    model_vals: Counter = Counter()
    model_docs: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)
    note_vals: Counter = Counter()
    note_docs: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for corpus, record_id, result in examples:
        attr = attr_lookup.get((corpus, record_id, result.value))
        if attr:
            if attr.first_failed_model is not None:
                model_vals[attr.first_failed_model] += 1
                model_docs[attr.first_failed_model].add((corpus, record_id))
            elif attr.attribution_note:
                note_vals[attr.attribution_note] += 1
                note_docs[attr.attribution_note].add((corpus, record_id))
    if not model_vals and not note_vals:
        return []
    lines = [
        '### Attribution summary',
        '',
        '| Attribution | Docs | Values |',
        '| --- | ---: | ---: |',
    ]
    for model, count in sorted(model_vals.items(), key=lambda x: -x[1]):
        lines.append(f'| `{model}` | {len(model_docs[model])} | {count} |')
    for note, count in sorted(note_vals.items(), key=lambda x: -x[1]):
        lines.append(f'| _{note}_ | {len(note_docs[note])} | {count} |')
    lines.append('')
    return lines


def _render_label_breakdown(
    examples: List[Tuple[str, str, GoldValueResult]],
    attr_lookup: Dict[Tuple[str, str, str], PipelineAttribution],
) -> List[str]:
    """Breakdown by (model, predicted, expected) for attributed rows."""
    breakdown: Counter = Counter()
    breakdown_docs: DefaultDict[tuple, Set[Tuple[str, str]]] = defaultdict(set)
    for corpus, record_id, result in examples:
        attr = attr_lookup.get((corpus, record_id, result.value))
        if attr and attr.first_failed_model is not None:
            key = (attr.first_failed_model, attr.predicted_label, attr.expected_label)
            breakdown[key] += 1
            breakdown_docs[key].add((corpus, record_id))
    if not breakdown:
        return []
    lines = [
        '### Label breakdown',
        '',
        '| Model | Predicted ≠ Expected | Docs | Values |',
        '| --- | --- | ---: | ---: |',
    ]
    for (model, predicted, expected), count in sorted(breakdown.items(), key=lambda x: -x[1]):
        docs = len(breakdown_docs[(model, predicted, expected)])
        lines.append(
            f'| `{model}` | {_format_label_info(predicted, expected)} | {docs} | {count} |'
        )
    lines.append('')
    return lines


def _render_ef_row(
    record_id: str,
    corpus: str,
    gold_value: str,
    attr: Optional[PipelineAttribution],
) -> str:
    if attr and attr.first_failed_model is not None:
        failed_model = f'`{attr.first_failed_model}`'
        label_info = _format_label_info(attr.predicted_label, attr.expected_label)
        data_text = _format_candidate_text(attr.candidate_text)
        sim = _format_sim(gold_value, attr.candidate_text)
    elif attr and attr.attribution_note:
        failed_model = f'_{attr.attribution_note}_'
        label_info = '—'
        data_text = '—'
        sim = '—'
    else:
        failed_model = '—'
        label_info = '—'
        data_text = '—'
        sim = '—'
    return (
        f'| {record_id} | {corpus}'
        f' | {failed_model} | {label_info}'
        f' | {gold_value} | {data_text} | {sim} |'
    )


_EF_TABLE_HEADER = [
    '| Doc | Corpus | First failed model | Predicted ≠ Expected | Gold value | Data text | Sim |',
    '| --- | --- | --- | --- | --- | --- | ---: |',
]


def _see_all_link(mode: FailureMode, total: int) -> str:
    filename = _MODE_DETAIL_FILE[mode]
    return f'_(showing {SAMPLE_SIZE} of {total} — [see all]({filename}))_'


_NR_TABLE_HEADER_NARROW = [
    '| Doc | Corpus | Gold value | Raw sim |',
    '| --- | --- | --- | ---: |',
]
_NR_TABLE_HEADER_WIDE = [
    '| Doc | Corpus | Gold value | Raw sim | Extracted | Extr sim |',
    '| --- | --- | --- | ---: | --- | ---: |',
]

# Similarity threshold for treating a NOT_IN_RAW_TEXT result as a gold/PDF form
# mismatch rather than a genuine absence.  Above this the extracted field value is
# clearly the same content rendered differently; below it the two strings are
# unrelated and the value is treated as truly absent.
_NR_DIFF_FORM_THRESHOLD = 0.6


def _render_not_in_raw_rows(
    examples: List[Tuple[str, str, GoldValueResult]],
    wide: bool,
) -> List[str]:
    lines = []
    for corpus, record_id, result in examples:
        raw_sim = (
            f'{result.best_raw_similarity:.2f}'
            if result.best_raw_similarity is not None else '—'
        )
        if wide:
            extracted = result.best_sb_match or '—'
            extr_sim = (
                f'{result.best_sb_similarity:.2f}'
                if result.best_sb_similarity is not None else '—'
            )
            lines.append(
                f'| {record_id} | {corpus} | {result.value}'
                f' | {raw_sim} | {extracted} | {extr_sim} |'
            )
        else:
            lines.append(f'| {record_id} | {corpus} | {result.value} | {raw_sim} |')
    return lines


def _split_not_in_raw(
    examples: List[Tuple[str, str, GoldValueResult]],
) -> Tuple[List[Tuple[str, str, GoldValueResult]], List[Tuple[str, str, GoldValueResult]]]:
    """Partition NOT_IN_RAW_TEXT examples into (absent, diff_form).

    diff_form: best_sb_similarity >= _NR_DIFF_FORM_THRESHOLD — the text was
               extracted as the correct field type but in a different form.
    absent:    everything else — no similar extraction found, or nothing extracted.
    """
    absent, diff_form = [], []
    for item in examples:
        _, _, result = item
        sim = result.best_sb_similarity or 0.0
        if sim >= _NR_DIFF_FORM_THRESHOLD:
            diff_form.append(item)
        else:
            absent.append(item)
    return absent, diff_form


def _render_not_in_raw_section(
    agg: FailureModeAggregate,
    summary_total: int,
) -> List[str]:
    label = FAILURE_MODE_LABEL[FailureMode.NOT_IN_RAW_TEXT]
    lines = [
        f'## {label} ({agg.total_values} value(s),'
        f' {agg.docs_affected} doc(s),'
        f' {_pct(agg.total_values, summary_total)})',
        '',
    ]
    examples = agg.examples
    if not examples:
        lines.append('_None._')
        lines.append('')
        return lines

    absent, diff_form = _split_not_in_raw(examples)

    if absent:
        lines += [
            f'### Absent from raw text ({len(absent)} value(s))',
            '',
            '_Gold values not found in the raw ScienceBeam TEI output and no similar '
            'value was extracted into the field. '
            'This may indicate the value is missing from the PDF, or that raw text '
            'extraction renders it too differently to detect. '
            'Model training is unlikely to recover these._',
            '',
        ]
        sample = absent[:SAMPLE_SIZE]
        lines += _NR_TABLE_HEADER_NARROW + _render_not_in_raw_rows(sample, wide=False)
        if len(absent) > SAMPLE_SIZE:
            lines += ['', _see_all_link(FailureMode.NOT_IN_RAW_TEXT, len(examples))]
        lines.append('')

    if diff_form:
        lines += [
            f'### Extracted with different form ({len(diff_form)} value(s))',
            '',
            '_The gold value was not found in the raw text, but a similar value was '
            'extracted into the field. '
            'This is a gold-label / PDF-rendering mismatch '
            '(e.g. different wording, extra words, ALL CAPS) rather than a model error. '
            'Consider normalising the gold labels or the extraction post-processing._',
            '',
        ]
        sample = diff_form[:SAMPLE_SIZE]
        lines += _NR_TABLE_HEADER_WIDE + _render_not_in_raw_rows(sample, wide=True)
        if len(diff_form) > SAMPLE_SIZE:
            lines += ['', _see_all_link(FailureMode.NOT_IN_RAW_TEXT, len(examples))]
        lines.append('')

    return lines


def _render_extraction_failed_section(
    agg: FailureModeAggregate,
    summary_total: int,
    _attribution_aggregates: List,  # kept for API compatibility, no longer used for display
    is_online: bool,
    attr_lookup: Dict[Tuple[str, str, str], PipelineAttribution],
) -> List[str]:
    label = FAILURE_MODE_LABEL[FailureMode.EXTRACTION_FAILED]
    lines = [
        f'## {label} ({agg.total_values} value(s),'
        f' {agg.docs_affected} doc(s),'
        f' {_pct(agg.total_values, summary_total)})',
        '',
        '_Gold values found in the raw TEI output but not predicted into the target field '
        '— the model saw this text but did not assign it the correct label. '
        'This is a model training issue, not a pipeline error._',
        '',
    ]

    examples = agg.examples
    if not examples:
        lines.append('_None._')
    elif is_online:
        lines += _render_attribution_summary(examples, attr_lookup)
        lines += _render_label_breakdown(examples, attr_lookup)
        lines += [
            f'_Context window (surrounding model-data lines): '
            f'[{_CONTEXT_WINDOW_DETAIL_FILE}]({_CONTEXT_WINDOW_DETAIL_FILE})_',
            '',
        ]
        sample = examples[:SAMPLE_SIZE]
        lines += ['### Extraction failures sample', ''] + _EF_TABLE_HEADER
        for corpus, record_id, result in sample:
            attr = attr_lookup.get((corpus, record_id, result.value))
            lines.append(_render_ef_row(record_id, corpus, result.value, attr))
        if len(examples) > SAMPLE_SIZE:
            lines.append('')
            lines.append(_see_all_link(FailureMode.EXTRACTION_FAILED, len(examples)))
    else:
        lines += [
            '_Run with `--parser-url` to identify the responsible model layer._',
            '',
        ]
        sample = examples[:SAMPLE_SIZE]
        lines += ['| Doc | Corpus | Gold value |', '| --- | --- | --- |']
        for corpus, record_id, result in sample:
            lines.append(f'| {record_id} | {corpus} | {result.value} |')
        if len(examples) > SAMPLE_SIZE:
            lines.append('')
            lines.append(_see_all_link(FailureMode.EXTRACTION_FAILED, len(examples)))

    lines.append('')
    return lines


def _render_partial_wrong_section(
    agg: FailureModeAggregate,
    summary_total: int,
) -> List[str]:
    label = FAILURE_MODE_LABEL[FailureMode.PARTIAL_WRONG]
    lines = [
        f'## {label} ({agg.total_values} value(s),'
        f' {agg.docs_affected} doc(s),'
        f' {_pct(agg.total_values, summary_total)})',
        '',
        '_Gold values present in ScienceBeam field output '
        'but with similarity below threshold._',
        '',
    ]
    examples = agg.examples
    if not examples:
        lines.append('_None._')
    else:
        sample = examples[:SAMPLE_SIZE]
        lines += [
            '| Doc | Corpus | Gold value | Best match | Similarity |',
            '| --- | --- | --- | --- | --- |',
        ]
        for corpus, record_id, result in sample:
            sim = (
                f'{result.best_sb_similarity:.2f}'
                if result.best_sb_similarity is not None
                else '—'
            )
            lines.append(
                f'| {record_id} | {corpus} | {result.value}'
                f' | {result.best_sb_match or "—"} | {sim} |'
            )
        if len(examples) > SAMPLE_SIZE:
            lines.append('')
            lines.append(_see_all_link(FailureMode.PARTIAL_WRONG, len(examples)))
    lines.append('')
    return lines


def _render_near_miss_rows(
    near_misses: List[Tuple[str, str, GoldValueResult]],
) -> List[str]:
    lines = []
    for corpus, record_id, result in near_misses:
        sim = f'{result.best_sb_similarity:.2f}' if result.best_sb_similarity is not None else '—'
        kind = _classify_near_miss(result.value, result.best_sb_match)
        lines.append(
            f'| {record_id} | {corpus} | {result.value}'
            f' | {result.best_sb_match or "—"} | {sim} | {kind} |'
        )
    return lines


_NEAR_MISS_TABLE_HEADER = [
    '| Doc | Corpus | Gold value | Extracted | Sim | Type |',
    '| --- | --- | --- | --- | ---: | --- |',
]


def _render_near_miss_cluster_summary(
    near_misses: List[Tuple[str, str, GoldValueResult]],
) -> List[str]:
    cluster_vals: Counter = Counter()
    cluster_docs: DefaultDict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for corpus, record_id, result in near_misses:
        kind = _classify_near_miss(result.value, result.best_sb_match)
        cluster_vals[kind] += 1
        cluster_docs[kind].add((corpus, record_id))
    lines = [
        '### By type',
        '',
        '| Type | Docs | Values |',
        '| --- | ---: | ---: |',
    ]
    for kind, count in sorted(cluster_vals.items(), key=lambda x: -x[1]):
        lines.append(f'| {kind} | {len(cluster_docs[kind])} | {count} |')
    lines.append('')
    return lines


def _render_near_miss_section(
    near_misses: List[Tuple[str, str, GoldValueResult]],
) -> List[str]:
    """Section for correctly-extracted values whose text differs slightly from gold."""
    if not near_misses:
        return []
    lines = [
        f'## Near-miss extractions ({len(near_misses)} value(s))',
        '',
        '_Gold values extracted correctly but with minor text differences '
        '(capitalisation, dashes, encoding variants). '
        'These are not failures, but indicate where gold labels and PDF-parsed text diverge._',
        '',
    ]
    lines += _render_near_miss_cluster_summary(near_misses)
    lines += ['### Near-miss sample', ''] + _NEAR_MISS_TABLE_HEADER
    sample = near_misses[:SAMPLE_SIZE]
    lines += _render_near_miss_rows(sample)
    if len(near_misses) > SAMPLE_SIZE:
        lines.append('')
        lines.append(
            f'_(showing {SAMPLE_SIZE} of {len(near_misses)}'
            f' — [see all]({_NEAR_MISS_DETAIL_FILE}))_'
        )
    lines.append('')
    return lines


def render_report(
    summary: FieldFailureSummary,
    doc_summaries: List[DocumentSummary],
) -> str:
    skipped_note = (
        f' ({summary.n_docs_skipped} skipped — no gold annotations for this field)'
        if summary.n_docs_skipped
        else ''
    )
    lines: List[str] = [
        f'# Gold-Label Failure Analysis: {summary.field}',
        '',
        f'- **Run**: `{summary.run_sb}`',
        f'- **Score method**: `{summary.method}`',
        f'- **Documents with gold annotations**: {summary.total_docs}{skipped_note}',
        f'- **Total gold values**: {summary.total_gold}',
        '',
        '## Summary',
        '',
    ]

    near_misses = _find_near_misses(doc_summaries)
    near_miss_docs = len({(c, r) for c, r, _ in near_misses})
    lines += _render_summary_table(summary, len(near_misses), near_miss_docs)
    lines += [
        '',
        f'**Recommended next step**: {summary.recommended_action}',
        '',
        f'## Analyzed documents ({len(doc_summaries)})',
        '',
    ]
    sample_docs = doc_summaries[:SAMPLE_SIZE]
    lines += _render_doc_table(sample_docs)
    if len(doc_summaries) > SAMPLE_SIZE:
        lines.append('')
        lines.append(
            f'_(showing {SAMPLE_SIZE} of {len(doc_summaries)}'
            f' — [see all]({_ANALYZED_DOCS_DETAIL_FILE}))_'
        )
    lines.append('')

    attr_lookup = _build_attr_lookup(doc_summaries)

    nr_agg = summary.mode_aggregates[int(FailureMode.NOT_IN_RAW_TEXT)]
    ef_agg = summary.mode_aggregates[int(FailureMode.EXTRACTION_FAILED)]
    pw_agg = summary.mode_aggregates[int(FailureMode.PARTIAL_WRONG)]

    if nr_agg.total_values:
        lines += _render_not_in_raw_section(nr_agg, summary.total_gold)

    if ef_agg.total_values:
        lines += _render_extraction_failed_section(
            ef_agg, summary.total_gold,
            summary.attribution_aggregates, summary.is_online, attr_lookup,
        )

    if pw_agg.total_values:
        lines += _render_partial_wrong_section(pw_agg, summary.total_gold)

    lines += _render_near_miss_section(near_misses)

    lines.append(_PRESENCE_FOOTNOTE)
    lines.append('')

    return '\n'.join(lines)


def _render_not_in_raw_full(
    agg: FailureModeAggregate,
    summary_total: int,
) -> str:
    label = FAILURE_MODE_LABEL[FailureMode.NOT_IN_RAW_TEXT]
    absent, diff_form = _split_not_in_raw(agg.examples)
    lines = [
        f'# {label} — full list',
        f'_{agg.total_values} value(s), {_pct(agg.total_values, summary_total)} of gold_',
        '',
    ]
    if absent:
        lines += [
            f'## Absent from raw text ({len(absent)} value(s))',
            '',
        ] + _NR_TABLE_HEADER_NARROW + _render_not_in_raw_rows(absent, wide=False)
        lines.append('')
    if diff_form:
        lines += [
            f'## Extracted with different form ({len(diff_form)} value(s))',
            '',
        ] + _NR_TABLE_HEADER_WIDE + _render_not_in_raw_rows(diff_form, wide=True)
        lines.append('')
    return '\n'.join(lines)


def _render_extraction_failed_full(
    agg: FailureModeAggregate,
    summary_total: int,
    is_online: bool,
    attr_lookup: Dict[Tuple[str, str, str], PipelineAttribution],
) -> str:
    label = FAILURE_MODE_LABEL[FailureMode.EXTRACTION_FAILED]
    lines = [
        f'# {label} — full list',
        f'_{agg.total_values} value(s), {_pct(agg.total_values, summary_total)} of gold_',
        '',
    ]
    if is_online:
        lines += _render_attribution_summary(agg.examples, attr_lookup)
        lines += _render_label_breakdown(agg.examples, attr_lookup)
        lines += _EF_TABLE_HEADER
        for corpus, record_id, result in agg.examples:
            attr = attr_lookup.get((corpus, record_id, result.value))
            lines.append(_render_ef_row(record_id, corpus, result.value, attr))
    else:
        lines += ['| Doc | Corpus | Gold value |', '| --- | --- | --- |']
        for corpus, record_id, result in agg.examples:
            lines.append(f'| {record_id} | {corpus} | {result.value} |')
    lines.append('')
    return '\n'.join(lines)


def _render_partial_wrong_full(
    agg: FailureModeAggregate,
    summary_total: int,
) -> str:
    label = FAILURE_MODE_LABEL[FailureMode.PARTIAL_WRONG]
    lines = [
        f'# {label} — full list',
        f'_{agg.total_values} value(s), {_pct(agg.total_values, summary_total)} of gold_',
        '',
        '| Doc | Corpus | Gold value | Best match | Similarity |',
        '| --- | --- | --- | --- | --- |',
    ]
    for corpus, record_id, result in agg.examples:
        sim = (
            f'{result.best_sb_similarity:.2f}'
            if result.best_sb_similarity is not None
            else '—'
        )
        lines.append(
            f'| {record_id} | {corpus} | {result.value}'
            f' | {result.best_sb_match or "—"} | {sim} |'
        )
    lines.append('')
    return '\n'.join(lines)


def _render_analyzed_docs_full(doc_summaries: List[DocumentSummary]) -> str:
    lines = [
        '# Analyzed documents — full list',
        f'_{len(doc_summaries)} document(s)_',
        '',
    ] + list(_DOC_TABLE_HEADER)
    for doc in doc_summaries:
        lines.append(_render_doc_row(doc))
    lines.append('')
    return '\n'.join(lines)


def _render_near_miss_full(
    near_misses: List[Tuple[str, str, GoldValueResult]],
) -> str:
    lines = [
        '# Near-miss extractions — full list',
        f'_{len(near_misses)} value(s)_',
        '',
    ]
    lines += _render_near_miss_cluster_summary(near_misses)
    lines += _NEAR_MISS_TABLE_HEADER + _render_near_miss_rows(near_misses) + ['']
    return '\n'.join(lines)


def _render_context_window_entry(  # pylint: disable=too-many-locals,too-many-branches
    gold_value: str,
    attr: PipelineAttribution,
    corpus: str,
    record_id: str,
    available_screenshots: Set[str],
    tei_snippets: Optional[Dict[str, str]] = None,
) -> List[str]:
    from ._screenshots import screenshot_relpath  # pylint: disable=import-outside-toplevel
    label_info = _format_label_info(attr.predicted_label, attr.expected_label)
    lines: List[str] = [
        # Include record_id to guarantee unique headings across all docs (MD024).
        f'### `{gold_value}` — {record_id}',
        '',
        f'_Model: `{attr.first_failed_model}` — {label_info}_',
        '',
    ]

    img_path = screenshot_relpath(corpus, record_id, gold_value)
    if img_path in available_screenshots:
        lines += [f'![{gold_value}]({img_path})', '']

    # Detail block: expected vs actual for this case.
    detail_parts = [f'**Gold:** `{gold_value}`']
    if attr.candidate_text:
        detail_parts.append(f'**Data text:** `{attr.candidate_text}`')
    if attr.predicted_label:
        detail_parts.append(f'**Predicted:** `{attr.predicted_label}`')
    if attr.expected_label:
        detail_parts.append(f'**Expected:** `{attr.expected_label}`')
    lines += [' · '.join(detail_parts), '']

    # Collapsible TEI XML fragment from the prediction output.
    if tei_snippets and img_path in tei_snippets:
        snippet = tei_snippets[img_path]
        lines += [
            '<details>',
            '<summary>TEI XML fragment</summary>',
            '',
            '```xml',
            snippet,
            '```',
            '',
            '</details>',
            '',
        ]

    if attr.context_window:
        if attr.context_is_block_level:
            header = ['| Label | Block text |', '| --- | --- |']
        else:
            header = ['| Token | Label |', '| --- | --- |']
        rows = []
        for text, label, is_match in attr.context_window:
            stripped = _strip_bio(label)
            if is_match:
                if attr.context_is_block_level:
                    rows.append(f'| **`{stripped}` ← predicted** | **{text}** |')
                else:
                    rows.append(f'| **{text}** | **`{stripped}` ← predicted** |')
            else:
                if attr.context_is_block_level:
                    rows.append(f'| `{stripped}` | {text} |')
                else:
                    rows.append(f'| {text} | `{stripped}` |')
        lines += [
            '<details>',
            '<summary>Context window</summary>',
            '',
        ] + header + rows + [
            '',
            '</details>',
            '',
        ]

    return lines


def _render_context_window_file(  # pylint: disable=too-many-locals
    summary: FieldFailureSummary,
    _doc_summaries: List[DocumentSummary],
    attr_lookup: Dict[Tuple[str, str, str], PipelineAttribution],
    available_screenshots: Optional[Set[str]] = None,
    tei_snippets: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Return context window detail file content, or None if no attributed contexts exist."""
    ef_agg = summary.mode_aggregates[int(FailureMode.EXTRACTION_FAILED)]
    if not ef_agg.examples:
        return None

    screenshots: Set[str] = available_screenshots or set()
    lines: List[str] = [
        '# Context Window — Extraction Failures',
        '',
        '_Surrounding model-data lines at the point of mislabelling. '
        'Highlighted rows show the matched span with the wrong label. '
        'Screenshots show the PDF region where the text appears._',
        '',
        '## How to read the context window table',
        '',
        'Each entry shows a small window of model-data lines around the point where '
        'the gold value was found with the wrong label.',
        '',
        '**Word-level models** (header, citation, …) show one row per token:',
        '',
        '| Token | Label |',
        '| --- | --- |',
        '| previous | `<other>` |',
        '| **Introduction** | **`<other>` ← predicted** |',
        '| next | `<other>` |',
        '',
        'The **bold** row is the matched token span. '
        '`← predicted` is the label assigned by the model; '
        'the expected label is shown in the entry heading.',
        '',
        '**Block-level models** (segmentation) show one row per text block. '
        'The full block text is shown so you can see what surrounds the target heading:',
        '',
        '| Label | Block text |',
        '| --- | --- |',
        '| `body` | This study describes … |',
        '| **`body` ← predicted** | **Animals and experimental conditions …** |',
        '| `body` | The following antibodies … |',
        '',
        '**Screenshot note:** The highlighted PDF region is found by searching the TEI '
        'XML output for the element whose text best matches the gold value. '
        'For short values (e.g. "Animals") the match may land on a different element '
        'than expected if the same word appears elsewhere on the page — '
        'check the TEI XML fragment below the screenshot to verify.',
        '',
        '---',
        '',
    ]
    has_any = False

    # Group by doc so navigating the file matches navigating the report.
    docs_seen: Dict[Tuple[str, str], List[Tuple[str, PipelineAttribution]]] = {}
    for corpus, record_id, result in ef_agg.examples:
        attr = attr_lookup.get((corpus, record_id, result.value))
        if attr and (attr.context_window or attr.first_failed_model):
            key = (corpus, record_id)
            docs_seen.setdefault(key, []).append((result.value, attr))

    for (corpus, record_id), value_attrs in docs_seen.items():
        has_any = True
        lines += [f'## {record_id} ({corpus})', '']
        for gold_value, attr in value_attrs:
            lines += _render_context_window_entry(
                gold_value, attr, corpus, record_id, screenshots, tei_snippets
            )

    return '\n'.join(lines) if has_any else None


def render_mode_detail_reports(
    summary: FieldFailureSummary,
    doc_summaries: List[DocumentSummary],
    available_screenshots: Optional[Set[str]] = None,
    tei_snippets: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return {filename: content} for per-mode full-list detail reports."""
    attr_lookup = _build_attr_lookup(doc_summaries)
    reports: Dict[str, str] = {}

    nr_agg = summary.mode_aggregates[int(FailureMode.NOT_IN_RAW_TEXT)]
    ef_agg = summary.mode_aggregates[int(FailureMode.EXTRACTION_FAILED)]
    pw_agg = summary.mode_aggregates[int(FailureMode.PARTIAL_WRONG)]

    if nr_agg.total_values > SAMPLE_SIZE:
        reports[_MODE_DETAIL_FILE[FailureMode.NOT_IN_RAW_TEXT]] = (
            _render_not_in_raw_full(nr_agg, summary.total_gold)
        )
    if ef_agg.total_values > SAMPLE_SIZE:
        reports[_MODE_DETAIL_FILE[FailureMode.EXTRACTION_FAILED]] = (
            _render_extraction_failed_full(
                ef_agg, summary.total_gold, summary.is_online, attr_lookup
            )
        )
    if pw_agg.total_values > SAMPLE_SIZE:
        reports[_MODE_DETAIL_FILE[FailureMode.PARTIAL_WRONG]] = (
            _render_partial_wrong_full(pw_agg, summary.total_gold)
        )

    near_misses = _find_near_misses(doc_summaries)
    if len(near_misses) > SAMPLE_SIZE:
        reports[_NEAR_MISS_DETAIL_FILE] = _render_near_miss_full(near_misses)

    if len(doc_summaries) > SAMPLE_SIZE:
        reports[_ANALYZED_DOCS_DETAIL_FILE] = _render_analyzed_docs_full(doc_summaries)

    ctx_content = _render_context_window_file(
        summary, doc_summaries, attr_lookup, available_screenshots, tei_snippets
    )
    if ctx_content is not None:
        reports[_CONTEXT_WINDOW_DETAIL_FILE] = ctx_content

    return reports
