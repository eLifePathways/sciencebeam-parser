from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Set

from ._types import (
    AttributionAggregate,
    DocumentSummary,
    FailureMode,
    FailureModeAggregate,
    FieldFailureSummary,
)


def _derive_recommendation(
    mode_aggregates: List[FailureModeAggregate],
    attribution_aggregates: List[AttributionAggregate],
    is_online: bool,
) -> str:
    failures = [a for a in mode_aggregates if a.mode != FailureMode.CORRECT]
    if not failures or all(a.total_values == 0 for a in failures):
        return 'No failures detected.'

    dominant = max(failures, key=lambda a: a.total_values)

    if dominant.mode == FailureMode.NOT_IN_RAW_TEXT:
        return (
            f'Investigate PDF layout extraction — {dominant.total_values} gold value(s) '
            f'absent from raw text across {dominant.docs_affected} document(s).'
        )
    if dominant.mode == FailureMode.EXTRACTION_FAILED:
        if is_online and attribution_aggregates:
            top = attribution_aggregates[0]
            return (
                f'Add training examples to `{top.model}` model — '
                f'{dominant.total_values} value(s) present in raw text but not extracted '
                f'across {dominant.docs_affected} document(s).'
            )
        return (
            f'{dominant.total_values} value(s) present in raw text but not extracted '
            f'across {dominant.docs_affected} document(s). '
            f'Run with `--parser-url` to identify the responsible model layer.'
        )
    if dominant.mode == FailureMode.PARTIAL_WRONG:
        return (
            f'Investigate boundary/partial extraction — {dominant.total_values} value(s) '
            f'found but below similarity threshold across {dominant.docs_affected} document(s).'
        )
    return 'No clear recommendation.'


def _build_mode_aggregates(
    doc_summaries: List[DocumentSummary],
) -> List[FailureModeAggregate]:
    result = []
    for mode in FailureMode:
        examples = []
        docs_affected = 0
        for doc in doc_summaries:
            doc_examples = [
                (doc.corpus, doc.record_id, r)
                for r in doc.results
                if r.mode == mode
            ]
            if doc_examples:
                docs_affected += 1
                examples.extend(doc_examples)
        result.append(FailureModeAggregate(
            mode=mode,
            total_values=len(examples),
            docs_affected=docs_affected,
            examples=examples,
        ))
    return result


def _build_attribution_aggregates(
    doc_summaries: List[DocumentSummary],
    model_chain: List[str],
) -> List[AttributionAggregate]:
    value_counter: Counter = Counter()
    doc_sets: Dict[str, Set] = defaultdict(set)
    for doc in doc_summaries:
        for attr in doc.attributions.values():
            model = attr.first_failed_model
            if model is None:
                continue
            value_counter[model] += 1
            doc_sets[model].add((doc.corpus, doc.record_id))
    return [
        AttributionAggregate(
            model=model_name,
            failure_count=count,
            doc_count=len(doc_sets[model_name]),
        )
        for model_name, count in value_counter.most_common()
        if model_name in model_chain
    ]


def aggregate_results(
    doc_summaries: List[DocumentSummary],
    model_chain: List[str],
    field: str,
    run_sb: str,
    is_online: bool,
    n_docs_skipped: int = 0,
    method: str = 'edit_sim',
) -> FieldFailureSummary:
    total_gold = sum(d.total_gold for d in doc_summaries)
    mode_aggregates = _build_mode_aggregates(doc_summaries)
    attribution_aggregates = _build_attribution_aggregates(doc_summaries, model_chain)
    recommendation = _derive_recommendation(mode_aggregates, attribution_aggregates, is_online)
    return FieldFailureSummary(
        field=field,
        run_sb=run_sb,
        total_docs=len(doc_summaries),
        total_gold=total_gold,
        mode_aggregates=mode_aggregates,
        attribution_aggregates=attribution_aggregates,
        recommended_action=recommendation,
        is_online=is_online,
        n_docs_skipped=n_docs_skipped,
        method=method,
    )
