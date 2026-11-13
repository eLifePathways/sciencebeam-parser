"""Score every column of a comparison on the same documents.

A delta across different document sets mixes which documents were measured with
how a tool did on them, so each corpus is narrowed to the documents every column
scored. A column with no documents for a corpus is left out of that corpus only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from benchmarks.score import summarise_scored_documents

SCORED_DOCUMENTS_KEY = "scored_documents"


def _corpora(summaries: Sequence[Dict[str, Any]]) -> List[str]:
    return list(dict.fromkeys(
        corpus for summary in summaries for corpus in summary[SCORED_DOCUMENTS_KEY]
    ))


def shared_documents(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Per corpus, the documents every column that scored the corpus has."""
    shared: Dict[str, Set[str]] = {}
    for corpus in _corpora(summaries):
        having = [
            set(summary[SCORED_DOCUMENTS_KEY][corpus]) for summary in summaries
            if summary[SCORED_DOCUMENTS_KEY].get(corpus)
        ]
        shared[corpus] = set.intersection(*having) if having else set()
    return shared


def on_shared_documents(
    labeled_summaries: Sequence[Tuple[str, Dict[str, Any]]],
    summary_paths: Sequence[Path],
) -> Tuple[List[Tuple[str, Dict[str, Any]]], Optional[str]]:
    """The summaries narrowed to the shared documents, and a note on what was left out.

    Summaries written before scored documents were recorded can't be narrowed, so
    they are returned as they are, with a note saying so.
    """
    summaries = [summary for _, summary in labeled_summaries]
    if not summaries:
        return list(labeled_summaries), None
    if any(SCORED_DOCUMENTS_KEY not in summary for summary in summaries):
        return list(labeled_summaries), (
            "> **Documents:** some summaries don't record which documents they "
            "scored, so the columns may cover different documents."
        )
    shared = shared_documents(summaries)
    narrowed = []
    for (label, summary), path in zip(labeled_summaries, summary_paths):
        corpora = dict(summary.get("corpora") or {})
        for corpus, scored in summary[SCORED_DOCUMENTS_KEY].items():
            if scored and set(scored) != shared[corpus]:
                corpora[corpus] = summarise_scored_documents(
                    summary, path.parent, corpus, shared[corpus]
                )
        narrowed.append((label, {**summary, "corpora": corpora}))
    return narrowed, _left_out_note(labeled_summaries, shared)


def _left_out_note(
    labeled_summaries: Sequence[Tuple[str, Dict[str, Any]]],
    shared: Dict[str, Set[str]],
) -> Optional[str]:
    lines = []
    for corpus, kept in shared.items():
        missing = []
        for label, summary in labeled_summaries:
            scored = set(summary[SCORED_DOCUMENTS_KEY].get(corpus) or [])
            if not scored:
                missing.append(f"{label} (none)")
                continue
            left_out = len(scored - kept)
            if left_out:
                missing.append(f"{label} has {left_out} the others lack")
        everywhere = set().union(*(
            set(summary[SCORED_DOCUMENTS_KEY].get(corpus) or [])
            for _, summary in labeled_summaries
        ))
        if len(everywhere) > len(kept) or any("(none)" in m for m in missing):
            lines.append(
                f"> - `{corpus}`: compared on {len(kept)} of {len(everywhere)}; "
                + "; ".join(missing)
            )
    if not lines:
        return None
    return "\n".join([
        "> **Documents:** each corpus is compared on the documents every column"
        " scored.",
        *lines,
    ])
