from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence
from unittest.mock import patch

from benchmarks.score import _doc_scores_to_dict, run_score
from benchmarks.shared_documents import on_shared_documents, shared_documents


def _fields(tp: int) -> dict:
    return _doc_scores_to_dict([{
        "field_name": "title", "scoring_type": "string", "scoring_method": "exact",
        "match_score": {"true_positive": tp, "false_positive": 0, "false_negative": 1 - tp},
    }])


def _scored_run(run_dir: Path, scored: Dict[str, Dict[str, int]]) -> dict:
    """A run with one score file per document, and the summary that lists them."""
    for corpus, tps in scored.items():
        scores_dir = run_dir / "scores" / corpus
        scores_dir.mkdir(parents=True, exist_ok=True)
        for record_id, tp in tps.items():
            (scores_dir / f"{record_id}.json").write_text(json.dumps({
                "record_id": record_id, "corpus": corpus, "fields": _fields(tp),
            }))
    return {
        "fields": ["title"],
        "field_measures": {"title": ["exact"]},
        "corpora": {corpus: {"n": len(tps)} for corpus, tps in scored.items()},
        "scored_documents": {corpus: sorted(tps) for corpus, tps in scored.items()},
    }


def _narrow(tmp_path: Path, runs: Sequence[Dict[str, Dict[str, int]]]):
    paths: List[Path] = []
    labeled = []
    for index, scored in enumerate(runs):
        run_dir = tmp_path / f"run{index}"
        labeled.append((f"run{index}", _scored_run(run_dir, scored)))
        paths.append(run_dir / "summary.json")
    return on_shared_documents(labeled, paths)


def test_intersects_per_corpus():
    summaries = [
        {"scored_documents": {"a": ["d1", "d2"], "b": ["e1"]}},
        {"scored_documents": {"a": ["d2", "d3"], "b": []}},
    ]
    assert shared_documents(summaries) == {"a": {"d2"}, "b": {"e1"}}


def test_every_column_is_scored_on_the_shared_documents(tmp_path: Path):
    narrowed, note = _narrow(tmp_path, [
        {"c": {"d1": 1, "d2": 1, "d3": 0}},
        {"c": {"d1": 1, "d2": 1}},
    ])
    assert [summary["corpora"]["c"]["n"] for _, summary in narrowed] == [2, 2]
    assert note is not None
    assert "`c`: compared on 2 of 3" in note
    assert "run0 has 1 the others lack" in note


def test_a_column_without_a_corpus_leaves_it_to_the_others(tmp_path: Path):
    narrowed, note = _narrow(tmp_path, [
        {"c": {"d1": 1}, "plos": {"p1": 1, "p2": 1}},
        {"c": {"d1": 1}},
    ])
    assert narrowed[0][1]["corpora"]["plos"]["n"] == 2
    assert note is not None and "run1 (none)" in note


def test_no_note_when_every_column_has_the_same_documents(tmp_path: Path):
    _, note = _narrow(tmp_path, [{"c": {"d1": 1}}, {"c": {"d1": 0}}])
    assert note is None


def test_an_old_summary_is_compared_as_it_is_and_said_so(tmp_path: Path):
    old = {"corpora": {"c": {"n": 5}}}
    narrowed, note = on_shared_documents([("old", old)], [tmp_path / "summary.json"])
    assert narrowed == [("old", old)]
    assert note is not None and "may cover different documents" in note


def test_scoring_is_limited_to_the_mode_documents(tmp_path: Path):
    run_dir = tmp_path / "run"
    for record_id in ("d1", "d2"):
        pred = run_dir / "predictions" / "c" / f"{record_id}.tei.xml"
        pred.parent.mkdir(parents=True, exist_ok=True)
        pred.write_text("<TEI/>")
        gold = tmp_path / "data" / "train" / "c" / f"{record_id}.jats.xml"
        gold.parent.mkdir(parents=True, exist_ok=True)
        gold.write_text("<article/>")
    config = {
        "dataset": {"splits": {"train": {"c": {"file": "c.parquet"}}}},
        "fields": ["title"],
        "scoring": {"default_methods": ["exact"]},
    }
    score = {
        "field_name": "title", "scoring_type": "string", "scoring_method": "exact",
        "match_score": {"true_positive": 1, "false_positive": 0, "false_negative": 0},
    }
    with patch("benchmarks.score.prepare_judge"), \
            patch("benchmarks.score._score_pair", return_value=[score]):
        run_score(config, run_dir, tmp_path / "data", out_path=None,
                  split_override="train", record_ids={("c", "d1")})
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["scored_documents"] == {"c": ["d1"]}
    assert summary["corpora"]["c"]["n"] == 1
