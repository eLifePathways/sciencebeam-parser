from __future__ import annotations

from typing import Optional

from benchmarks.llm_usage import (
    aggregate_llm_usage,
    combine_usage,
    read_manifest_entries,
    usage_for_corpora,
)


def _usage(
    calls: int = 1,
    input_tokens: int = 100,
    output_tokens: int = 50,
    peak_output_tokens: Optional[int] = None,
    cost: Optional[float] = 0.0004,
    task: str = "citation",
    model: str = "qwen/qwen3.5-9b",
) -> dict:
    entry = {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": 0,
        "peak_output_tokens": (
            output_tokens if peak_output_tokens is None else peak_output_tokens
        ),
        "models": [model],
        "providers": ["SiliconFlow"],
    }
    if cost is not None:
        entry["cost_credits"] = cost
    entry["by_task"] = {task: dict(entry)}
    return entry


def _manifest_entry(
    corpus: str = "biorxiv",
    record_id: str = "doc1",
    status: str = "ok",
    usage: Optional[dict] = None,
) -> dict:
    entry: dict = {"corpus": corpus, "record_id": record_id, "status": status}
    if usage is not None:
        entry["llm_usage"] = usage
    return entry


class TestCombineUsage:
    def test_should_sum_tokens_and_calls(self):
        combined = combine_usage([_usage(), _usage()])
        assert combined["calls"] == 2
        assert combined["input_tokens"] == 200
        assert combined["output_tokens"] == 100

    def test_should_keep_the_peak_as_a_peak(self):
        combined = combine_usage([
            _usage(output_tokens=50), _usage(output_tokens=16000)
        ])
        assert combined["output_tokens"] == 16050
        assert combined["peak_output_tokens"] == 16000

    def test_should_sum_cost_where_stated(self):
        combined = combine_usage([_usage(cost=0.001), _usage(cost=0.002)])
        assert combined["cost_credits"] == 0.003

    def test_should_leave_cost_absent_where_never_stated(self):
        combined = combine_usage([_usage(cost=None), _usage(cost=None)])
        assert "cost_credits" not in combined
        assert combined["output_tokens"] == 100

    def test_should_combine_by_task(self):
        combined = combine_usage([
            _usage(task="citation", output_tokens=50),
            _usage(task="reference_segmenter", output_tokens=900),
        ])
        assert combined["by_task"]["citation"]["output_tokens"] == 50
        assert combined["by_task"]["reference_segmenter"]["output_tokens"] == 900

    def test_should_list_each_model_once(self):
        combined = combine_usage([_usage(model="a"), _usage(model="a"), _usage(model="b")])
        assert combined["models"] == ["a", "b"]

    def test_should_be_empty_for_no_entries(self):
        combined = combine_usage([])
        assert combined["calls"] == 0
        assert "cost_credits" not in combined


class TestAggregateLlmUsage:
    def test_should_be_empty_when_no_entry_carries_usage(self):
        entries = [_manifest_entry(), _manifest_entry(record_id="doc2")]
        assert not aggregate_llm_usage(entries, ["biorxiv"])

    def test_should_aggregate_per_corpus(self):
        entries = [
            _manifest_entry(corpus="biorxiv", record_id="doc1", usage=_usage()),
            _manifest_entry(corpus="plos", record_id="doc2", usage=_usage(calls=2)),
        ]
        usage = aggregate_llm_usage(entries, ["biorxiv", "plos"])
        assert usage["biorxiv"]["calls"] == 1
        assert usage["plos"]["calls"] == 2

    def test_should_include_a_document_that_errored(self):
        entries = [
            _manifest_entry(record_id="doc1", usage=_usage()),
            _manifest_entry(record_id="doc2", status="error", usage=_usage()),
        ]
        usage = aggregate_llm_usage(entries, ["biorxiv"])
        assert usage["biorxiv"]["n_attempted"] == 2
        assert usage["biorxiv"]["n_with_usage"] == 2
        assert usage["biorxiv"]["calls"] == 2

    def test_should_count_documents_without_usage_as_attempted(self):
        entries = [
            _manifest_entry(record_id="doc1", usage=_usage()),
            _manifest_entry(record_id="doc2"),
        ]
        usage = aggregate_llm_usage(entries, ["biorxiv"])
        assert usage["biorxiv"]["n_attempted"] == 2
        assert usage["biorxiv"]["n_with_usage"] == 1

    def test_should_count_a_document_once_but_sum_every_attempt(self):
        entries = [
            _manifest_entry(record_id="doc1", status="error", usage=_usage()),
            _manifest_entry(record_id="doc1", usage=_usage()),
        ]
        usage = aggregate_llm_usage(entries, ["biorxiv"])
        assert usage["biorxiv"]["n_attempted"] == 1
        assert usage["biorxiv"]["calls"] == 2

    def test_should_leave_out_a_corpus_without_entries(self):
        entries = [_manifest_entry(usage=_usage())]
        usage = aggregate_llm_usage(entries, ["biorxiv", "plos"])
        assert list(usage) == ["biorxiv"]


class TestReadManifestEntries:
    def test_should_be_empty_without_a_manifest(self, tmp_path):
        assert read_manifest_entries(tmp_path) == []

    def test_should_read_entries_and_ignore_blank_lines(self, tmp_path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            '{"corpus": "biorxiv", "record_id": "doc1"}\n\n'
            '{"corpus": "biorxiv", "record_id": "doc2"}\n'
        )
        assert [entry["record_id"] for entry in read_manifest_entries(tmp_path)] == [
            "doc1", "doc2"
        ]


class TestUsageForCorpora:
    def test_should_be_none_without_recorded_usage(self):
        assert usage_for_corpora({"corpora": {}}, ["biorxiv"]) is None

    def test_should_sum_across_the_named_corpora(self):
        summary = {
            "llm_usage": {
                "biorxiv": {"n_attempted": 2, "n_with_usage": 2, **_usage(calls=2)},
                "plos": {"n_attempted": 1, "n_with_usage": 1, **_usage(calls=1)},
            }
        }
        usage = usage_for_corpora(summary, ["biorxiv", "plos"])
        assert usage is not None
        assert usage["calls"] == 3
        assert usage["n_attempted"] == 3
        assert usage["n_with_usage"] == 3

    def test_should_ignore_a_corpus_it_has_no_usage_for(self):
        summary = {
            "llm_usage": {
                "biorxiv": {"n_attempted": 2, "n_with_usage": 1, **_usage()},
            }
        }
        usage = usage_for_corpora(summary, ["biorxiv", "plos"])
        assert usage is not None
        assert usage["n_attempted"] == 2
