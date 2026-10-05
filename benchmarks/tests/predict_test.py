from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from sciencebeam_parser.models.llm.usage import USAGE_HEADER_NAME

from benchmarks.compute_cost import format_duration
from benchmarks.predict import (
    _append_manifest,
    _llm_usage_entry,
    _load_done,
    _Progress,
    _resolve_concurrency,
    _run_predict_async,
    _summarise_manifest,
    run_predict,
)

LLM_USAGE_1 = {
    "calls": 4,
    "input_tokens": 400,
    "output_tokens": 200,
    "cost_credits": 0.0016,
    "by_task": {"citation": {"calls": 4, "output_tokens": 200}},
}


class TestLoadDone:
    def test_should_return_empty_set_when_manifest_absent(self, tmp_path: Path):
        assert _load_done(tmp_path) == set()

    def test_should_include_ok_record(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}) + "\n"
        )
        assert _load_done(tmp_path) == {("biorxiv", "r1")}

    def test_should_exclude_error_record(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "error"}) + "\n"
        )
        assert _load_done(tmp_path) == set()

    def test_should_handle_mixed_ok_and_error(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        lines = [
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}),
            json.dumps({"corpus": "biorxiv", "record_id": "r2", "status": "error"}),
            json.dumps({"corpus": "biorxiv", "record_id": "r3", "status": "ok"}),
        ]
        manifest.write_text("\n".join(lines) + "\n")
        assert _load_done(tmp_path) == {("biorxiv", "r1"), ("biorxiv", "r3")}

    def test_should_ignore_blank_lines(self, tmp_path: Path):
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"corpus": "biorxiv", "record_id": "r1", "status": "ok"}) + "\n\n"
        )
        assert _load_done(tmp_path) == {("biorxiv", "r1")}


class TestAppendManifest:
    def test_should_create_parent_dirs_and_write_entry(self, tmp_path: Path):
        entry = {"corpus": "biorxiv", "record_id": "r1", "status": "ok", "elapsed_ms": 500}
        _append_manifest(tmp_path, entry)
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        assert manifest.exists()
        written = json.loads(manifest.read_text().strip())
        assert written == entry

    def test_should_append_successive_entries(self, tmp_path: Path):
        _append_manifest(tmp_path, {"corpus": "biorxiv", "record_id": "r1", "status": "ok"})
        _append_manifest(tmp_path, {"corpus": "biorxiv", "record_id": "r2", "status": "error"})
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        lines = [line for line in manifest.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["record_id"] == "r1"
        assert json.loads(lines[1])["record_id"] == "r2"


class TestResolveConcurrency:
    def test_explicit_value_returned_unchanged(self):
        assert _resolve_concurrency(4) == 4

    def test_zero_returns_at_least_two(self):
        assert _resolve_concurrency(0) >= 2

    def test_zero_returns_cpu_count_or_higher(self):
        expected = max(2, os.cpu_count() or 2)
        assert _resolve_concurrency(0) == expected


class TestFormatDuration:
    def test_seconds(self):
        assert format_duration(45.0) == "45s"

    def test_minutes(self):
        assert format_duration(90.0) == "1m30s"

    def test_hours(self):
        assert format_duration(7200.0) == "2.0h"

    def test_boundary_one_minute(self):
        assert format_duration(60.0) == "1m00s"

    def test_boundary_one_hour(self):
        assert format_duration(3600.0) == "1.0h"


class TestProgress:
    def test_initial_state(self):
        p = _Progress(10)
        assert p.total == 10
        assert p.n_ok == 0
        assert p.n_err == 0
        assert p.completed == 0

    def test_record_ok_increments(self):
        p = _Progress(10)
        p.record_ok("biorxiv", "doc1", 1000)
        assert p.n_ok == 1
        assert p.n_err == 0
        assert p.completed == 1

    def test_record_err_increments(self):
        p = _Progress(10)
        p.record_err("biorxiv", "doc1")
        assert p.n_ok == 0
        assert p.n_err == 1
        assert p.completed == 1

    def test_completed_sums_ok_and_err(self):
        p = _Progress(10)
        p.record_ok("biorxiv", "doc1", 500)
        p.record_err("biorxiv", "doc2")
        assert p.completed == 2


def _make_record(tmp_path: Path, corpus: str = "biorxiv", record_id: str = "doc1") -> dict:
    pdf = tmp_path / "data" / corpus / f"{record_id}.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"fake pdf")
    return {"corpus": corpus, "record_id": record_id, "pdf_path": str(pdf)}


def _usage_headers(usage: Optional[dict]) -> dict:
    return {} if usage is None else {USAGE_HEADER_NAME: json.dumps(usage)}


def _mock_client(
    content: bytes = b"<tei/>", usage: Optional[dict] = None
) -> AsyncMock:
    mock_response = MagicMock()
    mock_response.content = content
    mock_response.headers = _usage_headers(usage)
    mock_response.raise_for_status = MagicMock(return_value=None)
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=mock_response)
    return client


def _mock_client_http_error(
    status_code: int = 500,
    body: str = "server error",
    usage: Optional[dict] = None,
) -> AsyncMock:
    error_response = MagicMock()
    error_response.text = body
    error_response.headers = _usage_headers(usage)
    exc = httpx.HTTPStatusError(
        f"Server error '{status_code}'",
        request=MagicMock(),
        response=error_response,
    )
    mock_response = MagicMock()
    mock_response.headers = _usage_headers(usage)
    mock_response.raise_for_status = MagicMock(side_effect=exc)
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=mock_response)
    return client


def _mock_client_timeout() -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))
    return client


class TestRunPredictAsync:
    def _read_manifest(self, run_dir: Path) -> list:
        path = run_dir / "predictions" / "manifest.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    def test_successful_prediction_writes_file_and_manifest(self, tmp_path: Path):
        records = [_make_record(tmp_path)]
        run_dir = tmp_path / "run"
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=_mock_client(b"<tei/>")):
            n_ok, n_err = asyncio.run(
                _run_predict_async(records, set(), run_dir, "http://localhost:8080", 60, 1)
            )
        assert n_ok == 1 and n_err == 0
        assert (run_dir / "predictions" / "biorxiv" / "doc1.tei.xml").read_bytes() == b"<tei/>"
        entries = self._read_manifest(run_dir)
        assert len(entries) == 1
        assert entries[0]["status"] == "ok"
        assert entries[0]["record_id"] == "doc1"

    def test_http_error_records_error_and_body_in_manifest(self, tmp_path: Path):
        records = [_make_record(tmp_path)]
        run_dir = tmp_path / "run"
        client = _mock_client_http_error(500, "traceback here")
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            n_ok, n_err = asyncio.run(
                _run_predict_async(records, set(), run_dir, "http://localhost:8080", 60, 1)
            )
        assert n_ok == 0 and n_err == 1
        entries = self._read_manifest(run_dir)
        assert entries[0]["status"] == "error"
        assert entries[0]["error_body"] == "traceback here"

    def test_already_done_records_are_skipped(self, tmp_path: Path):
        records = [_make_record(tmp_path)]
        run_dir = tmp_path / "run"
        done = {("biorxiv", "doc1")}
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=_mock_client()):
            n_ok, n_err = asyncio.run(
                _run_predict_async(records, done, run_dir, "http://localhost:8080", 60, 1)
            )
        assert n_ok == 0 and n_err == 0
        assert not (run_dir / "predictions" / "biorxiv" / "doc1.tei.xml").exists()

    def test_multiple_records_all_processed(self, tmp_path: Path):
        records = [
            _make_record(tmp_path, "biorxiv", "doc1"),
            _make_record(tmp_path, "biorxiv", "doc2"),
            _make_record(tmp_path, "ore", "doc3"),
        ]
        run_dir = tmp_path / "run"
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=_mock_client(b"<tei/>")):
            n_ok, n_err = asyncio.run(
                _run_predict_async(records, set(), run_dir, "http://localhost:8080", 60, 2)
            )
        assert n_ok == 3 and n_err == 0

    def test_generic_exception_records_error(self, tmp_path: Path):
        records = [_make_record(tmp_path)]
        run_dir = tmp_path / "run"
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(side_effect=Exception("connection refused"))
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            n_ok, n_err = asyncio.run(
                _run_predict_async(records, set(), run_dir, "http://localhost:8080", 60, 1)
            )
        assert n_ok == 0 and n_err == 1
        entries = self._read_manifest(run_dir)
        assert entries[0]["status"] == "error"
        assert "connection refused" in entries[0]["error"]


class TestLlmUsageEntry:
    def test_should_be_empty_without_a_response(self):
        assert not _llm_usage_entry(None)

    def test_should_be_empty_without_the_header(self):
        response = MagicMock()
        response.headers = {}
        assert not _llm_usage_entry(response)

    def test_should_return_the_header_as_given(self):
        response = MagicMock()
        response.headers = {USAGE_HEADER_NAME: json.dumps(LLM_USAGE_1)}
        assert _llm_usage_entry(response) == {"llm_usage": LLM_USAGE_1}

    def test_should_be_empty_for_an_unparseable_header(self):
        response = MagicMock()
        response.headers = {USAGE_HEADER_NAME: "{not json"}
        assert not _llm_usage_entry(response)


class TestManifestLlmUsage:
    def _read_manifest(self, run_dir: Path) -> list:
        path = run_dir / "predictions" / "manifest.jsonl"
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    def test_should_record_usage_of_a_successful_document(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        client = _mock_client(usage=LLM_USAGE_1)
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            asyncio.run(_run_predict_async(
                [_make_record(tmp_path)], set(), run_dir, "http://localhost:8080", 60, 1
            ))
        assert self._read_manifest(run_dir)[0]["llm_usage"] == LLM_USAGE_1

    def test_should_record_usage_of_a_document_that_errored(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        client = _mock_client_http_error(500, "traceback here", usage=LLM_USAGE_1)
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            asyncio.run(_run_predict_async(
                [_make_record(tmp_path)], set(), run_dir, "http://localhost:8080", 60, 1
            ))
        entry = self._read_manifest(run_dir)[0]
        assert entry["status"] == "error"
        assert entry["llm_usage"] == LLM_USAGE_1

    def test_should_omit_usage_rather_than_zero_it_without_a_header(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        with patch(
            "benchmarks.predict.httpx.AsyncClient", return_value=_mock_client()
        ):
            asyncio.run(_run_predict_async(
                [_make_record(tmp_path)], set(), run_dir, "http://localhost:8080", 60, 1
            ))
        assert "llm_usage" not in self._read_manifest(run_dir)[0]

    def test_should_omit_usage_when_no_response_arrived(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        with patch(
            "benchmarks.predict.httpx.AsyncClient", return_value=_mock_client_timeout()
        ):
            asyncio.run(_run_predict_async(
                [_make_record(tmp_path)], set(), run_dir, "http://localhost:8080", 60, 1
            ))
        entry = self._read_manifest(run_dir)[0]
        assert entry["status"] == "error"
        assert "llm_usage" not in entry


def _mock_client_failing_then_ok(failures: int = 1) -> AsyncMock:
    """Fails the first `failures` times and succeeds after, as a transient error does."""
    calls = {"n": 0}
    error_response = MagicMock()
    error_response.text = "server error"
    error_response.headers = {}

    def _post(*_args, **_kwargs):
        calls["n"] += 1
        response = MagicMock()
        response.headers = {}
        response.content = b"<tei/>"
        if calls["n"] <= failures:
            response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError(
                "Server error '500'", request=MagicMock(), response=error_response
            ))
        else:
            response.raise_for_status = MagicMock(return_value=None)
        return response

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(side_effect=_post)
    return client


class TestSummariseManifest:
    def _write(self, tmp_path: Path, entries: list) -> Path:
        manifest = tmp_path / "predictions" / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("".join(json.dumps(e) + "\n" for e in entries))
        return tmp_path

    def test_should_count_a_document_that_failed_and_then_succeeded_as_recovered(
        self, tmp_path: Path
    ):
        self._write(tmp_path, [
            {"corpus": "biorxiv", "record_id": "r1", "status": "error", "pass": 1},
            {"corpus": "biorxiv", "record_id": "r1", "status": "ok", "pass": 2},
        ])
        records = [{"corpus": "biorxiv", "record_id": "r1"}]
        assert _summarise_manifest(tmp_path, records) == (1, 0, 1)

    def test_should_count_a_document_that_never_succeeded_as_missing(self, tmp_path: Path):
        self._write(tmp_path, [
            {"corpus": "biorxiv", "record_id": "r1", "status": "error", "pass": 1},
            {"corpus": "biorxiv", "record_id": "r1", "status": "error", "pass": 2},
        ])
        records = [{"corpus": "biorxiv", "record_id": "r1"}]
        assert _summarise_manifest(tmp_path, records) == (0, 1, 0)

    def test_should_not_count_a_first_time_success_as_recovered(self, tmp_path: Path):
        self._write(tmp_path, [
            {"corpus": "biorxiv", "record_id": "r1", "status": "ok", "pass": 1},
        ])
        records = [{"corpus": "biorxiv", "record_id": "r1"}]
        assert _summarise_manifest(tmp_path, records) == (1, 0, 0)

    def test_should_count_a_record_the_run_never_reached_as_missing(self, tmp_path: Path):
        self._write(tmp_path, [
            {"corpus": "biorxiv", "record_id": "r1", "status": "ok", "pass": 1},
        ])
        records = [
            {"corpus": "biorxiv", "record_id": "r1"},
            {"corpus": "biorxiv", "record_id": "r2"},
        ]
        assert _summarise_manifest(tmp_path, records) == (1, 1, 0)

    def test_should_ignore_a_manifest_entry_outside_the_records_asked_for(
        self, tmp_path: Path
    ):
        self._write(tmp_path, [
            {"corpus": "ore", "record_id": "other", "status": "ok", "pass": 1},
        ])
        records = [{"corpus": "biorxiv", "record_id": "r1"}]
        assert _summarise_manifest(tmp_path, records) == (0, 1, 0)


class TestRunPredictRetryPasses:
    def _run(self, tmp_path: Path, client: AsyncMock, retry_passes: int) -> dict:
        records = [_make_record(tmp_path)]
        run_dir = tmp_path / "run"
        with patch("benchmarks.predict.fetch_data", return_value=records), \
                patch("benchmarks.predict.resolved_sources", return_value={"biorxiv": {}}), \
                patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            run_predict(
                config={"fields": ["title"]}, mode="smoke", split="train",
                data_dir=tmp_path / "data", run_dir=run_dir,
                parser_url="http://localhost:8080", parser_image=None,
                profile=None, concurrency=1, retry_passes=retry_passes,
            )
        return json.loads((run_dir / "run.json").read_text())

    def _manifest(self, tmp_path: Path) -> list:
        """The documents, without the record the invocation makes of itself."""
        path = tmp_path / "run" / "predictions" / "manifest.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [entry for entry in entries if entry.get("record_id")]

    def test_should_recover_a_document_that_fails_once_and_report_it(self, tmp_path: Path):
        run_record = self._run(tmp_path, _mock_client_failing_then_ok(1), retry_passes=2)
        assert run_record["n_records"] == 1
        assert run_record["n_errors"] == 0
        assert run_record["n_recovered"] == 1
        assert run_record["retry_passes"] == 2

    def test_should_record_which_pass_produced_each_manifest_entry(self, tmp_path: Path):
        self._run(tmp_path, _mock_client_failing_then_ok(1), retry_passes=2)
        entries = self._manifest(tmp_path)
        assert [(e["status"], e["pass"]) for e in entries] == [("error", 1), ("ok", 2)]

    def test_should_leave_the_failure_when_no_retry_was_asked_for(self, tmp_path: Path):
        run_record = self._run(tmp_path, _mock_client_failing_then_ok(1), retry_passes=1)
        assert run_record["n_records"] == 0
        assert run_record["n_errors"] == 1
        assert run_record["n_recovered"] == 0

    def test_should_report_a_document_that_fails_every_pass_as_an_error(self, tmp_path: Path):
        run_record = self._run(tmp_path, _mock_client_failing_then_ok(5), retry_passes=3)
        assert run_record["n_errors"] == 1
        assert run_record["n_recovered"] == 0
        assert len(self._manifest(tmp_path)) == 3

    def test_should_not_ask_again_once_every_document_has_a_prediction(self, tmp_path: Path):
        client = _mock_client_failing_then_ok(0)
        run_record = self._run(tmp_path, client, retry_passes=3)
        assert run_record["n_records"] == 1
        assert client.post.await_count == 1


class TestErrorTiming:
    def _entry(self, tmp_path: Path, client: AsyncMock) -> dict:
        run_dir = tmp_path / "run"
        with patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            asyncio.run(_run_predict_async(
                [_make_record(tmp_path)], set(), run_dir, "http://localhost:8080", 60, 1
            ))
        path = run_dir / "predictions" / "manifest.jsonl"
        return json.loads(path.read_text().splitlines()[0])

    def test_should_record_how_long_an_http_error_took(self, tmp_path: Path):
        entry = self._entry(tmp_path, _mock_client_http_error(500, "boom"))
        assert entry["status"] == "error"
        assert entry["elapsed_ms"] >= 0

    def test_should_record_how_long_a_timeout_took(self, tmp_path: Path):
        entry = self._entry(tmp_path, _mock_client_timeout())
        assert entry["status"] == "error"
        assert entry["elapsed_ms"] >= 0


class TestRequestedProfile:
    """A run says which profile served it, and the request is what makes that
    true: naming one the deployment does not offer is refused rather than served
    by its default."""

    def _run(self, tmp_path: Path, client: AsyncMock, profile: Optional[str]) -> dict:
        records = [_make_record(tmp_path)]
        run_dir = tmp_path / "run"
        with patch("benchmarks.predict.fetch_data", return_value=records), \
                patch("benchmarks.predict.resolved_sources", return_value={"biorxiv": {}}), \
                patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            run_predict(
                config={"fields": ["title"]}, mode="smoke", split="train",
                data_dir=tmp_path / "data", run_dir=run_dir,
                parser_url="http://localhost:8080", parser_image=None,
                profile=profile, concurrency=1, retry_passes=1,
            )
        return json.loads((run_dir / "run.json").read_text())

    def test_should_ask_for_the_profile_it_records(self, tmp_path: Path):
        client = _mock_client()
        run_record = self._run(tmp_path, client, "llm_segmentation_named")
        assert client.post.call_args.kwargs["params"] == {
            "profile": "llm_segmentation_named"
        }
        assert run_record["profile"] == "llm_segmentation_named"

    def test_should_leave_the_deployment_its_own_profile_when_none_is_given(
        self, tmp_path: Path
    ):
        client = _mock_client()
        self._run(tmp_path, client, None)
        assert client.post.call_args.kwargs["params"] is None


class TestRunPredictCostRecord:
    def _run(
        self,
        tmp_path: Path,
        client: AsyncMock,
        parser_url: str = "http://localhost:8080",
        concurrency: int = 1,
        records: Optional[list] = None,
    ) -> dict:
        records = records if records is not None else [_make_record(tmp_path)]
        run_dir = tmp_path / "run"
        with patch("benchmarks.predict.fetch_data", return_value=records), \
                patch("benchmarks.predict.resolved_sources", return_value={"biorxiv": {}}), \
                patch("benchmarks.predict.httpx.AsyncClient", return_value=client):
            run_predict(
                config={"fields": ["title"]}, mode="smoke", split="train",
                data_dir=tmp_path / "data", run_dir=run_dir,
                parser_url=parser_url, parser_image=None,
                profile=None, concurrency=concurrency,
            )
        return json.loads((run_dir / "run.json").read_text())

    def test_should_record_the_resolved_concurrency(self, tmp_path: Path):
        run_record = self._run(tmp_path, _mock_client(), concurrency=0)
        assert run_record["concurrency"] == _resolve_concurrency(0)

    def test_should_record_the_machine_it_ran_on(self, tmp_path: Path):
        machine = self._run(tmp_path, _mock_client())["machine"]
        assert machine["cpu_count"] > 0
        assert machine["cpu_seconds"] >= 0

    def test_should_record_no_cpu_figure_where_the_parser_ran_elsewhere(
        self, tmp_path: Path
    ):
        machine = self._run(
            tmp_path, _mock_client(), parser_url="http://parser.example:8080"
        )["machine"]
        assert "cpu_seconds" not in machine
        assert machine["cpu_count"] > 0

    def test_should_count_the_documents_this_invocation_processed(self, tmp_path: Path):
        records = [
            _make_record(tmp_path, record_id="doc1"),
            _make_record(tmp_path, record_id="doc2"),
        ]
        run_record = self._run(tmp_path, _mock_client(), records=records)
        assert run_record["n_processed"] == 2

    def test_should_count_a_failed_document_the_machine_still_paid_for(
        self, tmp_path: Path
    ):
        run_record = self._run(tmp_path, _mock_client_http_error())
        assert run_record["n_records"] == 0
        assert run_record["n_processed"] == 1

    def test_should_not_count_documents_an_earlier_invocation_processed(
        self, tmp_path: Path
    ):
        records = [_make_record(tmp_path)]
        self._run(tmp_path, _mock_client(), records=records)
        resumed = self._run(tmp_path, _mock_client(), records=records)
        assert resumed["n_records"] == 1
        assert resumed["n_processed"] == 0

    def _manifest(self, tmp_path: Path) -> list:
        path = tmp_path / "run" / "predictions" / "manifest.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_should_stamp_what_this_invocation_wrote(self, tmp_path: Path):
        run_record = self._run(tmp_path, _mock_client())
        documents = [e for e in self._manifest(tmp_path) if e.get("record_id")]
        assert [e["run_started_at"] for e in documents] == [run_record["started_at"]]

    def test_should_record_what_the_invocation_cost_beside_its_documents(
        self, tmp_path: Path
    ):
        run_record = self._run(tmp_path, _mock_client())
        runs = [e for e in self._manifest(tmp_path) if e.get("type") == "run"]
        assert len(runs) == 1
        assert runs[0]["started_at"] == run_record["started_at"]
        assert runs[0]["n_processed"] == 1
        assert runs[0]["concurrency"] == run_record["concurrency"]
        assert runs[0]["machine"]["cpu_count"] > 0

    def test_should_record_each_invocation_that_generated_something(
        self, tmp_path: Path
    ):
        records = [_make_record(tmp_path, record_id="doc1")]
        self._run(tmp_path, _mock_client(), records=records)
        records.append(_make_record(tmp_path, record_id="doc2"))
        self._run(tmp_path, _mock_client(), records=records)
        runs = [e for e in self._manifest(tmp_path) if e.get("type") == "run"]
        assert [run["n_processed"] for run in runs] == [1, 1]
        assert len({run["started_at"] for run in runs}) == 2

    def test_should_record_no_run_where_the_invocation_generated_nothing(
        self, tmp_path: Path
    ):
        records = [_make_record(tmp_path)]
        self._run(tmp_path, _mock_client(), records=records)
        self._run(tmp_path, _mock_client(), records=records)
        runs = [e for e in self._manifest(tmp_path) if e.get("type") == "run"]
        assert len(runs) == 1
