from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.show_cases import (
    _comparison_label,
    _copy_if_exists,
    _fetch_pdfalto_xml,
    _get_doc_score,
    _run_label,
    export_case,
    find_cases,
    word_diff,
)


class TestGetDocScore:
    def test_binary_returns_f1(self):
        ms = {"true_positive": 1, "false_positive": 0, "false_negative": 0, "f1": 1.0}
        assert _get_doc_score({"levenshtein": ms}, "levenshtein") == 1.0

    def test_continuous_computes_from_sim_sum(self):
        ms = {"sim_sum": 0.8, "expected_count": 1, "predicted_count": 1, "f1": 0.0}
        assert _get_doc_score({"edit_sim": ms}, "edit_sim") == pytest.approx(0.8)

    def test_continuous_returns_zero_when_both_absent(self):
        ms = {"sim_sum": 0.0, "expected_count": 0, "predicted_count": 0, "f1": 0.0}
        assert _get_doc_score({"edit_sim": ms}, "edit_sim") == 0.0

    def test_returns_none_for_missing_method(self):
        assert _get_doc_score({}, "edit_sim") is None

    def test_continuous_precision_recall_harmonic(self):
        # sim_sum=0.6, ec=1, pc=2 → precision=0.3, recall=0.6 → f1=0.4
        ms = {"sim_sum": 0.6, "expected_count": 1, "predicted_count": 2, "f1": 0.0}
        result = _get_doc_score({"edit_sim": ms}, "edit_sim")
        assert result == pytest.approx(0.4, abs=1e-6)


class TestWordDiff:
    def test_identical_strings_show_no_markers(self):
        result = word_diff("hello world", "hello world")
        assert "[-" not in result
        assert "{+" not in result

    def test_substitution_shows_red_and_green(self):
        result = word_diff("hello world", "hello earth")
        assert "[-world-]" in result
        assert "{+earth+}" in result

    def test_deletion_shows_red(self):
        result = word_diff("hello world", "hello")
        assert "[-world-]" in result

    def test_insertion_shows_green(self):
        result = word_diff("hello", "hello world")
        assert "{+world+}" in result

    def test_both_none(self):
        assert word_diff(None, None) == "(both absent)"

    def test_gold_none(self):
        result = word_diff(None, "prediction")
        assert "gold absent" in result
        assert "prediction" in result

    def test_candidate_none(self):
        result = word_diff("gold", None)
        assert "prediction absent" in result


class TestRunLabel:
    def test_extracts_path_after_runs(self, tmp_path: Path):
        run = tmp_path / "benchmarks" / "runs" / "train"
        assert _run_label(run) == "train"

    def test_extracts_baseline_path(self, tmp_path: Path):
        run = tmp_path / "benchmarks" / "runs" / "baselines" / "grobid" / "0.9.0-crf" / "train"
        assert _run_label(run) == "baselines/grobid/0.9.0-crf/train"

    def test_falls_back_to_name_when_no_runs_segment(self, tmp_path: Path):
        run = tmp_path / "some" / "other" / "path"
        assert _run_label(run) == "path"


class TestComparisonLabel:
    def test_baseline_grobid_strips_prefix_and_split(self, tmp_path: Path):
        run_b = tmp_path / "runs" / "baselines" / "grobid" / "0.9.0-crf" / "train"
        assert _comparison_label(run_b) == "grobid-0.9.0-crf"

    def test_non_baseline_strips_split_only(self, tmp_path: Path):
        run_b = tmp_path / "runs" / "train"
        assert _comparison_label(run_b) == "train"

    def test_no_runs_segment_uses_name(self, tmp_path: Path):
        run_b = tmp_path / "some" / "path" / "train"
        assert _comparison_label(run_b) == "train"


def _make_export_case_call(
    out_dir: Path, tmp_path: Path,
    gold_text=None, text_a=None, text_b=None,
    with_source_files=False,
    alto_xml: Optional[bytes] = None,
):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    data_dir = tmp_path / "data"
    if with_source_files:
        (data_dir / "train" / "biorxiv").mkdir(parents=True, exist_ok=True)
        (data_dir / "train" / "biorxiv" / "doc1.pdf").write_bytes(b"pdf")
        (data_dir / "train" / "biorxiv" / "doc1.jats.xml").write_bytes(b"<jats/>")
        (run_a / "predictions" / "biorxiv").mkdir(parents=True, exist_ok=True)
        (run_a / "predictions" / "biorxiv" / "doc1.tei.xml").write_bytes(b"<tei-a/>")
        (run_b / "predictions" / "biorxiv").mkdir(parents=True, exist_ok=True)
        (run_b / "predictions" / "biorxiv" / "doc1.tei.xml").write_bytes(b"<tei-b/>")
        (run_a / "scores" / "biorxiv").mkdir(parents=True, exist_ok=True)
        (run_a / "scores" / "biorxiv" / "doc1.json").write_bytes(b'{"run":"a"}')
        (run_b / "scores" / "biorxiv").mkdir(parents=True, exist_ok=True)
        (run_b / "scores" / "biorxiv" / "doc1.json").write_bytes(b'{"run":"b"}')
    export_case(
        out_dir, "doc1", "biorxiv", "title",
        gold_text, text_a, text_b,
        run_a, run_b, data_dir, "train",
        alto_xml=alto_xml,
    )


class TestFetchPdfaltoXml:
    def _mock_client(self, content: bytes):
        mock_response = MagicMock()
        mock_response.content = content
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        return mock_client

    def test_returns_content_on_success(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"pdf")
        with patch(
            "benchmarks.show_cases.httpx.Client", return_value=self._mock_client(b"<alto/>")
        ):
            result = _fetch_pdfalto_xml("http://localhost:8080", pdf)
        assert result == b"<alto/>"

    def test_returns_none_on_error(self, tmp_path: Path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"pdf")
        with patch("benchmarks.show_cases.httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__.return_value.post.side_effect = Exception("conn error")
            result = _fetch_pdfalto_xml("http://localhost:8080", pdf)
        assert result is None


class TestCopyIfExists:
    def test_copies_when_src_exists(self, tmp_path: Path):
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dst = tmp_path / "dst.txt"
        _copy_if_exists(src, dst)
        assert dst.read_text() == "hello"

    def test_does_nothing_when_src_missing(self, tmp_path: Path):
        _copy_if_exists(tmp_path / "missing.txt", tmp_path / "dst.txt")
        assert not (tmp_path / "dst.txt").exists()


class TestExportCase:
    def test_writes_text_files(self, tmp_path: Path):
        out_dir = tmp_path / "examples"
        _make_export_case_call(out_dir, tmp_path, "gold text", "pred a", "pred b")
        assert (out_dir / "doc1.gold.title.txt").read_text() == "gold text"
        assert (out_dir / "doc1.run-a.title.txt").read_text() == "pred a"
        assert (out_dir / "doc1.run-b.title.txt").read_text() == "pred b"

    def test_creates_output_directory(self, tmp_path: Path):
        out_dir = tmp_path / "deep" / "nested" / "dir"
        _make_export_case_call(out_dir, tmp_path)
        assert out_dir.is_dir()

    def test_writes_empty_string_for_none(self, tmp_path: Path):
        out_dir = tmp_path / "examples"
        _make_export_case_call(out_dir, tmp_path)
        assert (out_dir / "doc1.gold.title.txt").read_text() == ""
        assert (out_dir / "doc1.run-a.title.txt").read_text() == ""
        assert (out_dir / "doc1.run-b.title.txt").read_text() == ""

    def test_copies_source_files_when_present(self, tmp_path: Path):
        out_dir = tmp_path / "examples"
        _make_export_case_call(out_dir, tmp_path, with_source_files=True)
        assert (out_dir / "doc1.pdf").read_bytes() == b"pdf"
        assert (out_dir / "doc1.jats.xml").read_bytes() == b"<jats/>"
        assert (out_dir / "doc1.run-a.tei.xml").read_bytes() == b"<tei-a/>"
        assert (out_dir / "doc1.run-b.tei.xml").read_bytes() == b"<tei-b/>"
        assert (out_dir / "doc1.run-a.scores.json").read_bytes() == b'{"run":"a"}'
        assert (out_dir / "doc1.run-b.scores.json").read_bytes() == b'{"run":"b"}'

    def test_skips_missing_source_files(self, tmp_path: Path):
        out_dir = tmp_path / "examples"
        _make_export_case_call(out_dir, tmp_path, with_source_files=False)
        assert not (out_dir / "doc1.pdf").exists()
        assert not (out_dir / "doc1.jats.xml").exists()
        assert not (out_dir / "doc1.run-a.tei.xml").exists()
        assert not (out_dir / "doc1.run-b.tei.xml").exists()
        assert not (out_dir / "doc1.run-a.scores.json").exists()
        assert not (out_dir / "doc1.run-b.scores.json").exists()

    def test_writes_alto_xml_when_provided(self, tmp_path: Path):
        out_dir = tmp_path / "examples"
        _make_export_case_call(out_dir, tmp_path, alto_xml=b"<alto><Page/></alto>")
        content = (out_dir / "doc1.alto.xml").read_bytes()
        assert b"<alto>" in content
        assert b"<Page/>" in content
        assert b"\n" in content  # formatted

    def test_skips_alto_xml_when_none(self, tmp_path: Path):
        out_dir = tmp_path / "examples"
        _make_export_case_call(out_dir, tmp_path, alto_xml=None)
        assert not (out_dir / "doc1.alto.xml").exists()


class TestFindCases:
    def _write_score(self, path: Path, field: str, method: str, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ms: dict
        if method == "edit_sim":
            ms = {"sim_sum": value, "expected_count": 1, "predicted_count": 1, "f1": 0.0}
        else:
            tp = 1 if value == 1.0 else 0
            ms = {"true_positive": tp, "false_positive": 1 - tp,
                  "false_negative": 1 - tp, "f1": value}
        path.write_text(json.dumps({
            "record_id": path.stem,
            "corpus": path.parent.name,
            "fields": {field: {"scoring_type": "string", method: ms}},
        }))

    def test_finds_regressions(self, tmp_path: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        self._write_score(run_a / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.5)
        self._write_score(run_b / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.8)

        cases = find_cases(run_a, run_b, "title", "edit_sim", None, "regression")
        assert len(cases) == 1
        delta, corpus, record_id, score_a, score_b = cases[0]
        assert corpus == "biorxiv"
        assert record_id == "doc1"
        assert delta == pytest.approx(-0.3, abs=1e-6)
        assert score_a == pytest.approx(0.5)
        assert score_b == pytest.approx(0.8)

    def test_finds_improvements(self, tmp_path: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        self._write_score(run_a / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.9)
        self._write_score(run_b / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.6)

        cases = find_cases(run_a, run_b, "title", "edit_sim", None, "improvement")
        assert len(cases) == 1
        assert cases[0][0] == pytest.approx(0.3, abs=1e-6)

    def test_excludes_equal_scores(self, tmp_path: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        self._write_score(run_a / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.8)
        self._write_score(run_b / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.8)

        assert not find_cases(run_a, run_b, "title", "edit_sim", None, "regression")
        assert not find_cases(run_a, run_b, "title", "edit_sim", None, "improvement")

    def test_skips_when_run_b_score_missing(self, tmp_path: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        self._write_score(run_a / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.5)

        assert not find_cases(run_a, run_b, "title", "edit_sim", None, "regression")

    def test_corpus_filter(self, tmp_path: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        self._write_score(run_a / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.5)
        self._write_score(run_b / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.9)
        self._write_score(run_a / "scores" / "pmc" / "doc2.json", "title", "edit_sim", 0.5)
        self._write_score(run_b / "scores" / "pmc" / "doc2.json", "title", "edit_sim", 0.9)

        cases = find_cases(run_a, run_b, "title", "edit_sim", "biorxiv", "regression")
        assert len(cases) == 1
        assert cases[0][1] == "biorxiv"

    def test_regressions_sorted_worst_first(self, tmp_path: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        self._write_score(run_a / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.7)
        self._write_score(run_b / "scores" / "biorxiv" / "doc1.json", "title", "edit_sim", 0.9)
        self._write_score(run_a / "scores" / "biorxiv" / "doc2.json", "title", "edit_sim", 0.4)
        self._write_score(run_b / "scores" / "biorxiv" / "doc2.json", "title", "edit_sim", 0.9)

        cases = find_cases(run_a, run_b, "title", "edit_sim", None, "regression")
        assert cases[0][2] == "doc2"  # worse regression first

    def test_returns_empty_when_no_scores_dir(self, tmp_path: Path):
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        assert not find_cases(run_a, run_b, "title", "edit_sim", None, "regression")
