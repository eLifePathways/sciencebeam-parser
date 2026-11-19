from pathlib import Path
from unittest.mock import patch

from lxml import etree

from benchmarks.jats_conversion import (
    CONFIG_PATH,
    STYLESHEET_PATH,
    checkout_converter,
    commit_converter,
    compare_formats,
    convert_predictions_to_jats,
)
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.resources.xslt import TEI_TO_JATS_XSLT_FILE


TEI_XML = (
    '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>'
    '<title level="a" type="main">An invented title</title>'
    '</titleStmt></fileDesc></teiHeader><text><body/></text></TEI>'
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _article_title(path: Path) -> str:
    return etree.parse(str(path)).getroot().xpath(
        "string(front/article-meta/title-group/article-title)"
    )


class TestConvertPredictionsToJats:
    def test_should_write_jats_beside_each_tei(self, tmp_path: Path):
        _write(tmp_path / "biorxiv" / "doc-1.tei.xml", TEI_XML)
        failed = convert_predictions_to_jats(tmp_path, checkout_converter())
        assert not failed
        assert _article_title(tmp_path / "biorxiv" / "doc-1.jats.xml") == "An invented title"

    def test_should_write_into_another_directory_where_given(self, tmp_path: Path):
        _write(tmp_path / "in" / "biorxiv" / "doc-1.tei.xml", TEI_XML)
        convert_predictions_to_jats(tmp_path / "in", checkout_converter(), tmp_path / "out")
        assert (tmp_path / "out" / "biorxiv" / "doc-1.jats.xml").exists()
        assert not (tmp_path / "in" / "biorxiv" / "doc-1.jats.xml").exists()

    def test_should_leave_no_jats_for_tei_that_does_not_parse(self, tmp_path: Path):
        broken = _write(tmp_path / "biorxiv" / "doc-1.tei.xml", "<TEI")
        stale = _write(tmp_path / "biorxiv" / "doc-1.jats.xml", "<article/>")
        _write(tmp_path / "biorxiv" / "doc-2.tei.xml", TEI_XML)
        failed = convert_predictions_to_jats(tmp_path, checkout_converter())
        assert failed == [broken]
        assert not stale.exists()
        assert (tmp_path / "biorxiv" / "doc-2.jats.xml").exists()


class TestCommitConverter:
    def test_should_convert_with_the_stylesheet_and_config_of_a_commit(
        self, tmp_path: Path
    ):
        files_by_path = {STYLESHEET_PATH: TEI_TO_JATS_XSLT_FILE, CONFIG_PATH: DEFAULT_CONFIG_FILE}

        def git_show(_commit: str, path: str) -> bytes:
            return Path(files_by_path[path]).read_bytes()

        jats_path = tmp_path / "doc-1.jats.xml"
        with patch("benchmarks.jats_conversion._git_show", side_effect=git_show) as show:
            converter = commit_converter("abc1234")
        converter.write_jats(etree.fromstring(TEI_XML), str(jats_path))
        assert _article_title(jats_path) == "An invented title"
        assert [call.args for call in show.call_args_list] == [
            ("abc1234", STYLESHEET_PATH), ("abc1234", CONFIG_PATH),
        ]


class TestCompareFormats:
    def test_should_score_the_tei_and_the_converted_jats_side_by_side(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        _write(run_dir / "predictions" / "biorxiv" / "doc-1.tei.xml", TEI_XML)
        _write(run_dir / "predictions" / "manifest.jsonl", "")
        out_dir = tmp_path / "out"
        with patch("benchmarks.jats_conversion.run_score") as run_score, \
                patch("benchmarks.jats_conversion.run_compare") as run_compare:
            compare_formats({}, run_dir, out_dir, tmp_path / "data", checkout_converter())
        assert [
            (call.args[1], call.kwargs["prediction_format"])
            for call in run_score.call_args_list
        ] == [(out_dir / "tei", "tei"), (out_dir / "jats", "jats")]
        assert (out_dir / "tei" / "predictions" / "biorxiv" / "doc-1.tei.xml").exists()
        assert (out_dir / "jats" / "predictions" / "biorxiv" / "doc-1.jats.xml").exists()
        assert (out_dir / "jats" / "predictions" / "manifest.jsonl").exists()
        assert run_compare.call_args.args[0] == [
            ("TEI", out_dir / "tei" / "summary.json"),
            ("JATS", out_dir / "jats" / "summary.json"),
        ]
        assert run_compare.call_args.kwargs["baseline"] == 0
