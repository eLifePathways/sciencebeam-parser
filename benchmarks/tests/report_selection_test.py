from __future__ import annotations

import json
from typing import List

import pytest

from benchmarks.report import ChartOutput, _render_comparison_report, run_compare
from benchmarks.report_grid import (
    ChartConfig,
    FieldsChartConfig,
    Selection,
    SelectionError,
)


def _agg(scoring_type: str, method: str, by_field: dict) -> dict:
    return {
        "scoring_type": scoring_type,
        "scoring_method": method,
        "summary_scores": {
            "by-field": {field: {"scores": {"f1": f1}} for field, f1 in by_field.items()}
        },
    }


def _summary(fields, field_measures, field_scoring_types, corpora) -> dict:
    return {
        "fields": fields,
        "field_measures": field_measures,
        "field_scoring_types": field_scoring_types,
        "corpora": corpora,
    }


def _two_field_summary(bump: float = 0.0, corpora=("biorxiv", "ore")) -> dict:
    """Two fields over two corpora, with `title` carrying two methods."""
    return _summary(
        fields=["title", "abstract"],
        field_measures={"title": ["exact", "levenshtein"], "abstract": ["levenshtein"]},
        field_scoring_types={"title": "string", "abstract": "string"},
        corpora={
            name: {
                "n": 10,
                "aggregated": [
                    _agg("string", "exact", {"title": 0.4 + bump}),
                    _agg("string", "levenshtein", {"title": 0.5 + bump, "abstract": 0.6 + bump}),
                ],
            }
            for name in corpora
        },
    )


def _rows(report: str) -> List[str]:
    return [line for line in report.splitlines() if line.startswith("| title") or
            line.startswith("| abstract")]


class TestFieldSelection:
    def test_shows_only_the_selected_field(self):
        report = _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(fields=("abstract",)),
        )
        assert all(row.startswith("| abstract") for row in _rows(report))

    def test_keeps_the_order_the_fields_were_asked_for(self):
        report = _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(fields=("abstract", "title")),
        )
        assert _rows(report)[0].startswith("| abstract")

    def test_narrows_to_the_selected_method(self):
        report = _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(fields=("title",), methods=("exact",)),
        )
        assert [row.split("|")[1].strip() for row in _rows(report)] == [
            "title (exact)", "title (exact)", "title (exact)",
        ]

    def test_shows_only_the_selected_corpus(self):
        report = _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(corpora=("ore",)),
        )
        assert "<b>ore</b>" in report and "<b>biorxiv</b>" not in report

    def test_renders_the_same_numbers_as_the_full_view(self):
        labeled = [("base", _two_field_summary()), ("head", _two_field_summary(0.05))]
        full = _rows(_render_comparison_report(labeled))
        narrowed = _rows(_render_comparison_report(
            labeled, selection=Selection(fields=("abstract",))
        ))
        assert narrowed == [row for row in full if row.startswith("| abstract")]

    def test_rejects_a_field_no_summary_scored(self):
        with pytest.raises(SelectionError, match="keywords"):
            _render_comparison_report(
                [("base", _two_field_summary()), ("head", _two_field_summary())],
                selection=Selection(fields=("keywords",)),
            )


class TestCharts:
    def _charted(self, tmp_path, **kwargs):
        return _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(fields=("abstract",), charts=("abstract",)),
            charts=ChartOutput(out_dir=tmp_path, **kwargs),
        )

    def test_adds_a_chart_section(self, tmp_path):
        assert "### Charts" in self._charted(tmp_path)

    def test_writes_the_image_beside_the_report(self, tmp_path):
        self._charted(tmp_path)
        assert (tmp_path / "abstract-levenshtein-all.png").exists()

    def test_links_the_image_relatively_by_default(self, tmp_path):
        assert "(charts/abstract-levenshtein-all.png)" in self._charted(tmp_path)

    def test_links_under_a_base_url_where_one_is_given(self, tmp_path):
        report = self._charted(tmp_path, base_url="https://example.org/r")
        assert "(https://example.org/r/abstract-levenshtein-all.png)" in report

    def test_keeps_a_runs_charts_apart_from_another_runs(self, tmp_path):
        report = self._charted(tmp_path, prefix="sha1-")
        assert "sha1-abstract-levenshtein-all.png" in report
        assert (tmp_path / "sha1-abstract-levenshtein-all.png").exists()

    def test_draws_nothing_without_a_chart_selection(self, tmp_path):
        report = _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            charts=ChartOutput(out_dir=tmp_path),
        )
        assert "### Charts" not in report
        assert not list(tmp_path.iterdir())

    def test_draws_nothing_for_a_single_corpus(self, tmp_path):
        report = _render_comparison_report(
            [
                ("base", _two_field_summary(corpora=("ore",))),
                ("head", _two_field_summary(0.05, corpora=("ore",))),
            ],
            selection=Selection(charts=("abstract",)),
            charts=ChartOutput(out_dir=tmp_path),
        )
        assert "### Charts" not in report

    def test_charts_each_method_and_scope_of_the_field(self, tmp_path):
        _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(charts=("title",)),
            charts=ChartOutput(out_dir=tmp_path),
        )
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            "title-exact-all.png", "title-levenshtein-all.png",
        ]

    def test_charts_only_the_methods_asked_for(self, tmp_path):
        _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(charts=("title",), chart_methods=("exact",)),
            charts=ChartOutput(out_dir=tmp_path),
        )
        assert [path.name for path in tmp_path.iterdir()] == ["title-exact-all.png"]

    def test_leaves_the_tables_alone_when_narrowing_the_charts(self, tmp_path):
        labeled = [("base", _two_field_summary()), ("head", _two_field_summary(0.05))]
        full = _rows(_render_comparison_report(labeled))
        charted = _rows(_render_comparison_report(
            labeled,
            selection=Selection(charts=("title",), chart_methods=("exact",)),
            charts=ChartOutput(out_dir=tmp_path),
        ))
        assert charted == full


class TestDeclaredCharts:
    def _charted(self, tmp_path, **chart):
        return _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(declared_charts=(ChartConfig(**chart),)),
            charts=ChartOutput(out_dir=tmp_path),
        )

    def test_draws_the_row_it_names(self, tmp_path):
        self._charted(tmp_path, field="title", method="exact")
        assert [path.name for path in tmp_path.iterdir()] == ["title-exact-all.png"]

    def test_uses_the_title_it_was_given(self, tmp_path):
        report = self._charted(
            tmp_path, field="title", method="exact", title="How titles fare"
        )
        assert "![How titles fare, exact match, over all" in report

    def test_narrows_to_the_corpora_it_names(self, tmp_path):
        self._charted(tmp_path, field="title", method="exact", corpora=("ore",))
        assert (tmp_path / "title-exact-all.png").exists()

    def test_fails_naming_a_row_the_tables_do_not_show(self, tmp_path):
        with pytest.raises(SelectionError, match="keywords"):
            self._charted(tmp_path, field="keywords", method="exact")


class TestRowFilter:
    def _report(self, row_filter):
        return _rows(_render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(
                fields=("title", "abstract"), row_filter=row_filter,
            ),
        ))

    def test_keeps_only_the_named_method(self):
        rows = self._report({"title": (("exact", ""),)})
        assert [row.split("|")[1].strip() for row in rows] == [
            "title (exact)", "title (exact)", "title (exact)",
        ]

    def test_an_empty_method_keeps_every_one(self):
        rows = self._report({"title": (("", ""),)})
        assert {row.split("|")[1].strip() for row in rows} == {
            "title (exact)", "title (levenshtein)",
        }

    def test_fails_naming_a_scope_no_summary_has(self):
        with pytest.raises(SelectionError, match="gold"):
            self._report({"title": (("exact", "gold"),)})


class TestExpectedTypes:
    def test_passes_where_the_summary_agrees(self):
        report = _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(expected_types={"title": "string"}),
        )
        assert "| title (exact)" in report

    def test_fails_stating_both_types(self):
        with pytest.raises(SelectionError, match="'string'.*'partial_list'"):
            _render_comparison_report(
                [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
                selection=Selection(expected_types={"title": "partial_list"}),
            )


class TestWritingTheReport:
    def test_makes_the_output_directory(self, tmp_path):
        summary = tmp_path / "summary.json"
        summary.write_text(json.dumps(_two_field_summary()))
        run_compare(
            [("base", summary), ("head", summary)], tmp_path / "new/dir/comparison.md",
        )
        assert (tmp_path / "new/dir/comparison.md").exists()


class TestPrimaryColumn:
    """Which column the deltas measure against is stated, not taken from the order."""

    def _labeled(self):
        return [
            ("a", _two_field_summary()),
            ("b", _two_field_summary(0.05)),
            ("c", _two_field_summary(0.10)),
        ]

    def _header(self, report: str) -> list:
        line = next(
            line for line in report.splitlines() if line.startswith("| Field (method)")
        )
        return [cell.strip() for cell in line.split("|")[1:-1]]

    def test_defaults_to_the_last_column(self):
        header = self._header(_render_comparison_report(self._labeled()))
        assert header[-2:] == ["Δ a", "Δ b"]

    def test_keeps_every_column_in_the_order_given(self):
        header = self._header(
            _render_comparison_report(self._labeled(), primary=0)
        )
        assert header[3:6] == ["a", "b", "c"]

    def test_gives_no_delta_column_to_the_primary(self):
        header = self._header(
            _render_comparison_report(self._labeled(), primary=0)
        )
        assert "Δ a" not in header and ["Δ b", "Δ c"] == header[-2:]

    def test_measures_the_deltas_against_the_named_column(self):
        rows = _rows(_render_comparison_report(self._labeled(), primary=0))
        # a is 0.5 on title (levenshtein); b is 0.55, c is 0.60.
        cells = [cell.strip() for cell in rows[1].split("|")[1:-1]]
        assert cells[-2:] == ["-0.050", "-0.100"]

    def test_moving_a_column_does_not_move_the_reference(self):
        labeled = self._labeled()
        moved = [labeled[0], labeled[2], labeled[1]]
        first = _rows(_render_comparison_report(labeled, primary=0))[1]
        second = _rows(_render_comparison_report(moved, primary=0))[1]
        # Same reference, so the same deltas -- in the order the columns now sit.
        assert sorted(first.split("|")[-3:]) == sorted(second.split("|")[-3:])


class TestFieldsChart:
    def _charted(self, tmp_path, rows, **kwargs):
        return _render_comparison_report(
            [("base", _two_field_summary()), ("head", _two_field_summary(0.05))],
            selection=Selection(declared_charts=(FieldsChartConfig(rows=rows, **kwargs),)),
            charts=ChartOutput(out_dir=tmp_path),
        )

    _ROWS = (("title", "levenshtein", "all"), ("abstract", "levenshtein", "all"))

    def test_draws_one_chart_for_the_whole_set(self, tmp_path):
        self._charted(tmp_path, self._ROWS)
        assert len(list(tmp_path.iterdir())) == 1

    def test_names_the_file_after_its_title(self, tmp_path):
        self._charted(tmp_path, self._ROWS, title="Key fields")
        assert (tmp_path / "fields-key-fields.png").exists()

    def test_names_the_file_after_its_fields_without_a_title(self, tmp_path):
        self._charted(tmp_path, self._ROWS)
        assert (tmp_path / "fields-title-abstract.png").exists()

    def test_links_it_from_the_report(self, tmp_path):
        report = self._charted(tmp_path, self._ROWS, title="Key fields")
        assert "![Key fields, matched at 80% edit similarity, over all" in report

    def test_fails_naming_a_row_the_tables_do_not_show(self, tmp_path):
        with pytest.raises(SelectionError, match="keywords"):
            self._charted(
                tmp_path,
                (("keywords", "levenshtein", "all"), ("title", "levenshtein", "all")),
            )

    def test_draws_for_a_single_corpus(self, tmp_path):
        # No corpus axis, so the two-corpus floor the per-corpus charts have
        # does not apply.
        _render_comparison_report(
            [
                ("base", _two_field_summary(corpora=("ore",))),
                ("head", _two_field_summary(0.05, corpora=("ore",))),
            ],
            selection=Selection(declared_charts=(FieldsChartConfig(rows=self._ROWS),)),
            charts=ChartOutput(out_dir=tmp_path),
        )
        assert list(tmp_path.iterdir())


class TestVariantsSection:
    _DESCRIBED = [
        ("GROBID", "`grobid` `0.9.1-crf`, profile `default`"),
        ("ScienceBeam", "`sciencebeam-parser` `main`, profile `grobid_crf`"),
    ]

    def _report(self, **kwargs):
        return _render_comparison_report(
            [("GROBID", _two_field_summary()), ("ScienceBeam", _two_field_summary(0.05))],
            **kwargs,
        )

    def test_says_nothing_where_no_variant_was_described(self):
        assert "What each column is" not in self._report()

    def test_names_the_profile_behind_each_column(self):
        report = self._report(variants=self._DESCRIBED)
        assert "profile `grobid_crf`" in report

    def test_collapses_it(self):
        report = self._report(variants=self._DESCRIBED)
        assert "<summary>What each column is (2 variants)</summary>" in report

    def test_marks_the_column_the_deltas_measure_against(self):
        report = self._report(variants=self._DESCRIBED, primary=0)
        line = next(
            line for line in report.splitlines() if line.startswith("| GROBID |")
        )
        assert "deltas are measured against this" in line

    def test_leaves_the_others_unmarked(self):
        report = self._report(variants=self._DESCRIBED, primary=0)
        line = next(
            line for line in report.splitlines() if line.startswith("| ScienceBeam |")
        )
        assert "deltas are measured against" not in line
