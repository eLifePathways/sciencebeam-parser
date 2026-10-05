from __future__ import annotations

import pytest
import yaml

from benchmarks.comparison_config import (
    available_baselines,
    load_comparison,
    parse_comparison,
    resolve_variants,
    to_selection,
)
from benchmarks.report_grid import SelectionError

TWO_VARIANTS = """
variants:
  - {label: grobid, tool: grobid, version: 0.9.1-crf, profile: default}
  - {label: head, current: true}
"""


def _parse(text: str):
    return parse_comparison(yaml.safe_load(text))


class TestVariants:
    def test_should_keep_the_declared_order(self):
        assert [v.label for v in _parse(TWO_VARIANTS).variants] == ["grobid", "head"]

    def test_should_default_the_profile(self):
        config = _parse("""
variants:
  - {label: a, tool: grobid, version: 1}
  - {label: b, current: true}
""")
        assert config.variants[0].profile == "default"

    def test_should_need_at_least_two(self):
        with pytest.raises(SelectionError, match="at least two variants"):
            _parse("variants:\n  - {label: a, current: true}\n")

    def test_should_name_a_column_for_its_coordinates_without_a_label(self):
        config = _parse("""
variants:
  - {tool: sciencebeam-parser, version: main, profile: grobid_crf}
  - {tool: sciencebeam-parser, version: main, profile: llm_all}
""")
        assert [v.label for v in config.variants] == [
            "sciencebeam-parser main (grobid_crf)",
            "sciencebeam-parser main (llm_all)",
        ]

    def test_should_name_the_run_under_test_by_its_profile(self):
        config = _parse(
            "variants:\n  - {current: true, profile: llm_all}\n"
            "  - {tool: grobid, version: 1}\n"
        )
        assert config.variants[0].label == "this run (llm_all)"

    def test_should_keep_a_label_that_was_given(self):
        config = _parse(
            "variants:\n  - {label: mine, tool: grobid, version: 1}\n"
            "  - {current: true}\n"
        )
        assert config.variants[0].label == "mine"

    def test_should_reject_a_variant_naming_nothing(self):
        with pytest.raises(SelectionError, match="exactly one of"):
            _parse("variants:\n  - {label: a}\n  - {label: b, current: true}\n")

    def test_should_reject_a_variant_naming_two_things(self):
        with pytest.raises(SelectionError, match="exactly one of"):
            _parse(
                "variants:\n  - {label: a, current: true, summary: x.json}\n"
                "  - {label: b, current: true}\n"
            )

    def test_should_reject_an_unknown_key(self):
        with pytest.raises(SelectionError, match="profil"):
            _parse(
                "variants:\n  - {label: a, tool: t, version: v, profil: x}\n"
                "  - {label: b, current: true}\n"
            )


class TestRows:
    def test_should_accept_a_single_method(self):
        config = _parse(TWO_VARIANTS + "rows:\n  - {field: title, method: exact}\n")
        assert config.rows[0].methods == ("exact",)

    def test_should_accept_several_methods(self):
        config = _parse(
            TWO_VARIANTS + "rows:\n  - {field: title, methods: [exact, levenshtein]}\n"
        )
        assert config.rows[0].methods == ("exact", "levenshtein")

    def test_should_reject_method_and_methods_together(self):
        with pytest.raises(SelectionError, match="use one"):
            _parse(
                TWO_VARIANTS
                + "rows:\n  - {field: title, method: exact, methods: [exact]}\n"
            )

    def test_should_reject_an_unknown_scope(self):
        with pytest.raises(SelectionError, match="scope"):
            _parse(TWO_VARIANTS + "rows:\n  - {field: title, scope: some}\n")

    def test_should_need_a_field(self):
        with pytest.raises(SelectionError, match="needs a field"):
            _parse(TWO_VARIANTS + "rows:\n  - {method: exact}\n")


class TestToSelection:
    def test_should_take_the_fields_from_the_rows_in_order(self):
        selection = to_selection(_parse(
            TWO_VARIANTS + "rows:\n  - {field: abstract}\n  - {field: title}\n"
        ))
        assert selection.fields == ("abstract", "title")

    def test_should_leave_the_fields_open_without_rows(self):
        assert to_selection(_parse(TWO_VARIANTS)).fields is None

    def test_should_not_constrain_the_scope_by_default(self):
        selection = to_selection(_parse(
            TWO_VARIANTS + "rows:\n  - {field: title, method: exact}\n"
        ))
        assert selection.row_filter == {"title": (("exact", ""),)}

    def test_should_constrain_an_explicit_scope(self):
        selection = to_selection(_parse(
            TWO_VARIANTS + "rows:\n  - {field: title, method: exact, scope: gold}\n"
        ))
        assert selection.row_filter == {"title": (("exact", "gold"),)}

    def test_should_stand_for_every_method_where_none_is_named(self):
        selection = to_selection(_parse(TWO_VARIANTS + "rows:\n  - {field: title}\n"))
        assert selection.row_filter == {"title": (("", ""),)}

    def test_should_carry_an_asserted_type(self):
        selection = to_selection(_parse(
            TWO_VARIANTS + "rows:\n  - {field: title, type: string}\n"
        ))
        assert selection.expected_types == {"title": "string"}

    def test_should_carry_the_corpora(self):
        selection = to_selection(_parse(TWO_VARIANTS + "corpora: [biorxiv, pkp]\n"))
        assert selection.corpora == ("biorxiv", "pkp")

    def test_should_carry_each_declared_chart(self):
        selection = to_selection(_parse(
            TWO_VARIANTS
            + "charts:\n  - {row: {field: title, method: exact}, title: Titles}\n"
        ))
        assert (selection.chart_configs[0].field, selection.chart_configs[0].title) == (
            "title", "Titles",
        )

    def test_should_default_a_chart_to_the_all_documents_row(self):
        selection = to_selection(_parse(
            TWO_VARIANTS + "charts:\n  - {row: {field: title, method: exact}}\n"
        ))
        assert selection.chart_configs[0].scope == "all"

    def test_should_reject_a_chart_without_a_method(self):
        with pytest.raises(SelectionError, match="field and a method"):
            _parse(TWO_VARIANTS + "charts:\n  - {row: {field: title}}\n")


class TestResolveVariants:
    def _config(self):
        return _parse(TWO_VARIANTS)

    def test_should_find_a_named_variant_where_the_run_put_it(self, tmp_path):
        stored = tmp_path / "baselines/grobid/0.9.1-crf/default/train"
        stored.mkdir(parents=True)
        (stored / "summary.json").write_text("{}")
        current = tmp_path / "train"
        current.mkdir()
        (current / "summary.json").write_text("{}")
        resolved = resolve_variants(self._config(), tmp_path, "train", current)
        assert [label for label, _ in resolved] == ["grobid", "head"]

    def test_should_fail_naming_what_it_could_not_find(self, tmp_path):
        with pytest.raises(SelectionError, match="grobid"):
            resolve_variants(self._config(), tmp_path, "train", tmp_path)

    def test_should_say_that_nothing_is_generated(self, tmp_path):
        (tmp_path / "baselines").mkdir()
        with pytest.raises(SelectionError, match="Nothing is generated"):
            resolve_variants(self._config(), tmp_path, "train", tmp_path)

    def test_should_say_when_there_is_no_runs_directory_at_all(self, tmp_path):
        with pytest.raises(SelectionError, match="does not get one"):
            resolve_variants(self._config(), tmp_path / "absent", "train", tmp_path)

    def test_should_list_the_baselines_that_were_scored(self, tmp_path):
        stored = tmp_path / "baselines/grobid/0.9.0-crf/default/train"
        stored.mkdir(parents=True)
        (stored / "summary.json").write_text("{}")
        with pytest.raises(SelectionError, match="grobid/0.9.0-crf/default/train"):
            resolve_variants(self._config(), tmp_path, "train", tmp_path)

    def test_should_say_a_current_variant_needs_a_run_under_test(self, tmp_path):
        stored = tmp_path / "baselines/grobid/0.9.1-crf/default/train"
        stored.mkdir(parents=True)
        (stored / "summary.json").write_text("{}")
        with pytest.raises(SelectionError, match="pass --current-run"):
            resolve_variants(self._config(), tmp_path, "train", None)

    def test_should_take_an_explicit_summary_path(self, tmp_path):
        summary = tmp_path / "given.json"
        summary.write_text("{}")
        config = _parse(
            f"variants:\n  - {{label: a, summary: {summary}}}\n"
            f"  - {{label: b, summary: {summary}}}\n"
        )
        assert [path for _, path in resolve_variants(config, tmp_path, "train")] == [
            summary, summary,
        ]


class TestLoadComparison:
    def test_should_resolve_a_name_under_the_comparisons_directory(self, tmp_path):
        (tmp_path / "models.yml").write_text(TWO_VARIANTS)
        assert load_comparison("models", tmp_path).name == "models"

    def test_should_accept_a_path(self, tmp_path):
        path = tmp_path / "models.yml"
        path.write_text(TWO_VARIANTS)
        assert load_comparison(str(path), tmp_path).name == "models"

    def test_should_say_where_it_looked(self, tmp_path):
        with pytest.raises(SelectionError, match="No comparison file at"):
            load_comparison("nope", tmp_path)


class TestShippedComparisons:
    def test_every_checked_in_comparison_parses(self):
        from pathlib import Path  # pylint: disable=import-outside-toplevel
        paths = sorted(Path("benchmarks/comparisons").glob("*.yml"))
        assert paths, "expected at least one worked example"
        for path in paths:
            assert load_comparison(str(path)).variants


class TestAvailableBaselines:
    def test_should_be_empty_without_a_baselines_directory(self, tmp_path):
        assert not available_baselines(tmp_path)

    def test_should_name_a_scored_baseline_by_its_coordinates(self, tmp_path):
        stored = tmp_path / "baselines/grobid/0.9.0-crf/default/train"
        stored.mkdir(parents=True)
        (stored / "summary.json").write_text("{}")
        assert available_baselines(tmp_path) == ["grobid/0.9.0-crf/default/train"]

    def test_should_skip_a_baseline_that_was_never_scored(self, tmp_path):
        (tmp_path / "baselines/grobid/0.9.0-crf/default/train").mkdir(parents=True)
        assert not available_baselines(tmp_path)
