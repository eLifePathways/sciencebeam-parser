from __future__ import annotations

from typing import Optional

import pytest

from benchmarks.report_grid import (
    Selection,
    SelectionError,
    build_grid,
    resolve_corpora,
    resolve_fields,
    resolve_measures,
)

MEASURES = {"title": ["exact", "levenshtein"], "abstract": ["levenshtein"]}


def _summary(fields=None, measures=None, corpora=None) -> dict:
    return {
        "fields": fields if fields is not None else ["title", "abstract"],
        "field_measures": measures if measures is not None else MEASURES,
        "corpora": {name: {"n": 1} for name in (corpora or ["biorxiv", "ore"])},
    }


def _labeled(*summaries) -> list:
    return [(f"run{index}", summary) for index, summary in enumerate(summaries)]


def _f1(_summary_: dict, field: str, method: str) -> Optional[float]:
    return {"title": 0.5, "abstract": 0.25}[field] + (0.1 if method == "exact" else 0)


def _gold_f1(_summary_: dict, field: str, method: str) -> Optional[float]:
    value = _f1(_summary_, field, method)
    return value + 0.05 if value is not None else None


class TestBuildGrid:
    def test_should_give_one_row_per_field_and_method(self):
        rows = build_grid(_labeled(_summary()), ["title", "abstract"], MEASURES, _f1)
        assert [(row.field, row.method) for row in rows] == [
            ("title", "exact"), ("title", "levenshtein"), ("abstract", "levenshtein"),
        ]

    def test_should_keep_the_order_the_fields_were_given_in(self):
        rows = build_grid(_labeled(_summary()), ["abstract", "title"], MEASURES, _f1)
        assert [row.field for row in rows][0] == "abstract"

    def test_should_carry_a_value_per_summary(self):
        rows = build_grid(_labeled(_summary(), _summary()), ["abstract"], MEASURES, _f1)
        assert rows[0].values == (0.25, 0.25)

    def test_should_report_a_bare_document_count_where_the_field_does_not_split(self):
        rows = build_grid(
            _labeled(_summary()), ["abstract"], MEASURES, _f1, _gold_f1,
            lambda field: (40, None),
        )
        assert [row.docs_label for row in rows] == ["40"]

    def test_should_add_a_gold_row_where_the_field_splits(self):
        rows = build_grid(
            _labeled(_summary()), ["abstract"], MEASURES, _f1, _gold_f1,
            lambda field: (40, 12),
        )
        assert [(row.scope, row.docs_label) for row in rows] == [
            ("all", "all 40"), ("gold", "gold 12"),
        ]

    def test_should_score_the_gold_row_with_the_gold_getter(self):
        rows = build_grid(
            _labeled(_summary()), ["abstract"], MEASURES, _f1, _gold_f1,
            lambda field: (40, 12),
        )
        assert (rows[0].values, rows[1].values) == ((0.25,), (0.3,))

    def test_should_scope_an_unsplit_row_as_all_so_a_chart_can_find_it(self):
        rows = build_grid(_labeled(_summary()), ["abstract"], MEASURES, _f1)
        assert (rows[0].scope, rows[0].split) == ("all", False)


class TestResolveFields:
    def test_should_default_to_the_primary_summary(self):
        labeled = _labeled(_summary(fields=["title", "keywords"]), _summary(fields=["title"]))
        assert resolve_fields(labeled, Selection()) == ["title"]

    def test_should_keep_the_order_asked_for(self):
        labeled = _labeled(_summary())
        assert resolve_fields(labeled, Selection(fields=("abstract", "title"))) == [
            "abstract", "title",
        ]

    def test_should_accept_a_field_only_a_baseline_scored(self):
        labeled = _labeled(_summary(fields=["title", "keywords"]), _summary(fields=["title"]))
        assert resolve_fields(labeled, Selection(fields=("keywords",))) == ["keywords"]

    def test_should_reject_a_field_no_summary_scored(self):
        with pytest.raises(SelectionError, match="keywrods"):
            resolve_fields(_labeled(_summary()), Selection(fields=("keywrods",)))

    def test_should_name_what_was_scored_in_the_error(self):
        with pytest.raises(SelectionError, match="'title', 'abstract'"):
            resolve_fields(_labeled(_summary()), Selection(fields=("nope",)))

    def test_should_validate_a_charted_field_too(self):
        with pytest.raises(SelectionError, match="nope"):
            resolve_fields(_labeled(_summary()), Selection(charts=("nope",)))

    def test_should_reject_charting_a_field_the_table_leaves_out(self):
        with pytest.raises(SelectionError, match="without showing it"):
            resolve_fields(
                _labeled(_summary()), Selection(fields=("title",), charts=("abstract",))
            )

    def test_should_allow_charting_a_field_that_is_shown(self):
        assert resolve_fields(
            _labeled(_summary()), Selection(fields=("title",), charts=("title",))
        ) == ["title"]


class TestResolveMeasures:
    def test_should_default_to_every_method_the_field_carries(self):
        measures = resolve_measures(_labeled(_summary()), ["title"], Selection())
        assert measures == {"title": ["exact", "levenshtein"]}

    def test_should_narrow_to_the_methods_asked_for(self):
        measures = resolve_measures(
            _labeled(_summary()), ["title"], Selection(methods=("levenshtein",))
        )
        assert measures == {"title": ["levenshtein"]}

    def test_should_keep_the_fields_own_method_order(self):
        measures = resolve_measures(
            _labeled(_summary()), ["title"], Selection(methods=("levenshtein", "exact"))
        )
        assert measures == {"title": ["exact", "levenshtein"]}

    def test_should_use_the_primarys_list_where_the_summaries_disagree(self):
        labeled = _labeled(
            _summary(measures={"title": ["exact", "levenshtein"]}),
            _summary(measures={"title": ["levenshtein"]}),
        )
        assert resolve_measures(labeled, ["title"], Selection()) == {
            "title": ["levenshtein"],
        }

    def test_should_reach_a_method_only_a_baseline_measured_when_asked(self):
        labeled = _labeled(
            _summary(measures={"title": ["exact", "levenshtein"]}),
            _summary(measures={"title": ["levenshtein"]}),
        )
        assert resolve_measures(labeled, ["title"], Selection(methods=("exact",))) == {
            "title": ["exact"],
        }

    def test_should_reject_a_method_no_selected_field_measures(self):
        with pytest.raises(SelectionError, match="levenshtien"):
            resolve_measures(
                _labeled(_summary()), ["title"],
                Selection(methods=("levenshtein", "levenshtien")),
            )

    def test_should_reject_a_selected_field_left_without_any_method(self):
        with pytest.raises(SelectionError, match="abstract"):
            resolve_measures(
                _labeled(_summary()), ["title", "abstract"], Selection(methods=("exact",))
            )


class TestResolveCorpora:
    def test_should_default_to_the_primary_summary(self):
        labeled = _labeled(_summary(corpora=["pkp"]), _summary(corpora=["biorxiv"]))
        assert resolve_corpora(labeled, Selection()) == ["biorxiv"]

    def test_should_keep_the_order_asked_for(self):
        labeled = _labeled(_summary(corpora=["biorxiv", "ore", "pkp"]))
        assert resolve_corpora(labeled, Selection(corpora=("pkp", "biorxiv"))) == [
            "pkp", "biorxiv",
        ]

    def test_should_reject_a_corpus_no_summary_scored(self):
        with pytest.raises(SelectionError, match="biorxv"):
            resolve_corpora(_labeled(_summary()), Selection(corpora=("biorxv",)))
