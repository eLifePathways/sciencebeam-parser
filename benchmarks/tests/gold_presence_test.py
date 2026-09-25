from __future__ import annotations

from benchmarks.gold_presence import (
    gold_records_field,
    has_gold,
    is_split_worth_reporting,
    predicted_value_count,
    produced_counts,
    produced_row,
    summarise_gold_presence,
)


def _continuous(expected_count: int, predicted_count: int = 0) -> dict:
    return {
        "scoring_type": "string",
        "edit_sim": {
            "sim_sum": 0.0,
            "expected_count": expected_count,
            "predicted_count": predicted_count,
        },
    }


def _presence(n: int, n_gold: int, no_gold_docs: int = 0, no_gold_values: int = 0) -> dict:
    return {
        "n": n,
        "n_gold": n_gold,
        "no_gold_docs": no_gold_docs,
        "no_gold_values": no_gold_values,
    }


class TestGoldRecordsField:
    def test_should_return_true_when_gold_has_a_value(self):
        assert gold_records_field(_continuous(1)) is True

    def test_should_return_false_when_gold_has_no_value(self):
        assert gold_records_field(_continuous(0)) is False

    def test_should_ignore_expected_something_of_a_padded_list(self):
        # 40 references without a DOI: the gold list exists and holds no gold value.
        entry = {
            "scoring_type": "partial_ulist",
            "levenshtein": {
                "expected_something": True,
                "binary_expected": 1,
                "true_negative": 40,
            },
            "edit_sim": {"sim_sum": 0.0, "expected_count": 0, "predicted_count": 0},
        }
        assert gold_records_field(entry) is False

    def test_should_return_none_when_no_continuous_method(self):
        entry = {"scoring_type": "string", "levenshtein": {"expected_something": True}}
        assert gold_records_field(entry) is None


class TestPredictedValueCount:
    def test_should_return_predicted_count(self):
        assert predicted_value_count(_continuous(0, predicted_count=7)) == 7

    def test_should_return_zero_when_no_continuous_method(self):
        assert predicted_value_count({"scoring_type": "string", "exact": {"score": 1}}) == 0


class TestSummariseGoldPresence:
    def test_should_count_documents_the_gold_records_the_field_for(self):
        documents = [
            {"acknowledgement": _continuous(1)},
            {"acknowledgement": _continuous(0)},
            {"acknowledgement": _continuous(0)},
        ]
        result = summarise_gold_presence(documents, ["acknowledgement"])
        assert result["acknowledgement"]["n"] == 3
        assert result["acknowledgement"]["n_gold"] == 1

    def test_should_count_documents_and_values_produced_where_gold_records_nothing(self):
        documents = [
            {"body_section_titles": _continuous(0, predicted_count=5)},
            {"body_section_titles": _continuous(0, predicted_count=3)},
            {"body_section_titles": _continuous(0, predicted_count=0)},
        ]
        result = summarise_gold_presence(documents, ["body_section_titles"])
        assert result["body_section_titles"]["no_gold_docs"] == 2
        assert result["body_section_titles"]["no_gold_values"] == 8

    def test_should_not_count_values_produced_where_the_gold_records_the_field(self):
        documents = [{"title": _continuous(1, predicted_count=1)}]
        result = summarise_gold_presence(documents, ["title"])
        assert result["title"]["no_gold_docs"] == 0
        assert result["title"]["no_gold_values"] == 0

    def test_should_omit_a_field_no_method_can_answer_for(self):
        documents = [{"title": {"scoring_type": "string", "exact": {"score": 1}}}]
        assert summarise_gold_presence(documents, ["title"]) == {}

    def test_should_skip_a_document_missing_the_field(self):
        result = summarise_gold_presence([{}, {"title": _continuous(1)}], ["title"])
        assert result["title"]["n"] == 1


class TestHasGold:
    def test_should_be_false_only_where_the_corpus_records_nothing(self):
        assert has_gold(_presence(n=10, n_gold=0)) is False

    def test_should_be_true_where_some_document_has_gold(self):
        assert has_gold(_presence(n=10, n_gold=1)) is True

    def test_should_be_true_where_presence_is_unknown(self):
        assert has_gold(None) is True


class TestIsSplitWorthReporting:
    def test_should_report_where_a_value_was_produced_against_no_gold(self):
        assert is_split_worth_reporting([_presence(10, 8, no_gold_docs=2, no_gold_values=2)])

    def test_should_report_where_the_corpus_records_no_gold_at_all(self):
        assert is_split_worth_reporting([_presence(10, 0)])

    def test_should_not_report_where_every_variant_abstained(self):
        assert not is_split_worth_reporting([_presence(10, 8), _presence(10, 8)])

    def test_should_report_where_any_variant_produced_something(self):
        assert is_split_worth_reporting(
            [_presence(10, 8), _presence(10, 8, no_gold_docs=1, no_gold_values=1)]
        )

    def test_should_not_report_for_a_summary_without_the_split(self):
        assert not is_split_worth_reporting([None])


class TestProducedCounts:
    def test_should_state_documents_and_values(self):
        assert produced_counts(
            _presence(39, 37, no_gold_docs=2, no_gold_values=118)
        ) == "2 docs, 118 values"

    def test_should_use_the_singular_for_one(self):
        assert produced_counts(
            _presence(10, 9, no_gold_docs=1, no_gold_values=1)
        ) == "1 doc, 1 value"

    def test_should_say_none_where_the_variant_abstained(self):
        assert produced_counts(_presence(40, 0)) == "none"

    def test_should_distinguish_a_variant_without_the_split_from_one_that_abstained(self):
        assert produced_counts(None) == "unknown"


class TestProducedRow:
    def test_should_give_the_field_its_no_gold_count_and_a_cell_per_variant(self):
        row = produced_row("reference_title", [
            _presence(39, 37, no_gold_docs=2, no_gold_values=118),
            _presence(39, 37, no_gold_docs=2, no_gold_values=139),
        ])
        assert row == ["reference_title", "2", "2 docs, 118 values", "2 docs, 139 values"]

    def test_should_count_the_no_gold_documents_of_the_variant_that_scored_most(self):
        row = produced_row("acknowledgement", [_presence(20, 2), _presence(40, 5)])
        assert row is not None
        assert row[1] == "35"

    def test_should_return_none_where_the_gold_records_every_document(self):
        assert produced_row("title", [_presence(10, 10)]) is None
