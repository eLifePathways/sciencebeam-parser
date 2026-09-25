from __future__ import annotations

from benchmarks.variant_match import (
    matched_variant_index,
    merge_variant_matches,
    render_variant_match_table,
    summarise_variant_matches,
    variant_match_row,
)


def _document(variant_count: int, matched_index: int = 0) -> dict:
    return {
        "abstract": {
            "scoring_type": "variants",
            "edit_sim": {
                "sim_sum": 1.0,
                "expected_count": 1,
                "predicted_count": 1,
                "variant_count": variant_count,
                "matched_variant_index": matched_index,
            },
        }
    }


class TestSummariseVariantMatches:
    def test_should_count_the_documents_offering_a_choice(self):
        documents = [_document(1), _document(2), _document(3)]
        assert summarise_variant_matches(documents, ["abstract"]) == {
            "abstract": {"n_variants": 2, "n_translation": 0}
        }

    def test_should_count_a_credited_translation(self):
        documents = [_document(2, 1), _document(2, 0), _document(3, 2)]
        assert summarise_variant_matches(documents, ["abstract"]) == {
            "abstract": {"n_variants": 3, "n_translation": 2}
        }

    def test_should_report_nothing_for_a_field_scored_as_a_string(self):
        documents = [{"title": {"scoring_type": "string", "edit_sim": {"sim_sum": 1.0}}}]
        assert summarise_variant_matches(documents, ["title"]) == {}


class TestMergeVariantMatches:
    def test_should_sum_the_counts_of_several_corpora(self):
        merged = merge_variant_matches([
            {"n_variants": 2, "n_translation": 1},
            {"n_variants": 3, "n_translation": 0},
        ])
        assert merged == {"n_variants": 5, "n_translation": 1}

    def test_should_return_none_where_no_corpus_reports_one(self):
        assert merge_variant_matches([None, None]) is None


class TestVariantMatchRow:
    def test_should_return_none_where_no_document_offers_a_choice(self):
        assert variant_match_row("abstract", [None]) is None

    def test_should_report_each_run(self):
        assert variant_match_row("abstract", [
            {"n_variants": 4, "n_translation": 1},
            {"n_variants": 4, "n_translation": 3},
        ]) == ["abstract", "4", "1", "3"]

    def test_should_report_a_run_without_a_count_as_unknown(self):
        assert variant_match_row("abstract", [
            {"n_variants": 4, "n_translation": 1}, None,
        ]) == ["abstract", "4", "1", "unknown"]


class TestRenderVariantMatchTable:
    def test_should_render_nothing_where_no_field_carries_variants(self):
        assert render_variant_match_table({}, ["abstract"]) == []

    def test_should_render_a_row_per_field(self):
        lines = render_variant_match_table(
            {"abstract": {"n_variants": 27, "n_translation": 7}}, ["abstract"]
        )
        assert "| abstract | 27 | 7 |" in lines


class TestMatchedVariantIndex:
    def test_should_return_the_recorded_index(self):
        assert matched_variant_index(_document(2, 1)["abstract"]) == 1

    def test_should_return_zero_for_a_field_scored_as_a_string(self):
        assert matched_variant_index(
            {"scoring_type": "string", "edit_sim": {"sim_sum": 1.0}}
        ) == 0

    def test_should_return_zero_where_the_field_is_absent(self):
        assert matched_variant_index({}) == 0
