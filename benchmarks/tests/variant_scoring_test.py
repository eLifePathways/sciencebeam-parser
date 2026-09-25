from __future__ import annotations

from sciencebeam_judge.evaluation.scoring_types.string import STRING_SCORING_TYPE

from benchmarks.variant_scoring import (
    FIRST_VARIANT_SCORING_TYPE,
    MATCHED_VARIANT_INDEX,
    VARIANT_COUNT,
    VARIANTS_SCORING_TYPE,
    select_variant_indices,
)

MEASURES = ["levenshtein", "edit_sim"]

ABSTRACT_1 = "the abstract of the article, as its publisher recorded it"
ABSTRACT_2 = "el resumen del artículo, tal y como lo registró su editorial"


def _score(expected, actual):
    return VARIANTS_SCORING_TYPE.score(expected, actual, measures=MEASURES)


def _without_variant_keys(scores: dict) -> dict:
    return {
        method: {
            key: value for key, value in score.items()
            if key not in (VARIANT_COUNT, MATCHED_VARIANT_INDEX)
        }
        for method, score in scores.items()
    }


class TestSelectVariantIndices:
    def test_should_select_the_only_pair(self):
        assert select_variant_indices([ABSTRACT_1], [ABSTRACT_1]) == (0, 0)

    def test_should_select_the_variant_the_prediction_matches(self):
        assert select_variant_indices([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2]) == (1, 0)

    def test_should_prefer_the_first_variant_where_both_match_as_well(self):
        assert select_variant_indices([ABSTRACT_1, ABSTRACT_1], [ABSTRACT_1]) == (0, 0)

    def test_should_select_the_predicted_value_that_matches_best(self):
        assert select_variant_indices([ABSTRACT_2], [ABSTRACT_1, ABSTRACT_2]) == (0, 1)

    def test_should_select_the_first_pair_where_either_side_is_empty(self):
        assert select_variant_indices([], [ABSTRACT_1]) == (0, 0)
        assert select_variant_indices([ABSTRACT_1], []) == (0, 0)


class TestVariantsScoringType:
    def test_should_score_a_single_variant_exactly_as_a_string(self):
        expected, actual = [ABSTRACT_1], [ABSTRACT_1[:40]]
        assert _without_variant_keys(_score(expected, actual)) == (
            STRING_SCORING_TYPE.score(expected, actual, measures=MEASURES)
        )

    def test_should_score_no_gold_exactly_as_a_string(self):
        expected, actual = [], [ABSTRACT_1]
        assert _without_variant_keys(_score(expected, actual)) == (
            STRING_SCORING_TYPE.score(expected, actual, measures=MEASURES)
        )

    def test_should_credit_a_prediction_matching_the_translation(self):
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2])
        assert scores["edit_sim"]["sim_sum"] == 1.0

    def test_should_record_which_variant_was_credited(self):
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2])
        assert scores["edit_sim"][MATCHED_VARIANT_INDEX] == 1
        assert scores["edit_sim"][VARIANT_COUNT] == 2

    def test_should_record_the_first_variant_where_it_was_credited(self):
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert scores["edit_sim"][MATCHED_VARIANT_INDEX] == 0

    def test_should_not_score_a_concatenation_of_the_variants(self):
        glued = STRING_SCORING_TYPE.score(
            [ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1], measures=MEASURES
        )
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert scores["edit_sim"]["sim_sum"] == 1.0
        assert glued["edit_sim"]["sim_sum"] < 1.0


def _score_first_variant(expected, actual):
    return FIRST_VARIANT_SCORING_TYPE.score(expected, actual, measures=MEASURES)


class TestFirstVariantScoringType:
    def test_should_credit_a_prediction_matching_the_article_own_abstract(self):
        scores = _score_first_variant([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert scores["edit_sim"]["sim_sum"] == 1.0

    def test_should_not_credit_a_prediction_matching_the_translation(self):
        scores = _score_first_variant([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2])
        assert scores["edit_sim"]["sim_sum"] < 0.5

    def test_should_credit_it_wherever_the_prediction_filed_it(self):
        scores = _score_first_variant([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2, ABSTRACT_1])
        assert scores["edit_sim"]["sim_sum"] == 1.0

    def test_should_not_score_a_concatenation_of_the_gold_variants(self):
        scores = _score_first_variant([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1 + ABSTRACT_2])
        assert scores["edit_sim"]["sim_sum"] < 1.0

    def test_should_record_no_variant_match_since_only_one_can_be_credited(self):
        scores = _score_first_variant([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert VARIANT_COUNT not in scores["edit_sim"]
        assert MATCHED_VARIANT_INDEX not in scores["edit_sim"]
