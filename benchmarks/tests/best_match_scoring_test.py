from __future__ import annotations

from sciencebeam_judge.evaluation.scoring_types.string import STRING_SCORING_TYPE

from benchmarks.best_match_scoring import (
    BEST_MATCH_FROM_FIRST_SCORING_TYPE,
    MATCHED_EXPECTED_INDEX,
    N_EXPECTED_VALUES,
    BEST_MATCH_SCORING_TYPE,
    select_best_match_indices,
)

MEASURES = ["levenshtein", "edit_sim"]

ABSTRACT_1 = "the abstract of the article, as its publisher recorded it"
ABSTRACT_2 = "el resumen del artículo, tal y como lo registró su editorial"


def _score(expected, actual):
    return BEST_MATCH_SCORING_TYPE.score(expected, actual, measures=MEASURES)


def _without_variant_keys(scores: dict) -> dict:
    return {
        method: {
            key: value for key, value in score.items()
            if key not in (N_EXPECTED_VALUES, MATCHED_EXPECTED_INDEX)
        }
        for method, score in scores.items()
    }


class TestSelectVariantIndices:
    def test_should_select_the_only_pair(self):
        assert select_best_match_indices([ABSTRACT_1], [ABSTRACT_1]) == (0, 0)

    def test_should_select_the_variant_the_prediction_matches(self):
        assert select_best_match_indices([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2]) == (1, 0)

    def test_should_prefer_the_first_expected_value_where_both_match_as_well(self):
        assert select_best_match_indices([ABSTRACT_1, ABSTRACT_1], [ABSTRACT_1]) == (0, 0)

    def test_should_select_the_predicted_value_that_matches_best(self):
        assert select_best_match_indices([ABSTRACT_2], [ABSTRACT_1, ABSTRACT_2]) == (0, 1)

    def test_should_select_the_first_pair_where_either_side_is_empty(self):
        assert select_best_match_indices([], [ABSTRACT_1]) == (0, 0)
        assert select_best_match_indices([ABSTRACT_1], []) == (0, 0)


class TestBestMatchScoringType:
    def test_should_score_a_single_variant_exactly_as_a_string(self):
        expected, actual = [ABSTRACT_1], [ABSTRACT_1[:40]]
        assert _without_variant_keys(_score(expected, actual)) == (
            STRING_SCORING_TYPE.score(expected, actual, measures=MEASURES)
        )

    def test_should_score_no_gold_exactly_as_a_string(self):
        expected: list = []
        actual = [ABSTRACT_1]
        assert _without_variant_keys(_score(expected, actual)) == (
            STRING_SCORING_TYPE.score(expected, actual, measures=MEASURES)
        )

    def test_should_credit_a_prediction_matching_the_translation(self):
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2])
        assert scores["edit_sim"]["sim_sum"] == 1.0

    def test_should_record_which_variant_was_credited(self):
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2])
        assert scores["edit_sim"][MATCHED_EXPECTED_INDEX] == 1
        assert scores["edit_sim"][N_EXPECTED_VALUES] == 2

    def test_should_record_the_first_expected_value_where_it_was_credited(self):
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert scores["edit_sim"][MATCHED_EXPECTED_INDEX] == 0

    def test_should_not_score_a_concatenation_of_the_variants(self):
        glued = STRING_SCORING_TYPE.score(
            [ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1], measures=MEASURES
        )
        scores = _score([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert scores["edit_sim"]["sim_sum"] == 1.0
        assert glued["edit_sim"]["sim_sum"] < 1.0


def _score_from_first(expected, actual):
    return BEST_MATCH_FROM_FIRST_SCORING_TYPE.score(expected, actual, measures=MEASURES)


class TestBestMatchFromFirstScoringType:
    def test_should_credit_a_prediction_matching_the_article_own_abstract(self):
        scores = _score_from_first([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert scores["edit_sim"]["sim_sum"] == 1.0

    def test_should_not_credit_a_prediction_matching_the_translation(self):
        scores = _score_from_first([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2])
        assert scores["edit_sim"]["sim_sum"] < 0.5

    def test_should_credit_it_wherever_the_prediction_filed_it(self):
        scores = _score_from_first([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_2, ABSTRACT_1])
        assert scores["edit_sim"]["sim_sum"] == 1.0

    def test_should_not_score_a_concatenation_of_the_gold_variants(self):
        scores = _score_from_first([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1 + ABSTRACT_2])
        assert scores["edit_sim"]["sim_sum"] < 1.0

    def test_should_record_no_variant_match_since_only_one_can_be_credited(self):
        scores = _score_from_first([ABSTRACT_1, ABSTRACT_2], [ABSTRACT_1])
        assert N_EXPECTED_VALUES not in scores["edit_sim"]
        assert MATCHED_EXPECTED_INDEX not in scores["edit_sim"]
