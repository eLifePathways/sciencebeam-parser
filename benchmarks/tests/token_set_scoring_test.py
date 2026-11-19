from __future__ import annotations

from benchmarks.token_set_scoring import (
    TOKEN_SET_EDIT_SIM_SCORING_METHOD,
    sorted_unique_tokens,
)


def _score(expected: str, actual: str) -> float:
    return TOKEN_SET_EDIT_SIM_SCORING_METHOD.distance_measure(expected, actual)


class TestSortedUniqueTokens:
    def test_should_sort_the_words(self):
        assert sorted_unique_tokens("gamma alpha beta") == "alpha beta gamma"

    def test_should_keep_one_of_each_word(self):
        assert sorted_unique_tokens("alpha beta alpha") == "alpha beta"

    def test_should_split_on_punctuation(self):
        assert sorted_unique_tokens("alpha, beta. gamma") == "alpha beta gamma"

    def test_should_keep_accents(self):
        assert sorted_unique_tokens("Fundación") == "Fundación"


class TestTokenSetEditSim:
    def test_should_ignore_the_order_of_the_words(self):
        assert _score(
            "Department of Imaginary Studies, University of Nowhere",
            "University of Nowhere, Department of Imaginary Studies",
        ) == 1.0

    def test_should_ignore_a_repeated_phrase(self):
        assert _score(
            "University of Nowhere, Department of Imaginary Studies,"
            " University of Nowhere, Nowhere",
            "University of Nowhere, Department of Imaginary Studies, Nowhere",
        ) == 1.0

    def test_should_score_a_misspelt_word_in_part(self):
        score = _score("University of Nowhere", "Universty of Nowhere")
        assert 0.8 < score < 1.0

    def test_should_score_an_unrelated_value_low(self):
        assert _score("University of Nowhere", "Institute of Elsewhere") < 0.5
