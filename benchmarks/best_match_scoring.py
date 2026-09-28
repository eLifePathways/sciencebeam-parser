"""Scoring a field where either side may hold several values for one right answer.

`best_match` credits the best matching pair, any gold value against any predicted one.
`best_match_from_first` credits the best match the *first* gold value makes with any
predicted one, so a document offering alternatives has one right answer while a prediction
offering several is still read. The mapping cannot express the second on its own, since
one field name is read on both sides: restricting the gold to `abstract` would restrict
the prediction to its own main abstract with it.

Neither type is told what the first gold value means, and neither can find out — the judge
hands a scoring type the values of one field and nothing else — so both are named for the
position they read rather than for what a field puts there.
`variant_xpath.abstract_variants` puts the article's own abstract first for that reason,
building its list from `main_abstract`, and `judge_setup_test` asserts the result through
the mapping. A field mapped without that ordering must not use `best_match_from_first`,
and `matched_expected_index` means nothing for it.

Written against sciencebeam-judge's `ScoringType` interface so it can move there once the
rule has settled.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from sciencebeam_judge.evaluation.normalization import normalize_whitespace
from sciencebeam_judge.evaluation.scoring_methods.scoring_methods import (
    ScoringMethod,
    get_scoring_method,
)
from sciencebeam_judge.evaluation.scoring_types.scoring_type import ScoringType
from sciencebeam_judge.evaluation.scoring_types.string import STRING_SCORING_TYPE

BEST_MATCH_SCORING_TYPE_NAME = "best_match"
BEST_MATCH_FROM_FIRST_SCORING_TYPE_NAME = "best_match_from_first"

N_EXPECTED_VALUES = "n_expected_values"
MATCHED_EXPECTED_INDEX = "matched_expected_index"
MATCHED_CONCATENATION = "matched_concatenation"

SELECTION_METHOD = "edit_sim"


def _preprocessed(method: ScoringMethod, value: str, convert_to_lower: bool) -> str:
    value = normalize_whitespace(value)
    if convert_to_lower:
        value = value.lower()
    return method.preprocessing_fn(value)


def _matches_concatenation_better(
    expected: Sequence[str],
    actual: str,
    best_score: float,
    convert_to_lower: bool,
) -> bool:
    """Whether the prediction is every gold value at once rather than any one of them.

    A prediction holding two languages in a single value can only half match either, which
    reads as a poor extraction rather than as the unsegmented one it is.
    """
    method = get_scoring_method(SELECTION_METHOD)
    concatenated = _preprocessed(method, "".join(expected), convert_to_lower)
    return method.scoring_fn(concatenated, _preprocessed(method, actual, convert_to_lower)) > (
        best_score
    )


def _best_match_score(
    expected: str, actual: str, convert_to_lower: bool
) -> float:
    method = get_scoring_method(SELECTION_METHOD)
    return method.scoring_fn(
        _preprocessed(method, expected, convert_to_lower),
        _preprocessed(method, actual, convert_to_lower),
    )


def select_best_match_indices(
    expected: Sequence[str],
    actual: Sequence[str],
    convert_to_lower: bool = False,
) -> Tuple[int, int]:
    """The gold and predicted values that match each other best, the first gold on a tie."""
    if not expected or not actual or (len(expected) == 1 and len(actual) == 1):
        return 0, 0
    method = get_scoring_method(SELECTION_METHOD)
    expected_values = [_preprocessed(method, value, convert_to_lower) for value in expected]
    actual_values = [_preprocessed(method, value, convert_to_lower) for value in actual]
    return max(
        (
            (expected_index, actual_index)
            for expected_index in range(len(expected_values))
            for actual_index in range(len(actual_values))
        ),
        key=lambda pair: method.scoring_fn(expected_values[pair[0]], actual_values[pair[1]]),
    )


class BestMatchScoringType(ScoringType):
    def __init__(self, first_expected_only: bool = False):
        self.first_expected_only = first_expected_only

    def score(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        expected: Sequence[str],
        actual: Sequence[str],
        include_values: bool = False,
        measures: Optional[List[str]] = None,
        convert_to_lower: bool = False,
    ) -> Dict[str, Any]:
        all_expected = expected
        if self.first_expected_only:
            expected = expected[:1]
        expected_index, actual_index = select_best_match_indices(
            expected, actual, convert_to_lower
        )
        scores = STRING_SCORING_TYPE.score(
            expected[expected_index:expected_index + 1],
            actual[actual_index:actual_index + 1],
            include_values=include_values,
            measures=measures,
            convert_to_lower=convert_to_lower,
        )
        if self.first_expected_only:
            return scores
        extra: Dict[str, Any] = {
            N_EXPECTED_VALUES: len(expected),
            MATCHED_EXPECTED_INDEX: expected_index,
        }
        if len(all_expected) > 1 and actual:
            extra[MATCHED_CONCATENATION] = _matches_concatenation_better(
                all_expected, actual[actual_index],
                _best_match_score(
                    expected[expected_index], actual[actual_index], convert_to_lower
                ),
                convert_to_lower,
            )
        return {
            method: {**score, **extra}
            for method, score in scores.items()
        }


BEST_MATCH_SCORING_TYPE = BestMatchScoringType()
BEST_MATCH_FROM_FIRST_SCORING_TYPE = BestMatchScoringType(first_expected_only=True)
