"""Scoring a field where either side may hold several values for one right answer.

The best matching pair is credited, any gold value against any predicted one, and the
score records which gold value it was. A field that wants one gold value against several
predicted ones says so in `eval.yml` by naming a different mapping entry per side, so
nothing here has to read a position and trust what put the value there.

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
    def score(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        expected: Sequence[str],
        actual: Sequence[str],
        include_values: bool = False,
        measures: Optional[List[str]] = None,
        convert_to_lower: bool = False,
    ) -> Dict[str, Any]:
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
        extra: Dict[str, Any] = {
            N_EXPECTED_VALUES: len(expected),
            MATCHED_EXPECTED_INDEX: expected_index,
        }
        if len(expected) > 1 and actual:
            extra[MATCHED_CONCATENATION] = _matches_concatenation_better(
                expected, actual[actual_index],
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
