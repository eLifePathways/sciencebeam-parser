"""Scoring a field whose gold carries the same value in more than one language.

The gold values are the variants a document carries, the first of them the article's own,
and a prediction may carry several values of its own. The best matching pair of the two is
credited, and the score records which gold variant that was. `main_variant` restricts the
gold to its first value and leaves the prediction side alone, which the mapping cannot
express: the judge reads one field name on both sides, so `abstract` limits the prediction
as well.

That first value is the article's own because the field's mapping puts it there, which is
a contract between the two rather than something either can check alone. For `abstracts`
it is `variant_xpath.abstract_variants`, and `judge_setup_test` asserts that the first of
`abstracts` is what `abstract` reads alone. A field mapped without that ordering must not
use these types. Written against sciencebeam-judge's `ScoringType` interface so it can move there
once the rule has settled.
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

SCORING_TYPE_NAME = "variants"
MAIN_VARIANT_SCORING_TYPE_NAME = "main_variant"

VARIANT_COUNT = "variant_count"
MATCHED_VARIANT_INDEX = "matched_variant_index"

SELECTION_METHOD = "edit_sim"


def _preprocessed(method: ScoringMethod, value: str, convert_to_lower: bool) -> str:
    value = normalize_whitespace(value)
    if convert_to_lower:
        value = value.lower()
    return method.preprocessing_fn(value)


def select_variant_indices(
    expected: Sequence[str],
    actual: Sequence[str],
    convert_to_lower: bool = False,
) -> Tuple[int, int]:
    """The gold and predicted values that match each other best, the main variant on a tie."""
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


class VariantsScoringType(ScoringType):
    def __init__(self, main_only: bool = False):
        self.main_only = main_only

    def score(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        expected: Sequence[str],
        actual: Sequence[str],
        include_values: bool = False,
        measures: Optional[List[str]] = None,
        convert_to_lower: bool = False,
    ) -> Dict[str, Any]:
        if self.main_only:
            expected = expected[:1]
        expected_index, actual_index = select_variant_indices(
            expected, actual, convert_to_lower
        )
        scores = STRING_SCORING_TYPE.score(
            expected[expected_index:expected_index + 1],
            actual[actual_index:actual_index + 1],
            include_values=include_values,
            measures=measures,
            convert_to_lower=convert_to_lower,
        )
        if self.main_only:
            return scores
        return {
            method: {
                **score,
                VARIANT_COUNT: len(expected),
                MATCHED_VARIANT_INDEX: expected_index,
            }
            for method, score in scores.items()
        }


VARIANTS_SCORING_TYPE = VariantsScoringType()
MAIN_VARIANT_SCORING_TYPE = VariantsScoringType(main_only=True)
