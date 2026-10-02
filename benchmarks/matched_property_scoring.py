"""Scoring a property of each value rather than the value itself.

A field whose values carry a property — the language of an abstract, the type of a
reference — wants that property scored against the gold's, and the two sides have to be
paired by the value before the properties can be compared: position cannot be trusted
when either side may hold a different number of values in a different order.

The mapping composes each value as `property`, a separator, then the text that identifies
it, which the judge's xpath language can already express. This splits the pair, aligns on
the text, and scores the properties. A pair whose gold property is empty is left out: a
publisher that records the language of a translation and not of the article's own
abstract should neither credit nor penalise what was produced for the one it omitted, and
that is a decision per value rather than per document.

The denominator is therefore pairs, not documents, and a document carrying three
properties weighs three times one carrying a single property.

Written against sciencebeam-judge's `ScoringType` interface so it can move there once the
rule has settled. It needs no change to that interface, which is the reason for composing
the pair in the mapping rather than naming a second field in `eval.yml`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from sciencebeam_judge.evaluation.normalization import normalize_whitespace
from sciencebeam_judge.evaluation.scoring_methods.scoring_methods import get_scoring_method
from sciencebeam_judge.evaluation.scoring_types.scoring_type import ScoringType
from sciencebeam_judge.evaluation.scoring_types.string import STRING_SCORING_TYPE

MATCHED_PROPERTY_SCORING_TYPE_NAME = "matched_property"

# The mapping puts this between a value's property and its text. U+2063 is a formatting
# character no document text carries, and unlike the ASCII unit separator it is valid in
# XML, which the judge writes field values back into.
PROPERTY_SEPARATOR = "\u2063"

N_MATCHED_PAIRS = "n_matched_pairs"
N_EXPECTED_PROPERTIES = "n_expected_properties"

SELECTION_METHOD = "edit_sim"

# How alike two values have to be before their properties are treated as describing the
# same thing. Below it the gold value was not extracted at all, and scoring its property
# against whatever is nearest would measure the pairing rather than the property.
MIN_SELECTION_SCORE = 0.5


def split_property(value: str) -> Tuple[str, str]:
    prop, _, text = value.partition(PROPERTY_SEPARATOR)
    return prop.strip(), text


def _selection_score(expected_text: str, actual_text: str) -> float:
    method = get_scoring_method(SELECTION_METHOD)
    return method.scoring_fn(
        method.preprocessing_fn(normalize_whitespace(expected_text)),
        method.preprocessing_fn(normalize_whitespace(actual_text)),
    )


def iter_matched_properties(
    expected: Sequence[str],
    actual: Sequence[str],
) -> List[Tuple[str, str]]:
    """The gold and predicted property of each gold value that declares one and was found."""
    actual_pairs = [split_property(value) for value in actual]
    matched: List[Tuple[str, str]] = []
    for expected_value in expected:
        expected_property, expected_text = split_property(expected_value)
        if not expected_property:
            continue
        best_property, best_score = "", 0.0
        for actual_property, actual_text in actual_pairs:
            selection_score = _selection_score(expected_text, actual_text)
            if selection_score > best_score:
                best_property, best_score = actual_property, selection_score
        if best_score < MIN_SELECTION_SCORE:
            matched.append((expected_property, ""))
            continue
        matched.append((expected_property, best_property))
    return matched


# Counts to add up across pairs, rather than take from the last of them. `score` is
# averaged instead, and the booleans are true where any pair is true.
SUMMED_KEYS = (
    "true_positive", "true_negative", "false_positive", "false_negative",
    "binary_expected", "binary_actual", "sim_sum", "expected_count", "predicted_count",
)


def _combine(pair_scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One score per pair, added up, so a document of three properties is three
    observations rather than one all-or-nothing comparison."""
    combined: Dict[str, Any] = {}
    for key in SUMMED_KEYS:
        values = [score[key] for score in pair_scores if key in score]
        if values:
            combined[key] = sum(values)
    for key in ("expected_something", "actual_something"):
        values = [score[key] for score in pair_scores if key in score]
        if values:
            combined[key] = any(values)
    scores = [score["score"] for score in pair_scores if "score" in score]
    if scores:
        combined["score"] = sum(scores) / len(scores)
    return combined


class MatchedPropertyScoringType(ScoringType):
    def score(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        expected: Sequence[str],
        actual: Sequence[str],
        include_values: bool = False,
        measures: Optional[List[str]] = None,
        convert_to_lower: bool = False,
    ) -> Dict[str, Any]:
        matched = iter_matched_properties(expected, actual)
        if not matched:
            # Every document the gold declares no property for still needs an entry per
            # method, or it cannot be aggregated with the ones that do.
            scores = STRING_SCORING_TYPE.score(
                [], [], include_values=include_values, measures=measures,
                convert_to_lower=convert_to_lower
            )
            return {
                method: {**score, N_MATCHED_PAIRS: 0, N_EXPECTED_PROPERTIES: 0}
                for method, score in scores.items()
            }
        pair_scores_by_method: Dict[str, List[Dict[str, Any]]] = {}
        for expected_property, actual_property in matched:
            for method, score in STRING_SCORING_TYPE.score(
                [expected_property],
                [actual_property] if actual_property else [],
                include_values=include_values,
                measures=measures,
                convert_to_lower=convert_to_lower,
            ).items():
                pair_scores_by_method.setdefault(method, []).append(score)
        scores = {
            method: _combine(pair_scores)
            for method, pair_scores in pair_scores_by_method.items()
        }
        extra = {
            N_MATCHED_PAIRS: len(matched),
            N_EXPECTED_PROPERTIES: sum(
                1 for value in expected if split_property(value)[0]
            ),
        }
        return {method: {**score, **extra} for method, score in scores.items()}


MATCHED_PROPERTY_SCORING_TYPE = MatchedPropertyScoringType()
