"""A scoring method comparing the words of two values rather than their sequence.

Written against sciencebeam-judge's scoring method interface, and registered by
`benchmarks.judge_setup`, so it can move there once it has settled.
"""
from __future__ import annotations

import re

from sciencebeam_judge.evaluation.scoring_methods.scoring_methods import (
    EDIT_DISTANCE_APPROXIMATE_FN_LIST,
    ScoringMethod,
    levenshtein_score,
)
from sciencebeam_judge.utils.distance_matching import (
    Value_Types,
    type_checked_distance_function,
)

TOKEN_SET_EDIT_SIM_METHOD_NAME = "token_set_edit_sim"

TOKEN_SEPARATOR_PATTERN = re.compile(r"[\W_]+")


def sorted_unique_tokens(value: str) -> str:
    """The distinct words of `value`, sorted, so neither order nor repetition counts."""
    return " ".join(sorted(set(TOKEN_SEPARATOR_PATTERN.sub(" ", value).split())))


TOKEN_SET_EDIT_SIM_SCORING_METHOD = ScoringMethod(
    TOKEN_SET_EDIT_SIM_METHOD_NAME,
    type_checked_distance_function(levenshtein_score, Value_Types),
    approximate_scoring_fn_list=EDIT_DISTANCE_APPROXIMATE_FN_LIST,
    preprocessing_fn=sorted_unique_tokens,
    continuous=True,
)
