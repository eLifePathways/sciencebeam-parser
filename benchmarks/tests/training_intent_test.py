from __future__ import annotations

from typing import Any, Dict

import pytest

from benchmarks.corpus_source import CorpusConfigError
from benchmarks.training_intent import (
    PairIntent,
    get_declared_pairs,
    group_by_corpus_and_mode,
    validate_intent,
)

SPLIT = "train"


def _config(generate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": {
            "splits": {
                SPLIT: {
                    "ore": {"file": "ore.parquet"},
                    "scielo": {"file": "scielo.parquet"},
                }
            }
        },
        "cc_by_corpora": ["ore", "scielo"],
        "sampling": {
            "smoke": {"ore": 10, "scielo": 10},
            "medium": {"ore": 50, "scielo": 50},
        },
        "generate": generate,
    }


class TestGetDeclaredPairs:
    def test_returns_a_pair_per_declared_model(self):
        pairs = get_declared_pairs(
            _config({"ore": {"segmentation": "smoke", "citation": "medium"}})
        )
        assert pairs == [
            PairIntent(corpus="ore", model="segmentation", mode="smoke"),
            PairIntent(corpus="ore", model="citation", mode="medium"),
        ]

    def test_leaves_out_a_model_declared_none(self):
        pairs = get_declared_pairs(
            _config({"ore": {"segmentation": "smoke", "affiliation-address": "none"}})
        )
        assert [pair.model for pair in pairs] == ["segmentation"]

    def test_leaves_out_a_model_declared_empty(self):
        pairs = get_declared_pairs(_config({"ore": {"affiliation-address": None}}))
        assert not pairs

    def test_returns_nothing_without_a_generate_block(self):
        assert not get_declared_pairs({"cc_by_corpora": ["ore"]})

    def test_rejects_a_corpus_that_is_not_a_mapping_of_models(self):
        with pytest.raises(CorpusConfigError, match="must map each model to a mode"):
            get_declared_pairs(_config({"ore": ["segmentation"]}))


class TestGroupByCorpusAndMode:
    def test_groups_the_models_a_single_run_covers(self):
        grouped = group_by_corpus_and_mode(
            get_declared_pairs(
                _config(
                    {
                        "ore": {
                            "segmentation": "smoke",
                            "header": "smoke",
                            "citation": "medium",
                        },
                        "scielo": {"segmentation": "smoke"},
                    }
                )
            )
        )
        assert grouped == {
            ("ore", "smoke"): ["segmentation", "header"],
            ("ore", "medium"): ["citation"],
            ("scielo", "smoke"): ["segmentation"],
        }


class TestValidateIntent:
    def test_accepts_a_declaration_the_config_defines(self):
        validate_intent(
            _config({"ore": {"segmentation": "smoke", "citation": "medium"}}), SPLIT
        )

    def test_rejects_a_corpus_outside_cc_by_corpora(self):
        config = _config({"plos": {"segmentation": "smoke"}})
        config["dataset"]["splits"][SPLIT]["plos"] = {"file": "plos.parquet"}
        with pytest.raises(CorpusConfigError, match="not in `cc_by_corpora`"):
            validate_intent(config, SPLIT)

    def test_rejects_a_corpus_the_split_does_not_have(self):
        config = _config({"ore": {"segmentation": "smoke"}})
        del config["dataset"]["splits"][SPLIT]["ore"]
        with pytest.raises(CorpusConfigError, match="does not have"):
            validate_intent(config, SPLIT)

    def test_rejects_a_mode_sampling_does_not_define(self):
        with pytest.raises(CorpusConfigError, match="`sampling` does not"):
            validate_intent(_config({"ore": {"segmentation": "enormous"}}), SPLIT)

    def test_rejects_a_mode_that_gives_the_corpus_no_sample_size(self):
        config = _config({"scielo": {"segmentation": "medium"}})
        del config["sampling"]["medium"]["scielo"]
        with pytest.raises(CorpusConfigError, match="no sample size"):
            validate_intent(config, SPLIT)

    def test_rejects_an_unknown_model_when_the_names_are_given(self):
        with pytest.raises(CorpusConfigError, match="generation does not"):
            validate_intent(
                _config({"ore": {"segmentaion": "smoke"}}),
                SPLIT,
                known_model_names=["segmentation", "citation"],
            )

    def test_does_not_check_model_names_when_they_are_not_given(self):
        validate_intent(_config({"ore": {"segmentaion": "smoke"}}), SPLIT)

    def test_ignores_a_model_declared_none_in_an_unlisted_corpus(self):
        validate_intent(_config({"plos": {"segmentation": "none"}}), SPLIT)
