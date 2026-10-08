from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.labels import (
    CORPUS_LABELS,
    FIELD_LABELS,
    METHOD_LABELS,
    corpus_label,
    field_label,
    method_label,
)


class TestCorpusLabel:
    def test_should_name_a_corpus_the_way_a_reader_would(self):
        assert corpus_label("scielo_preprints-jats") == "SciELO Preprints"

    def test_should_capitalise_a_name_the_identifier_lowercases(self):
        assert corpus_label("biorxiv") == "bioRxiv"

    def test_should_capitalise_an_acronym_the_identifier_lowercases(self):
        assert corpus_label("ore") == "ORE"

    def test_should_fall_back_to_the_identifier(self):
        assert corpus_label("something-new") == "something-new"

    def test_should_cover_every_corpus_eval_yml_declares(self):
        config = yaml.safe_load(Path("benchmarks/eval.yml").read_text(encoding="utf-8"))
        declared = {
            corpus
            for split in config["dataset"]["splits"].values()
            for corpus in split
        }
        assert not declared - set(CORPUS_LABELS)


class TestFieldLabel:
    def test_should_name_a_field_the_way_a_reader_would(self):
        assert field_label("author_full_names") == "Authors"

    def test_should_keep_a_measure_apart_from_the_field_it_relaxes(self):
        assert field_label("abstract_any_language") == "Abstract (any language)"

    def test_should_fall_back_to_the_identifier(self):
        assert field_label("something_new") == "something_new"

    def test_should_cover_every_field_eval_yml_scores(self):
        config = yaml.safe_load(Path("benchmarks/eval.yml").read_text(encoding="utf-8"))
        assert not set(config["fields"]) - set(FIELD_LABELS)


class TestMethodLabel:
    def test_should_say_what_the_method_does(self):
        assert method_label("exact") == "exact match"

    def test_should_tell_the_two_edit_distances_apart(self):
        # Both run the same normalised edit distance; `edit_sim` strips punctuation and
        # whitespace from both sides first.
        assert method_label("levenshtein") == "edit similarity"
        assert method_label("edit_sim") == "edit similarity, ignoring punctuation"

    def test_should_fall_back_to_the_identifier(self):
        assert method_label("something_new") == "something_new"

    def test_should_cover_every_method_eval_yml_asks_for(self):
        config = yaml.safe_load(Path("benchmarks/eval.yml").read_text(encoding="utf-8"))
        scoring = config["scoring"]
        asked = set(scoring["default_methods"])
        for entry in (scoring.get("per_field") or {}).values():
            asked |= set(entry.get("methods") or [])
        assert not asked - set(METHOD_LABELS)
