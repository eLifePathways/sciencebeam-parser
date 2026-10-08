from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.corpus_labels import CORPUS_LABELS, corpus_label


class TestCorpusLabel:
    def test_should_name_a_corpus_the_way_a_reader_would(self):
        assert corpus_label("scielo_preprints-jats") == "SciELO Preprints"

    def test_should_capitalise_a_name_the_identifier_lowercases(self):
        assert corpus_label("biorxiv") == "bioRxiv"

    def test_should_expand_an_acronym_the_identifier_hides(self):
        assert corpus_label("ore") == "Open Research Europe"

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
