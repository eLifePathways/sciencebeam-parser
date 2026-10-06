from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest

from benchmarks.generate_training_data_cli import clear_pair, main, select_pairs
from benchmarks.training_intent import PairIntent

CONFIG = """\
dataset:
  repo_id: org/repo
  revision: main
  splits:
    train:
      ore:
        file: ore/train.parquet
        id_column: id
      scielo:
        file: scielo/train.parquet
        id_column: id

cc_by_corpora:
  - ore
  - scielo

sampling:
  smoke:
    ore: 10
    scielo: 10
  medium:
    ore: 50
    scielo: 50

seeds:
  sample: 42

generate:
  ore:
    segmentation: smoke
    header: smoke
    citation: medium
    affiliation-address: none
  scielo:
    segmentation: smoke
"""


@pytest.fixture(name="config_file")
def _config_file(tmp_path: Path) -> Path:
    config_file = tmp_path / "training-source.yml"
    config_file.write_text(textwrap.dedent(CONFIG), encoding="utf-8")
    return config_file


def _make_source_trees(tmp_path: Path) -> None:
    """What a fetch would have left on disk for the declared pairs."""
    for mode, corpora in [("smoke", ["ore", "scielo"]), ("medium", ["ore"])]:
        for corpus in corpora:
            (tmp_path / "source" / mode / "train" / corpus).mkdir(parents=True)


def _run(config_file: Path, tmp_path: Path, *extra: str):
    _make_source_trees(tmp_path)
    with patch(
        "benchmarks.generate_training_data_cli.fetch_training_source"
    ) as mock_fetch, patch(
        "benchmarks.generate_training_data_cli.generate_from_tree_main"
    ) as mock_generate:
        main([
            "--config", str(config_file),
            "--source-root", str(tmp_path / "source"),
            "--output-path", str(tmp_path / "out"),
            *extra,
        ])
    return mock_fetch, mock_generate


class TestSelectPairs:
    PAIRS = [
        PairIntent("ore", "segmentation", "smoke"),
        PairIntent("ore", "citation", "medium"),
        PairIntent("scielo", "segmentation", "smoke"),
    ]

    def test_covers_everything_when_nothing_is_named(self):
        assert select_pairs(self.PAIRS, None, None) == self.PAIRS

    def test_narrows_to_a_corpus(self):
        assert select_pairs(self.PAIRS, ["scielo"], None) == [self.PAIRS[2]]

    def test_narrows_to_a_model(self):
        assert [pair.corpus for pair in select_pairs(self.PAIRS, None, ["segmentation"])] == [
            "ore",
            "scielo",
        ]

    def test_narrows_by_both(self):
        assert select_pairs(self.PAIRS, ["ore"], ["citation"]) == [self.PAIRS[1]]


class TestClearPair:
    def test_removes_what_generation_wrote(self, tmp_path: Path):
        pair = tmp_path / "segmentation"
        (pair / "corpus" / "tei").mkdir(parents=True)
        (pair / "corpus" / "tei" / "a.tei.xml").write_text("<x/>", encoding="utf-8")
        (pair / "quality").mkdir()
        (pair / "quality" / "a.segmentation.quality.json").write_text("{}", encoding="utf-8")
        (pair / "provenance.json").write_text("{}\n", encoding="utf-8")

        clear_pair(pair)

        assert not (pair / "corpus").exists()
        assert not (pair / "quality").exists()
        assert not (pair / "provenance.json").exists()

    def test_leaves_what_a_person_wrote(self, tmp_path: Path):
        pair = tmp_path / "segmentation"
        (pair / "verdicts").mkdir(parents=True)
        (pair / "verdicts" / "a.json").write_text("{}", encoding="utf-8")
        (pair / "corpus").mkdir()

        clear_pair(pair)

        assert (pair / "verdicts" / "a.json").is_file()

    def test_is_quiet_about_a_pair_that_does_not_exist_yet(self, tmp_path: Path):
        clear_pair(tmp_path / "never-generated")


class TestRegenerate:
    def test_fetches_each_declared_mode_for_the_corpora_that_want_it(
        self, config_file: Path, tmp_path: Path
    ):
        mock_fetch, _ = _run(config_file, tmp_path)

        by_mode = {
            call.args[1]: (call.args[3], call.kwargs["include"])
            for call in mock_fetch.call_args_list
        }
        assert set(by_mode) == {"smoke", "medium"}
        assert by_mode["smoke"][1] == ["ore", "scielo"]
        assert by_mode["medium"][1] == ["ore"]
        assert by_mode["smoke"][0] == tmp_path / "source" / "smoke"

    def test_generates_one_run_per_corpus_and_mode(
        self, config_file: Path, tmp_path: Path
    ):
        _, mock_generate = _run(config_file, tmp_path)

        runs = []
        for call in mock_generate.call_args_list:
            argv = call.args[0]
            models = argv[argv.index("--models") + 1:]
            runs.append((argv[argv.index("--source-data") + 1], tuple(models)))
        assert sorted(runs) == sorted([
            (str(tmp_path / "source" / "smoke"), ("segmentation", "header")),
            (str(tmp_path / "source" / "medium"), ("citation",)),
            (str(tmp_path / "source" / "smoke"), ("segmentation",)),
        ])

    def test_never_generates_a_model_declared_none(
        self, config_file: Path, tmp_path: Path
    ):
        _, mock_generate = _run(config_file, tmp_path)

        every_argv = [arg for call in mock_generate.call_args_list for arg in call.args[0]]
        assert "affiliation-address" not in every_argv

    def test_clears_a_declared_pair_before_generating_it(
        self, config_file: Path, tmp_path: Path
    ):
        stale = tmp_path / "out" / "train" / "ore" / "citation" / "corpus"
        stale.mkdir(parents=True)
        (stale / "old.tei.xml").write_text("<x/>", encoding="utf-8")

        _run(config_file, tmp_path)

        assert not stale.exists()

    def test_leaves_an_undeclared_pair_alone(self, config_file: Path, tmp_path: Path):
        undeclared = tmp_path / "out" / "train" / "ore" / "affiliation-address" / "corpus"
        undeclared.mkdir(parents=True)
        (undeclared / "kept.tei.xml").write_text("<x/>", encoding="utf-8")

        _run(config_file, tmp_path)

        assert (undeclared / "kept.tei.xml").is_file()

    def test_narrowing_fetches_and_generates_only_what_was_named(
        self, config_file: Path, tmp_path: Path
    ):
        mock_fetch, mock_generate = _run(
            config_file, tmp_path, "--corpus", "ore", "--model", "citation"
        )

        assert [call.args[1] for call in mock_fetch.call_args_list] == ["medium"]
        assert mock_generate.call_count == 1

    def test_skip_fetch_generates_from_what_is_on_disk(
        self, config_file: Path, tmp_path: Path
    ):
        mock_fetch, mock_generate = _run(config_file, tmp_path, "--skip-fetch")

        mock_fetch.assert_not_called()
        assert mock_generate.call_count == 3

    def test_forwards_extra_arguments_to_generation(
        self, config_file: Path, tmp_path: Path
    ):
        _, mock_generate = _run(config_file, tmp_path, "--num-workers", "4")

        argv = mock_generate.call_args_list[0].args[0]
        assert argv[argv.index("--num-workers") + 1] == "4"

    def test_refuses_a_mode_the_config_does_not_define(
        self, config_file: Path, tmp_path: Path
    ):
        config_file.write_text(
            config_file.read_text(encoding="utf-8").replace(
                "segmentation: smoke", "segmentation: enormous", 1
            ),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="`sampling` does not"):
            _run(config_file, tmp_path)

    def test_refuses_a_model_generation_does_not_produce(
        self, config_file: Path, tmp_path: Path
    ):
        config_file.write_text(
            config_file.read_text(encoding="utf-8").replace(
                "segmentation: smoke", "segmentaion: smoke", 1
            ),
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="generation does not"):
            _run(config_file, tmp_path)

    def test_a_failed_group_exits_non_zero_and_the_rest_still_run(
        self, config_file: Path, tmp_path: Path
    ):
        _make_source_trees(tmp_path)

        def _side_effect(argv):
            if "citation" in argv:
                raise RuntimeError("boom")

        with patch("benchmarks.generate_training_data_cli.fetch_training_source"), patch(
            "benchmarks.generate_training_data_cli.generate_from_tree_main",
            side_effect=_side_effect,
        ) as mock_generate:
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--config", str(config_file),
                    "--source-root", str(tmp_path / "source"),
                    "--output-path", str(tmp_path / "out"),
                ])

        assert exc_info.value.code == 1
        assert mock_generate.call_count == 3


class TestOneCorpusPerRun:
    def test_each_run_names_the_corpus_it_is_for(self, config_file: Path, tmp_path: Path):
        """Generation iterates every allowed corpus in the source tree it is given.

        Both corpora are fetched into the smoke tree, so a run that did not name
        its corpus would generate each pair twice -- the second time over a pair
        the rebuild had already cleared and filled.
        """
        _, mock_generate = _run(config_file, tmp_path)

        pairs: List[Tuple[str, str]] = []
        for call in mock_generate.call_args_list:
            argv = call.args[0]
            assert argv.count("--corpus") == 1
            corpus = argv[argv.index("--corpus") + 1]
            models = argv[argv.index("--models") + 1:]
            pairs.extend((corpus, model) for model in models)

        assert len(pairs) == len(set(pairs))
        assert sorted(pairs) == [
            ("ore", "citation"),
            ("ore", "header"),
            ("ore", "segmentation"),
            ("scielo", "segmentation"),
        ]


class TestMissingSource:
    def test_does_not_clear_a_pair_it_cannot_rebuild(
        self, config_file: Path, tmp_path: Path
    ):
        """Clearing comes first, so a missing source must stop before it.

        Otherwise a rebuild against a source tree that was never fetched empties
        every declared pair and generates nothing into them.
        """
        existing = tmp_path / "out" / "train" / "ore" / "citation" / "corpus"
        existing.mkdir(parents=True)
        (existing / "kept.tei.xml").write_text("<x/>", encoding="utf-8")

        with patch(
            "benchmarks.generate_training_data_cli.generate_from_tree_main"
        ) as mock_generate:
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "--config", str(config_file),
                    "--source-root", str(tmp_path / "never-fetched"),
                    "--output-path", str(tmp_path / "out"),
                    "--skip-fetch",
                ])

        assert exc_info.value.code == 1
        assert (existing / "kept.tei.xml").is_file()
        mock_generate.assert_not_called()


class TestKeepingWhatCannotBeRebuilt:
    def test_keeps_the_data_of_a_document_whose_source_has_gone(self, tmp_path: Path):
        """A rebuild is not a reason to lose a document the dataset stopped carrying.

        It cannot be generated again, so clearing the pair would be the only thing
        that removed it -- and it may have been reviewed.
        """
        pair = tmp_path / "segmentation"
        (pair / "corpus" / "tei").mkdir(parents=True)
        (pair / "corpus" / "tei" / "gone.segmentation.tei.xml").write_text(
            "<x/>", encoding="utf-8"
        )
        (pair / "corpus" / "tei" / "here.segmentation.tei.xml").write_text(
            "<x/>", encoding="utf-8"
        )
        (pair / "quality").mkdir()
        (pair / "quality" / "gone.segmentation.quality.json").write_text(
            "{}", encoding="utf-8"
        )
        (pair / "quality" / "here.segmentation.quality.json").write_text(
            "{}", encoding="utf-8"
        )

        clear_pair(pair, ["gone"])

        assert (pair / "corpus" / "tei" / "gone.segmentation.tei.xml").is_file()
        assert (pair / "quality" / "gone.segmentation.quality.json").is_file()
        assert not (pair / "corpus" / "tei" / "here.segmentation.tei.xml").exists()
        assert not (pair / "quality" / "here.segmentation.quality.json").exists()

    def test_clears_everything_when_nothing_has_gone(self, tmp_path: Path):
        pair = tmp_path / "segmentation"
        (pair / "corpus" / "tei").mkdir(parents=True)
        (pair / "corpus" / "tei" / "a.segmentation.tei.xml").write_text("<x/>", encoding="utf-8")

        clear_pair(pair, [])

        assert not (pair / "corpus").exists()

    def test_a_document_is_not_kept_by_another_whose_id_starts_the_same(
        self, tmp_path: Path
    ):
        pair = tmp_path / "segmentation"
        (pair / "corpus" / "tei").mkdir(parents=True)
        (pair / "corpus" / "tei" / "2-114_v1.segmentation.tei.xml").write_text(
            "<x/>", encoding="utf-8"
        )
        (pair / "corpus" / "tei" / "2-114_v10.segmentation.tei.xml").write_text(
            "<x/>", encoding="utf-8"
        )

        clear_pair(pair, ["2-114_v1"])

        assert (pair / "corpus" / "tei" / "2-114_v1.segmentation.tei.xml").is_file()
        assert not (pair / "corpus" / "tei" / "2-114_v10.segmentation.tei.xml").exists()
