import pytest
import yaml

from sciencebeam_parser.training.quality.requirement import (
    Requirement,
    RequirementConfigError,
    get_requirement,
    get_requirement_by_corpus_and_model,
    load_requirement_by_corpus_and_model
)


ORE = 'ore'
SEGMENTATION = 'segmentation'


CONFIG_JSON = {
    'require': {
        ORE: {SEGMENTATION: Requirement.ACCEPTED, 'header': Requirement.MEASURED},
        'scielo_preprints-jats': {SEGMENTATION: Requirement.NOT_REJECTED},
    }
}


class TestGetRequirementByCorpusAndModel:
    def test_should_read_every_declared_pair(self):
        requirement_by_pair = get_requirement_by_corpus_and_model(CONFIG_JSON)
        assert requirement_by_pair[(ORE, SEGMENTATION)] == Requirement.ACCEPTED
        assert requirement_by_pair[(ORE, 'header')] == Requirement.MEASURED
        assert requirement_by_pair[
            ('scielo_preprints-jats', SEGMENTATION)
        ] == Requirement.NOT_REJECTED

    def test_should_read_a_config_declaring_nothing_as_declaring_nothing(self):
        assert not get_requirement_by_corpus_and_model({'generate': {}})

    def test_should_refuse_a_requirement_the_gate_does_not_know(self):
        with pytest.raises(RequirementConfigError) as exc_info:
            get_requirement_by_corpus_and_model(
                {'require': {ORE: {SEGMENTATION: 'reviewed'}}}
            )
        assert 'reviewed' in str(exc_info.value)

    def test_should_refuse_a_block_that_is_not_per_corpus(self):
        with pytest.raises(RequirementConfigError):
            get_requirement_by_corpus_and_model({'require': [SEGMENTATION]})

    def test_should_refuse_a_corpus_that_is_not_per_model(self):
        with pytest.raises(RequirementConfigError):
            get_requirement_by_corpus_and_model({'require': {ORE: Requirement.ACCEPTED}})

    def test_should_read_it_from_a_file(self, tmp_path):
        config_file_path = tmp_path / 'training-source.yml'
        config_file_path.write_text(yaml.safe_dump(CONFIG_JSON), encoding='utf-8')
        assert load_requirement_by_corpus_and_model(
            str(config_file_path)
        )[(ORE, SEGMENTATION)] == Requirement.ACCEPTED


class TestGetRequirement:
    def test_should_demand_nothing_where_no_corpus_declares_anything(self):
        # Assembly is run over corpora that carry no declaration of their own.
        assert get_requirement(None, ORE, SEGMENTATION) == Requirement.MEASURED

    def test_should_give_what_the_pair_declares(self):
        assert get_requirement(
            get_requirement_by_corpus_and_model(CONFIG_JSON), ORE, SEGMENTATION
        ) == Requirement.ACCEPTED

    def test_should_refuse_a_pair_the_declaration_leaves_out(self):
        # A corpus that has declared some pairs has said which it has an opinion
        # about, so a missing one is an omission rather than a decision.
        with pytest.raises(RequirementConfigError) as exc_info:
            get_requirement(
                get_requirement_by_corpus_and_model(CONFIG_JSON), ORE, 'citation'
            )
        assert 'citation' in str(exc_info.value)

    def test_should_refuse_a_corpus_the_declaration_does_not_name(self):
        with pytest.raises(RequirementConfigError):
            get_requirement(
                get_requirement_by_corpus_and_model(CONFIG_JSON), 'other', SEGMENTATION
            )
