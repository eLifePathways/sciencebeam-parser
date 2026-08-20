from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.models.llm.config import LlmEngineConfig
from sciencebeam_parser.models.llm.prompt import get_prompt_template


CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent
    / 'sciencebeam_parser' / 'resources' / 'default_config' / 'config.yml'
)

LLM_MODELS_BY_PROFILE: Dict[str, List[str]] = {
    'llm_reference_segmenter': ['reference_segmenter'],
    'llm_citation': ['citation'],
    'llm_references': ['citation', 'reference_segmenter'],
}

SHAPE_BY_TASK = {'reference_segmenter': 'lines', 'citation': 'values'}


def get_shipped_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))


def get_resolved_models(profile_name: str) -> dict:
    return AppConfig(get_shipped_config()).resolve_profile(profile_name)['models']


class TestShippedDefaults:
    def test_should_not_change_the_default_profile(self):
        assert get_shipped_config()['profile'] == 'grobid_crf'

    def test_should_not_use_the_llm_engine_in_the_default_models(self):
        engines = {
            model_config.get('engine')
            for model_config in get_shipped_config()['models'].values()
            if isinstance(model_config, dict)
        }
        assert 'llm' not in engines


class TestShippedLlmProfiles:
    @pytest.mark.parametrize('profile_name', sorted(LLM_MODELS_BY_PROFILE))
    def test_should_define_the_profile(self, profile_name: str):
        config = get_shipped_config()
        assert profile_name in config['sequence_model_profiles']
        assert config['profiles'][profile_name]['sequence_models'] == profile_name

    @pytest.mark.parametrize('profile_name,expected', sorted(LLM_MODELS_BY_PROFILE.items()))
    def test_should_use_the_llm_engine_for_those_models_only(
        self, profile_name: str, expected: List[str]
    ):
        models = get_resolved_models(profile_name)
        llm_models = sorted(
            name for name, model_config in models.items()
            if isinstance(model_config, dict) and model_config.get('engine') == 'llm'
        )
        assert llm_models == expected

    @pytest.mark.parametrize('profile_name', sorted(LLM_MODELS_BY_PROFILE))
    def test_should_leave_every_other_model_on_the_crf_engine(self, profile_name: str):
        models = get_resolved_models(profile_name)
        llm_models = set(LLM_MODELS_BY_PROFILE[profile_name])
        other_engines = {
            model_config.get('engine')
            for name, model_config in models.items()
            if isinstance(model_config, dict) and name not in llm_models
        }
        assert other_engines == {'wapiti'}

    @pytest.mark.parametrize('profile_name', sorted(LLM_MODELS_BY_PROFILE))
    def test_should_parse_and_reference_a_prompt_that_exists(self, profile_name: str):
        models = get_resolved_models(profile_name)
        for task in LLM_MODELS_BY_PROFILE[profile_name]:
            config = LlmEngineConfig.from_model_config(models[task])
            assert config.task == task
            assert config.response_shape == SHAPE_BY_TASK[task]
            assert config.provider
            assert get_prompt_template(config.task, config.prompt_version)
