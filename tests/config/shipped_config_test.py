from pathlib import Path
from typing import Dict

import pytest
import yaml

from sciencebeam_parser.config.config import AppConfig


CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / 'sciencebeam_parser' / 'resources' / 'default_config' / 'config.yml'
)

SHARED_BASE_PROFILE = 'grobid_crf_0_9_0'

# Profiles combining single-model profiles, and the profile each model comes from.
COMBINED_PROFILES: Dict[str, Dict[str, str]] = {
    'wapiti_scielo_preprints_ore': {
        'reference_segmenter': 'wapiti_refseg_scielo_preprints_ore',
        'citation': 'wapiti_citation_scielo_preprints_ore',
    },
    'llm_references': {
        'reference_segmenter': 'llm_reference_segmenter',
        'citation': 'llm_citation',
    },
    'llm_all': {
        'segmentation': 'llm_segmentation',
        'reference_segmenter': 'llm_reference_segmenter',
        'citation': 'llm_citation',
    },
}

COMBINED_MODEL_PARAMS = [
    (combined_profile_name, model_name, profile_name)
    for combined_profile_name, profile_name_by_model in sorted(COMBINED_PROFILES.items())
    for model_name, profile_name in sorted(profile_name_by_model.items())
]

COMPOSED_MODEL_PARAMS = sorted({
    (model_name, profile_name)
    for _, model_name, profile_name in COMBINED_MODEL_PARAMS
})


def get_shipped_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))


def get_resolved_models(profile_name: str) -> dict:
    return AppConfig(get_shipped_config()).resolve_profile(profile_name)['models']


class TestShippedCombinedProfiles:
    def test_should_cover_every_profile_that_composes_others(self):
        composing_profile_names = {
            profile_name
            for profile_name, profile in get_shipped_config()['sequence_model_profiles'].items()
            if isinstance(profile.get('extends'), list)
        }
        assert composing_profile_names == set(COMBINED_PROFILES)

    @pytest.mark.parametrize('combined_profile_name,model_name,profile_name', COMBINED_MODEL_PARAMS)
    def test_should_serve_what_the_profile_it_composes_serves(
        self, combined_profile_name: str, model_name: str, profile_name: str
    ):
        assert (
            get_resolved_models(combined_profile_name)[model_name]
            == get_resolved_models(profile_name)[model_name]
        )

    @pytest.mark.parametrize('model_name,profile_name', COMPOSED_MODEL_PARAMS)
    def test_should_compose_profiles_that_change_the_model_they_are_named_for(
        self, model_name: str, profile_name: str
    ):
        assert (
            get_resolved_models(profile_name)[model_name]
            != get_resolved_models(SHARED_BASE_PROFILE)[model_name]
        )
