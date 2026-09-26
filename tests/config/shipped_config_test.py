import os
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest
import yaml

from sciencebeam_parser.app.profiles import ProfileRegistry
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.models.model_impl_factory import EngineNames, get_engine_name_for_config
from sciencebeam_parser.processors.fulltext.models import SEQUENCE_MODEL_CLASS_BY_NAME


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
    'delft_scielo_preprints_ore': {
        'reference_segmenter': 'delft_refseg_scielo_preprints_ore',
        'citation': 'delft_citation_scielo_preprints_ore',
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

# The profile serving models from the Hugging Face Hub, and the base it extends.
HUB_PROFILE = 'delft_hub'

# Left out of the budget the other profiles are held to. `delft_hub` extends
# `biorxiv_elife` and shares eight of its ten models with it, adding only the
# two it takes from the Hub -- the same sharing the budget is there to check --
# but the budget is a fixed multiple of one profile's models, so it is an
# allowance every profile spends rather than a measure of how well each shares.
PROFILES_OUTSIDE_THE_SHARING_BUDGET = frozenset({HUB_PROFILE})
HUB_PROFILE_BASE = 'biorxiv_elife'

# What the delft engine recognises as a Hub reference
HUB_PATH_PREFIX = 'hf://'


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


class TestShippedProfilesOnlyChangeWhatIsPerRequest:
    def test_should_only_set_keys_a_profile_may_set(self):
        AppConfig(get_shipped_config()).validate_profiles()

    def test_should_declare_no_extra_selectable_profiles(self):
        """What production serves is unchanged by adding a per-request parameter."""
        assert get_shipped_config()['selectable_profiles'] == []


class TestShippedProfilesShareTheirModels:
    """What holding more than one shipped profile costs.

    The `wapiti_*` and `llm_*` profiles extend `grobid_crf_0_9_0` and change one
    or two models of ten, so a process serving several of them holds one
    instance per distinct configuration rather than ten per profile.
    """

    @pytest.fixture(name='registry')
    def _registry(self, monkeypatch: pytest.MonkeyPatch) -> ProfileRegistry:
        for name in list(os.environ):
            if name.startswith('SCIENCEBEAM_PARSER__'):
                monkeypatch.delenv(name)
        base_config = AppConfig(get_shipped_config())
        return ProfileRegistry(
            base_config=base_config,
            app_context=MagicMock(name='app_context'),
            default_profile_name=SHARED_BASE_PROFILE,
            selectable_profile_names=base_config.get_profile_names(),
            max_models=1000
        )

    def test_should_hold_one_extra_model_per_model_a_profile_changes(
        self, registry: ProfileRegistry
    ):
        registry.get_bundle('wapiti_refseg_scielo_preprints_ore')
        registry.get_bundle('wapiti_citation_scielo_preprints_ore')
        assert registry.model_cache.get_loaded_model_count() == len(
            SEQUENCE_MODEL_CLASS_BY_NAME
        ) + 2

    def test_should_share_the_models_two_profiles_agree_on(
        self, registry: ProfileRegistry
    ):
        refseg = registry.get_bundle('wapiti_refseg_scielo_preprints_ore')
        citation = registry.get_bundle('wapiti_citation_scielo_preprints_ore')
        assert (
            refseg.fulltext_models.segmentation_model
            is citation.fulltext_models.segmentation_model
        )
        assert (
            refseg.fulltext_models.reference_segmenter_model
            is not citation.fulltext_models.reference_segmenter_model
        )

    def test_should_hold_every_shipped_profile_for_less_than_three_unshared_ones(
        self, registry: ProfileRegistry
    ):
        for profile_name in AppConfig(get_shipped_config()).get_profile_names():
            if profile_name in PROFILES_OUTSIDE_THE_SHARING_BUDGET:
                continue
            registry.get_bundle(profile_name)
        assert registry.model_cache.get_loaded_model_count() < 3 * len(
            SEQUENCE_MODEL_CLASS_BY_NAME
        )

    def test_should_attribute_two_profiles_to_different_digests(
        self, registry: ProfileRegistry
    ):
        assert (
            registry.get_bundle('wapiti_refseg_scielo_preprints_ore').models_digest
            != registry.get_bundle('wapiti_citation_scielo_preprints_ore').models_digest
        )


class TestShippedDelftHubProfile:
    """
    Guards `delft_hub`, which overrides only the models published to the Hub and
    takes the rest from the profile it extends.

    A profile is deep merged over its base (`_deep_merge` in
    `sciencebeam_parser.config.config`), so an override carrying only a `path`
    keeps every other key of the entry underneath it -- `engine` included. Point
    a hub path at an entry whose base is a wapiti one and the config stays valid
    while the wapiti loader is handed a delft model directory, which only fails
    at load time, in a deployment.
    """

    def test_should_serve_the_models_of_its_base_that_it_does_not_override(self):
        hub_models = get_resolved_models(HUB_PROFILE)
        base_models = get_resolved_models(HUB_PROFILE_BASE)
        assert set(hub_models) == set(base_models)
        unchanged = {
            model_name: entry
            for model_name, entry in hub_models.items()
            if not entry['path'].startswith(HUB_PATH_PREFIX)
        }
        assert unchanged == {
            model_name: entry
            for model_name, entry in base_models.items()
            if model_name in unchanged
        }

    def test_should_override_at_least_one_model_with_a_hub_path(self):
        assert [
            model_name
            for model_name, entry in get_resolved_models(HUB_PROFILE).items()
            if entry['path'].startswith(HUB_PATH_PREFIX)
        ]

    def test_should_serve_every_hub_path_with_the_delft_engine(self):
        """No shipped profile pairs a Hub path with a loader that cannot read one.

        Swept over every profile rather than `delft_hub` alone: the deep merge
        makes this silent wherever it happens, and a profile added later on a
        wapiti base would be just as wrong.
        """
        wrongly_served = {
            (profile_name, model_name): get_engine_name_for_config(entry)
            for profile_name in get_shipped_config()['sequence_model_profiles']
            for model_name, entry in get_resolved_models(profile_name).items()
            if entry.get('path', '').startswith(HUB_PATH_PREFIX)
            and get_engine_name_for_config(entry) != EngineNames.DELFT
        }
        assert not wrongly_served
