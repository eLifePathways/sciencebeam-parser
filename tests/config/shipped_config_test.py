import os
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import yaml

from sciencebeam_parser.app.profiles import (
    ProfileRegistry,
    get_canonical_config_json,
    get_selectable_profile_names
)
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.models.header.extract import ABSTRACTS_MODES
from sciencebeam_parser.models.model_impl_factory import EngineNames, get_engine_name_for_config
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig
from sciencebeam_parser.processors.fulltext.models import SEQUENCE_MODEL_CLASS_BY_NAME


CONFIG_PATH = (
    Path(__file__).parent.parent.parent
    / 'sciencebeam_parser' / 'resources' / 'default_config' / 'config.yml'
)

SHARED_BASE_PROFILE = 'grobid_crf_0_9_0'

# The names a deployment points at, and the profile each one currently names.
POINTER_PROFILES: Dict[str, str] = {
    'grobid_crf': 'grobid_crf_0_9_0',
    'grobid_dl': 'delft_hub',
    'recommended': 'delft_all_scielo_preprints_ore',
    'recommended_cv': 'delft_all_scielo_preprints_ore_cv',
    'lightweight': 'wapiti_all_scielo_preprints_ore',
}

# Profiles combining single-model profiles, and the profile each model comes from.
COMBINED_PROFILES: Dict[str, Dict[str, str]] = {
    'wapiti_references_scielo_preprints_ore': {
        'reference_segmenter': 'wapiti_refseg_scielo_preprints_ore',
        'citation': 'wapiti_citation_scielo_preprints_ore',
    },
    'delft_references_scielo_preprints_ore': {
        'reference_segmenter': 'delft_refseg_scielo_preprints_ore',
        'citation': 'delft_citation_scielo_preprints_ore',
    },
    'wapiti_all_scielo_preprints_ore': {
        'segmentation': 'wapiti_seg_scielo_preprints_ore',
        'reference_segmenter': 'wapiti_refseg_scielo_preprints_ore',
        'citation': 'wapiti_citation_scielo_preprints_ore',
    },
    'delft_all_scielo_preprints_ore': {
        'segmentation': 'delft_seg_scielo_preprints_ore',
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

HUB_PROFILE_BASE = 'grobid_crf_0_9_0'

# What the delft engine recognises as a Hub reference
HUB_PATH_PREFIX = 'hf://'


def get_shipped_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))


def get_shipped_selectable_profile_names() -> List[str]:
    config = AppConfig(get_shipped_config())
    return get_selectable_profile_names(config, config.get_active_profile_name())


def get_resolved_models(profile_name: Optional[str]) -> dict:
    return AppConfig(get_shipped_config()).resolve_profile(profile_name)['models']


def get_fulltext_processor_config(profile_name: Optional[str]) -> FullTextProcessorConfig:
    return FullTextProcessorConfig.from_app_config(
        AppConfig(get_shipped_config()).resolve_profile(profile_name)
    )


ABSTRACTS_MODE_PROFILES = [
    'grobid_crf_abstract_variants',
    'grobid_crf_abstract_merged_by_language'
]


ABSTRACTS_MODE_BASE_PROFILE = 'grobid_crf_0_9_0'


class TestAbstractsModeProfiles:
    """Each exists to be compared with `grobid_crf_0_9_0`, so one setting may differ."""

    @pytest.mark.parametrize('profile_name', ABSTRACTS_MODE_PROFILES)
    def test_should_differ_from_its_base_profile_in_abstracts_mode_only(
        self, profile_name: str
    ):
        base_config = get_fulltext_processor_config(ABSTRACTS_MODE_BASE_PROFILE)
        variant_config = get_fulltext_processor_config(profile_name)
        assert variant_config.abstracts_mode != base_config.abstracts_mode
        assert variant_config._replace(  # pylint: disable=protected-access
            abstracts_mode=base_config.abstracts_mode
        ) == base_config

    @pytest.mark.parametrize('profile_name', ABSTRACTS_MODE_PROFILES)
    def test_should_serve_the_models_its_base_profile_serves(self, profile_name: str):
        assert (
            get_resolved_models(profile_name)
            == get_resolved_models(ABSTRACTS_MODE_BASE_PROFILE)
        )

    def test_should_cover_every_mode_other_than_its_base(self):
        base_mode = get_fulltext_processor_config(ABSTRACTS_MODE_BASE_PROFILE).abstracts_mode
        assert {
            get_fulltext_processor_config(name).abstracts_mode
            for name in ABSTRACTS_MODE_PROFILES
        } == set(ABSTRACTS_MODES) - {base_mode}


CV_PROFILE_PAIRS = [
    ('grobid_crf_cv', 'grobid_crf_0_9_0'),
    ('delft_all_scielo_preprints_ore_cv', 'delft_all_scielo_preprints_ore')
]


class TestCvProfiles:
    """Each exists to be compared with the profile it varies, so one setting may differ."""

    @pytest.mark.parametrize('profile_name,base_profile_name', CV_PROFILE_PAIRS)
    def test_should_differ_from_its_base_profile_in_use_cv_model_only(
        self, profile_name: str, base_profile_name: Optional[str]
    ):
        base_config = get_fulltext_processor_config(base_profile_name)
        variant_config = get_fulltext_processor_config(profile_name)
        assert variant_config.use_cv_model
        assert not base_config.use_cv_model
        assert variant_config._replace(  # pylint: disable=protected-access
            use_cv_model=base_config.use_cv_model
        ) == base_config

    @pytest.mark.parametrize('profile_name,base_profile_name', CV_PROFILE_PAIRS)
    def test_should_serve_the_models_its_base_profile_serves(
        self, profile_name: str, base_profile_name: Optional[str]
    ):
        assert get_resolved_models(profile_name) == get_resolved_models(base_profile_name)


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


class TestShippedPointerProfiles:
    """A name a deployment points at, and the combination it currently names."""

    def test_should_declare_every_pointer_the_shipped_config_names(self):
        config = AppConfig(get_shipped_config())
        assert {
            name: config.get_pinned_profile_name(name)
            for name in config.get_profile_names()
            if config.is_profile_pointer(name)
        } == POINTER_PROFILES

    @pytest.mark.parametrize('profile_name,target_name', sorted(POINTER_PROFILES.items()))
    def test_should_serve_exactly_what_it_names(self, profile_name: str, target_name: str):
        assert get_resolved_models(profile_name) == get_resolved_models(target_name)

    @pytest.mark.parametrize('profile_name,target_name', sorted(POINTER_PROFILES.items()))
    def test_should_process_a_document_as_what_it_names_does(
        self, profile_name: str, target_name: str
    ):
        assert (
            get_fulltext_processor_config(profile_name)
            == get_fulltext_processor_config(target_name)
        )

    def test_should_attribute_a_benchmark_run_where_it_already_attributed_it(self):
        """`grobid_crf` named a profile through an alias and now names one directly."""
        config = AppConfig(get_shipped_config())
        assert config.get_pinned_profile_name('grobid_crf') == 'grobid_crf_0_9_0'

    def test_should_fit_every_selectable_profile_under_the_model_cap(self):
        """A deployment taking the shipped config holds all of them without refusing one."""
        config = AppConfig(get_shipped_config())
        model_keys = {
            (model_name, get_canonical_config_json(model_config))
            for profile_name in get_shipped_selectable_profile_names()
            for model_name, model_config in config.resolve_profile(profile_name)['models'].items()
        }
        assert len(model_keys) <= get_shipped_config()['max_loaded_models']

    def test_should_serve_a_pointer_by_default(self):
        """A deployment taking the shipped default tracks what we recommend."""
        config = AppConfig(get_shipped_config())
        assert config.is_profile_pointer(config.get('profile'))


class TestShippedProfilesOnlyChangeWhatIsPerRequest:
    def test_should_only_set_keys_a_profile_may_set(self):
        AppConfig(get_shipped_config()).validate_profiles()

    def test_should_say_what_every_profile_is_for(self):
        """A caller choosing between them is not reading this file."""
        config = AppConfig(get_shipped_config())
        unlabelled = [
            name for name in config.get_profile_names()
            if not config.get_profile_label(name) or not config.get_profile_description(name)
        ]
        assert not unlabelled

    def test_should_declare_the_user_facing_profiles_selectable(self):
        """What a deployment offers is the set of names, not the set of combinations."""
        assert set(get_shipped_config()['selectable_profiles']) == set(POINTER_PROFILES)

    def test_should_serve_the_profile_every_selectable_pointer_names(self):
        """Attribution reports the pinned profile, which a caller must then be able to send."""
        assert set(get_shipped_selectable_profile_names()) == (
            set(POINTER_PROFILES) | set(POINTER_PROFILES.values())
        )


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

    def test_should_hold_every_shipped_profile_for_a_quarter_of_what_unshared_would_cost(
        self, registry: ProfileRegistry
    ):
        profile_names = AppConfig(get_shipped_config()).get_profile_names()
        for profile_name in profile_names:
            registry.get_bundle(profile_name)
        unshared_count = len(profile_names) * len(SEQUENCE_MODEL_CLASS_BY_NAME)
        assert registry.model_cache.get_loaded_model_count() < unshared_count / 4

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

    def test_should_give_grobid_features_to_its_grobid_header_model(self):
        assert get_resolved_models(HUB_PROFILE)['header'].get('feature_flavour') == 'grobid'

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
