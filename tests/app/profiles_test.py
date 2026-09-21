import logging
import threading
from typing import Dict, Iterator, List, Optional, Type
from unittest.mock import MagicMock, patch

import pytest

from sciencebeam_parser.app import profiles as profiles_module
from sciencebeam_parser.app.profiles import (
    ProfileNotSelectableError,
    ProfileRegistry,
    SequenceModelCache,
    TooManyLoadedModelsError,
    get_model_cache_key,
    get_selectable_profile_names
)
from sciencebeam_parser.config.config import AppConfig, UnknownProfileError
from sciencebeam_parser.models.model import Model
from sciencebeam_parser.processors.fulltext.models import SEQUENCE_MODEL_CLASS_BY_NAME


LOGGER = logging.getLogger(__name__)


MODEL_NAMES = sorted(SEQUENCE_MODEL_CLASS_BY_NAME)
MODEL_COUNT = len(MODEL_NAMES)


def get_models_config(prefix: str) -> dict:
    return {name: {'path': f'{prefix}/{name}'} for name in MODEL_NAMES}


# `b` extends `a` and changes one model of ten, the shape the shipped profiles
# have; `c` shares nothing with either.
BASE_CONFIG_PROPS = {
    'models': get_models_config('base'),
    'sequence_model_profiles': {
        'a': get_models_config('a'),
        'b': {'extends': 'a', 'citation': {'path': 'b/citation'}},
        'c': get_models_config('c'),
    },
    'profiles': {
        'a': {'sequence_models': 'a'},
        'b': {'sequence_models': 'b'},
        'c': {'sequence_models': 'c'},
    },
    'profile_aliases': {'alias_b': 'b'},
    'profile': 'a',
}


class RecordingModelFactory:
    """Stands in for building a model, counting what was built and when."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.lock = threading.Lock()
        self.created_paths: List[str] = []

    def __call__(
        self,
        model_config: dict,
        app_context,  # pylint: disable=unused-argument
        model_class: Type[Model]
    ) -> MagicMock:
        if self.delay:
            threading.Event().wait(self.delay)
        with self.lock:
            self.created_paths.append(model_config['path'])
        return MagicMock(name=f"model:{model_class.__name__}:{model_config['path']}")

    @property
    def created_count(self) -> int:
        with self.lock:
            return len(self.created_paths)


@pytest.fixture(name='model_factory')
def _model_factory() -> Iterator[RecordingModelFactory]:
    factory = RecordingModelFactory()
    with patch.object(profiles_module, 'create_model_for_config', factory):
        yield factory


@pytest.fixture(name='base_config')
def _base_config() -> AppConfig:
    return AppConfig(BASE_CONFIG_PROPS)


def create_registry(
    base_config: AppConfig,
    selectable_profile_names: Optional[List[str]] = None,
    max_models: int = 1000,
    default_profile_name: Optional[str] = 'a'
) -> ProfileRegistry:
    return ProfileRegistry(
        base_config=base_config,
        app_context=MagicMock(name='app_context'),
        default_profile_name=default_profile_name,
        selectable_profile_names=(
            selectable_profile_names
            if selectable_profile_names is not None
            else ['a', 'b', 'c']
        ),
        max_models=max_models
    )


@pytest.fixture(name='registry')
def _registry(base_config: AppConfig) -> ProfileRegistry:
    return create_registry(base_config)


class TestProfileRegistryDefault:
    def test_should_serve_the_deployment_profile_when_nothing_is_named(
        self, registry: ProfileRegistry, model_factory: RecordingModelFactory
    ):
        assert registry.get_bundle().name == 'a'
        assert sorted(model_factory.created_paths) == [
            f'a/{name}' for name in MODEL_NAMES
        ]

    def test_should_serve_the_same_bundle_the_default_name_resolves_to(
        self, registry: ProfileRegistry, model_factory: RecordingModelFactory
    ):
        assert registry.get_bundle() is registry.get_bundle('a')
        assert model_factory.created_count == MODEL_COUNT

    def test_should_resolve_an_alias(
        self, registry: ProfileRegistry, model_factory: RecordingModelFactory  # noqa pylint: disable=unused-argument
    ):
        assert registry.get_bundle('alias_b') is registry.get_bundle('b')


class TestProfileRegistrySharing:
    def test_should_share_a_model_two_profiles_agree_on(
        self, registry: ProfileRegistry, model_factory: RecordingModelFactory
    ):
        bundle_a = registry.get_bundle('a')
        bundle_b = registry.get_bundle('b')
        assert (
            bundle_a.fulltext_models.segmentation_model
            is bundle_b.fulltext_models.segmentation_model
        )
        # The one model they differ on, and nothing else.
        assert model_factory.created_count == MODEL_COUNT + 1

    def test_should_not_share_a_model_two_profiles_differ_on(
        self, registry: ProfileRegistry, model_factory: RecordingModelFactory  # noqa pylint: disable=unused-argument
    ):
        assert (
            registry.get_bundle('a').fulltext_models.citation_model
            is not registry.get_bundle('b').fulltext_models.citation_model
        )

    def test_should_share_nothing_between_unrelated_profiles(
        self, registry: ProfileRegistry, model_factory: RecordingModelFactory
    ):
        registry.get_bundle('a')
        registry.get_bundle('c')
        assert model_factory.created_count == 2 * MODEL_COUNT


class TestProfileRegistrySelection:
    def test_should_reject_an_unknown_name(self, registry: ProfileRegistry):
        with pytest.raises(UnknownProfileError) as exc_info:
            registry.get_bundle('nonexistent')
        assert 'nonexistent' in str(exc_info.value)

    def test_should_list_what_is_selectable_when_rejecting(
        self, base_config: AppConfig
    ):
        registry = create_registry(base_config, selectable_profile_names=['a'])
        with pytest.raises(UnknownProfileError) as exc_info:
            registry.get_bundle('nonexistent')
        assert "['a']" in str(exc_info.value)

    def test_should_reject_a_real_profile_the_deployment_does_not_serve(
        self, base_config: AppConfig
    ):
        registry = create_registry(base_config, selectable_profile_names=['a'])
        with pytest.raises(ProfileNotSelectableError) as exc_info:
            registry.get_bundle('b')
        assert "'b'" in str(exc_info.value)

    def test_should_always_serve_its_own_profile(self, base_config: AppConfig):
        registry = create_registry(base_config, selectable_profile_names=[])
        assert registry.get_available_profile_names() == ['a']


class TestProfileRegistryEnvironmentOverride:
    def test_should_let_an_override_win_over_the_requested_profile(
        self,
        registry: ProfileRegistry,
        model_factory: RecordingModelFactory,  # noqa pylint: disable=unused-argument
        monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(
            'SCIENCEBEAM_PARSER__MODELS__CITATION__PATH', 'override/citation'
        )
        assert (
            registry.get_config('b')['models']['citation']['path'] == 'override/citation'
        )

    def test_should_apply_the_override_to_every_profile(
        self,
        registry: ProfileRegistry,
        model_factory: RecordingModelFactory,
        monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(
            'SCIENCEBEAM_PARSER__MODELS__CITATION__PATH', 'override/citation'
        )
        registry.get_bundle('a')
        registry.get_bundle('b')
        # The override makes the two profiles agree on the model they differ on,
        # so there is nothing left to load a second time.
        assert model_factory.created_count == MODEL_COUNT


class TestProfileRegistryDigest:
    def test_should_differ_between_profiles_with_different_models(
        self, registry: ProfileRegistry, model_factory: RecordingModelFactory  # noqa pylint: disable=unused-argument
    ):
        assert (
            registry.get_bundle('a').models_digest
            != registry.get_bundle('b').models_digest
        )

    def test_should_record_an_environment_override(
        self,
        base_config: AppConfig,
        model_factory: RecordingModelFactory,  # noqa pylint: disable=unused-argument
        monkeypatch: pytest.MonkeyPatch
    ):
        without_override = create_registry(base_config).get_bundle('a').models_digest
        monkeypatch.setenv(
            'SCIENCEBEAM_PARSER__MODELS__CITATION__PATH', 'override/citation'
        )
        with_override = create_registry(base_config).get_bundle('a').models_digest
        assert with_override != without_override


class TestProfileRegistryCap:
    def test_should_refuse_a_profile_that_would_not_fit(
        self, base_config: AppConfig, model_factory: RecordingModelFactory
    ):
        registry = create_registry(base_config, max_models=MODEL_COUNT)
        registry.get_bundle('a')
        with pytest.raises(TooManyLoadedModelsError) as exc_info:
            registry.get_bundle('c')
        assert str(MODEL_COUNT) in str(exc_info.value)
        # Refused before any of it was built, rather than part way through.
        assert model_factory.created_count == MODEL_COUNT

    def test_should_admit_a_profile_that_mostly_shares(
        self, base_config: AppConfig, model_factory: RecordingModelFactory
    ):
        registry = create_registry(base_config, max_models=MODEL_COUNT + 1)
        registry.get_bundle('a')
        assert registry.get_bundle('b').name == 'b'
        assert model_factory.created_count == MODEL_COUNT + 1

    def test_should_keep_serving_what_it_already_holds(
        self, base_config: AppConfig, model_factory: RecordingModelFactory  # noqa pylint: disable=unused-argument
    ):
        registry = create_registry(base_config, max_models=MODEL_COUNT)
        registry.get_bundle('a')
        with pytest.raises(TooManyLoadedModelsError):
            registry.get_bundle('c')
        assert registry.get_bundle('a').name == 'a'


class TestSequenceModelCacheConcurrency:
    def _get_bundles_concurrently(
        self, registry: ProfileRegistry, profile_names: List[str]
    ) -> Dict[str, object]:
        results: Dict[str, object] = {}
        barrier = threading.Barrier(len(profile_names))

        def _run(profile_name: str) -> None:
            barrier.wait()
            results[profile_name] = registry.get_bundle(profile_name)

        threads = [
            threading.Thread(target=_run, args=(name,), name=f'get-{name}')
            for name in profile_names
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results

    def test_should_load_a_cold_model_once_under_concurrent_first_use(
        self, base_config: AppConfig
    ):
        factory = RecordingModelFactory(delay=0.01)
        with patch.object(profiles_module, 'create_model_for_config', factory):
            registry = create_registry(base_config)
            self._get_bundles_concurrently(registry, ['a', 'a', 'a', 'a'])
        assert sorted(factory.created_paths) == sorted(
            f'a/{name}' for name in MODEL_NAMES
        )

    def test_should_share_a_model_between_profiles_asked_for_at_once(
        self, base_config: AppConfig
    ):
        factory = RecordingModelFactory(delay=0.01)
        with patch.object(profiles_module, 'create_model_for_config', factory):
            registry = create_registry(base_config)
            results = self._get_bundles_concurrently(registry, ['a', 'b'])
        assert factory.created_count == MODEL_COUNT + 1
        assert (
            results['a'].fulltext_models.segmentation_model  # type: ignore[attr-defined]
            is results['b'].fulltext_models.segmentation_model  # type: ignore[attr-defined]
        )


class TestSequenceModelCacheKey:
    def test_should_tell_two_classes_apart(self):
        model_config = {'path': 'the/path'}
        assert (
            get_model_cache_key(model_config, SEQUENCE_MODEL_CLASS_BY_NAME['header'])
            != get_model_cache_key(
                model_config, SEQUENCE_MODEL_CLASS_BY_NAME['segmentation']
            )
        )

    def test_should_ignore_the_order_keys_were_written_in(self):
        model_class = SEQUENCE_MODEL_CLASS_BY_NAME['header']
        assert (
            get_model_cache_key({'path': 'p', 'engine': 'wapiti'}, model_class)
            == get_model_cache_key({'engine': 'wapiti', 'path': 'p'}, model_class)
        )

    def test_should_count_what_it_holds(self):
        cache = SequenceModelCache(MagicMock(name='app_context'), max_models=2)
        factory = RecordingModelFactory()
        with patch.object(profiles_module, 'create_model_for_config', factory):
            cache.get_model({'path': 'p1'}, SEQUENCE_MODEL_CLASS_BY_NAME['header'])
            cache.get_model({'path': 'p1'}, SEQUENCE_MODEL_CLASS_BY_NAME['header'])
        assert cache.get_loaded_model_count() == 1


class TestGetSelectableProfileNames:
    def test_should_default_to_the_deployment_profile_alone(self):
        assert get_selectable_profile_names(AppConfig(BASE_CONFIG_PROPS), 'a') == ['a']

    def test_should_include_what_the_config_declares(self):
        config = AppConfig({**BASE_CONFIG_PROPS, 'selectable_profiles': ['b', 'c']})
        assert get_selectable_profile_names(config, 'a') == ['a', 'b', 'c']

    def test_should_resolve_aliases(self):
        config = AppConfig({**BASE_CONFIG_PROPS, 'selectable_profiles': ['alias_b']})
        assert get_selectable_profile_names(config, 'a') == ['a', 'b']

    def test_should_accept_a_single_name(self):
        config = AppConfig({**BASE_CONFIG_PROPS, 'selectable_profiles': 'b'})
        assert get_selectable_profile_names(config, 'a') == ['a', 'b']

    def test_should_return_nothing_without_a_default_profile(self):
        assert get_selectable_profile_names(AppConfig(BASE_CONFIG_PROPS), None) == []
