import contextvars
import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type, cast

from sciencebeam_parser.app.context import AppContext
from sciencebeam_parser.config.config import (
    ALL_PROFILES,
    AppConfig,
    UnknownProfileError
)
from sciencebeam_parser.models.model import Model
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig
from sciencebeam_parser.processors.fulltext.models import (
    SEQUENCE_MODEL_CLASS_BY_NAME,
    FullTextModels,
    T_Model,
    create_model_for_config,
    load_models
)


LOGGER = logging.getLogger(__name__)


PROFILE_HEADER_NAME = 'X-ScienceBeam-Profile'
PROFILE_DIGEST_HEADER_NAME = 'X-ScienceBeam-Profile-Digest'


# Four profiles' worth of models, for a deployment that opted in without saying
# how much it wanted to hold. Counted in models rather than profiles because the
# profiles here overlap: three of them may be eleven models or thirty.
DEFAULT_MAX_LOADED_MODELS = 40


ModelCacheKey = Tuple[str, str]


class ProfileNotSelectableError(ValueError):
    """A real profile this deployment does not serve."""


class TooManyLoadedModelsError(RuntimeError):
    """The cap from the config, reached. A refusal rather than an OOM kill."""


def get_canonical_config_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def get_model_cache_key(model_config: dict, model_class: Type[Model]) -> ModelCacheKey:
    """What makes two models the same model.

    The class is in the key because a configuration does not name it: two
    entries may resolve to the same path and want different `Model` subclasses,
    and sharing an instance between them would serve one model's labels through
    the other's semantics.
    """
    return (model_class.__name__, get_canonical_config_json(model_config))


def get_models_digest(models_config: dict) -> str:
    return hashlib.sha256(
        get_canonical_config_json(models_config).encode('utf-8')
    ).hexdigest()[:12]


class SequenceModelCache:
    """One `Model` per resolved configuration, shared by every profile naming it.

    Locked per key rather than as a whole: the factory downloads and converts a
    model artefact, so a single lock would make one cold model block every
    request for every other.
    """

    def __init__(
        self,
        app_context: AppContext,
        max_models: int = DEFAULT_MAX_LOADED_MODELS
    ):
        self.app_context = app_context
        self.max_models = max_models
        self._lock = threading.Lock()
        self._models: Dict[ModelCacheKey, Model] = {}
        self._key_locks: Dict[ModelCacheKey, threading.Lock] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)

    def get_loaded_model_count(self) -> int:
        return len(self)

    def assert_capacity_for_keys(self, keys: Sequence[ModelCacheKey]) -> None:
        """Refuse a profile before any of it is loaded, rather than halfway through.

        The keys of a whole bundle are known before the first of them is built,
        so the request that does not fit fails without leaving a part-loaded
        profile behind it.
        """
        with self._lock:
            additional = {key for key in keys if key not in self._models}
            if len(self._models) + len(additional) <= self.max_models:
                return
            loaded_count = len(self._models)
        raise TooManyLoadedModelsError(
            f'{loaded_count} models are loaded and this profile needs '
            f'{len(additional)} more, over the limit of {self.max_models}. '
            'Restart with a higher `max_loaded_models` to serve more profiles '
            'from one process.'
        )

    def _get_key_lock(self, key: ModelCacheKey) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    def get_model(self, model_config: dict, model_class: Type[T_Model]) -> T_Model:
        key = get_model_cache_key(model_config, model_class)
        with self._lock:
            model = self._models.get(key)
        if model is not None:
            return cast(T_Model, model)
        with self._get_key_lock(key):
            with self._lock:
                model = self._models.get(key)
            if model is not None:
                return cast(T_Model, model)
            LOGGER.info('loading model: %s %s', model_class.__name__, model_config)
            new_model = create_model_for_config(
                model_config, app_context=self.app_context, model_class=model_class
            )
            with self._lock:
                self._models[key] = new_model
            return new_model


class ProfileBundle(NamedTuple):
    """Everything about serving a document that the profile decides."""
    name: Optional[str]
    fulltext_models: FullTextModels
    fulltext_processor_config: FullTextProcessorConfig
    models_digest: str

    def preload(self) -> None:
        self.fulltext_models.preload()


def get_selectable_profile_names(
    config: AppConfig,
    default_profile_name: Optional[str]
) -> List[str]:
    """What this deployment serves, which is its own profile until it says more.

    The default is the conservative one: adding a per-request parameter does not
    change what one process may be asked to hold until a deployment opts in.
    `all` opts in to every profile the config declares, which `max_loaded_models`
    still bounds.
    """
    configured = config.get('selectable_profiles') or []
    if isinstance(configured, str):
        configured = [configured]
    names: Set[Optional[str]]
    if ALL_PROFILES in configured:
        names = set(config.get_profile_names())
    else:
        names = {
            config.get_active_profile_name(name)
            for name in configured
        }
    names.discard(None)
    if default_profile_name:
        names.add(default_profile_name)
    return sorted(name for name in names if name)


class ProfileRegistry:
    """Which models and processor config a requested profile name resolves to.

    Holds the config as the file declares it rather than as the deployment
    resolved it, because resolving is what a request asks for. Environment
    variables are applied after the profile, so a deployment's override still
    wins over anything a caller names.
    """

    def __init__(
        self,
        base_config: AppConfig,
        app_context: AppContext,
        default_profile_name: Optional[str] = None,
        selectable_profile_names: Optional[Sequence[str]] = None,
        max_models: int = DEFAULT_MAX_LOADED_MODELS
    ):
        self.base_config = base_config
        self.default_profile_name = default_profile_name
        self.selectable_profile_names: Set[str] = set(selectable_profile_names or [])
        if default_profile_name:
            self.selectable_profile_names.add(default_profile_name)
        self.model_cache = SequenceModelCache(app_context, max_models=max_models)
        self._lock = threading.Lock()
        self._bundles: Dict[Optional[str], ProfileBundle] = {}
        self._bundle_locks: Dict[Optional[str], threading.Lock] = {}
        self._configs: Dict[Optional[str], AppConfig] = {}

    def get_available_profile_names(self) -> List[str]:
        return sorted(self.selectable_profile_names)

    def get_resolved_profile_name(self, profile_name: Optional[str]) -> Optional[str]:
        """The name a request asked for, after aliases, checked against what is served.

        Both refusals list what is selectable rather than what the config
        declares: a caller can only ask for the former, and a deployment serving
        one of nine profiles should not answer a typo with the other eight.
        """
        if not profile_name:
            return self.default_profile_name
        resolved_name = self.base_config.get_active_profile_name(profile_name)
        available = self.get_available_profile_names()
        if resolved_name not in self.base_config.get_profile_names():
            raise UnknownProfileError(
                f'Unknown profile {profile_name!r}. Available: {available}'
            )
        if resolved_name not in self.selectable_profile_names:
            raise ProfileNotSelectableError(
                f'Profile {profile_name!r} is not selectable. Available: {available}'
            )
        return resolved_name

    def get_config(self, resolved_name: Optional[str]) -> AppConfig:
        with self._lock:
            config = self._configs.get(resolved_name)
            if config is None:
                config = (
                    self.base_config
                    .resolve_profile(resolved_name)
                    .apply_environment_variables()
                )
                self._configs[resolved_name] = config
            return config

    def _get_bundle_lock(self, resolved_name: Optional[str]) -> threading.Lock:
        with self._lock:
            return self._bundle_locks.setdefault(resolved_name, threading.Lock())

    def _build_bundle(self, resolved_name: Optional[str]) -> ProfileBundle:
        config = self.get_config(resolved_name)
        models_config = config['models']
        self.model_cache.assert_capacity_for_keys([
            get_model_cache_key(models_config[name], model_class)
            for name, model_class in SEQUENCE_MODEL_CLASS_BY_NAME.items()
        ])
        fulltext_processor_config = FullTextProcessorConfig.from_app_config(config)
        return ProfileBundle(
            name=resolved_name,
            fulltext_models=load_models(
                config,
                app_context=self.model_cache.app_context,
                fulltext_processor_config=fulltext_processor_config,
                model_factory=self.model_cache.get_model
            ),
            fulltext_processor_config=fulltext_processor_config,
            models_digest=get_models_digest(models_config)
        )

    def get_bundle(self, profile_name: Optional[str] = None) -> ProfileBundle:
        resolved_name = self.get_resolved_profile_name(profile_name)
        with self._lock:
            bundle = self._bundles.get(resolved_name)
        if bundle is not None:
            return bundle
        with self._get_bundle_lock(resolved_name):
            with self._lock:
                bundle = self._bundles.get(resolved_name)
            if bundle is not None:
                return bundle
            LOGGER.info('resolving profile: %r', resolved_name)
            bundle = self._build_bundle(resolved_name)
            with self._lock:
                self._bundles[resolved_name] = bundle
            return bundle

    def get_default_bundle(self) -> ProfileBundle:
        return self.get_bundle(None)


@dataclass
class RequestProfile:
    """What served this request, filled in below and read above.

    Mutable and bound by the middleware rather than by the dependency that knows
    the answer: a dependency runs in a worker thread with its own copy of the
    context, so anything it binds there is invisible to the response.
    """
    name: Optional[str] = None
    digest: Optional[str] = None


REQUEST_PROFILE: contextvars.ContextVar[Optional[RequestProfile]] = (
    contextvars.ContextVar('request_profile', default=None)
)


def start_request_profile() -> RequestProfile:
    request_profile = RequestProfile()
    REQUEST_PROFILE.set(request_profile)
    return request_profile


def record_request_profile(bundle: ProfileBundle) -> None:
    request_profile = REQUEST_PROFILE.get()
    if request_profile is not None:
        request_profile.name = bundle.name
        request_profile.digest = bundle.models_digest


def get_request_profile_headers() -> Dict[str, str]:
    """The profile beside the body, with what it resolved to.

    The digest is there because the name on its own can mislead: an environment
    override wins over the requested profile, so two deployments can answer the
    same profile name with different models.
    """
    request_profile = REQUEST_PROFILE.get()
    if request_profile is None or request_profile.name is None:
        return {}
    headers = {PROFILE_HEADER_NAME: request_profile.name}
    if request_profile.digest:
        headers[PROFILE_DIGEST_HEADER_NAME] = request_profile.digest
    return headers
