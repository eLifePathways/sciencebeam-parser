import logging
import os
import copy
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import yaml


LOGGER = logging.getLogger(__name__)


DEFAULT_DOWNLOAD_DIR = 'data/download'


# What a profile is allowed to overlay. Everything outside this set is built once
# and shared by every profile, and nothing about those objects is keyed on the
# configuration they were built from -- so a profile setting `lookup` or
# `llm_response_cache_dir` would be served the deployment's value alongside its
# own models, with nothing able to observe the mismatch.
PROFILE_OVERLAY_KEYS = frozenset({'sequence_models', 'models', 'processors'})


# What `selectable_profiles` says to mean every declared profile. A word rather
# than `*`, which is an alias indicator in YAML and so cannot be passed through
# `SCIENCEBEAM_PARSER__SELECTABLE_PROFILES`.
ALL_PROFILES = 'all'


class UnknownProfileError(ValueError):
    """A name that is not a profile. Raised for a request, so it carries the list."""


class InvalidProfileError(ValueError):
    """A profile the process must not serve, found before it serves anything."""


def parse_env_value(value: str) -> Union[str, int]:
    return yaml.safe_load(value)


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _get_base_profile_names(name: str, extends: Any) -> List[str]:
    if extends is None:
        return []
    if isinstance(extends, str):
        return [extends]
    if not isinstance(extends, list):
        raise ValueError(
            f'Invalid extends in sequence_model_profile {name!r}: expected a profile name '
            f'or a list of profile names, but got {extends!r}'
        )
    for base_name in extends:
        if not isinstance(base_name, str):
            raise ValueError(
                f'Invalid extends entry in sequence_model_profile {name!r}: '
                f'expected a profile name, but got {base_name!r}'
            )
    return extends


def _get_sequence_model_profile_layers(
    seq_profiles: dict,
    name: str,
    _seen: Tuple[str, ...] = ()
) -> List[str]:
    if name not in seq_profiles:
        raise ValueError(
            f'Unknown sequence_model_profile {name!r}. Available: {sorted(seq_profiles)}'
        )
    if name in _seen:
        raise ValueError(
            f'Circular extends detected for sequence_model_profile {name!r} '
            f'(chain: {" -> ".join([*_seen, name])})'
        )
    layers: List[str] = []
    for base_name in _get_base_profile_names(name, seq_profiles[name].get('extends')):
        layers.extend(_get_sequence_model_profile_layers(seq_profiles, base_name, _seen + (name,)))
    layers.append(name)
    return layers


def _resolve_sequence_model_profile(seq_profiles: dict, name: str) -> dict:
    resolved: dict = {}
    for layer_name in dict.fromkeys(_get_sequence_model_profile_layers(seq_profiles, name)):
        resolved = _deep_merge(resolved, {
            key: value
            for key, value in seq_profiles[layer_name].items()
            if key != 'extends'
        })
    return resolved


class AppConfig:
    def __init__(self, props: dict):
        self.props = props

    def __repr__(self) -> str:
        return '%s(%r)' % (type(self).__name__, self.props)

    @staticmethod
    def load_yaml(file_path: str) -> 'AppConfig':
        return AppConfig(
            yaml.safe_load(Path(file_path).read_text(encoding='utf-8'))
        )

    def apply_environment_variables(self, prefix: str = 'SCIENCEBEAM_PARSER__') -> 'AppConfig':
        updated_props = copy.deepcopy(self.props)
        env_vars = os.environ
        if not env_vars:
            LOGGER.debug('no environment variables')
        LOGGER.debug('processing env vars: %r', env_vars)
        for env_name, env_value in env_vars.items():
            if not env_name.startswith(prefix):
                LOGGER.debug('ignoring: %r', env_name)
                continue
            key_path = env_name[len(prefix):].lower().split('__')
            LOGGER.debug('updating: %r -> %r', env_name, key_path)
            parent_key_path = key_path[:-1]
            leaf_key = key_path[-1]
            parent_props = updated_props
            for parent_key in parent_key_path:
                parent_props = parent_props.setdefault(parent_key, {})
            parent_props[leaf_key] = parse_env_value(env_value)
        return AppConfig(updated_props)

    def resolve_profile(self, profile_name: Optional[str] = None) -> 'AppConfig':
        name = profile_name or self.props.get('profile')
        if not name:
            return self

        aliases = self.props.get('profile_aliases', {})
        resolved = aliases.get(name, name)

        profiles = self.props.get('profiles', {})
        if resolved not in profiles:
            available = sorted(profiles)
            suffix = f' (alias for {resolved!r})' if resolved != name else ''
            raise UnknownProfileError(
                f'Unknown profile {name!r}{suffix}. Available: {available}'
            )

        profile = profiles[resolved]
        overlay: dict = {}

        seq_name = profile.get('sequence_models')
        if seq_name:
            seq_profiles = self.props.get('sequence_model_profiles', {})
            if seq_name not in seq_profiles:
                raise ValueError(
                    f'Profile {resolved!r} references unknown sequence_model_profile '
                    f'{seq_name!r}. Available: {sorted(seq_profiles)}'
                )
            overlay['models'] = _resolve_sequence_model_profile(seq_profiles, seq_name)

        for key, value in profile.items():
            if key != 'sequence_models':
                overlay[key] = value

        return AppConfig(_deep_merge(self.props, overlay))

    def get_active_profile_name(self, profile_name: Optional[str] = None) -> Optional[str]:
        name = profile_name or self.props.get('profile')
        if not name:
            return None
        aliases = self.props.get('profile_aliases', {})
        return aliases.get(name, name)

    def get_profile_names(self) -> List[str]:
        return sorted(self.props.get('profiles', {}))

    def validate_profiles(self) -> 'AppConfig':
        profiles = self.props.get('profiles', {})
        if ALL_PROFILES in profiles:
            raise InvalidProfileError(
                f'A profile may not be named {ALL_PROFILES!r}: that is what '
                '`selectable_profiles` says to mean every profile'
            )
        for name, profile in sorted(profiles.items()):
            invalid_keys = sorted(set(profile) - PROFILE_OVERLAY_KEYS)
            if invalid_keys:
                raise InvalidProfileError(
                    f'Profile {name!r} may not set {invalid_keys}. '
                    f'A profile may only set {sorted(PROFILE_OVERLAY_KEYS)}'
                )
        return self

    def get(self, key: str, default_value: Optional[Any] = None):
        return self.props.get(key, default_value)

    def __getitem__(self, key: str):
        return self.props[key]


def get_download_dir(config: Union[dict, AppConfig]) -> str:
    return os.path.expanduser(
        config.get('download_dir', DEFAULT_DOWNLOAD_DIR)
    )


def get_llm_response_cache_dir(config: Union[dict, AppConfig]) -> Optional[str]:
    """Where the llm engine replays completions from, or None for off.

    One directory for every llm model rather than a setting per model: it is
    storage rather than something that shapes an answer, and entries are keyed by
    the request, so two models cannot collide in it.

    `None` rather than an empty string, because no path is what is meant and the
    config already spells that as an unset key — as `pdfalto.path` does. An empty
    `SCIENCEBEAM_PARSER__LLM_RESPONSE_CACHE_DIR` parses to `None` too, so turning
    it off by environment and by config reach the same value.
    """
    value = config.get('llm_response_cache_dir')
    return os.path.expanduser(value) if value else None
