import logging
import os
import copy
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import yaml


LOGGER = logging.getLogger(__name__)


DEFAULT_DOWNLOAD_DIR = 'data/download'


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


def _resolve_sequence_model_profile(
    seq_profiles: dict,
    name: str,
    _seen: Tuple[str, ...] = ()
) -> dict:
    if name not in seq_profiles:
        raise ValueError(
            f'Unknown sequence_model_profile {name!r}. Available: {sorted(seq_profiles)}'
        )
    if name in _seen:
        raise ValueError(
            f'Circular extends detected for sequence_model_profile {name!r} '
            f'(chain: {" -> ".join([*_seen, name])})'
        )
    profile = seq_profiles[name]
    base_name = profile.get('extends')
    overlay = {key: value for key, value in profile.items() if key != 'extends'}
    if not base_name:
        return overlay
    base = _resolve_sequence_model_profile(seq_profiles, base_name, _seen + (name,))
    return _deep_merge(base, overlay)


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
            raise ValueError(
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

    def get(self, key: str, default_value: Optional[Any] = None):
        return self.props.get(key, default_value)

    def __getitem__(self, key: str):
        return self.props[key]


def get_download_dir(config: Union[dict, AppConfig]) -> str:
    return os.path.expanduser(
        config.get('download_dir', DEFAULT_DOWNLOAD_DIR)
    )
