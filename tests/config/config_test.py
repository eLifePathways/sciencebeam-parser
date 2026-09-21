from pathlib import Path
from typing import Iterable, Optional
from unittest.mock import patch

import pytest
import yaml

from sciencebeam_parser.config.config import (
    AppConfig,
    InvalidProfileError,
    UnknownProfileError,
    _deep_merge,
    _resolve_sequence_model_profile,
    get_llm_response_cache_dir
)
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE


MINIMAL_PROFILE_CONFIG = {
    'sequence_model_profiles': {
        'profile_a': {
            'segmentation': {'path': 'path_a/segmentation'},
            'header': {'path': 'path_a/header'},
        },
        'profile_b': {
            'segmentation': {'path': 'path_b/segmentation', 'engine': 'wapiti'},
            'header': {'path': 'path_b/header', 'engine': 'wapiti'},
        },
        'profile_b_extended': {
            'extends': 'profile_b',
            'header': {'path': 'path_b_extended/header', 'engine': 'wapiti'},
        },
    },
    'profiles': {
        'profile_a': {'sequence_models': 'profile_a'},
        'profile_b': {'sequence_models': 'profile_b'},
        'profile_b_extended': {'sequence_models': 'profile_b_extended'},
        'profile_with_extra': {
            'sequence_models': 'profile_a',
            'processors': {'fulltext': {'use_cv_model': True}},
        },
    },
    'profile_aliases': {
        'alias_a': 'profile_a',
    },
    'profile': 'profile_a',
    'models': {
        'segmentation': {'path': 'base/segmentation', 'use_first_token_of_block': False},
        'header': {'path': 'base/header'},
    },
}


class TestDeepMerge:
    def test_scalar_overlay_wins(self):
        result = _deep_merge({'a': 1}, {'a': 2})
        assert result['a'] == 2

    def test_base_key_preserved_when_not_in_overlay(self):
        result = _deep_merge({'a': 1, 'b': 2}, {'a': 99})
        assert result['b'] == 2

    def test_nested_dict_merged_recursively(self):
        base = {'models': {'segmentation': {'path': 'old', 'use_first_token_of_block': False}}}
        overlay = {'models': {'segmentation': {'path': 'new'}}}
        result = _deep_merge(base, overlay)
        assert result['models']['segmentation']['path'] == 'new'
        assert result['models']['segmentation']['use_first_token_of_block'] is False

    def test_overlay_adds_new_nested_key(self):
        base = {'models': {'segmentation': {'path': 'old'}}}
        overlay = {'models': {'header': {'path': 'new_header'}}}
        result = _deep_merge(base, overlay)
        assert result['models']['segmentation']['path'] == 'old'
        assert result['models']['header']['path'] == 'new_header'

    def test_does_not_mutate_base(self):
        base = {'a': {'b': 1}}
        _deep_merge(base, {'a': {'b': 2}})
        assert base['a']['b'] == 1


class TestResolveSequenceModelProfile:
    def test_returns_profile_without_extends_unchanged(self):
        seq_profiles = {
            'base': {'segmentation': {'path': 'base/segmentation'}},
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'base')
        assert result == {'segmentation': {'path': 'base/segmentation'}}

    def test_merges_extended_profile(self):
        seq_profiles = {
            'base': {
                'segmentation': {'path': 'base/segmentation', 'engine': 'wapiti'},
                'header': {'path': 'base/header', 'engine': 'wapiti'},
            },
            'child': {
                'extends': 'base',
                'header': {'path': 'child/header'},
            },
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'child')
        assert result['segmentation'] == {'path': 'base/segmentation', 'engine': 'wapiti'}
        assert result['header'] == {'path': 'child/header', 'engine': 'wapiti'}
        assert 'extends' not in result

    def test_supports_chained_extends(self):
        seq_profiles = {
            'grandparent': {'segmentation': {'path': 'gp/segmentation'}},
            'parent': {'extends': 'grandparent', 'header': {'path': 'p/header'}},
            'child': {'extends': 'parent', 'table': {'path': 'c/table'}},
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'child')
        assert result['segmentation'] == {'path': 'gp/segmentation'}
        assert result['header'] == {'path': 'p/header'}
        assert result['table'] == {'path': 'c/table'}

    def test_merges_a_list_of_bases(self):
        seq_profiles = {
            'base_1': {'segmentation': {'path': 'base_1/segmentation'}},
            'base_2': {'header': {'path': 'base_2/header'}},
            'child': {'extends': ['base_1', 'base_2'], 'table': {'path': 'child/table'}},
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'child')
        assert result['segmentation'] == {'path': 'base_1/segmentation'}
        assert result['header'] == {'path': 'base_2/header'}
        assert result['table'] == {'path': 'child/table'}

    def test_later_base_wins_over_earlier_base(self):
        seq_profiles = {
            'base_1': {'header': {'path': 'base_1/header', 'engine': 'wapiti'}},
            'base_2': {'header': {'path': 'base_2/header'}},
            'child': {'extends': ['base_1', 'base_2']},
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'child')
        assert result['header'] == {'path': 'base_2/header', 'engine': 'wapiti'}

    def test_own_keys_win_over_every_base(self):
        seq_profiles = {
            'base_1': {'header': {'path': 'base_1/header'}},
            'base_2': {'header': {'path': 'base_2/header'}},
            'child': {'extends': ['base_1', 'base_2'], 'header': {'path': 'child/header'}},
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'child')
        assert result['header'] == {'path': 'child/header'}

    def test_applies_a_shared_base_once_before_the_profiles_overriding_it(self):
        seq_profiles = {
            'shared': {
                'segmentation': {'path': 'shared/segmentation', 'engine': 'wapiti'},
                'header': {'path': 'shared/header', 'engine': 'wapiti'},
                'citation': {'path': 'shared/citation', 'engine': 'wapiti'},
            },
            'header_only': {'extends': 'shared', 'header': {'path': 'header_only/header'}},
            'citation_only': {'extends': 'shared', 'citation': {'engine': 'llm'}},
            'combined': {'extends': ['header_only', 'citation_only']},
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'combined')
        assert result['segmentation'] == {'path': 'shared/segmentation', 'engine': 'wapiti'}
        assert result['header'] == {'path': 'header_only/header', 'engine': 'wapiti'}
        assert result['citation'] == {'path': 'shared/citation', 'engine': 'llm'}

    def test_ignores_a_repeated_base_name(self):
        seq_profiles = {
            'base_1': {'header': {'path': 'base_1/header'}},
            'base_2': {'header': {'path': 'base_2/header'}},
            'child': {'extends': ['base_1', 'base_2', 'base_1']},
        }
        result = _resolve_sequence_model_profile(seq_profiles, 'child')
        assert result['header'] == {'path': 'base_2/header'}

    def test_treats_an_empty_list_as_no_base(self):
        seq_profiles = {'child': {'extends': [], 'header': {'path': 'child/header'}}}
        result = _resolve_sequence_model_profile(seq_profiles, 'child')
        assert result == {'header': {'path': 'child/header'}}

    def test_raises_on_unknown_profile(self):
        with pytest.raises(ValueError, match='Unknown sequence_model_profile'):
            _resolve_sequence_model_profile({}, 'missing')

    def test_raises_on_unknown_extends_target(self):
        seq_profiles = {'child': {'extends': 'missing'}}
        with pytest.raises(ValueError, match='Unknown sequence_model_profile'):
            _resolve_sequence_model_profile(seq_profiles, 'child')

    def test_raises_on_circular_extends(self):
        seq_profiles = {
            'a': {'extends': 'b'},
            'b': {'extends': 'a'},
        }
        with pytest.raises(ValueError, match='Circular extends'):
            _resolve_sequence_model_profile(seq_profiles, 'a')

    def test_reports_the_circular_chain_in_the_order_it_was_followed(self):
        seq_profiles = {
            'alpha': {'extends': 'beta'},
            'beta': {'extends': 'gamma'},
            'gamma': {'extends': 'alpha'},
        }
        with pytest.raises(ValueError) as exc_info:
            _resolve_sequence_model_profile(seq_profiles, 'alpha')
        assert 'chain: alpha -> beta -> gamma -> alpha' in str(exc_info.value)

    def test_raises_on_unknown_name_in_a_list(self):
        seq_profiles = {
            'base': {'header': {'path': 'base/header'}},
            'child': {'extends': ['base', 'missing']},
        }
        with pytest.raises(ValueError, match="Unknown sequence_model_profile 'missing'"):
            _resolve_sequence_model_profile(seq_profiles, 'child')

    def test_raises_on_circular_extends_through_a_list_entry(self):
        seq_profiles = {
            'base': {'header': {'path': 'base/header'}},
            'child': {'extends': ['base', 'cycle']},
            'cycle': {'extends': 'child'},
        }
        with pytest.raises(ValueError) as exc_info:
            _resolve_sequence_model_profile(seq_profiles, 'child')
        assert 'chain: child -> cycle -> child' in str(exc_info.value)

    def test_raises_on_non_string_entry_in_a_list(self):
        seq_profiles = {'child': {'extends': ['base', 123]}}
        with pytest.raises(ValueError) as exc_info:
            _resolve_sequence_model_profile(seq_profiles, 'child')
        assert "sequence_model_profile 'child'" in str(exc_info.value)
        assert '123' in str(exc_info.value)

    @pytest.mark.parametrize('extends', [{'name': 'base'}, {}, 123])
    def test_raises_on_extends_that_is_neither_a_string_nor_a_list(self, extends):
        seq_profiles = {'child': {'extends': extends}}
        with pytest.raises(ValueError) as exc_info:
            _resolve_sequence_model_profile(seq_profiles, 'child')
        assert "sequence_model_profile 'child'" in str(exc_info.value)
        assert repr(extends) in str(exc_info.value)


class TestAppConfigResolveProfile:
    def _make_config(self, extra: Optional[dict] = None) -> AppConfig:
        props = dict(MINIMAL_PROFILE_CONFIG)
        if extra:
            props = {**props, **extra}
        return AppConfig(props)

    def test_applies_sequence_model_profile(self):
        config = self._make_config().resolve_profile('profile_b')
        assert config['models']['segmentation']['path'] == 'path_b/segmentation'
        assert config['models']['segmentation']['engine'] == 'wapiti'
        assert config['models']['header']['path'] == 'path_b/header'

    def test_applies_extended_sequence_model_profile(self):
        config = self._make_config().resolve_profile('profile_b_extended')
        assert config['models']['segmentation']['path'] == 'path_b/segmentation'
        assert config['models']['header']['path'] == 'path_b_extended/header'

    def test_inherits_base_model_keys_not_in_profile(self):
        config = self._make_config().resolve_profile('profile_a')
        assert config['models']['segmentation']['path'] == 'path_a/segmentation'
        assert config['models']['segmentation']['use_first_token_of_block'] is False

    def test_uses_default_profile_when_no_name_given(self):
        config = self._make_config().resolve_profile()
        assert config['models']['segmentation']['path'] == 'path_a/segmentation'

    def test_resolves_alias(self):
        config = self._make_config().resolve_profile('alias_a')
        assert config['models']['segmentation']['path'] == 'path_a/segmentation'

    def test_profile_with_extra_config_section(self):
        config = self._make_config().resolve_profile('profile_with_extra')
        assert config['models']['segmentation']['path'] == 'path_a/segmentation'
        assert config['processors']['fulltext']['use_cv_model'] is True

    def test_returns_self_when_no_profile_configured(self):
        props = {k: v for k, v in MINIMAL_PROFILE_CONFIG.items() if k != 'profile'}
        config = AppConfig(props)
        result = config.resolve_profile()
        assert result is config

    def test_raises_on_unknown_profile(self):
        config = self._make_config()
        with pytest.raises(ValueError, match='Unknown profile'):
            config.resolve_profile('nonexistent')

    def test_raises_on_unknown_alias_target(self):
        props = {
            **MINIMAL_PROFILE_CONFIG,
            'profile_aliases': {'broken_alias': 'does_not_exist'},
        }
        config = AppConfig(props)
        with pytest.raises(ValueError, match="alias for 'does_not_exist'"):
            config.resolve_profile('broken_alias')

    def test_raises_on_unknown_sequence_model_profile(self):
        props = {
            **MINIMAL_PROFILE_CONFIG,
            'profiles': {'bad': {'sequence_models': 'missing_seq_profile'}},
        }
        config = AppConfig(props)
        with pytest.raises(ValueError, match='sequence_model_profile'):
            config.resolve_profile('bad')

    def test_get_active_profile_name_with_explicit_name(self):
        config = self._make_config()
        assert config.get_active_profile_name('alias_a') == 'profile_a'

    def test_get_active_profile_name_from_default(self):
        config = self._make_config()
        assert config.get_active_profile_name() == 'profile_a'

    def test_get_active_profile_name_returns_none_when_not_configured(self):
        props = {k: v for k, v in MINIMAL_PROFILE_CONFIG.items() if k != 'profile'}
        config = AppConfig(props)
        assert config.get_active_profile_name() is None

    def test_raises_unknown_profile_error_on_unknown_profile(self):
        config = self._make_config()
        with pytest.raises(UnknownProfileError):
            config.resolve_profile('nonexistent')


class TestAppConfigValidateProfiles:
    """A profile may only change what a request is served, not how it is served.

    Everything outside `PROFILE_OVERLAY_KEYS` is built once and shared, and
    nothing about those objects is keyed on the configuration they came from, so
    a profile able to set one would be served the deployment's value with no
    test able to see the difference.
    """

    def test_accepts_a_profile_that_only_names_models(self):
        config = AppConfig(MINIMAL_PROFILE_CONFIG)
        assert config.validate_profiles() is config

    def test_accepts_processors(self):
        config = AppConfig({
            **MINIMAL_PROFILE_CONFIG,
            'profiles': {'a': {'processors': {'fulltext': {'use_cv_model': True}}}}
        })
        assert config.validate_profiles() is config

    def test_rejects_a_profile_setting_shared_configuration(self):
        config = AppConfig({
            **MINIMAL_PROFILE_CONFIG,
            'profiles': {'a': {'lookup': {'country': {}}}}
        })
        with pytest.raises(InvalidProfileError) as exc_info:
            config.validate_profiles()
        assert "'a'" in str(exc_info.value)
        assert 'lookup' in str(exc_info.value)

    def test_names_every_key_it_rejects(self):
        config = AppConfig({
            **MINIMAL_PROFILE_CONFIG,
            'profiles': {'a': {'download_dir': '/tmp', 'llm_response_cache_dir': '/tmp'}}
        })
        with pytest.raises(InvalidProfileError) as exc_info:
            config.validate_profiles()
        assert 'download_dir' in str(exc_info.value)
        assert 'llm_response_cache_dir' in str(exc_info.value)

    def test_get_profile_names_lists_what_the_config_declares(self):
        assert AppConfig(MINIMAL_PROFILE_CONFIG).get_profile_names() == [
            'profile_a', 'profile_b', 'profile_b_extended', 'profile_with_extra'
        ]


@pytest.fixture(name='env_vars_mock')
def _env_vars_mock() -> Iterable[dict]:
    mock: dict
    with patch('os.environ', {}) as mock:
        yield mock


class TestGetLlmResponseCacheDir:
    def test_should_be_none_when_the_key_is_absent(self):
        assert get_llm_response_cache_dir({}) is None

    def test_should_be_none_when_the_key_is_set_to_nothing(self):
        # How the shipped config spells it, and what an empty
        # `SCIENCEBEAM_PARSER__LLM_RESPONSE_CACHE_DIR` parses to, so turning it
        # off by config and by environment reach the same value.
        assert get_llm_response_cache_dir({'llm_response_cache_dir': None}) is None

    def test_should_expand_a_home_relative_path(self):
        expanded = get_llm_response_cache_dir({'llm_response_cache_dir': '~/cache'})
        assert expanded and not expanded.startswith('~')

    def test_should_read_it_from_an_app_config(self, tmp_path: Path):
        config_path = tmp_path / 'config.yml'
        config_path.write_text(yaml.dump({'llm_response_cache_dir': '/tmp/llm'}))
        assert get_llm_response_cache_dir(AppConfig.load_yaml(str(config_path))) == '/tmp/llm'

    def test_should_be_none_in_the_shipped_config(self):
        # Requirement: enabling it is deliberate, since the entries hold
        # document text.
        assert get_llm_response_cache_dir(
            AppConfig.load_yaml(DEFAULT_CONFIG_FILE)
        ) is None


class TestAppConfig:
    def test_should_load_yaml(self, tmp_path: Path):
        config_path = tmp_path / 'config.yml'
        config_path.write_text(yaml.dump({
            'key1': 'value1'
        }))
        config = AppConfig.load_yaml(str(config_path))
        assert config.props['key1'] == 'value1'

    def test_should_override_top_level_value_with_env_var(
        self,
        tmp_path: Path,
        env_vars_mock: dict
    ):
        env_vars_mock['SCIENCEBEAM_PARSER__KEY1'] = 'updated value1'
        config_path = tmp_path / 'config.yml'
        config_path.write_text(yaml.dump({
            'key1': 'value1'
        }))
        config = AppConfig.load_yaml(str(config_path))
        config = config.apply_environment_variables()
        assert config.props['key1'] == 'updated value1'

    def test_should_override_nested_value_with_env_var(
        self,
        tmp_path: Path,
        env_vars_mock: dict
    ):
        env_vars_mock['SCIENCEBEAM_PARSER__PARENT1__KEY1'] = 'updated value1'
        config_path = tmp_path / 'config.yml'
        config_path.write_text(yaml.dump({
            'parent1': {
                'key1': 'original value1'
            }
        }))
        original_config = AppConfig.load_yaml(str(config_path))
        config = original_config.apply_environment_variables()
        assert config.props['parent1']['key1'] == 'updated value1'
        assert original_config.props['parent1']['key1'] == 'original value1'

    def test_should_override_int_value_with_env_var(
        self,
        tmp_path: Path,
        env_vars_mock: dict
    ):
        env_vars_mock['SCIENCEBEAM_PARSER__KEY1'] = '222'
        config_path = tmp_path / 'config.yml'
        config_path.write_text(yaml.dump({
            'key1': 111
        }))
        config = AppConfig.load_yaml(str(config_path))
        config = config.apply_environment_variables()
        assert config.props['key1'] == 222

    def test_should_override_bool_value_with_env_var(
        self,
        tmp_path: Path,
        env_vars_mock: dict
    ):
        env_vars_mock['SCIENCEBEAM_PARSER__KEY1'] = 'false'
        config_path = tmp_path / 'config.yml'
        config_path.write_text(yaml.dump({
            'key1': True
        }))
        config = AppConfig.load_yaml(str(config_path))
        config = config.apply_environment_variables()
        assert config.props['key1'] is False
