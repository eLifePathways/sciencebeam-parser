from unittest.mock import MagicMock

import pytest

from sciencebeam_parser.models.model_impl import ModelImpl
from sciencebeam_parser.models.model_impl_factory import (
    EngineNames,
    get_engine_name_for_config,
    get_model_impl_for_config
)


class TestGetEngineNameForConfig:
    def test_should_return_wapiti_if_engine_prop_is_wapiti(self):
        assert get_engine_name_for_config({
            'engine': EngineNames.WAPITI
        }) == EngineNames.WAPITI

    def test_should_return_delft_without_engine_prop(self):
        assert get_engine_name_for_config({}) == EngineNames.DELFT

    def test_should_return_delft_if_engine_prop_is_blank(self):
        assert get_engine_name_for_config({
            'engine': ''
        }) == EngineNames.DELFT


class TestGetModelImplForConfig:
    def test_should_not_require_a_path_for_the_llm_engine(self):
        model_impl = get_model_impl_for_config({
            'engine': EngineNames.LLM,
            'task': 'reference_segmenter',
            'model': 'qwen/qwen3.5-9b',
            'prompt_version': 'lines-v1',
        }, app_context=MagicMock(name='app_context'))
        assert isinstance(model_impl, ModelImpl)

    def test_should_reject_an_invalid_engine_name(self):
        with pytest.raises(RuntimeError):
            get_model_impl_for_config(
                {'engine': 'other', 'path': 'x'},
                app_context=MagicMock(name='app_context')
            )
