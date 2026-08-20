import pytest

from sciencebeam_parser.models.llm.config import LlmConfigError, LlmEngineConfig


MINIMAL_CONFIG = {
    'task': 'reference_segmenter',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'lines-v1',
}


class TestLlmEngineConfig:
    def test_should_read_minimal_config(self):
        config = LlmEngineConfig.from_model_config(MINIMAL_CONFIG)
        assert config.task == 'reference_segmenter'
        assert config.model == 'qwen/qwen3.5-9b'
        assert config.response_shape == 'lines'
        assert config.temperature == 0.0

    def test_should_ignore_unrelated_keys(self):
        config = LlmEngineConfig.from_model_config({**MINIMAL_CONFIG, 'engine': 'llm'})
        assert config.model == 'qwen/qwen3.5-9b'

    @pytest.mark.parametrize('missing', ['task', 'model', 'prompt_version'])
    def test_should_reject_missing_required_key(self, missing: str):
        config = {key: value for key, value in MINIMAL_CONFIG.items() if key != missing}
        with pytest.raises(LlmConfigError):
            LlmEngineConfig.from_model_config(config)

    def test_should_reject_free_model_id(self):
        with pytest.raises(LlmConfigError):
            LlmEngineConfig.from_model_config({
                **MINIMAL_CONFIG, 'model': 'qwen/qwen3.5-9b:free'
            })


class TestProviderRouting:
    def test_should_enforce_zero_retention_and_fail_closed(self):
        routing = LlmEngineConfig.from_model_config(MINIMAL_CONFIG).provider_routing
        assert routing['zdr'] is True
        assert routing['data_collection'] == 'deny'
        assert routing['allow_fallbacks'] is False
        assert routing['require_parameters'] is True

    def test_should_pin_provider_when_configured(self):
        routing = LlmEngineConfig.from_model_config({
            **MINIMAL_CONFIG, 'provider': 'siliconflow'
        }).provider_routing
        assert routing['only'] == ['siliconflow']

    def test_should_not_pin_provider_when_not_configured(self):
        assert 'only' not in LlmEngineConfig.from_model_config(MINIMAL_CONFIG).provider_routing
