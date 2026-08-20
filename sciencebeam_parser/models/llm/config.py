import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional


API_KEY_ENV_NAMES = ('SCIENCEBEAM_LLM_API_KEY', 'OPENROUTER_API_KEY')

DEFAULT_ENDPOINT = 'https://openrouter.ai/api/v1'


class LlmConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LlmEngineConfig:
    task: str
    model: str
    prompt_version: str
    response_shape: str = 'lines'
    endpoint: str = DEFAULT_ENDPOINT
    provider: Optional[str] = None
    reasoning: str = ''
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    max_output_tokens: int = 8000
    max_attempts: int = 4
    record_trace_content: bool = True
    warn_input_lines: int = 300
    max_input_lines: int = 0
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_model_config(config: Mapping[str, Any]) -> 'LlmEngineConfig':
        for required in ('task', 'model', 'prompt_version'):
            if not config.get(required):
                raise LlmConfigError(f'llm engine requires {required!r} in the model config')
        model = config['model']
        if model.endswith(':free'):
            raise LlmConfigError(
                f'refusing model id {model!r}: the free tier requires allowing training on'
                ' prompts, which the zero-retention requirement forbids'
            )
        known = {field_.name for field_ in fields(LlmEngineConfig)}
        return LlmEngineConfig(**{
            key: value for key, value in config.items()
            if key in known
        })

    @property
    def provider_routing(self) -> Dict[str, Any]:
        routing: Dict[str, Any] = {
            'zdr': True,
            'data_collection': 'deny',
            'allow_fallbacks': False,
            'require_parameters': True,
        }
        if self.provider:
            routing['only'] = [self.provider]
        return routing


def get_api_key() -> str:
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    raise LlmConfigError(
        'no api key: set one of ' + ', '.join(API_KEY_ENV_NAMES)
    )
