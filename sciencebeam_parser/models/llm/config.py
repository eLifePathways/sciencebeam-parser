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
    reasoning_enabled: Optional[bool] = None
    temperature: float = 0.0
    timeout_seconds: float = 300.0
    max_output_tokens: int = 16000
    max_attempts: int = 5
    record_trace_content: bool = True
    warn_input_lines: int = 300
    max_input_lines: int = 0
    mark_furniture: bool = False
    mark_blocks: bool = False
    warn_unclaimed_line_share: float = 0.1
    max_references_per_request: int = 10
    max_missing_reference_retries: int = 1
    max_malformed_response_retries: int = 1
    max_concurrent_requests: int = 4
    evidence_mismatch_raises: bool = False
    dropped_field_raises: bool = False
    unanswered_reference_raises: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reasoning_enabled is not None and not isinstance(self.reasoning_enabled, bool):
            raise LlmConfigError(
                'reasoning_enabled must be true or false, but is'
                f' {self.reasoning_enabled!r}: write it unquoted, since yaml reads a quoted'
                " 'false' as a string. Provider parameters the engine does not model go under"
                " extra_body, e.g. extra_body: {reasoning: {effort: 'low'}}"
            )
        if self.reasoning_enabled is not None and 'reasoning' in self.extra_body:
            raise LlmConfigError(
                'reasoning_enabled and extra_body.reasoning both set the reasoning request'
                ' parameter, and the first silently wins: keep whichever one is meant'
            )

    @staticmethod
    def from_model_config(config: Mapping[str, Any]) -> 'LlmEngineConfig':
        for required in ('task', 'model', 'prompt_version'):
            if not config.get(required):
                raise LlmConfigError(f'llm engine requires {required!r} in the model config')
        if 'reasoning' in config:
            raise LlmConfigError(
                "'reasoning' is no longer accepted, because every value except 'off' was"
                ' ignored: write reasoning_enabled: false (unquoted) for what was'
                " reasoning: 'off', and put an effort level under extra_body, e.g."
                " extra_body: {reasoning: {effort: 'low'}}"
            )
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
