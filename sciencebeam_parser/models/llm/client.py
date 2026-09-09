import logging
import random
import time
from typing import Any, Dict, Mapping, Optional, Protocol

import httpx

from sciencebeam_parser.models.llm.config import LlmEngineConfig, get_api_key


LOGGER = logging.getLogger(__name__)

RETRY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})

RATE_LIMIT_STATUS_CODE = 429

RETRY_BASE_SECONDS = 2.0
# A rate limit is a window to outlast rather than a blip to retry through: the
# provider throttles this model upstream for around a minute at a time, and an
# exponential backoff from 2s spends every attempt inside the same window.
RATE_LIMIT_BASE_SECONDS = 8.0
MAX_RETRY_SECONDS = 30.0


def get_retry_after_seconds(headers: Mapping[str, str]) -> Optional[float]:
    """The provider's own answer to when to come back, when it gives one."""
    value = headers.get('retry-after') or headers.get('Retry-After')
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # an HTTP-date form, which the backoff covers anyway


def get_retry_delay(
    attempt: int,
    status_code: Optional[int] = None,
    retry_after: Optional[float] = None,
    jitter: Optional[float] = None
) -> float:
    """Seconds to wait before `attempt` (1 for the first retry).

    Jittered because requests are issued in parallel: workers throttled together
    would otherwise back off by the same amount and collide on every retry, which
    is what turns one rate-limited moment into a whole batch of lost documents.
    """
    if retry_after is not None:
        return min(retry_after, MAX_RETRY_SECONDS)
    base = (
        RATE_LIMIT_BASE_SECONDS if status_code == RATE_LIMIT_STATUS_CODE
        else RETRY_BASE_SECONDS
    )
    delay = min(base * (2 ** (attempt - 1)), MAX_RETRY_SECONDS)
    spread = jitter if jitter is not None else random.random()
    return delay * (0.5 + 0.5 * spread)


def get_error_status_code(response_json: Mapping[str, Any]) -> Optional[int]:
    """OpenRouter returns some upstream failures as http 200 with an error body,
    so the status code alone does not say whether a call is worth retrying."""
    error = response_json.get('error')
    if not isinstance(error, dict):
        return None
    code = error.get('code')
    return code if isinstance(code, int) else 0


class LlmRequestError(RuntimeError):
    pass


class LlmCompletionClient(Protocol):
    def validate_configuration(self) -> None:
        ...

    def get_completion(
        self, prompt: str, response_schema: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        ...


class LlmClient:
    def __init__(self, config: LlmEngineConfig):
        self.config = config

    def _headers(self) -> Dict[str, str]:
        return {'Authorization': f'Bearer {get_api_key()}'}

    def _request_body(self, prompt: str, response_schema: Mapping[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            'model': self.config.model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_output_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'sciencebeam_labels',
                    'strict': True,
                    'schema': response_schema,
                },
            },
            'provider': self.config.provider_routing,
            **self.config.extra_body,
        }
        if self.config.reasoning == 'off':
            body['reasoning'] = {'enabled': False}
        return body

    def validate_configuration(self) -> None:
        """Fails at load rather than at first request, and spends no tokens."""
        url = f'{self.config.endpoint.rstrip("/")}/models'
        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise LlmRequestError(f'{self.config.endpoint} is not reachable: {exc}') from exc
        if response.status_code != 200:
            raise LlmRequestError(
                f'{url} returned {response.status_code}: {response.text[:200]}'
            )
        model_ids = {
            entry.get('id') for entry in response.json().get('data', [])
            if isinstance(entry, dict)
        }
        if model_ids and self.config.model not in model_ids:
            raise LlmRequestError(
                f'{self.config.endpoint} does not offer model {self.config.model!r}'
            )
        LOGGER.info(
            'llm engine configured: model=%r provider=%r prompt=%r shape=%r',
            self.config.model, self.config.provider, self.config.prompt_version,
            self.config.response_shape
        )

    def get_completion(self, prompt: str, response_schema: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._post_with_retry(prompt, response_schema)

    def _post_with_retry(
        self, prompt: str, response_schema: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        url = f'{self.config.endpoint.rstrip("/")}/chat/completions'
        body = self._request_body(prompt, response_schema)
        last_error = ''
        retry_status: Optional[int] = None
        retry_after: Optional[float] = None
        for attempt in range(self.config.max_attempts):
            if attempt:
                delay = get_retry_delay(attempt, retry_status, retry_after)
                LOGGER.info(
                    'llm retrying in %.1fs (attempt %d of %d, last: %s)',
                    delay, attempt + 1, self.config.max_attempts, last_error[:80]
                )
                time.sleep(delay)
            try:
                with httpx.Client(timeout=self.config.timeout_seconds) as client:
                    response = client.post(url, headers=self._headers(), json=body)
            except httpx.HTTPError as exc:
                last_error = f'{type(exc).__name__}: {exc}'
                retry_status, retry_after = None, None
                continue
            if response.status_code in RETRY_STATUS_CODES:
                last_error = f'http {response.status_code}: {response.text[:200]}'
                retry_status = response.status_code
                retry_after = get_retry_after_seconds(response.headers)
                continue
            if response.status_code != 200:
                raise LlmRequestError(
                    f'http {response.status_code}: {response.text[:200]}'
                )
            response_json = response.json()
            error_code = get_error_status_code(response_json)
            if error_code is None:
                return response_json
            last_error = f'error body {error_code}: {str(response_json)[:200]}'
            if error_code in RETRY_STATUS_CODES:
                retry_status = error_code
                retry_after = get_retry_after_seconds(response.headers)
                continue
            raise LlmRequestError(last_error)
        raise LlmRequestError(
            f'giving up after {self.config.max_attempts} attempts: {last_error}'
        )


class LlmTruncatedResponseError(LlmRequestError):
    pass


def get_response_content(response_json: Mapping[str, Any]) -> str:
    choices = response_json.get('choices')
    if not choices:
        raise LlmRequestError(f'response has no choices: {str(response_json)[:200]}')
    choice = choices[0]
    content = choice.get('message', {}).get('content')
    finish_reason = choice.get('finish_reason') or choice.get('native_finish_reason')
    if finish_reason == 'length':
        completion_tokens = (
            response_json.get('usage', {}).get('completion_tokens')
        )
        raise LlmTruncatedResponseError(
            'response hit the output token limit'
            f' (finish_reason={finish_reason!r},'
            f' completion_tokens={completion_tokens},'
            f' chars={len(content or "")});'
            ' raise max_output_tokens or reduce the input per request'
        )
    if not content:
        raise LlmRequestError(
            f'response content is empty (finish_reason={finish_reason!r})'
        )
    return content
