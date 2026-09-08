from typing import Any, Dict

import pytest

from sciencebeam_parser.models.llm.client import (
    MAX_RETRY_SECONDS,
    LlmRequestError,
    LlmTruncatedResponseError,
    get_error_status_code,
    get_response_content,
    get_retry_after_seconds,
    get_retry_delay
)


def get_response(content: Any = '{"starts": [0]}', **choice_extra) -> Dict[str, Any]:
    return {
        'choices': [{'message': {'content': content}, **choice_extra}],
        'usage': {'completion_tokens': 8000},
    }


class TestGetResponseContent:
    def test_should_return_the_content(self):
        assert get_response_content(get_response()) == '{"starts": [0]}'

    def test_should_raise_a_truncation_error_when_the_output_limit_was_hit(self):
        with pytest.raises(LlmTruncatedResponseError, match='output token limit'):
            get_response_content(get_response(finish_reason='length'))

    def test_should_name_the_completion_token_count_in_a_truncation_error(self):
        with pytest.raises(LlmTruncatedResponseError, match='completion_tokens=8000'):
            get_response_content(get_response(finish_reason='length'))

    def test_should_detect_truncation_reported_only_as_native_finish_reason(self):
        with pytest.raises(LlmTruncatedResponseError):
            get_response_content(get_response(native_finish_reason='length'))

    def test_should_not_treat_a_normal_stop_as_truncation(self):
        assert get_response_content(get_response(finish_reason='stop'))

    def test_should_raise_for_no_choices(self):
        with pytest.raises(LlmRequestError, match='no choices'):
            get_response_content({'choices': []})

    def test_should_raise_for_empty_content(self):
        with pytest.raises(LlmRequestError, match='empty'):
            get_response_content(get_response(content=''))


class TestGetErrorStatusCode:
    def test_should_be_none_for_a_normal_response(self):
        assert get_error_status_code(get_response()) is None

    def test_should_return_the_code_from_an_error_body(self):
        assert get_error_status_code(
            {'error': {'message': 'error code: 524', 'code': 504}}
        ) == 504

    def test_should_return_zero_for_an_error_body_without_a_code(self):
        assert get_error_status_code({'error': {'message': 'nope'}}) == 0


class TestGetRetryAfterSeconds:
    def test_should_read_the_header(self):
        assert get_retry_after_seconds({'retry-after': '12'}) == 12.0

    def test_should_read_it_case_insensitively(self):
        assert get_retry_after_seconds({'Retry-After': '3'}) == 3.0

    def test_should_be_none_when_absent(self):
        assert get_retry_after_seconds({}) is None

    def test_should_be_none_for_an_http_date(self):
        # the backoff covers this case; parsing dates is not worth the surface
        assert get_retry_after_seconds({'retry-after': 'Wed, 21 Oct 2026 07:28:00 GMT'}) is None


class TestGetRetryDelay:
    """A rate limit is a window to outlast, not a blip to retry through, and
    parallel workers must not come back in lockstep.
    """
    def test_should_wait_longer_for_a_rate_limit_than_for_a_server_error(self):
        assert (
            get_retry_delay(1, 429, jitter=1.0)
            > get_retry_delay(1, 500, jitter=1.0)
        )

    def test_should_outlast_a_minute_of_rate_limiting_across_the_attempts(self):
        total = sum(get_retry_delay(a, 429, jitter=1.0) for a in (1, 2, 3, 4))
        assert total >= 60

    def test_should_grow_with_each_attempt(self):
        delays = [get_retry_delay(a, 429, jitter=1.0) for a in (1, 2, 3)]
        assert delays == sorted(delays)
        assert delays[0] < delays[-1]

    def test_should_cap_the_wait(self):
        assert get_retry_delay(99, 429, jitter=1.0) == MAX_RETRY_SECONDS

    def test_should_prefer_the_provider_retry_after(self):
        assert get_retry_delay(1, 429, retry_after=11.0) == 11.0

    def test_should_cap_an_implausible_retry_after(self):
        assert get_retry_delay(1, 429, retry_after=8000.0) == MAX_RETRY_SECONDS

    def test_should_vary_with_jitter_so_workers_do_not_collide(self):
        assert get_retry_delay(2, 429, jitter=0.0) != get_retry_delay(2, 429, jitter=1.0)

    def test_should_never_return_more_than_the_undelayed_delay(self):
        for attempt in (1, 2, 3, 4):
            assert (
                get_retry_delay(attempt, 429, jitter=0.0)
                <= get_retry_delay(attempt, 429, jitter=1.0)
            )
