from typing import Any, Dict

import pytest

from sciencebeam_parser.models.llm.client import (
    LlmRequestError,
    LlmTruncatedResponseError,
    get_error_status_code,
    get_response_content
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
