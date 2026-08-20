import json

import pytest

from sciencebeam_parser.models.llm.decode import (
    LlmResponseError,
    decode_line_starts_response,
    get_line_numbers,
    get_lines,
    render_numbered_lines,
    snap_starts_to_label_lines
)


# "1 ." on its own line, then two lines of reference text, then a second reference
TOKENS = [
    '1', '.', 'Fleming', 'PS', ',', 'Koletsi', 'D', 'High', 'quality',
    '2', '.', 'Howick', 'J',
]
LINE_STATUS = [
    'LINESTART', 'LINEEND',
    'LINESTART', 'LINEIN', 'LINEIN', 'LINEIN', 'LINEEND',
    'LINESTART', 'LINEEND',
    'LINESTART', 'LINEEND',
    'LINESTART', 'LINEEND',
]


def response(starts) -> str:
    return json.dumps({'starts': starts})


class TestGetLineNumbers:
    def test_should_increment_on_each_line_start(self):
        assert get_line_numbers(LINE_STATUS) == [0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4]


class TestRenderNumberedLines:
    def test_should_prefix_each_line_with_its_number(self):
        rendered = render_numbered_lines(TOKENS, get_line_numbers(LINE_STATUS))
        assert rendered.splitlines()[0] == '0\t1 .'
        assert rendered.splitlines()[1] == '1\tFleming PS , Koletsi D'


class TestDecodeLineStartsResponse:
    def test_should_label_a_bare_number_line_as_label_and_the_rest_as_reference(self):
        labeled = decode_line_starts_response(response([0, 3]), TOKENS, LINE_STATUS)
        assert [label for _, label in labeled] == [
            'B-<label>', 'I-<label>',
            'B-<reference>', 'I-<reference>', 'I-<reference>', 'I-<reference>', 'I-<reference>',
            'I-<reference>', 'I-<reference>',
            'B-<label>', 'I-<label>',
            'B-<reference>', 'I-<reference>',
        ]

    def test_should_return_the_input_tokens_unchanged(self):
        labeled = decode_line_starts_response(response([0, 3]), TOKENS, LINE_STATUS)
        assert [token for token, _ in labeled] == TOKENS

    def test_should_label_tokens_before_the_first_reference_as_other(self):
        labeled = decode_line_starts_response(response([3]), TOKENS, LINE_STATUS)
        assert [label for _, label in labeled][:9] == ['O'] * 9

    def test_should_raise_for_line_number_out_of_range(self):
        with pytest.raises(LlmResponseError, match='out of range'):
            decode_line_starts_response(response([0, 99]), TOKENS, LINE_STATUS)

    def test_should_raise_for_negative_line_number(self):
        with pytest.raises(LlmResponseError, match='out of range'):
            decode_line_starts_response(response([-1]), TOKENS, LINE_STATUS)

    def test_should_raise_for_non_ascending_line_numbers(self):
        with pytest.raises(LlmResponseError, match='ascending'):
            decode_line_starts_response(response([3, 0]), TOKENS, LINE_STATUS)

    def test_should_raise_for_duplicate_line_numbers(self):
        with pytest.raises(LlmResponseError, match='ascending'):
            decode_line_starts_response(response([0, 0]), TOKENS, LINE_STATUS)

    def test_should_raise_for_malformed_json(self):
        with pytest.raises(LlmResponseError, match='not json'):
            decode_line_starts_response('{"starts": [0', TOKENS, LINE_STATUS)

    def test_should_raise_for_truncated_response_missing_starts(self):
        with pytest.raises(LlmResponseError, match='no "starts"'):
            decode_line_starts_response('{}', TOKENS, LINE_STATUS)

    def test_should_raise_for_empty_starts(self):
        with pytest.raises(LlmResponseError, match='empty'):
            decode_line_starts_response(response([]), TOKENS, LINE_STATUS)

    def test_should_raise_for_non_integer_line_number(self):
        with pytest.raises(LlmResponseError, match='not an integer'):
            decode_line_starts_response(response(['0']), TOKENS, LINE_STATUS)

    def test_should_raise_when_feature_rows_do_not_match_tokens(self):
        with pytest.raises(LlmResponseError, match='does not match'):
            decode_line_starts_response(response([0]), TOKENS, LINE_STATUS[:-1])


class TestSnapStartsToLabelLines:
    def test_should_move_a_start_onto_a_preceding_bare_label_line(self):
        labeled = decode_line_starts_response(response([1, 4]), TOKENS, LINE_STATUS)
        assert [label for _, label in labeled][:3] == [
            'B-<label>', 'I-<label>', 'B-<reference>'
        ]

    def test_should_recover_the_label_for_every_reference(self):
        labeled = decode_line_starts_response(response([1, 4]), TOKENS, LINE_STATUS)
        labels = [label for _, label in labeled]
        assert labels.count('B-<label>') == 2
        assert labels.count('B-<reference>') == 2

    def test_should_leave_a_start_that_is_already_on_the_label_line(self):
        labeled = decode_line_starts_response(response([0, 3]), TOKENS, LINE_STATUS)
        assert [label for _, label in labeled][:2] == ['B-<label>', 'I-<label>']

    def test_should_not_snap_onto_a_line_that_is_itself_a_start(self):
        lines = get_lines(TOKENS, get_line_numbers(LINE_STATUS))
        assert snap_starts_to_label_lines([3, 4], lines) == [3, 4]

    def test_should_leave_starts_with_no_preceding_label_line(self):
        lines = get_lines(TOKENS, get_line_numbers(LINE_STATUS))
        assert snap_starts_to_label_lines([2], lines) == [2]

    def test_should_not_snap_two_starts_onto_the_same_label_line(self):
        lines = get_lines(TOKENS, get_line_numbers(LINE_STATUS))
        assert snap_starts_to_label_lines([1, 2], lines) == [0, 2]
