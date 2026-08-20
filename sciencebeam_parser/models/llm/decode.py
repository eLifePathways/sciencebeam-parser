import json
import re
from typing import Any, List, Mapping, Sequence, Tuple


LINE_START = 'LINESTART'

LABEL_ONLY_LINE = re.compile(r'^[\[(]?\d{1,3}[\])]?[.)]?$')

LINES_RESPONSE_SCHEMA: Mapping[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['starts'],
    'properties': {
        'starts': {
            'type': 'array',
            'items': {'type': 'integer'},
        },
    },
}


class LlmResponseError(ValueError):
    pass


class LlmInputTooLargeError(ValueError):
    pass


def get_line_numbers(line_status_values: Sequence[str]) -> List[int]:
    line_numbers: List[int] = []
    current = -1
    for index, status in enumerate(line_status_values):
        if status == LINE_START or index == 0:
            current += 1
        line_numbers.append(current)
    return line_numbers


def get_lines(tokens: Sequence[str], line_numbers: Sequence[int]) -> List[List[str]]:
    lines: List[List[str]] = []
    for token, line_number in zip(tokens, line_numbers):
        while len(lines) <= line_number:
            lines.append([])
        lines[line_number].append(token)
    return lines


def render_numbered_lines(tokens: Sequence[str], line_numbers: Sequence[int]) -> str:
    return '\n'.join(
        f'{number}\t' + ' '.join(line_tokens)
        for number, line_tokens in enumerate(get_lines(tokens, line_numbers))
    )


def parse_line_starts(content: str, line_count: int) -> List[int]:
    try:
        payload = json.loads(content)
    except ValueError as exc:
        raise LlmResponseError(f'response is not json: {exc}') from exc
    if not isinstance(payload, dict) or 'starts' not in payload:
        raise LlmResponseError('response has no "starts"')
    starts = payload['starts']
    if not isinstance(starts, list) or not starts:
        raise LlmResponseError('"starts" is empty or not a list')
    resolved: List[int] = []
    for value in starts:
        if isinstance(value, bool) or not isinstance(value, int):
            raise LlmResponseError(f'line number is not an integer: {value!r}')
        if not 0 <= value < line_count:
            raise LlmResponseError(
                f'line number {value} out of range for {line_count} lines'
            )
        resolved.append(value)
    if resolved != sorted(set(resolved)):
        raise LlmResponseError(f'line numbers are not strictly ascending: {resolved}')
    return resolved


def snap_starts_to_label_lines(
    line_starts: Sequence[int],
    lines: Sequence[Sequence[str]]
) -> List[int]:
    existing = set(line_starts)
    snapped: List[int] = []
    for start in line_starts:
        previous = start - 1
        if (
            previous >= 0
            and previous not in existing
            and previous not in snapped
            and LABEL_ONLY_LINE.match(''.join(lines[previous]))
        ):
            snapped.append(previous)
            continue
        snapped.append(start)
    return sorted(snapped)


def iter_labels_for_line_starts(
    tokens: Sequence[str],
    line_numbers: Sequence[int],
    line_starts: Sequence[int]
) -> List[str]:
    lines = get_lines(tokens, line_numbers)
    starts = set(line_starts)
    labels: List[str] = []
    started = False
    for token_index, line_number in enumerate(line_numbers):
        is_line_start = token_index == 0 or line_numbers[token_index - 1] != line_number
        if line_number in starts and is_line_start:
            started = True
            if LABEL_ONLY_LINE.match(''.join(lines[line_number])):
                labels.append('B-<label>')
                continue
            labels.append('B-<reference>')
            continue
        if not started:
            labels.append('O')
            continue
        previous = labels[-1]
        if previous in ('B-<label>', 'I-<label>'):
            labels.append('I-<label>' if line_numbers[token_index - 1] == line_number
                          else 'B-<reference>')
            continue
        labels.append('I-<reference>')
    return labels


def decode_line_starts_response(
    content: str,
    tokens: Sequence[str],
    line_status_values: Sequence[str]
) -> List[Tuple[str, str]]:
    if len(tokens) != len(line_status_values):
        raise LlmResponseError(
            f'token count {len(tokens)} does not match feature rows'
            f' {len(line_status_values)}'
        )
    line_numbers = get_line_numbers(line_status_values)
    line_starts = parse_line_starts(content, line_count=max(line_numbers) + 1)
    line_starts = snap_starts_to_label_lines(
        line_starts, get_lines(tokens, line_numbers)
    )
    labels = iter_labels_for_line_starts(tokens, line_numbers, line_starts)
    if len(labels) != len(tokens):
        raise LlmResponseError(
            f'produced {len(labels)} labels for {len(tokens)} tokens'
        )
    return list(zip(tokens, labels))
