import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from sciencebeam_parser.models.llm.decode import LlmResponseError


WORD_SEPARATOR = re.compile(r'[^0-9A-Za-zÀ-ɏ]+')


def iter_words(text: str) -> List[str]:
    return [word for word in WORD_SEPARATOR.split(text) if word]


def get_values_response_schema(labels: Sequence[str]) -> Mapping[str, Any]:
    return {
        'type': 'object',
        'additionalProperties': False,
        'required': ['fields'],
        'properties': {
            'fields': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'required': ['label', 'text'],
                    'properties': {
                        'label': {'type': 'string', 'enum': list(labels)},
                        'text': {'type': 'string'},
                    },
                },
            },
        },
    }


def get_word_positions(tokens: Sequence[str]) -> List[Tuple[int, str]]:
    return [
        (token_index, word)
        for token_index, token in enumerate(tokens)
        for word in iter_words(token)
    ]


def find_unclaimed_span(
    wanted: Sequence[str],
    word_positions: Sequence[Tuple[int, str]],
    claimed: Sequence[bool]
) -> Tuple[int, int]:
    """The earliest occurrence whose tokens are all unclaimed.

    Document order is not assumed: models emit `date` out of position often
    enough that requiring it rejects values the source does contain.
    """
    seen_but_claimed = False
    for start in range(len(word_positions) - len(wanted) + 1):
        window = word_positions[start:start + len(wanted)]
        if [word for _, word in window] != list(wanted):
            continue
        first_token, last_token = window[0][0], window[-1][0]
        if any(claimed[first_token:last_token + 1]):
            seen_but_claimed = True
            continue
        return first_token, last_token
    snippet = ' '.join(wanted)[:60]
    if seen_but_claimed:
        raise LlmResponseError(
            f'value claimed by an earlier field: {snippet}'
        )
    raise LlmResponseError(f'value not found in source: {snippet}')


def parse_values(content: str, labels: Sequence[str]) -> List[Dict[str, str]]:
    try:
        payload = json.loads(content)
    except ValueError as exc:
        raise LlmResponseError(f'response is not json: {exc}') from exc
    if not isinstance(payload, dict) or 'fields' not in payload:
        raise LlmResponseError('response has no "fields"')
    fields = payload['fields']
    if not isinstance(fields, list):
        raise LlmResponseError('"fields" is not a list')
    known = set(labels)
    parsed: List[Dict[str, str]] = []
    for entry in fields:
        if not isinstance(entry, dict) or 'label' not in entry or 'text' not in entry:
            raise LlmResponseError(f'field entry is malformed: {entry!r}')
        label = entry['label']
        if label not in known:
            raise LlmResponseError(f'unknown label {label!r}')
        parsed.append({'label': label, 'text': entry['text']})
    return parsed


def decode_values_response(
    content: str,
    tokens: Sequence[str],
    labels: Sequence[str]
) -> List[Tuple[str, str]]:
    fields = parse_values(content, labels)
    word_positions = get_word_positions(tokens)
    claimed = [False] * len(tokens)
    token_labels = ['O'] * len(tokens)
    for entry in fields:
        wanted = iter_words(entry['text'])
        if not wanted:
            continue
        first_token, last_token = find_unclaimed_span(wanted, word_positions, claimed)
        for token_index in range(first_token, last_token + 1):
            claimed[token_index] = True
            token_labels[token_index] = (
                f'B-<{entry["label"]}>' if token_index == first_token
                else f'I-<{entry["label"]}>'
            )
    return list(zip(tokens, token_labels))
