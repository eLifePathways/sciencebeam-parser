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


def get_batched_values_response_schema(labels: Sequence[str]) -> Mapping[str, Any]:
    """One entry per input reference, so a batch does not grow the index space.

    Requiring exactly one entry per reference also makes a merged or omitted
    reference fail validation rather than score.
    """
    single = get_values_response_schema(labels)
    return {
        'type': 'object',
        'additionalProperties': False,
        'required': ['references'],
        'properties': {
            'references': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'required': ['index', 'fields'],
                    'properties': {
                        'index': {'type': 'integer'},
                        'fields': single['properties']['fields'],
                    },
                },
            },
        },
    }


def render_numbered_references(token_lists: Sequence[Sequence[str]]) -> str:
    return '\n'.join(
        f'[{index}] ' + ' '.join(tokens)
        for index, tokens in enumerate(token_lists)
    )


def get_labels_for_fields(
    fields: Sequence[Mapping[str, str]],
    tokens: Sequence[str]
) -> List[str]:
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
    return token_labels


def decode_values_response(
    content: str,
    tokens: Sequence[str],
    labels: Sequence[str]
) -> List[Tuple[str, str]]:
    fields = parse_values(content, labels)
    return list(zip(tokens, get_labels_for_fields(fields, tokens)))


def parse_batched_values(
    content: str,
    reference_count: int,
    labels: Sequence[str]
) -> List[List[Dict[str, str]]]:
    try:
        payload = json.loads(content)
    except ValueError as exc:
        raise LlmResponseError(f'response is not json: {exc}') from exc
    if not isinstance(payload, dict) or 'references' not in payload:
        raise LlmResponseError('response has no "references"')
    entries = payload['references']
    if not isinstance(entries, list):
        raise LlmResponseError('"references" is not a list')
    by_index: Dict[int, List[Dict[str, str]]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or 'index' not in entry:
            raise LlmResponseError(f'reference entry is malformed: {entry!r}')
        index = entry['index']
        if isinstance(index, bool) or not isinstance(index, int):
            raise LlmResponseError(f'reference index is not an integer: {index!r}')
        if not 0 <= index < reference_count:
            raise LlmResponseError(
                f'reference index {index} out of range for {reference_count}'
                ' references sent'
            )
        if index in by_index:
            raise LlmResponseError(f'reference index {index} appears twice')
        by_index[index] = parse_values(
            json.dumps({'fields': entry.get('fields') or []}), labels
        )
    missing = sorted(set(range(reference_count)) - set(by_index))
    if missing:
        raise LlmResponseError(
            f'no answer for reference(s) {missing} of {reference_count} sent'
        )
    return [by_index[index] for index in range(reference_count)]


def decode_batched_values_response(
    content: str,
    token_lists: Sequence[Sequence[str]],
    labels: Sequence[str]
) -> List[List[Tuple[str, str]]]:
    fields_per_reference = parse_batched_values(content, len(token_lists), labels)
    return [
        list(zip(tokens, get_labels_for_fields(fields, tokens)))
        for fields, tokens in zip(fields_per_reference, token_lists)
    ]
