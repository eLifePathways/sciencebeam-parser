import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple
)

from sciencebeam_parser.models.llm.decode import LlmResponseError, iter_words


LOGGER = logging.getLogger(__name__)


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


def get_character_positions(tokens: Sequence[str]) -> Tuple[str, List[int], List[int]]:
    """Alphanumeric characters of every token, and where each came from.

    Compared as characters rather than as words because models join and split
    words differently from the tokeniser: `Moreno-San Segundo` comes back as
    `Moreno SanSegundo`, which no word-sequence comparison can match.
    """
    characters: List[str] = []
    token_of_character: List[int] = []
    first_character_of_token: List[int] = []
    for token_index, token in enumerate(tokens):
        first_character_of_token.append(len(characters))
        for character in token:
            if character.isalnum():
                characters.append(character.lower())
                token_of_character.append(token_index)
    return ''.join(characters), token_of_character, first_character_of_token


def find_unclaimed_span(
    wanted: str,
    positions: Tuple[str, List[int], List[int]],
    claimed: Sequence[bool],
    context: str = ''
) -> Optional[Tuple[int, int]]:
    """The earliest occurrence whose tokens are all unclaimed.

    Document order is not assumed: models emit `date` out of position often
    enough that requiring it rejects values the source does contain. A match must
    begin at a token boundary, so a value cannot be located mid-word.
    """
    characters, token_of_character, first_character_of_token = positions
    token_starts = set(first_character_of_token)
    seen_but_claimed = False
    start = characters.find(wanted)
    while start >= 0:
        if start in token_starts:
            first_token = token_of_character[start]
            last_token = token_of_character[start + len(wanted) - 1]
            if not any(claimed[first_token:last_token + 1]):
                return first_token, last_token
            seen_but_claimed = True
        start = characters.find(wanted, start + 1)
    suffix = f' [{context}]' if context else ''
    if seen_but_claimed:
        # Two fields over the same span is the model being wrong, not the response
        # being malformed, and the guarantee is untouched: the text is in the
        # source either way. Dropping the later claim costs one field; raising
        # would cost every reference in the document.
        LOGGER.warning(
            'llm dropping a field whose text an earlier field already claimed:'
            ' %r%s', wanted[:60], suffix
        )
        return None
    raise LlmResponseError(f'value not found in source: {wanted[:60]!r}{suffix}')


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
    """`REFERENCE n` on its own line, not `[n]`.

    A bracketed number is what a numbered reference list looks like, so the
    marker was being returned as a `note` — including for unnumbered lists, where
    the value then appears nowhere in the reference.
    """
    return '\n'.join(
        f'REFERENCE {index}\n' + ' '.join(tokens)
        for index, tokens in enumerate(token_lists)
    )


def get_labels_for_fields(
    fields: Sequence[Mapping[str, str]],
    tokens: Sequence[str],
    context: str = ''
) -> List[str]:
    positions = get_character_positions(tokens)
    claimed = [False] * len(tokens)
    token_labels = ['O'] * len(tokens)
    for entry in fields:
        wanted = ''.join(iter_words(entry['text'])).lower()
        if not wanted:
            continue
        span = find_unclaimed_span(
            wanted, positions, claimed,
            context=f'{context}label={entry["label"]}'
            f' reference={" ".join(tokens)[:60]!r}'
        )
        if span is None:
            continue
        first_token, last_token = span
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
        list(zip(tokens, get_labels_for_fields(
            fields, tokens, context=f'reference index {index}, '
        )))
        for index, (fields, tokens) in enumerate(
            zip(fields_per_reference, token_lists)
        )
    ]
