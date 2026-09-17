import json
import re
from typing import Any, List, Mapping, Sequence, Tuple


LINE_START = 'LINESTART'

LABEL_ONLY_LINE = re.compile(r'^[\[(]?\d{1,3}[\])]?[.)]?$')

WORD_SEPARATOR = re.compile(r'[^0-9A-Za-zÀ-ɏ]+')


def iter_words(text: str) -> List[str]:
    """Words with punctuation removed, so a quote can be compared with tokens.

    One implementation, imported by the value shapes too: two of these drifting
    apart is how a match silently stops matching.
    """
    return [word for word in WORD_SEPARATOR.split(text) if word]


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


EVIDENCE_RESPONSE_SCHEMA: Mapping[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['references'],
    'properties': {
        'references': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': ['line', 'starts_with'],
                'properties': {
                    'line': {'type': 'integer'},
                    'starts_with': {'type': 'string', 'maxLength': 48},
                },
            },
        },
    },
}


def get_regions_response_schema(labels: Sequence[str]) -> Mapping[str, Any]:
    """One entry per region: where it starts, where it ends, and what it is.

    The payload is two indices and a label from a closed set, so there is nothing
    a model could invent that reaches the output. Every guarantee below is
    enforced again on decode, because a provider that ignores the schema would
    otherwise be trusted.

    The end is redundant with the next region's start, and that is the point: a
    model that drops a region has to contradict itself to hide it. The worst
    document measured returned three regions where four were needed, and the
    surviving one silently swallowed 172 lines because a region ran until the
    next one started.
    """
    return {
        'type': 'object',
        'additionalProperties': False,
        'required': ['regions'],
        'properties': {
            'regions': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'additionalProperties': False,
                    'required': ['start', 'end', 'label'],
                    'properties': {
                        'start': {'type': 'integer'},
                        'end': {'type': 'integer'},
                        'label': {'type': 'string', 'enum': list(labels)},
                    },
                },
            },
        },
    }


class LlmResponseError(ValueError):
    pass


class LlmMalformedResponseError(LlmResponseError):
    """The response could not be parsed, as distinct from a claim it makes that
    the engine will not honour.

    Only this one is worth asking again for. A strictness setting that fires is a
    decision rather than a bad sample, and repeating the request would only spend
    tokens on the same answer.
    """


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


def get_json_payload(content: str):
    """Parsed json, or an error carrying enough of the response to diagnose it.

    The content is otherwise only on the telemetry span, so a run without a
    collector — CI, for one — leaves no way to tell a response truncated
    mid-array from one with a stray token in it.
    """
    try:
        return json.loads(content)
    except ValueError as exc:
        position = getattr(exc, 'pos', None)
        excerpt = (
            content[max(0, position - 60):position + 20] if position is not None
            else content[:120]
        )
        raise LlmMalformedResponseError(
            f'response is not json: {exc}; {len(content)} chars,'
            f' around the failure: ...{excerpt!r}'
        ) from exc


def parse_line_starts(content: str, line_count: int) -> List[int]:
    payload = get_json_payload(content)
    if not isinstance(payload, dict) or 'starts' not in payload:
        raise LlmMalformedResponseError('response has no "starts"')
    starts = payload['starts']
    if not isinstance(starts, list):
        raise LlmMalformedResponseError('"starts" is not a list')
    resolved: List[int] = []
    for value in starts:
        if isinstance(value, bool) or not isinstance(value, int):
            raise LlmMalformedResponseError(f'line number is not an integer: {value!r}')
        if not 0 <= value < line_count:
            raise LlmMalformedResponseError(
                f'line number {value} out of range for {line_count} lines'
            )
        resolved.append(value)
    if resolved != sorted(set(resolved)):
        raise LlmMalformedResponseError(f'line numbers are not strictly ascending: {resolved}')
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


def parse_evidence_line_starts(
    content: str,
    line_count: int
) -> Tuple[List[int], List[str]]:
    """Line numbers are the answer; the quoted words are only evidence for them.

    Keeping the payload an index means a wrong quote costs a check rather than
    the reference, which is the failure mode the anchor shape had.
    """
    payload = get_json_payload(content)
    if not isinstance(payload, dict) or 'references' not in payload:
        raise LlmMalformedResponseError('response has no "references"')
    entries = payload['references']
    if not isinstance(entries, list):
        raise LlmMalformedResponseError('"references" is not a list')
    starts: List[int] = []
    claims: List[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or 'line' not in entry:
            raise LlmMalformedResponseError(f'reference entry is malformed: {entry!r}')
        line = entry['line']
        if isinstance(line, bool) or not isinstance(line, int):
            raise LlmMalformedResponseError(f'line number is not an integer: {line!r}')
        if not 0 <= line < line_count:
            raise LlmMalformedResponseError(
                f'line number {line} out of range for {line_count} lines'
            )
        starts.append(line)
        claims.append(str(entry.get('starts_with') or ''))
    if starts != sorted(set(starts)):
        raise LlmMalformedResponseError(f'line numbers are not strictly ascending: {starts}')
    return starts, claims


def count_evidence_mismatches(
    starts: Sequence[int],
    claims: Sequence[str],
    lines: Sequence[Sequence[str]]
) -> int:
    """A quote is accepted against the line it names or the one below it.

    Models name the line holding the reference number while quoting the words on
    the line under it, which is the training convention rather than an error.
    """
    mismatches = 0
    for line, claim in zip(starts, claims):
        wanted = iter_words(claim)
        if not wanted:
            continue
        candidates = []
        for offset in (0, 1):
            if 0 <= line + offset < len(lines):
                candidates.append(
                    iter_words(' '.join(lines[line + offset]))[:len(wanted)]
                )
        if wanted not in candidates:
            mismatches += 1
    return mismatches


def decode_evidence_response(
    content: str,
    tokens: Sequence[str],
    line_status_values: Sequence[str]
) -> Tuple[List[Tuple[str, str]], int]:
    if len(tokens) != len(line_status_values):
        raise LlmMalformedResponseError(
            f'token count {len(tokens)} does not match feature rows'
            f' {len(line_status_values)}'
        )
    line_numbers = get_line_numbers(line_status_values)
    lines = get_lines(tokens, line_numbers)
    starts, claims = parse_evidence_line_starts(content, line_count=len(lines))
    mismatches = count_evidence_mismatches(starts, claims, lines)
    snapped = snap_starts_to_label_lines(starts, lines)
    labels = iter_labels_for_line_starts(tokens, line_numbers, snapped)
    return list(zip(tokens, labels)), mismatches


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


def render_numbered_line_texts(line_texts: Sequence[str]) -> str:
    """Segmentation's rows are already lines, so the number is the row index."""
    return '\n'.join(f'{number}\t{text}' for number, text in enumerate(line_texts))


def _get_line_index(value: Any, field_name: str, line_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LlmMalformedResponseError(
            f'region {field_name} is not an integer: {value!r}'
        )
    if not 0 <= value < line_count:
        raise LlmMalformedResponseError(
            f'region {field_name} {value} out of range for {line_count} lines'
        )
    return value


def parse_regions(
    content: str,
    line_count: int,
    region_names: Sequence[str],
    max_regions: int
) -> List[Tuple[int, int, str]]:
    payload = get_json_payload(content)
    if not isinstance(payload, dict) or 'regions' not in payload:
        raise LlmMalformedResponseError('response has no "regions"')
    entries = payload['regions']
    if not isinstance(entries, list):
        raise LlmMalformedResponseError('"regions" is not a list')
    if not entries:
        raise LlmMalformedResponseError('"regions" is empty, so no line has a label')
    if len(entries) > max_regions:
        # Over-segmentation is what truncates a response: one pilot answer ran to
        # 204 regions on a 604-line document and was cut off mid-json.
        raise LlmMalformedResponseError(
            f'{len(entries)} regions exceeds max_regions={max_regions}'
        )
    allowed = set(region_names)
    regions: List[Tuple[int, int, str]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise LlmMalformedResponseError(f'region entry is malformed: {entry!r}')
        unexpected = sorted(set(entry) - {'start', 'end', 'label'})
        if unexpected:
            raise LlmMalformedResponseError(
                f'region entry has unexpected key(s) {unexpected}: {entry!r}'
            )
        start = _get_line_index(entry.get('start'), 'start', line_count)
        end = _get_line_index(entry.get('end'), 'end', line_count)
        name = entry.get('label')
        if end < start:
            raise LlmMalformedResponseError(
                f'region ends at line {end}, before it starts at {start}'
            )
        if not isinstance(name, str) or name not in allowed:
            raise LlmMalformedResponseError(
                f'region label {name!r} is not one of {sorted(allowed)}'
            )
        regions.append((start, end, name))
    _check_regions_cover_every_line(regions, line_count)
    return regions


def _check_regions_cover_every_line(
    regions: Sequence[Tuple[int, int, str]], line_count: int
) -> None:
    """Every line belongs to exactly one region, and the response says so twice.

    Asking for an end as well as a start makes a dropped region visible: a model
    that leaves one out has to either leave a gap here or claim the neighbouring
    region covers lines it already said it ended before.
    """
    if regions[0][0] != 0:
        raise LlmMalformedResponseError(
            f'the first region starts at line {regions[0][0]} rather than 0,'
            ' so the lines before it have no label'
        )
    for (_, end, _), (next_start, _, _) in zip(regions, regions[1:]):
        if next_start != end + 1:
            gap = 'a gap' if next_start > end + 1 else 'an overlap'
            raise LlmMalformedResponseError(
                f'{gap} between a region ending at line {end} and the next'
                f' starting at line {next_start}'
            )
    last_end = regions[-1][1]
    if last_end != line_count - 1:
        raise LlmMalformedResponseError(
            f'the last region ends at line {last_end} rather than {line_count - 1},'
            ' so the lines after it have no label'
        )


def iter_labels_for_regions(
    regions: Sequence[Tuple[int, int, str]], line_count: int
) -> List[str]:
    from sciencebeam_parser.models.llm.tasks import (  # noqa pylint: disable=import-outside-toplevel
        get_segmentation_label
    )
    labels: List[str] = []
    for start, end, name in regions:
        label = get_segmentation_label(name)
        labels.extend(
            [f'B-<{label.strip("<>")}>']
            + [f'I-<{label.strip("<>")}>'] * (end - start)
        )
    if len(labels) != line_count:
        raise LlmMalformedResponseError(
            f'regions cover {len(labels)} lines rather than {line_count}'
        )
    return labels


def decode_regions_response(
    content: str,
    line_texts: Sequence[str],
    region_names: Sequence[str],
    max_regions: int
) -> List[str]:
    """One label per line, covering every line, from a response of region spans."""
    regions = parse_regions(
        content,
        line_count=len(line_texts),
        region_names=region_names,
        max_regions=max_regions
    )
    return iter_labels_for_regions(regions, len(line_texts))


def decode_line_starts_response(
    content: str,
    tokens: Sequence[str],
    line_status_values: Sequence[str]
) -> List[Tuple[str, str]]:
    if len(tokens) != len(line_status_values):
        raise LlmMalformedResponseError(
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
        raise LlmMalformedResponseError(
            f'produced {len(labels)} labels for {len(tokens)} tokens'
        )
    return list(zip(tokens, labels))
