import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from sciencebeam_parser.document.layout_noise_filter import LayoutNoiseFilterConfig

from ._types import DocumentSummary, RemovedBlock


def write_jsonl(path: Path, records: Sequence) -> None:
    with path.open('w', encoding='utf-8') as fp:
        for record in records:
            fp.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + '\n')


def _format_text(text: str, max_length: int = 300) -> str:
    text = ' '.join(text.split())
    if len(text) > max_length:
        text = text[:max_length] + '…'
    return text.replace('|', '\\|')


def _render_config(config: LayoutNoiseFilterConfig) -> List[str]:
    lines = ['## Filter configuration', '', '| setting | value |', '| --- | --- |']
    for field in dataclasses.fields(config):
        lines.append(f'| `{field.name}` | {getattr(config, field.name)} |')
    return lines + ['']


def _render_per_corpus(
    summaries: Sequence[DocumentSummary],
    removed: Sequence[RemovedBlock]
) -> List[str]:
    by_corpus: Dict[str, List[DocumentSummary]] = defaultdict(list)
    for summary in summaries:
        by_corpus[summary.corpus].append(summary)
    in_jats_by_corpus: Dict[str, int] = defaultdict(int)
    inconclusive_by_corpus: Dict[str, int] = defaultdict(int)
    for item in removed:
        if item.in_jats:
            in_jats_by_corpus[item.corpus] += 1
        elif item.in_jats is None:
            inconclusive_by_corpus[item.corpus] += 1
    lines = [
        '## Per corpus',
        '',
        '`in JATS` is the precision alarm: a removed line whose text is in the publisher\'s',
        'JATS is content. `too short` is the removals the alarm cannot look up, which is most',
        'page numbers. Absence from the JATS is not evidence of furniture.',
        '',
        '| corpus | docs | failed | lines | removed blocks | removed lines | % of lines |'
        ' in JATS | too short |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for corpus in sorted(by_corpus):
        corpus_summaries = by_corpus[corpus]
        ok = [item for item in corpus_summaries if not item.error]
        line_count = sum(item.line_count for item in ok)
        removed_lines = sum(item.removed_line_count for item in ok)
        percentage = 100 * removed_lines / line_count if line_count else 0
        lines.append(
            f'| {corpus} | {len(corpus_summaries)} '
            f'| {len(corpus_summaries) - len(ok)} '
            f'| {line_count} '
            f'| {sum(item.removed_block_count for item in ok)} '
            f'| {removed_lines} | {percentage:.2f}% '
            f'| {in_jats_by_corpus[corpus]} | {inconclusive_by_corpus[corpus]} |'
        )
    return lines + ['']


def _render_in_jats(removed: Sequence[RemovedBlock]) -> List[str]:
    alarms = [item for item in removed if item.in_jats]
    lines = [
        '## Removals whose text is in the JATS',
        '',
        f'{len(alarms)} of {len(removed)} removed blocks. Each one is content the filter takes,',
        'independently of any furniture label. Listed in full — this is the list to act on.',
        '',
    ]
    if not alarms:
        return lines + ['None.', '']
    lines += ['| corpus | document | page | type | text |', '| --- | --- | ---: | --- | --- |']
    for item in alarms:
        lines.append(
            f'| {item.corpus} | {item.document_id} | {item.page_number}/{item.page_count} '
            f'| {item.note_type} | {_format_text(item.text)} |'
        )
    return lines + ['']


def _render_samples(removed: Sequence[RemovedBlock], sample_size: int) -> List[str]:
    by_corpus: Dict[str, List[RemovedBlock]] = defaultdict(list)
    for item in removed:
        by_corpus[item.corpus].append(item)
    lines = [
        '## What is removed, as text',
        '',
        f'Up to {sample_size} blocks per corpus. A count cannot be judged; this can.',
        '',
    ]
    for corpus in sorted(by_corpus):
        items = by_corpus[corpus]
        lines += [f'### {corpus}', '']
        if len(items) > sample_size:
            lines += [f'Showing {sample_size} of {len(items)}.', '']
        lines += [
            '| document | page | type | y | text |',
            '| --- | ---: | --- | ---: | --- |',
        ]
        for item in items[:sample_size]:
            y_relative = f'{item.y_relative:.3f}' if item.y_relative is not None else '—'
            lines.append(
                f'| {item.document_id} | {item.page_number}/{item.page_count} '
                f'| {item.note_type} | {y_relative} | {_format_text(item.text)} |'
            )
        lines.append('')
    return lines


def _render_errors(summaries: Sequence[DocumentSummary]) -> List[str]:
    failed = [item for item in summaries if item.error]
    if not failed:
        return []
    lines = ['## Documents that failed', '', '| corpus | document | error |', '| --- | --- | --- |']
    for item in failed:
        lines.append(f'| {item.corpus} | {item.document_id} | {_format_text(item.error or "")} |')
    return lines + ['']


def render_report(
    summaries: Sequence[DocumentSummary],
    removed: Sequence[RemovedBlock],
    config: LayoutNoiseFilterConfig,
    sample_size: int
) -> str:
    ok = [item for item in summaries if not item.error]
    lines = [
        '# What the layout noise filter removes',
        '',
        f'{len(ok)} documents, {sum(item.line_count for item in ok)} layout lines, '
        f'{sum(item.removed_block_count for item in ok)} blocks removed.',
        '',
    ]
    lines += _render_config(config)
    lines += _render_per_corpus(summaries, removed)
    lines += _render_in_jats(removed)
    lines += _render_samples(removed, sample_size=sample_size)
    lines += _render_errors(summaries)
    return '\n'.join(lines)


def write_report(
    output_path: Path,
    summaries: Sequence[DocumentSummary],
    removed: Sequence[RemovedBlock],
    config: LayoutNoiseFilterConfig,
    sample_size: int
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path / 'removed-blocks.jsonl', removed)
    write_jsonl(output_path / 'documents.jsonl', summaries)
    report_path = output_path / 'report.md'
    report_path.write_text(
        render_report(summaries, removed, config=config, sample_size=sample_size),
        encoding='utf-8'
    )
    return report_path
