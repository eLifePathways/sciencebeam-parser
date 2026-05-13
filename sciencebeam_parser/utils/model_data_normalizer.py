import re
from pathlib import Path
from typing import Iterator

GROBID_MODEL_HEADER_RE = re.compile(r'^=== model: .+ ===$')
FEATURE_COUNT = 33


def _normalize_sciencebeam_line(line: str) -> str:
    parts = line.split(' ')
    if len(parts) <= FEATURE_COUNT:
        return line
    features = parts[:FEATURE_COUNT]
    label = parts[-1]
    return '\t'.join(features + [label])


def iter_normalized_lines(lines: Iterator[str]) -> Iterator[str]:
    format_detected: str | None = None
    for raw_line in lines:
        line = raw_line.rstrip('\n')
        if not line:
            yield ''
            continue
        if GROBID_MODEL_HEADER_RE.match(line):
            continue
        if format_detected is None:
            format_detected = 'grobid' if '\t' in line else 'sciencebeam'
        if format_detected == 'sciencebeam':
            line = _normalize_sciencebeam_line(line)
        yield line


def normalize_file(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as f_in, output_path.open('w') as f_out:
        for line in iter_normalized_lines(iter(f_in)):
            f_out.write(line + '\n')
