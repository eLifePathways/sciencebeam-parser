from pathlib import Path

import pytest

from sciencebeam_parser.utils.model_data_normalizer import (
    FEATURE_COUNT,
    iter_normalized_lines,
    normalize_file,
)

FEATURES = '\t'.join(['f'] * FEATURE_COUNT)
FEATURES_SPACE = ' '.join(['f'] * FEATURE_COUNT)


def _grobid_line(label: str) -> str:
    return f'{FEATURES}\t{label}'


def _sciencebeam_line(label: str, block_text: str = 'word') -> str:
    return f'{FEATURES_SPACE} {block_text} {label}'


def _normalize(lines: list) -> list:
    return list(iter_normalized_lines(iter(lines)))


class TestIterNormalizedLines:
    def test_should_pass_through_grobid_tab_separated_line(self):
        line = _grobid_line('I-<header>')
        assert _normalize([line]) == [line]

    def test_should_strip_block_text_from_sciencebeam_line(self):
        result = _normalize([_sciencebeam_line('B-<header>')])
        assert result == [_grobid_line('B-<header>')]

    def test_should_strip_multi_word_block_text_from_sciencebeam_line(self):
        result = _normalize([_sciencebeam_line('B-<header>', block_text='hello world foo')])
        assert result == [_grobid_line('B-<header>')]

    def test_should_preserve_blank_lines(self):
        lines = [_grobid_line('I-<header>'), '', _grobid_line('I-<body>')]
        assert _normalize(lines) == lines

    def test_should_strip_grobid_model_header(self):
        lines = ['=== model: segmentation ===', _grobid_line('I-<header>')]
        assert _normalize(lines) == [_grobid_line('I-<header>')]

    def test_should_strip_grobid_model_header_with_any_model_name(self):
        lines = ['=== model: header ===', _grobid_line('I-<title>')]
        assert _normalize(lines) == [_grobid_line('I-<title>')]

    def test_should_not_strip_line_that_looks_like_partial_header(self):
        line = _grobid_line('I-<header>')
        assert _normalize([line]) == [line]

    def test_should_detect_format_from_first_non_blank_line(self):
        lines = ['', _sciencebeam_line('B-<body>'), _sciencebeam_line('I-<body>')]
        result = _normalize(lines)
        assert result == ['', _grobid_line('B-<body>'), _grobid_line('I-<body>')]

    def test_should_handle_empty_input(self):
        assert not _normalize([])

    def test_should_handle_input_with_only_blank_lines(self):
        assert _normalize(['', '']) == ['', '']

    def test_should_strip_trailing_newline_from_input_lines(self):
        line = _grobid_line('I-<header>')
        assert _normalize([line + '\n']) == [line]


class TestNormalizeFile:
    def test_should_normalize_grobid_file(self, tmp_path: Path):
        line = _grobid_line('I-<header>')
        input_file = tmp_path / 'input.data'
        input_file.write_text(line + '\n')
        output_file = tmp_path / 'out' / 'output.data'

        normalize_file(input_file, output_file)

        assert output_file.read_text() == line + '\n'

    def test_should_normalize_sciencebeam_file(self, tmp_path: Path):
        input_file = tmp_path / 'input.data'
        input_file.write_text(_sciencebeam_line('B-<header>') + '\n')
        output_file = tmp_path / 'output.data'

        normalize_file(input_file, output_file)

        assert output_file.read_text() == _grobid_line('B-<header>') + '\n'

    def test_should_create_output_parent_directory(self, tmp_path: Path):
        input_file = tmp_path / 'input.data'
        input_file.write_text(_grobid_line('I-<header>') + '\n')
        output_file = tmp_path / 'nested' / 'deep' / 'output.data'

        normalize_file(input_file, output_file)

        assert output_file.exists()

    def test_should_preserve_blank_lines_in_file(self, tmp_path: Path):
        content = _grobid_line('I-<header>') + '\n\n' + _grobid_line('I-<body>') + '\n'
        input_file = tmp_path / 'input.data'
        input_file.write_text(content)
        output_file = tmp_path / 'output.data'

        normalize_file(input_file, output_file)

        assert output_file.read_text() == content

    def test_should_raise_if_input_file_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            normalize_file(tmp_path / 'missing.data', tmp_path / 'out.data')
