import os
from pathlib import Path
from unittest.mock import patch

from sciencebeam_parser.external.pdfalto.wrapper import (
    PDFALTO_VERSION,
    PdfAltoWrapper,
    get_default_pdfalto_binary_path
)


class TestGetDefaultPdfaltoBinaryPath:
    def test_should_use_binary_shipped_with_pdfalto_package(self):
        with patch('pdfalto.binary_path', return_value=Path('/opt/pdfalto/bin/pdfalto')):
            assert get_default_pdfalto_binary_path() == '/opt/pdfalto/bin/pdfalto'

    def test_default_binary_should_exist_and_be_executable(self):
        binary_path = get_default_pdfalto_binary_path()
        assert os.path.isfile(binary_path)
        assert os.access(binary_path, os.X_OK)

    def test_version_should_track_installed_package(self):
        assert PDFALTO_VERSION.startswith('v0.')


class TestPdfAltoWrapperEnsureExecutable:
    def test_should_add_exec_bit_to_non_executable_file(self, tmp_path: Path):
        fake_binary = tmp_path / 'pdfalto'
        fake_binary.write_text('#!/bin/sh\n', encoding='utf-8')
        fake_binary.chmod(0o644)
        PdfAltoWrapper(str(fake_binary)).ensure_executable()
        assert os.access(fake_binary, os.X_OK)

    def test_should_not_touch_already_executable_file(self, tmp_path: Path):
        fake_binary = tmp_path / 'pdfalto'
        fake_binary.write_text('#!/bin/sh\n', encoding='utf-8')
        fake_binary.chmod(0o755)
        with patch('os.chmod') as chmod_mock:
            PdfAltoWrapper(str(fake_binary)).ensure_executable()
        chmod_mock.assert_not_called()
