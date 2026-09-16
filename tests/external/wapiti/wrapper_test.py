import io
import os
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sciencebeam_parser.external.wapiti.wrapper import (
    install_wapiti_and_get_path_or_none,
    remove_source_entries_colliding_with_binary
)


# Stands in for the real Wapiti Makefile: the `wapiti` target just creates the file.
FAKE_MAKEFILE = 'wapiti:\n\t@touch wapiti\n'


def _create_fake_wapiti_source_tar_gz(tar_gz_path: Path, source_dir_name: str) -> None:
    with tarfile.open(tar_gz_path, mode='w:gz') as tar:
        makefile = tarfile.TarInfo(f'{source_dir_name}/Makefile')
        makefile.size = len(FAKE_MAKEFILE)
        tar.addfile(makefile, fileobj=_bytes_io(FAKE_MAKEFILE))
        xcode_dir = tarfile.TarInfo(f'{source_dir_name}/Wapiti')
        xcode_dir.type = tarfile.DIRTYPE
        xcode_dir.mode = 0o755
        tar.addfile(xcode_dir)
        xcode_file = tarfile.TarInfo(f'{source_dir_name}/Wapiti/project.pbxproj')
        xcode_file.size = 0
        tar.addfile(xcode_file, fileobj=_bytes_io(''))


def _bytes_io(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode('utf-8'))


@pytest.fixture(name='install_url')
def _install_url(tmp_path: Path) -> str:
    tar_gz_path = tmp_path / 'Wapiti-abc123.tar.gz'
    _create_fake_wapiti_source_tar_gz(tar_gz_path, 'Wapiti-abc123')
    return str(tar_gz_path)


@pytest.fixture(name='download_manager')
def _download_manager() -> MagicMock:
    download_manager = MagicMock(name='download_manager')
    download_manager.download_if_url.side_effect = lambda url, **_: url
    return download_manager


class TestRemoveSourceEntriesCollidingWithBinary:
    def test_should_remove_xcode_directory(self, tmp_path: Path):
        (tmp_path / 'Wapiti').mkdir()
        (tmp_path / 'src').mkdir()
        remove_source_entries_colliding_with_binary(str(tmp_path))
        assert not (tmp_path / 'Wapiti').exists()
        assert (tmp_path / 'src').exists()

    def test_should_not_fail_if_directory_is_absent(self, tmp_path: Path):
        remove_source_entries_colliding_with_binary(str(tmp_path))


class TestInstallWapitiAndGetPathOrNone:
    def test_should_return_none_without_install_url(self, download_manager: MagicMock):
        assert install_wapiti_and_get_path_or_none(None, download_manager) is None

    def test_should_reject_non_tar_gz(self, download_manager: MagicMock):
        with pytest.raises(ValueError):
            install_wapiti_and_get_path_or_none('wapiti.zip', download_manager)

    def test_should_reject_archive_member_escaping_target_directory(
        self,
        tmp_path: Path,
        download_manager: MagicMock
    ):
        tar_gz_path = tmp_path / 'evil.tar.gz'
        with tarfile.open(tar_gz_path, mode='w:gz') as tar:
            member = tarfile.TarInfo('../../escaped.txt')
            member.size = 0
            tar.addfile(member, fileobj=_bytes_io(''))
        with pytest.raises((ValueError, tarfile.TarError)):
            install_wapiti_and_get_path_or_none(str(tar_gz_path), download_manager)
        assert not (tmp_path.parent / 'escaped.txt').exists()

    def test_should_build_binary_after_removing_colliding_directory(
        self,
        install_url: str,
        download_manager: MagicMock
    ):
        binary_path = install_wapiti_and_get_path_or_none(install_url, download_manager)
        assert binary_path
        assert os.path.isfile(binary_path)
        # isdir rather than exists: on a case-insensitive filesystem 'Wapiti' now
        # resolves to the freshly built 'wapiti' file
        assert not os.path.isdir(os.path.join(os.path.dirname(binary_path), 'Wapiti'))

    def test_should_reuse_existing_binary_without_re_extracting(
        self,
        install_url: str,
        download_manager: MagicMock
    ):
        binary_path = install_wapiti_and_get_path_or_none(install_url, download_manager)
        assert binary_path
        Path(binary_path).chmod(0o755)
        Path(binary_path).write_text('built', encoding='utf-8')
        assert install_wapiti_and_get_path_or_none(install_url, download_manager) == binary_path
        assert Path(binary_path).read_text(encoding='utf-8') == 'built'
