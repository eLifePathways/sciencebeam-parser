from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import tarfile
from typing import Optional

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager


LOGGER = logging.getLogger(__name__)


DEFAULT_WAPITI_PATH = 'wapiti'

TAR_GZ_EXT = '.tar.gz'

# Entries of the Wapiti source tree whose name only differs from the `wapiti` make
# target by case. On a case-insensitive filesystem (the macOS default) `make` then
# reports the target as up to date, never compiles, and the parser ends up trying to
# exec a directory. They are an Xcode project and take no part in the build.
WAPITI_SOURCE_ENTRIES_COLLIDING_WITH_BINARY = ('Wapiti',)


def _safe_extractall(tar: tarfile.TarFile, target_directory: str) -> None:
    # Guard against tar-slip (members escaping the target directory, CVE-2007-4559).
    # The 'data' filter does this in the standard library on Python >= 3.12 (and
    # recent 3.10/3.11 patch releases); older interpreters get a manual check.
    if hasattr(tarfile, 'data_filter'):
        tar.extractall(target_directory, filter='data')
        return
    target_root = os.path.realpath(target_directory)
    for member in tar.getmembers():
        member_path = os.path.realpath(os.path.join(target_root, member.name))
        if member_path != target_root and not member_path.startswith(target_root + os.sep):
            raise ValueError(f'tar member escapes target directory: {member.name!r}')
        if member.issym() or member.islnk():
            raise ValueError(f'links not allowed in wapiti source archive: {member.name!r}')
    tar.extractall(target_directory)


def _get_wapiti_source_directory(extracted_directory: str) -> str:
    extracted_files = os.listdir(extracted_directory)
    if len(extracted_files) == 1:
        return os.path.join(extracted_directory, extracted_files[0])
    return extracted_directory


def remove_source_entries_colliding_with_binary(wapiti_source_directory: str) -> None:
    for entry_name in WAPITI_SOURCE_ENTRIES_COLLIDING_WITH_BINARY:
        entry_path = os.path.join(wapiti_source_directory, entry_name)
        if os.path.isdir(entry_path):
            LOGGER.info('removing %r (collides with the wapiti binary name)', entry_path)
            shutil.rmtree(entry_path)


def install_wapiti_and_get_path_or_none(
    install_url: Optional[str],
    download_manager: DownloadManager
) -> Optional[str]:
    if not install_url:
        return None
    if not install_url.endswith(TAR_GZ_EXT):
        raise ValueError(f'only supporting {TAR_GZ_EXT}')
    local_file = download_manager.download_if_url(
        install_url,
        auto_uncompress=False
    )
    extracted_directory = local_file[:-len(TAR_GZ_EXT)]
    if os.path.isdir(extracted_directory):
        wapiti_binary = os.path.join(
            _get_wapiti_source_directory(extracted_directory), DEFAULT_WAPITI_PATH
        )
        if os.path.isfile(wapiti_binary) and os.access(wapiti_binary, os.X_OK):
            LOGGER.info('wapiti already built: %s', wapiti_binary)
            return wapiti_binary
    LOGGER.debug('local_file: %s', local_file)
    LOGGER.debug('extracting to: %s', extracted_directory)
    with tarfile.open(local_file, mode='r') as tar:
        _safe_extractall(tar, extracted_directory)
    wapiti_source_directory = _get_wapiti_source_directory(extracted_directory)
    remove_source_entries_colliding_with_binary(wapiti_source_directory)
    LOGGER.info('running make in %s', wapiti_source_directory)
    subprocess.check_output(
        'make',
        cwd=wapiti_source_directory
    )
    wapiti_binary = os.path.join(wapiti_source_directory, DEFAULT_WAPITI_PATH)
    if not os.path.isfile(wapiti_binary):
        raise RuntimeError(
            f'make completed but produced no wapiti binary at {wapiti_binary!r}'
        )
    LOGGER.info('done, binary: %s', wapiti_binary)
    return wapiti_binary


@dataclass
class LazyWapitiBinaryWrapper:
    download_manager: DownloadManager
    install_url: Optional[str] = None
    _binary_path: Optional[str] = None

    def get_binary_path(self) -> str:
        if not self.install_url:
            return DEFAULT_WAPITI_PATH
        if self._binary_path:
            return self._binary_path
        self._binary_path = install_wapiti_and_get_path_or_none(
            self.install_url,
            download_manager=self.download_manager
        ) or DEFAULT_WAPITI_PATH
        return self._binary_path
