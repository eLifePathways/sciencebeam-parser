from dataclasses import dataclass
import os
import logging
from shutil import copyfileobj
from typing import Optional, Sequence
import zipfile

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager
from sciencebeam_trainer_delft.utils.io import is_external_location


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZipUrl:
    archive_url: str
    inner_path: Optional[str]


def download_if_url_from_alternatives(
    download_manager: DownloadManager,
    alternative_file_url_or_path_list: Sequence[str]
) -> str:
    for file_url_or_path in alternative_file_url_or_path_list:
        if not is_external_location(file_url_or_path):
            if os.path.exists(file_url_or_path):
                return file_url_or_path
            LOGGER.debug('local file doesnt exist: %r', file_url_or_path)
            continue
        local_file = download_manager.get_local_file(file_url_or_path)
        if os.path.exists(local_file):
            return local_file
    LOGGER.debug(
        'no existing local files found, downloading: %r', alternative_file_url_or_path_list
    )
    for file_url_or_path in alternative_file_url_or_path_list:
        try:
            local_file = download_manager.download_if_url(file_url_or_path)
            if os.path.exists(local_file):
                return local_file
            LOGGER.debug(
                'local file for %r not found: %r',
                file_url_or_path, local_file
            )
        except FileNotFoundError:
            LOGGER.debug('remote file not found: %r', file_url_or_path)
    raise FileNotFoundError('no file found for %r' % alternative_file_url_or_path_list)


def parse_zip_url(value: str) -> ZipUrl:
    if '!' not in value:
        return ZipUrl(archive_url=value, inner_path=None)
    archive_url, inner_path = value.split('!', 1)
    return ZipUrl(archive_url=archive_url, inner_path=inner_path or None)


def extract_file_from_zip_archive_to_file(
    archive_file: str,
    inner_path: str,
    local_file: str
) -> None:
    tmp_file = local_file + '.part'
    with zipfile.ZipFile(archive_file, 'r') as zf:
        try:
            member_name = inner_path.lstrip('/')
            if member_name not in zf.namelist():
                raise FileNotFoundError(
                    f'inner path {member_name!r} not found in archive {archive_file!r}'
                    + f' (available: {zf.namelist()!r})'
                )

            with zf.open(member_name, 'r') as source_fp, open(tmp_file, 'wb') as target_fp:
                copyfileobj(source_fp, target_fp)
            os.replace(tmp_file, local_file)
        except Exception:
            LOGGER.exception(
                'error extracting %r from archive %r',
                inner_path,
                archive_file
            )
            if os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except OSError:
                    LOGGER.warning(
                        'could not remove temporary file %r', tmp_file
                    )
            raise


def download_with_zip_path_support(
    download_manager: DownloadManager,
    url: str
) -> str:
    zip_url = parse_zip_url(url)
    if not zip_url.inner_path:
        return download_manager.download_if_url(url)
    local_file = download_manager.get_local_file(url)
    if os.path.exists(local_file):
        LOGGER.debug('local file for %r already exists: %r', url, local_file)
        return local_file
    archive_file = download_manager.download_if_url(zip_url.archive_url)
    extract_file_from_zip_archive_to_file(
        archive_file=archive_file,
        inner_path=zip_url.inner_path,
        local_file=local_file
    )
    return local_file
