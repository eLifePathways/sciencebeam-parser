import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_parser.utils.download import download_if_url_from_alternatives


LOGGER = logging.getLogger(__name__)


TEST_URL_1 = 'test://download/file1'
TEST_URL_2 = 'test://download/file2'

DATA_1 = b'data 1'


class MockDownloadManager(DownloadManager):
    def __init__(
        self,
        download_dir: str
    ):
        super().__init__(download_dir)
        self.download_dir = Path(download_dir)
        self.data_by_url: Dict[str, bytes] = {}
        self.downloaded_urls: List[str] = []
        self.failed_urls: List[str] = []

    def set_data_by_url(self, url: str, data: bytes):
        self.data_by_url[url] = data

    def download(
        self,
        file_url: str,
        local_file: Optional[str] = None,
        auto_uncompress: bool = True,
        skip_if_downloaded: bool = True
    ) -> str:
        assert skip_if_downloaded
        assert local_file is None
        if file_url not in self.data_by_url:
            LOGGER.debug('url %r not in predefined urls: %r', file_url, self.data_by_url.keys())
            self.failed_urls.append(file_url)
            raise FileNotFoundError('file not found: %r' % file_url)
        _data = self.data_by_url[file_url]
        local_file_path = Path(self.get_local_file(file_url))
        if local_file_path.exists():
            return str(local_file_path)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        local_file_path.write_bytes(_data)
        LOGGER.debug('provided downloaded data for %r -> %r', file_url, str(local_file_path))
        self.downloaded_urls.append(file_url)
        return str(local_file_path)


@pytest.fixture(name='download_manager_mock')
def _download_manager_mock(tmp_path: Path) -> MockDownloadManager:
    download_dir_path = tmp_path / 'downloads'
    mock = MockDownloadManager(str(download_dir_path))
    return mock


class TestDownloadIfUrlFromAlternatives:
    def test_should_return_local_existing_path_from_single_file_list(
        self,
        tmp_path: Path,
        download_manager_mock: MockDownloadManager
    ):
        local_file_path = tmp_path / 'existing.bin'
        local_file_path.write_bytes(DATA_1)
        result = download_if_url_from_alternatives(
            download_manager_mock,
            [str(local_file_path)]
        )
        assert result == str(local_file_path)
        assert not download_manager_mock.downloaded_urls

    def test_should_return_local_existing_path_ignoring_other_non_existing_files(
        self,
        tmp_path: Path,
        download_manager_mock: MockDownloadManager
    ):
        local_file_path = tmp_path / 'existing.bin'
        local_file_path.write_bytes(DATA_1)
        result = download_if_url_from_alternatives(
            download_manager_mock,
            [
                str(tmp_path / 'non-existing1.bin'),
                str(local_file_path),
                str(tmp_path / 'non-existing2.bin')
            ]
        )
        assert result == str(local_file_path)
        assert not download_manager_mock.downloaded_urls

    def test_should_return_download_first_file(
        self,
        download_manager_mock: MockDownloadManager
    ):
        download_manager_mock.set_data_by_url(TEST_URL_1, DATA_1)
        result = download_if_url_from_alternatives(
            download_manager_mock,
            [TEST_URL_1, TEST_URL_2]
        )
        assert result == download_manager_mock.get_local_file(TEST_URL_1)
        assert os.path.exists(result)
        assert download_manager_mock.downloaded_urls == [TEST_URL_1]

    def test_should_return_download_second_file(
        self,
        download_manager_mock: MockDownloadManager
    ):
        download_manager_mock.set_data_by_url(TEST_URL_2, DATA_1)
        result = download_if_url_from_alternatives(
            download_manager_mock,
            [TEST_URL_1, TEST_URL_2]
        )
        assert result == download_manager_mock.get_local_file(TEST_URL_2)
        assert os.path.exists(result)
        assert download_manager_mock.downloaded_urls == [TEST_URL_2]

    def test_should_not_download_if_already_downloaded_first_file(
        self,
        download_manager_mock: MockDownloadManager
    ):
        download_manager_mock.set_data_by_url(TEST_URL_1, DATA_1)
        download_manager_mock.download(TEST_URL_1)
        download_manager_mock.downloaded_urls.clear()

        result = download_if_url_from_alternatives(
            download_manager_mock,
            [TEST_URL_1, TEST_URL_2]
        )
        assert result == download_manager_mock.get_local_file(TEST_URL_1)
        assert os.path.exists(result)
        assert not download_manager_mock.downloaded_urls
        assert not download_manager_mock.failed_urls

    def test_should_not_download_if_already_downloaded_second_file(
        self,
        download_manager_mock: MockDownloadManager
    ):
        download_manager_mock.set_data_by_url(TEST_URL_2, DATA_1)
        download_manager_mock.download(TEST_URL_2)
        download_manager_mock.downloaded_urls.clear()

        result = download_if_url_from_alternatives(
            download_manager_mock,
            [TEST_URL_1, TEST_URL_2]
        )
        assert result == download_manager_mock.get_local_file(TEST_URL_2)
        assert os.path.exists(result)
        assert not download_manager_mock.downloaded_urls
        assert not download_manager_mock.failed_urls
