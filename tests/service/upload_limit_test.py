from io import BytesIO
from unittest.mock import MagicMock

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from sciencebeam_parser.app.profiles import ProfileBundle
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig
from sciencebeam_parser.service.server import create_app_for_parser
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.service.upload_limit import (
    add_upload_size_limit,
    get_max_upload_bytes
)


ORIGIN_1 = 'https://demo.example.org'

ALLOW_ORIGIN_HEADER_NAME = 'Access-Control-Allow-Origin'

MAX_BYTES_1 = 100

# comfortably above one multipart part's boundary and header overhead
MAX_BYTES_ROOMY = 10000


class TestGetMaxUploadBytes:
    def test_should_apply_the_limit_the_shipped_config_declares(self):
        """The default lives in config.yml alone, so nothing can drift from it."""
        assert get_max_upload_bytes(AppConfig.load_yaml(DEFAULT_CONFIG_FILE))

    def test_should_apply_no_limit_for_an_absent_key(self):
        assert get_max_upload_bytes(AppConfig({})) is None

    def test_should_read_a_configured_limit(self):
        assert get_max_upload_bytes(AppConfig({'max_upload_bytes': 123})) == 123

    def test_should_treat_zero_as_no_limit(self):
        assert get_max_upload_bytes(AppConfig({'max_upload_bytes': 0})) is None

    def test_should_treat_a_blank_value_as_no_limit(self):
        """What `SCIENCEBEAM_PARSER__MAX_UPLOAD_BYTES=` parses to."""
        assert get_max_upload_bytes(AppConfig({'max_upload_bytes': None})) is None


def get_middleware_classes(app: FastAPI) -> list:
    return [middleware.cls for middleware in app.user_middleware]


class TestAddUploadSizeLimit:
    def test_should_add_nothing_without_a_limit(self):
        app = FastAPI()
        add_upload_size_limit(app, AppConfig({'max_upload_bytes': 0}))
        assert get_middleware_classes(app) == []

    def test_should_add_the_limit_middleware(self):
        app = FastAPI()
        add_upload_size_limit(app, AppConfig({'max_upload_bytes': 123}))
        assert get_middleware_classes(app) == [RequestBodyLimitMiddleware]


@pytest.fixture(name='sciencebeam_parser_mock')
def _sciencebeam_parser_mock() -> MagicMock:
    mock = MagicMock(name='sciencebeam_parser_mock')
    mock.profile_registry.get_bundle.return_value = ProfileBundle(
        name='profile1',
        fulltext_models=MagicMock(name='fulltext_models'),
        fulltext_processor_config=FullTextProcessorConfig(),
        models_digest='profile1-digest'
    )
    mock.profile_registry.get_available_profile_names.return_value = ['profile1']
    return mock


def create_test_client(
    sciencebeam_parser_mock: MagicMock,
    max_upload_bytes: int
) -> TestClient:
    sciencebeam_parser_mock.config = AppConfig({
        'max_upload_bytes': max_upload_bytes,
        'cors': {'allowed_origins': [ORIGIN_1]}
    })
    return TestClient(
        create_app_for_parser(sciencebeam_parser_mock),
        raise_server_exceptions=False
    )


def post_document(client: TestClient, content: bytes) -> httpx2.Response:
    return client.post(
        '/api/convert',
        files={'input': ('test.pdf', BytesIO(content))},
        headers={'Origin': ORIGIN_1}
    )


class TestUploadSizeLimitInTheApp:
    def test_should_reject_an_upload_over_the_limit(
        self, sciencebeam_parser_mock: MagicMock
    ):
        response = post_document(
            create_test_client(sciencebeam_parser_mock, MAX_BYTES_1),
            b'x' * (MAX_BYTES_1 * 2)
        )
        assert response.status_code == 413

    def test_should_accept_an_upload_within_the_limit(
        self, sciencebeam_parser_mock: MagicMock
    ):
        response = post_document(
            create_test_client(sciencebeam_parser_mock, MAX_BYTES_ROOMY),
            b'x' * (MAX_BYTES_ROOMY // 2)
        )
        assert response.status_code != 413

    def test_should_count_the_whole_body_not_only_the_file(
        self, sciencebeam_parser_mock: MagicMock
    ):
        """Multipart boundaries and headers are inside the limit too."""
        response = post_document(
            create_test_client(sciencebeam_parser_mock, MAX_BYTES_1),
            b'x' * (MAX_BYTES_1 // 2)
        )
        assert response.status_code == 413

    def test_should_let_a_rejected_upload_be_read_cross_origin(
        self, sciencebeam_parser_mock: MagicMock
    ):
        """The limit sits inside the cross-origin wrapper so a page sees the 413."""
        response = post_document(
            create_test_client(sciencebeam_parser_mock, MAX_BYTES_1),
            b'x' * (MAX_BYTES_1 * 2)
        )
        assert response.status_code == 413
        assert response.headers[ALLOW_ORIGIN_HEADER_NAME] == ORIGIN_1
