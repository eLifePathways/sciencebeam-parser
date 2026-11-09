from io import BytesIO
from typing import Iterable, Optional, Sequence
from unittest.mock import MagicMock, patch

import httpx2
import pytest
from fastapi.testclient import TestClient

from sciencebeam_parser.app.parser import (
    UnsupportedRequestMediaTypeScienceBeamParserError
)
from sciencebeam_parser.app.profiles import (
    PROFILE_DIGEST_HEADER_NAME,
    PROFILE_HEADER_NAME,
    ProfileBundle
)
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.models.llm.usage import USAGE_HEADER_NAME
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.service.cors import (
    ANY_ORIGIN,
    get_allowed_origins
)
from sciencebeam_parser.service.server import create_app_for_parser


PERMITTED_ORIGIN = 'https://demo.example.org'
OTHER_ORIGIN = 'https://other.example.org'

PDF_FILENAME_1 = 'test.pdf'
PDF_CONTENT_1 = b'test pdf content'

ALLOW_ORIGIN_HEADER_NAME = 'Access-Control-Allow-Origin'
EXPOSE_HEADERS_HEADER_NAME = 'Access-Control-Expose-Headers'


def get_config_for_allowed_origins(allowed_origins) -> AppConfig:
    return AppConfig({'cors': {'allowed_origins': allowed_origins}})


@pytest.fixture(name='env_vars_mock')
def _env_vars_mock() -> Iterable[dict]:
    mock: dict
    with patch('os.environ', {}) as mock:
        yield mock


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


@pytest.fixture(name='get_local_file_for_response_media_type_mock')
def _get_local_file_for_response_media_type_mock(
    sciencebeam_parser_mock: MagicMock
) -> MagicMock:
    return (
        sciencebeam_parser_mock
        .get_new_session.return_value.__enter__.return_value
        .get_source.return_value
        .get_local_file_for_response_media_type
    )


def create_test_client(
    sciencebeam_parser_mock: MagicMock,
    allowed_origins: Optional[Sequence[str]] = None
) -> TestClient:
    sciencebeam_parser_mock.config = get_config_for_allowed_origins(
        list(allowed_origins or [])
    )
    return TestClient(
        create_app_for_parser(sciencebeam_parser_mock),
        raise_server_exceptions=False
    )


def post_document(client: TestClient, origin: Optional[str] = None) -> httpx2.Response:
    return client.post(
        '/api/convert',
        files={'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))},
        headers={'Origin': origin} if origin else {}
    )


class TestGetAllowedOrigins:
    def test_should_permit_nothing_by_default(self):
        assert not get_allowed_origins(AppConfig({}))

    def test_should_permit_nothing_for_an_empty_cors_block(self):
        assert not get_allowed_origins(AppConfig({'cors': None}))

    def test_should_permit_nothing_for_an_unset_value(self):
        assert not get_allowed_origins(get_config_for_allowed_origins(None))

    def test_should_read_a_list_of_origins(self):
        assert get_allowed_origins(
            get_config_for_allowed_origins([PERMITTED_ORIGIN, OTHER_ORIGIN])
        ) == [PERMITTED_ORIGIN, OTHER_ORIGIN]

    def test_should_accept_a_single_origin_without_a_list(self):
        """What `SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS=https://x` parses to."""
        assert get_allowed_origins(
            get_config_for_allowed_origins(PERMITTED_ORIGIN)
        ) == [PERMITTED_ORIGIN]

    def test_should_accept_any_origin_without_a_list(self):
        """What `SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS=*` parses to."""
        assert get_allowed_origins(
            get_config_for_allowed_origins(ANY_ORIGIN)
        ) == [ANY_ORIGIN]


class TestAllowedOriginsFromTheEnvironment:
    def test_should_read_a_list_of_origins(self, env_vars_mock: dict):
        env_vars_mock['SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS'] = (
            f'[{PERMITTED_ORIGIN}, {OTHER_ORIGIN}]'
        )
        config = get_config_for_allowed_origins([]).apply_environment_variables()
        assert get_allowed_origins(config) == [PERMITTED_ORIGIN, OTHER_ORIGIN]

    def test_should_read_a_single_origin(self, env_vars_mock: dict):
        env_vars_mock['SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS'] = PERMITTED_ORIGIN
        config = get_config_for_allowed_origins([]).apply_environment_variables()
        assert get_allowed_origins(config) == [PERMITTED_ORIGIN]

    def test_should_read_an_unquoted_asterisk(self, env_vars_mock: dict):
        """What a deployment manifest's `value: '*'` passes to the process."""
        env_vars_mock['SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS'] = ANY_ORIGIN
        config = get_config_for_allowed_origins([]).apply_environment_variables()
        assert get_allowed_origins(config) == [ANY_ORIGIN]

    def test_should_permit_nothing_for_an_empty_value(self, env_vars_mock: dict):
        env_vars_mock['SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS'] = ''
        config = get_config_for_allowed_origins([PERMITTED_ORIGIN]).apply_environment_variables()
        assert not get_allowed_origins(config)


class TestShippedConfig:
    def test_should_permit_no_origin(self):
        """Upgrading a deployment that changes nothing changes nothing."""
        assert not get_allowed_origins(AppConfig.load_yaml(DEFAULT_CONFIG_FILE))


class TestWithoutAnyPermittedOrigin:
    def test_should_not_let_a_page_read_the_response(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock
    ):
        get_local_file_for_response_media_type_mock.return_value = None
        client = create_test_client(sciencebeam_parser_mock)
        response = post_document(client, origin=PERMITTED_ORIGIN)
        assert ALLOW_ORIGIN_HEADER_NAME not in response.headers

    def test_should_not_answer_a_preflight(
        self,
        sciencebeam_parser_mock: MagicMock
    ):
        client = create_test_client(sciencebeam_parser_mock)
        response = client.options('/api/convert', headers={
            'Origin': PERMITTED_ORIGIN,
            'Access-Control-Request-Method': 'POST'
        })
        assert ALLOW_ORIGIN_HEADER_NAME not in response.headers


class TestWithAPermittedOrigin:
    def test_should_let_a_page_on_it_read_the_response(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock,
        tmp_path
    ):
        local_file = tmp_path / 'result.xml'
        local_file.write_bytes(b'<TEI></TEI>')
        get_local_file_for_response_media_type_mock.return_value = str(local_file)
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = post_document(client, origin=PERMITTED_ORIGIN)
        assert response.status_code == 200
        assert response.headers[ALLOW_ORIGIN_HEADER_NAME] == PERMITTED_ORIGIN

    def test_should_not_let_a_page_on_another_origin_read_the_response(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock,
        tmp_path
    ):
        local_file = tmp_path / 'result.xml'
        local_file.write_bytes(b'<TEI></TEI>')
        get_local_file_for_response_media_type_mock.return_value = str(local_file)
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = post_document(client, origin=OTHER_ORIGIN)
        assert ALLOW_ORIGIN_HEADER_NAME not in response.headers

    def test_should_answer_a_preflight_without_parsing_a_document(
        self,
        sciencebeam_parser_mock: MagicMock
    ):
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = client.options('/api/convert', headers={
            'Origin': PERMITTED_ORIGIN,
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type'
        })
        assert response.status_code == 200
        assert response.headers[ALLOW_ORIGIN_HEADER_NAME] == PERMITTED_ORIGIN
        sciencebeam_parser_mock.get_new_session.assert_not_called()

    def test_should_not_answer_a_preflight_from_another_origin(
        self,
        sciencebeam_parser_mock: MagicMock
    ):
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = client.options('/api/convert', headers={
            'Origin': OTHER_ORIGIN,
            'Access-Control-Request-Method': 'POST'
        })
        assert ALLOW_ORIGIN_HEADER_NAME not in response.headers

    def test_should_let_a_page_read_what_served_the_request(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock,
        tmp_path
    ):
        local_file = tmp_path / 'result.xml'
        local_file.write_bytes(b'<TEI></TEI>')
        get_local_file_for_response_media_type_mock.return_value = str(local_file)
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = post_document(client, origin=PERMITTED_ORIGIN)
        exposed = response.headers[EXPOSE_HEADERS_HEADER_NAME]
        for header_name in [
            PROFILE_HEADER_NAME, PROFILE_DIGEST_HEADER_NAME, USAGE_HEADER_NAME
        ]:
            assert header_name in exposed

    def test_should_let_a_page_read_an_unsupported_media_type_failure(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock
    ):
        get_local_file_for_response_media_type_mock.side_effect = (
            UnsupportedRequestMediaTypeScienceBeamParserError()
        )
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = post_document(client, origin=PERMITTED_ORIGIN)
        assert response.status_code == 406
        assert response.headers[ALLOW_ORIGIN_HEADER_NAME] == PERMITTED_ORIGIN

    def test_should_let_a_page_read_an_unhandled_failure(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock
    ):
        """The `500` is built by the api app's own error middleware."""
        get_local_file_for_response_media_type_mock.side_effect = RuntimeError('oh dear')
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = post_document(client, origin=PERMITTED_ORIGIN)
        assert response.status_code == 500
        assert response.headers[ALLOW_ORIGIN_HEADER_NAME] == PERMITTED_ORIGIN

    def test_should_not_permit_credentials(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock,
        tmp_path
    ):
        """An anonymous API says so, rather than leaving it to a default."""
        local_file = tmp_path / 'result.xml'
        local_file.write_bytes(b'<TEI></TEI>')
        get_local_file_for_response_media_type_mock.return_value = str(local_file)
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = post_document(client, origin=PERMITTED_ORIGIN)
        assert 'Access-Control-Allow-Credentials' not in response.headers

    def test_should_not_permit_credentials_on_a_preflight(
        self,
        sciencebeam_parser_mock: MagicMock
    ):
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = client.options('/api/convert', headers={
            'Origin': PERMITTED_ORIGIN,
            'Access-Control-Request-Method': 'POST'
        })
        assert 'Access-Control-Allow-Credentials' not in response.headers

    def test_should_not_answer_for_the_index_page(
        self,
        sciencebeam_parser_mock: MagicMock
    ):
        client = create_test_client(sciencebeam_parser_mock, [PERMITTED_ORIGIN])
        response = client.get('/', headers={'Origin': PERMITTED_ORIGIN})
        assert response.status_code == 200
        assert ALLOW_ORIGIN_HEADER_NAME not in response.headers


class TestWithAnyOriginPermitted:
    def test_should_let_a_page_on_any_origin_read_the_response(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock,
        tmp_path
    ):
        local_file = tmp_path / 'result.xml'
        local_file.write_bytes(b'<TEI></TEI>')
        get_local_file_for_response_media_type_mock.return_value = str(local_file)
        client = create_test_client(sciencebeam_parser_mock, [ANY_ORIGIN])
        response = post_document(client, origin=OTHER_ORIGIN)
        assert response.status_code == 200
        assert response.headers[ALLOW_ORIGIN_HEADER_NAME] == ANY_ORIGIN
