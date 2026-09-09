from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import logging
import threading
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import httpx
import pytest

from sciencebeam_parser.app.parser import (
    UnsupportedRequestMediaTypeScienceBeamParserError,
    UnsupportedResponseMediaTypeScienceBeamParserError
)
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.models.llm.decode import LlmResponseError
from sciencebeam_parser.models.llm.usage import USAGE_HEADER_NAME, record_llm_usage
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.service.api.app import create_api_app
from sciencebeam_parser.utils.media_types import MediaTypes


LOGGER = logging.getLogger(__name__)

PDF_FILENAME_1 = 'test.pdf'
PDF_CONTENT_1 = b'test pdf content'
DOCX_CONTENT_1 = b'test docx content 1'
XML_CONTENT_1 = b'<article></article>'

LLM_RESPONSE_1 = {
    'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'cost': 0.0001},
    'model': 'qwen/qwen3.5-9b',
    'provider': 'SiliconFlow',
}

TEI_XML_CONTENT_1 = b'<TEI>1</TEI>'
JATS_XML_CONTENT_1 = b'<article>1</article>'
IMAGE_DATA_1 = b'Image 1'
ZIP_CONTENT_1 = b'ZIP 1'


@pytest.fixture(name='request_temp_path')
def _request_temp_path(tmp_path: Path) -> Path:
    request_temp_path = tmp_path / 'request'
    request_temp_path.mkdir()
    return request_temp_path


@pytest.fixture(name='sciencebeam_parser_mock')
def _sciencebeam_parser_mock() -> MagicMock:
    mock = MagicMock(
        name='sciencebeam_parser_mock'
    )
    return mock


@pytest.fixture(name='sciencebeam_parser_session_mock')
def _sciencebeam_parser_session_mock(
    sciencebeam_parser_mock: MagicMock
) -> MagicMock:
    return sciencebeam_parser_mock.get_new_session.return_value.__enter__.return_value


@pytest.fixture(name='sciencebeam_parser_source_mock')
def _sciencebeam_parser_source_mock(
    sciencebeam_parser_session_mock: MagicMock
) -> MagicMock:
    return sciencebeam_parser_session_mock.get_source.return_value


@pytest.fixture(name='get_local_file_for_response_media_type_mock')
def _get_local_file_for_response_media_type_mock(
    sciencebeam_parser_source_mock: MagicMock
) -> MagicMock:
    return sciencebeam_parser_source_mock.get_local_file_for_response_media_type


@pytest.fixture(name='app_config', scope='session')
def _app_config() -> AppConfig:
    return AppConfig.load_yaml(DEFAULT_CONFIG_FILE)


def create_test_client(
    sciencebeam_parser_mock: MagicMock
) -> TestClient:
    app = create_api_app(
        sciencebeam_parser=sciencebeam_parser_mock
    )
    client = TestClient(app)
    return client


@pytest.fixture(name='test_client')
def _test_client(
    sciencebeam_parser_mock: MagicMock
) -> TestClient:
    return create_test_client(
        sciencebeam_parser_mock=sciencebeam_parser_mock
    )


def _get_ok_json(response: httpx.Response) -> dict:
    assert response.status_code == 200
    return response.json()


class TestApiApp:
    class TestRoot:
        def test_should_not_fail(self, test_client: TestClient):
            response = test_client.get('/')
            assert _get_ok_json(response)

    class TestIsAlive:
        def test_should_return_true(self, test_client: TestClient):
            response = test_client.get('/isalive')
            assert response.status_code == 200
            assert response.text == 'true'

    class TestPdfAlto:
        def test_should_reject_post_without_data(
            self,
            test_client: TestClient
        ):
            response = test_client.post('/pdfalto')
            assert response.status_code == 400

        @pytest.mark.parametrize(
            'post_data_key',
            ['file', 'input']
        )
        def test_should_accept_file_and_pass_to_convert_method(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path,
            post_data_key: str
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/pdfalto', files={
                post_data_key: (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1), 'application/pdf')
            })
            assert response.status_code == 200
            assert response.content == XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.ALTO_XML
            )

    class _AbstractPdfConversionApiTest(ABC):
        @abstractmethod
        def get_api_path(self) -> str:
            pass

        def test_should_reject_post_without_data(self, test_client: TestClient):
            response = test_client.post(self.get_api_path())
            assert response.status_code == 400

        def test_should_reject_unsupported_request_media_type(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock
        ):
            get_local_file_for_response_media_type_mock.side_effect = (
                UnsupportedRequestMediaTypeScienceBeamParserError()
            )
            response = test_client.post(self.get_api_path(), files={
                'input': ('test.unsupported', BytesIO(PDF_CONTENT_1))
            })
            assert response.status_code == 406

        def test_should_reject_unsupported_accept_media_type(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock
        ):
            get_local_file_for_response_media_type_mock.side_effect = (
                UnsupportedResponseMediaTypeScienceBeamParserError()
            )
            response = test_client.post(self.get_api_path(), files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            }, headers={'Accept': 'media/unsupported'})
            assert response.status_code == 406

        def test_should_log_the_uploaded_filename(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path,
            caplog: pytest.LogCaptureFixture
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(TEI_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(
                expected_output_path
            )
            with caplog.at_level(
                logging.INFO, logger='sciencebeam_parser.service.api.dependencies'
            ):
                response = test_client.post(self.get_api_path(), files={
                    'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
                })
            assert response.status_code == 200
            assert any(
                PDF_FILENAME_1 in message for message in caplog.messages
            ), caplog.messages

    class TestProcessHeaderDocument(_AbstractPdfConversionApiTest):
        def get_api_path(self) -> str:
            return '/processHeaderDocument'

        @pytest.mark.parametrize(
            'post_data_key',
            ['file', 'input']
        )
        def test_should_accept_file_and_pass_to_convert_method(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path,
            post_data_key: str
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(TEI_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processHeaderDocument', files={
                post_data_key: (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            })
            assert response.status_code == 200
            assert response.content == TEI_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.TEI_XML
            )

        def test_should_convert_to_jats(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(JATS_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processHeaderDocument', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            }, headers={'Accept': MediaTypes.JATS_XML})
            assert response.status_code == 200
            assert response.content == JATS_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.JATS_XML
            )

    class TestProcessFullTextDocument(_AbstractPdfConversionApiTest):
        def get_api_path(self) -> str:
            return '/processFulltextDocument'

        @pytest.mark.parametrize(
            'post_data_key',
            ['file', 'input']
        )
        def test_should_accept_file_and_pass_to_convert_method(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path,
            post_data_key: str
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(TEI_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processFulltextDocument', files={
                post_data_key: (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            })
            assert response.status_code == 200
            assert response.content == TEI_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.TEI_XML
            )

        def test_should_convert_to_jats(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(JATS_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processFulltextDocument', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            }, headers={'Accept': MediaTypes.JATS_XML})
            assert response.status_code == 200
            assert response.content == JATS_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.JATS_XML
            )

    class TestProcessReferences(_AbstractPdfConversionApiTest):
        def get_api_path(self) -> str:
            return '/processReferences'

        @pytest.mark.parametrize(
            'post_data_key',
            ['file', 'input']
        )
        def test_should_accept_file_and_pass_to_convert_method(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path,
            post_data_key: str
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(TEI_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processReferences', files={
                post_data_key: (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            })
            assert response.status_code == 200
            assert response.content == TEI_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.TEI_XML
            )

        def test_should_convert_to_jats(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(JATS_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processReferences', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            }, headers={'Accept': MediaTypes.JATS_XML})
            assert response.status_code == 200
            assert response.content == JATS_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.JATS_XML
            )

    class TestProcessFullTextAssetDocument(_AbstractPdfConversionApiTest):
        def get_api_path(self) -> str:
            return '/processFulltextAssetDocument'

        @pytest.mark.parametrize(
            'post_data_key',
            ['file', 'input']
        )
        def test_should_accept_file_and_pass_to_convert_method(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path,
            post_data_key: str
        ):
            expected_output_path = request_temp_path / 'result.zip'
            expected_output_path.write_bytes(ZIP_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processFulltextAssetDocument', files={
                post_data_key: (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            })
            assert response.status_code == 200
            assert response.headers.get('content-type') == 'application/zip'
            assert response.content == ZIP_CONTENT_1

        def test_should_convert_to_jats(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.zip'
            expected_output_path.write_bytes(ZIP_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/processFulltextAssetDocument', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            }, headers={'Accept': MediaTypes.JATS_ZIP})
            assert response.status_code == 200
            assert response.headers.get('content-type') == 'application/zip'
            assert response.content == ZIP_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.JATS_ZIP
            )

    class TestProcessConvert(_AbstractPdfConversionApiTest):
        def get_api_path(self) -> str:
            return '/convert'

        def test_should_convert_to_jats_by_default(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(JATS_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/convert', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            })
            assert response.status_code == 200
            assert response.content == JATS_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.JATS_XML
            )

        def test_should_return_tei_xml_if_requested(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(TEI_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/convert', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            }, headers={'Accept': MediaTypes.TEI_XML})
            assert response.status_code == 200
            assert response.content == TEI_XML_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.TEI_XML
            )

        def test_should_be_able_to_request_jats_zip(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.zip'
            expected_output_path.write_bytes(ZIP_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post('/convert', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            }, headers={'Accept': MediaTypes.JATS_ZIP})
            assert response.status_code == 200
            assert response.headers.get('content-type') == 'application/zip'
            assert response.content == ZIP_CONTENT_1
            get_local_file_for_response_media_type_mock.assert_called_with(
                MediaTypes.JATS_ZIP
            )

        def test_should_be_able_to_pass_includes_request_arg(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(JATS_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post(
                '/convert',
                files={'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))},
                headers={'Accept': MediaTypes.JATS_XML},
                params={'includes': 'title'}
            )
            assert response.status_code == 200
            assert response.content == JATS_XML_CONTENT_1

        def test_should_accept_docx(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'result.xml'
            expected_output_path.write_bytes(JATS_XML_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post(
                '/convert',
                files={'input': ('test.docx', BytesIO(DOCX_CONTENT_1))},
                headers={'Accept': MediaTypes.JATS_XML},
                params={'includes': 'title'}
            )
            assert response.status_code == 200
            assert response.content == JATS_XML_CONTENT_1

        def test_should_convert_docx_to_pdf(
            self,
            test_client: TestClient,
            get_local_file_for_response_media_type_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_output_path = request_temp_path / 'test.pdf'
            expected_output_path.write_bytes(PDF_CONTENT_1)
            get_local_file_for_response_media_type_mock.return_value = str(expected_output_path)
            response = test_client.post(
                '/convert',
                files={'input': ('test.docx', BytesIO(DOCX_CONTENT_1))},
                headers={'Accept': MediaTypes.PDF}
            )
            assert response.status_code == 200
            assert response.content == PDF_CONTENT_1


class TestLlmUsageHeader:
    """The response carries what the request spent, on both paths.

    `get_local_file_for_response_media_type` stands in for the work of the
    request: what matters here is that a completion recorded inside the handler
    reaches the response, and that a handler which raises still reports it.
    """
    def _usage_recording_side_effect(self, output_path: Path, calls: int):
        def side_effect(*_args, **_kwargs):
            for _ in range(calls):
                record_llm_usage('citation', LLM_RESPONSE_1)
            return str(output_path)
        return side_effect

    def test_should_report_what_the_request_spent(
        self,
        test_client: TestClient,
        get_local_file_for_response_media_type_mock: MagicMock,
        request_temp_path: Path
    ):
        output_path = request_temp_path / 'result.xml'
        output_path.write_bytes(TEI_XML_CONTENT_1)
        get_local_file_for_response_media_type_mock.side_effect = (
            self._usage_recording_side_effect(output_path, calls=3)
        )
        response = test_client.post('/processFulltextDocument', files={
            'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
        })
        assert response.status_code == 200
        assert response.content == TEI_XML_CONTENT_1
        usage = json.loads(response.headers[USAGE_HEADER_NAME])
        assert usage['calls'] == 3
        assert usage['input_tokens'] == 300
        assert usage['cost_credits'] == pytest.approx(0.0003)
        assert usage['by_task']['citation']['calls'] == 3

    def test_should_report_what_a_failed_request_had_already_spent(
        self,
        sciencebeam_parser_mock: MagicMock,
        get_local_file_for_response_media_type_mock: MagicMock
    ):
        def side_effect(*_args, **_kwargs):
            record_llm_usage('citation', LLM_RESPONSE_1)
            raise LlmResponseError('response could not be parsed')
        get_local_file_for_response_media_type_mock.side_effect = side_effect
        client = TestClient(
            create_api_app(sciencebeam_parser=sciencebeam_parser_mock),
            raise_server_exceptions=False
        )
        response = client.post('/processFulltextDocument', files={
            'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
        })
        assert response.status_code == 500
        usage = json.loads(response.headers[USAGE_HEADER_NAME])
        assert usage['calls'] == 1
        assert usage['input_tokens'] == 100

    def test_should_not_add_a_header_when_no_llm_engine_ran(
        self,
        test_client: TestClient,
        get_local_file_for_response_media_type_mock: MagicMock,
        request_temp_path: Path
    ):
        output_path = request_temp_path / 'result.xml'
        output_path.write_bytes(TEI_XML_CONTENT_1)
        get_local_file_for_response_media_type_mock.return_value = str(output_path)
        response = test_client.post('/processFulltextDocument', files={
            'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
        })
        assert response.status_code == 200
        assert USAGE_HEADER_NAME not in response.headers

    def test_should_keep_concurrent_requests_apart(
        self,
        test_client: TestClient,
        get_local_file_for_response_media_type_mock: MagicMock,
        request_temp_path: Path
    ):
        output_path = request_temp_path / 'result.xml'
        output_path.write_bytes(TEI_XML_CONTENT_1)
        barrier = threading.Barrier(2, timeout=5)
        calls_per_request = iter([3, 1])
        lock = threading.Lock()

        def side_effect(*_args, **_kwargs):
            with lock:
                calls = next(calls_per_request)
            # Both requests are inside the handler before either records, so a
            # shared accumulator would report 4 to both of them.
            barrier.wait()
            for _ in range(calls):
                record_llm_usage('citation', LLM_RESPONSE_1)
            return str(output_path)
        get_local_file_for_response_media_type_mock.side_effect = side_effect

        def post(_index: int) -> int:
            response = test_client.post('/processFulltextDocument', files={
                'input': (PDF_FILENAME_1, BytesIO(PDF_CONTENT_1))
            })
            assert response.status_code == 200
            return json.loads(response.headers[USAGE_HEADER_NAME])['calls']

        with ThreadPoolExecutor(max_workers=2) as pool:
            reported = list(pool.map(post, range(2)))

        assert sorted(reported) == [1, 3]
