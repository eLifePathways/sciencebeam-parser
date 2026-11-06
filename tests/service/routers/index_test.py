import json
import os
import re
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from sciencebeam_parser.app.parser import DOC_TO_PDF_SUPPORTED_MEDIA_TYPES
from sciencebeam_parser.processors.fulltext.config import (
    FRONT_FIELDS,
    RequestFieldNames
)
from sciencebeam_parser.service.api.routers.convert import CONVERT_RESPONSE_MEDIA_TYPES
from sciencebeam_parser.service.routers.index import (
    DEPLOYMENT_PLACEHOLDER,
    INCLUDES_FIELD_NAMES,
    INDEX_HTML_FILE,
    STATIC_DIRECTORY,
    create_index_router,
    get_accepted_source_media_types,
    get_deployment_props,
    get_embedded_json,
    get_index_html
)
from sciencebeam_parser.utils.media_types import MediaTypes


@pytest.fixture(name='sciencebeam_parser_mock')
def _sciencebeam_parser_mock() -> MagicMock:
    sciencebeam_parser = MagicMock(name='sciencebeam_parser')
    sciencebeam_parser.doc_to_pdf_enabled = True
    return sciencebeam_parser


def get_embedded_deployment_props(html: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="deployment">(.*?)</script>',
        html,
        re.DOTALL
    )
    assert match
    return json.loads(match.group(1))


def get_test_client(sciencebeam_parser: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(create_index_router(sciencebeam_parser))
    return TestClient(app)


class TestGetAcceptedSourceMediaTypes:
    def test_should_accept_pdf_and_every_doc_to_pdf_media_type(self):
        accepted_media_types = get_accepted_source_media_types(doc_to_pdf_enabled=True)
        assert set(accepted_media_types) == (
            {MediaTypes.PDF} | DOC_TO_PDF_SUPPORTED_MEDIA_TYPES
        )

    def test_should_accept_pdf_only_without_doc_to_pdf(self):
        assert get_accepted_source_media_types(doc_to_pdf_enabled=False) == [
            MediaTypes.PDF
        ]

    def test_should_offer_pdf_first(self):
        assert get_accepted_source_media_types(
            doc_to_pdf_enabled=True
        )[0] == MediaTypes.PDF


class TestIncludesFieldNames:
    def test_should_offer_every_field_a_request_may_ask_for(self):
        assert set(INCLUDES_FIELD_NAMES) == (
            FRONT_FIELDS | {RequestFieldNames.REFERENCES}
        )


class TestGetDeploymentProps:
    def test_should_describe_every_accepted_media_type_with_an_extension(
        self, sciencebeam_parser_mock: MagicMock
    ):
        props = get_deployment_props(sciencebeam_parser_mock)
        accepted = props['accepted_source_media_types']
        assert [item['media_type'] for item in accepted] == (
            get_accepted_source_media_types(doc_to_pdf_enabled=True)
        )
        assert all(item['extension'] for item in accepted)

    def test_should_offer_the_output_media_types_convert_responds_with(
        self, sciencebeam_parser_mock: MagicMock
    ):
        props = get_deployment_props(sciencebeam_parser_mock)
        assert [
            item['media_type'] for item in props['output_media_types']
        ] == list(CONVERT_RESPONSE_MEDIA_TYPES)

    def test_should_say_which_output_media_types_are_readable_as_text(
        self, sciencebeam_parser_mock: MagicMock
    ):
        props = get_deployment_props(sciencebeam_parser_mock)
        is_text_by_media_type = {
            item['media_type']: item['is_text']
            for item in props['output_media_types']
        }
        assert is_text_by_media_type[MediaTypes.JATS_XML] is True
        assert is_text_by_media_type[MediaTypes.TEI_XML] is True
        assert is_text_by_media_type[MediaTypes.JATS_ZIP] is False
        assert is_text_by_media_type[MediaTypes.PDF] is False


class TestGetEmbeddedJson:
    def test_should_escape_angle_brackets_that_would_end_the_script(self):
        embedded_json = get_embedded_json({'key': '</script>'})
        assert '</script>' not in embedded_json
        assert json.loads(embedded_json) == {'key': '</script>'}


class TestGetIndexHtml:
    def test_should_replace_the_placeholder_with_the_deployment_props(self):
        html = get_index_html({'key': 'value'})
        assert DEPLOYMENT_PLACEHOLDER not in html
        assert get_embedded_deployment_props(html) == {'key': 'value'}


class TestIndexHtmlFile:
    def test_should_reference_assets_that_are_shipped(self):
        with open(INDEX_HTML_FILE, 'r', encoding='utf-8') as html_fp:
            html = html_fp.read()
        referenced_paths = re.findall(r'(?:href|src)="/static/([^"]+)"', html)
        assert referenced_paths
        for referenced_path in referenced_paths:
            assert os.path.exists(os.path.join(STATIC_DIRECTORY, referenced_path))

    def test_should_reference_nothing_outside_the_service(self):
        with open(INDEX_HTML_FILE, 'r', encoding='utf-8') as html_fp:
            html = html_fp.read()
        assert not re.search(r'(?:href|src)="(?:https?:)?//', html)


class TestIndexRouter:
    def test_should_respond_with_html(self, sciencebeam_parser_mock: MagicMock):
        response = get_test_client(sciencebeam_parser_mock).get('/')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/html')

    def test_should_carry_what_the_deployment_accepts(
        self, sciencebeam_parser_mock: MagicMock
    ):
        sciencebeam_parser_mock.doc_to_pdf_enabled = False
        response = get_test_client(sciencebeam_parser_mock).get('/')
        props = get_embedded_deployment_props(response.text)
        assert [
            item['media_type'] for item in props['accepted_source_media_types']
        ] == [MediaTypes.PDF]

    def test_should_carry_no_elife_or_pathways_mark(
        self, sciencebeam_parser_mock: MagicMock
    ):
        response = get_test_client(sciencebeam_parser_mock).get('/')
        assert 'elife' not in response.text.lower()
        assert 'pathways' not in response.text.lower()
