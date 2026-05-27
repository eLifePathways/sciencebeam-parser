import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Iterable, Iterator
from zipfile import ZipFile

from lxml import etree

import pytest

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.document.layout_document import LayoutGraphic
from sciencebeam_parser.document.semantic_document import SemanticDocument, SemanticGraphic
from sciencebeam_parser.document.tei.document import TeiDocument
from sciencebeam_parser.utils.media_types import MediaTypes
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE

from sciencebeam_parser.app import parser as parser_module
from sciencebeam_parser.app.parser import (
    TEMP_ALTO_XML_FILENAME,
    ScienceBeamParser,
    ScienceBeamParserSession,
    UnsupportedRequestMediaTypeScienceBeamParserError,
    UnsupportedResponseMediaTypeScienceBeamParserError,
    serialize_xml_to_file
)


LOGGER = logging.getLogger(__name__)

PDF_FILENAME_1 = 'test.pdf'
PDF_CONTENT_1 = b'test pdf content'
DOCX_CONTENT_1 = b'test docx content 1'
XML_CONTENT_1 = b'<article></article>'

TEI_XML_CONTENT_1 = b'<TEI>1</TEI>'
JATS_XML_CONTENT_1 = b'<article>1</article>'
IMAGE_DATA_1 = b'Image 1'


@pytest.fixture(name='request_temp_path')
def _request_temp_path(tmp_path: Path) -> Iterator[Path]:
    request_temp_path = tmp_path / 'request'
    request_temp_path.mkdir()
    with patch.object(parser_module, 'TemporaryDirectory') as mock:
        mock.return_value.__enter__.return_value = request_temp_path
        yield request_temp_path


@pytest.fixture(name='download_manager_class_mock', autouse=True)
def _download_manager_class_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'DownloadManager') as mock:
        yield mock


@pytest.fixture(name='pdfalto_wrapper_class_mock')
def _pdfalto_wrapper_class_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'PdfAltoWrapper') as mock:
        yield mock


@pytest.fixture(name='pdfalto_wrapper_mock', autouse=True)
def _pdfalto_wrapper_mock(pdfalto_wrapper_class_mock: MagicMock) -> MagicMock:
    return pdfalto_wrapper_class_mock.return_value


@pytest.fixture(name='full_text_processor_class_mock')
def _fulltextprocessor_class_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'FullTextProcessor') as mock:
        yield mock


@pytest.fixture(name='full_text_processor_mock', autouse=True)
def _fulltextprocessor_mock(full_text_processor_class_mock: MagicMock) -> MagicMock:
    return full_text_processor_class_mock.return_value


@pytest.fixture(name='load_models_mock', autouse=True)
def _load_models_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'load_models') as mock:
        yield mock


@pytest.fixture(name='fulltext_models', autouse=True)
def _fulltext_models(load_models_mock: MagicMock) -> MagicMock:
    return load_models_mock.return_value


@pytest.fixture(name='get_tei_for_semantic_document_mock', autouse=True)
def _get_tei_for_semantic_document_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'get_tei_for_semantic_document') as mock:
        mock.return_value = (
            TeiDocument(etree.fromstring(TEI_XML_CONTENT_1))
        )
        yield mock


@pytest.fixture(name='load_app_features_context_mock', autouse=True)
def _load_app_features_context_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'load_app_features_context') as mock:
        yield mock


@pytest.fixture(name='xslt_transformer_wrapper_class_mock', autouse=True)
def _xslt_transformer_wrapper_class_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'XsltTransformerWrapper') as mock:
        yield mock


@pytest.fixture(name='xslt_transformer_wrapper_mock')
def _xslt_transformer_wrapper_mock(
    xslt_transformer_wrapper_class_mock: MagicMock
) -> MagicMock:
    return xslt_transformer_wrapper_class_mock.from_template_file.return_value


@pytest.fixture(name='doc_converter_wrapper_class_mock', autouse=True)
def _doc_converter_wrapper_class_mock() -> Iterator[MagicMock]:
    with patch.object(parser_module, 'DocConverterWrapper') as mock:
        yield mock


@pytest.fixture(name='doc_converter_wrapper_mock')
def _doc_converter_wrapper_mock(
    doc_converter_wrapper_class_mock: MagicMock
) -> MagicMock:
    return doc_converter_wrapper_class_mock.return_value


@pytest.fixture(name='app_config', scope='session')
def _app_config() -> AppConfig:
    return AppConfig.load_yaml(DEFAULT_CONFIG_FILE)


@pytest.fixture(name='sciencebeam_parser')
def _sciencebeam_parser(app_config: AppConfig) -> ScienceBeamParser:
    return ScienceBeamParser.from_config(app_config)


@pytest.fixture(name='sciencebeam_parser_session')
def _sciencebeam_parser_session(
    sciencebeam_parser: ScienceBeamParser
) -> Iterable[ScienceBeamParserSession]:
    with sciencebeam_parser.get_new_session() as session:
        yield session


class TestSerializeXmlToFile:
    def test_should_indent_structural_elements_when_pretty_print(
        self, tmp_path: Path
    ):
        root = etree.fromstring(
            b'<TEI><teiHeader><title>My Title</title></teiHeader></TEI>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(root, filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert '\n' in content
        assert 'My Title' in content

    def test_should_preserve_mixed_content_text_when_pretty_print(
        self, tmp_path: Path
    ):
        root = etree.fromstring(
            b'<TEI><p>some text <ref>1</ref> more text</p></TEI>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(root, filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert 'some text ' in content
        assert ' more text' in content

    def test_should_not_indent_when_pretty_print_disabled(
        self, tmp_path: Path
    ):
        root = etree.fromstring(
            b'<TEI><teiHeader><title>My Title</title></teiHeader></TEI>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(root, filename, pretty_print=False)
        content = Path(filename).read_text(encoding='utf-8')
        assert '\n' not in content


class TestScienceBeamParser:
    class TestStartup:
        def test_should_not_preload_if_disabled(
            self,
            app_config: AppConfig,
            fulltext_models: MagicMock
        ):
            ScienceBeamParser.from_config(
                AppConfig({
                    **app_config.props,
                    'preload_on_startup': False
                })
            )
            fulltext_models.preload.assert_not_called()

        def test_should_preload_if_enabled(
            self,
            app_config: AppConfig,
            fulltext_models: MagicMock
        ):
            ScienceBeamParser.from_config(
                AppConfig({
                    **app_config.props,
                    'preload_on_startup': True
                })
            )
            fulltext_models.preload.assert_called()

    class TestGetLocalFileForResponseMediaType:
        def test_should_raise_for_unsupported_request_media_type(
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            request_temp_path: Path
        ):
            expected_pdf_path = request_temp_path / 'test.pdf'
            expected_output_path = request_temp_path / TEMP_ALTO_XML_FILENAME
            expected_output_path.write_bytes(XML_CONTENT_1)
            with pytest.raises(UnsupportedRequestMediaTypeScienceBeamParserError):
                (
                    sciencebeam_parser_session.get_source(
                        str(expected_pdf_path),
                        'media/unsupported'
                    ).get_local_file_for_response_media_type(
                        MediaTypes.ALTO_XML
                    )
                )

        def test_should_raise_for_unsupported_response_media_type(
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            request_temp_path: Path
        ):
            expected_pdf_path = request_temp_path / 'test.pdf'
            expected_output_path = request_temp_path / TEMP_ALTO_XML_FILENAME
            expected_output_path.write_bytes(XML_CONTENT_1)
            with pytest.raises(UnsupportedResponseMediaTypeScienceBeamParserError):
                (
                    sciencebeam_parser_session.get_source(
                        str(expected_pdf_path),
                        MediaTypes.PDF
                    ).get_local_file_for_response_media_type(
                        'media/unsupported'
                    )
                )

        def test_should_convert_docx_to_pdf(
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            doc_converter_wrapper_mock: MagicMock,
            request_temp_path: Path
        ):
            docx_path = request_temp_path / 'test.docx'
            doc_converter_wrapper_mock.convert.return_value = 'test.pdf'
            result_file = (
                sciencebeam_parser_session.get_source(
                    str(docx_path),
                    MediaTypes.DOCX
                ).get_local_file_for_response_media_type(
                    MediaTypes.PDF
                )
            )
            assert result_file == doc_converter_wrapper_mock.convert.return_value
            assert doc_converter_wrapper_mock.convert.call_args[0][0] == (
                str(docx_path)
            )

        def test_should_not_convert_pdf_to_alto_xml(
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            pdfalto_wrapper_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_pdf_path = request_temp_path / 'test.pdf'
            expected_output_path = request_temp_path / TEMP_ALTO_XML_FILENAME
            expected_output_path.write_bytes(XML_CONTENT_1)
            result_file = (
                sciencebeam_parser_session.get_source(
                    str(expected_pdf_path),
                    MediaTypes.PDF
                ).get_local_file_for_response_media_type(
                    MediaTypes.ALTO_XML
                )
            )
            pdfalto_wrapper_mock.convert_pdf_to_pdfalto_xml.assert_called_with(
                str(expected_pdf_path),
                str(expected_output_path),
                first_page=None,
                last_page=None
            )
            assert result_file == str(expected_output_path)

        def test_should_pretty_print_tei_xml_when_configured(
            self,
            app_config: AppConfig,
            get_tei_for_semantic_document_mock: MagicMock,
            request_temp_path: Path
        ):
            parser = ScienceBeamParser.from_config(
                AppConfig({**app_config.props, 'output': {'pretty_print_xml': True}})
            )
            expected_pdf_path = request_temp_path / 'test.pdf'
            (request_temp_path / TEMP_ALTO_XML_FILENAME).write_bytes(XML_CONTENT_1)
            get_tei_for_semantic_document_mock.return_value = TeiDocument(
                etree.fromstring(
                    b'<TEI><teiHeader><title>My Title</title></teiHeader></TEI>'
                )
            )
            with parser.get_new_session() as session:
                result_file = (
                    session.get_source(str(expected_pdf_path), MediaTypes.PDF)
                    .get_local_file_for_response_media_type(MediaTypes.TEI_XML)
                )
            content = Path(result_file).read_text(encoding='utf-8')
            assert '\n' in content
            assert 'My Title' in content

        def test_should_not_pretty_print_tei_xml_when_disabled(
            self,
            app_config: AppConfig,
            get_tei_for_semantic_document_mock: MagicMock,
            request_temp_path: Path
        ):
            parser = ScienceBeamParser.from_config(
                AppConfig({**app_config.props, 'output': {'pretty_print_xml': False}})
            )
            expected_pdf_path = request_temp_path / 'test.pdf'
            (request_temp_path / TEMP_ALTO_XML_FILENAME).write_bytes(XML_CONTENT_1)
            get_tei_for_semantic_document_mock.return_value = TeiDocument(
                etree.fromstring(
                    b'<TEI><teiHeader><title>My Title</title></teiHeader></TEI>'
                )
            )
            with parser.get_new_session() as session:
                result_file = (
                    session.get_source(str(expected_pdf_path), MediaTypes.PDF)
                    .get_local_file_for_response_media_type(MediaTypes.TEI_XML)
                )
            content = Path(result_file).read_text(encoding='utf-8')
            assert '\n' not in content

        def test_should_not_convert_pdf_to_tei_xml(
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            get_tei_for_semantic_document_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_pdf_path = request_temp_path / 'test.pdf'
            expected_output_path = request_temp_path / TEMP_ALTO_XML_FILENAME
            expected_output_path.write_bytes(XML_CONTENT_1)
            get_tei_for_semantic_document_mock.return_value = (
                TeiDocument(etree.fromstring(TEI_XML_CONTENT_1))
            )
            result_file = (
                sciencebeam_parser_session.get_source(
                    str(expected_pdf_path),
                    MediaTypes.PDF
                ).get_local_file_for_response_media_type(
                    MediaTypes.TEI_XML
                )
            )
            assert Path(result_file).read_bytes() == TEI_XML_CONTENT_1

        def test_should_not_convert_pdf_to_jats_xml(
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            get_tei_for_semantic_document_mock: MagicMock,
            xslt_transformer_wrapper_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_pdf_path = request_temp_path / 'test.pdf'
            expected_output_path = request_temp_path / TEMP_ALTO_XML_FILENAME
            expected_output_path.write_bytes(XML_CONTENT_1)
            get_tei_for_semantic_document_mock.return_value = (
                TeiDocument(etree.fromstring(TEI_XML_CONTENT_1))
            )
            xslt_transformer_wrapper_mock.return_value = (
                etree.fromstring(JATS_XML_CONTENT_1)
            )
            result_file = (
                sciencebeam_parser_session.get_source(
                    str(expected_pdf_path),
                    MediaTypes.PDF
                ).get_local_file_for_response_media_type(
                    MediaTypes.JATS_XML
                )
            )
            assert Path(result_file).read_bytes() == JATS_XML_CONTENT_1

        def test_should_not_convert_pdf_to_tei_zip(  # pylint: disable=too-many-locals
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            get_tei_for_semantic_document_mock: MagicMock,
            full_text_processor_class_mock: MagicMock,
            full_text_processor_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_pdf_path = request_temp_path / 'test.pdf'
            expected_output_path = request_temp_path / TEMP_ALTO_XML_FILENAME
            graphic_local_file_path = request_temp_path / 'image1.png'
            graphic_relative_path = graphic_local_file_path.name
            expected_output_path.write_bytes(XML_CONTENT_1)
            graphic_local_file_path.write_bytes(IMAGE_DATA_1)
            get_tei_for_semantic_document_mock.return_value = (
                TeiDocument(etree.fromstring(TEI_XML_CONTENT_1))
            )
            semantic_document = SemanticDocument()
            semantic_document.back_section.add_content(
                SemanticGraphic(
                    layout_graphic=LayoutGraphic(
                        local_file_path=str(graphic_local_file_path)
                    ),
                    relative_path=graphic_relative_path
                )
            )
            full_text_processor_mock.get_semantic_document_for_layout_document.return_value = (
                semantic_document
            )
            result_file = (
                sciencebeam_parser_session.get_source(
                    str(expected_pdf_path),
                    MediaTypes.PDF
                ).get_local_file_for_response_media_type(
                    MediaTypes.TEI_ZIP
                )
            )
            with ZipFile(result_file, 'r') as zip_file:
                tei_xml_data = zip_file.read('tei.xml')
                assert tei_xml_data == TEI_XML_CONTENT_1
                image_data = zip_file.read(graphic_relative_path)
                assert image_data == IMAGE_DATA_1
            full_text_processor_kwargs = full_text_processor_class_mock.call_args[1]
            full_text_processor_config = full_text_processor_kwargs['config']
            assert full_text_processor_config.extract_graphic_assets is True

        def test_should_not_convert_pdf_to_jats_zip(  # pylint: disable=too-many-locals
            self,
            sciencebeam_parser_session: ScienceBeamParserSession,
            get_tei_for_semantic_document_mock: MagicMock,
            full_text_processor_class_mock: MagicMock,
            full_text_processor_mock: MagicMock,
            xslt_transformer_wrapper_mock: MagicMock,
            request_temp_path: Path
        ):
            expected_pdf_path = request_temp_path / 'test.pdf'
            expected_output_path = request_temp_path / TEMP_ALTO_XML_FILENAME
            graphic_local_file_path = request_temp_path / 'image1.png'
            graphic_relative_path = graphic_local_file_path.name
            expected_output_path.write_bytes(XML_CONTENT_1)
            graphic_local_file_path.write_bytes(IMAGE_DATA_1)
            get_tei_for_semantic_document_mock.return_value = (
                TeiDocument(etree.fromstring(TEI_XML_CONTENT_1))
            )
            xslt_transformer_wrapper_mock.return_value = (
                etree.fromstring(JATS_XML_CONTENT_1)
            )
            semantic_document = SemanticDocument()
            semantic_document.back_section.add_content(
                SemanticGraphic(
                    layout_graphic=LayoutGraphic(
                        local_file_path=str(graphic_local_file_path)
                    ),
                    relative_path=graphic_relative_path
                )
            )
            full_text_processor_mock.get_semantic_document_for_layout_document.return_value = (
                semantic_document
            )
            result_file = (
                sciencebeam_parser_session.get_source(
                    str(expected_pdf_path),
                    MediaTypes.PDF
                ).get_local_file_for_response_media_type(
                    MediaTypes.JATS_ZIP
                )
            )
            with ZipFile(result_file, 'r') as zip_file:
                jats_xml_data = zip_file.read('jats.xml')
                assert jats_xml_data == JATS_XML_CONTENT_1
                image_data = zip_file.read(graphic_relative_path)
                assert image_data == IMAGE_DATA_1
            full_text_processor_kwargs = full_text_processor_class_mock.call_args[1]
            full_text_processor_config = full_text_processor_kwargs['config']
            assert full_text_processor_config.extract_graphic_assets is True
