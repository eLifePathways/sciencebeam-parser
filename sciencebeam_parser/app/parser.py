import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import List, Optional, Set
from zipfile import ZipFile

from lxml import etree

from sciencebeam_trainer_delft.utils.download_manager import DownloadManager

from sciencebeam_parser.app.context import AppContext
from sciencebeam_parser.config.config import AppConfig, get_download_dir
from sciencebeam_parser.external.pdfalto.wrapper import PdfAltoWrapper, get_default_pdfalto_url
from sciencebeam_parser.external.pdfalto.parser import parse_alto_root
from sciencebeam_parser.external.wapiti.wrapper import LazyWapitiBinaryWrapper
from sciencebeam_parser.lookup.loader import load_lookup_from_config
from sciencebeam_parser.models.data import AppFeaturesContext
from sciencebeam_parser.document.layout_document import LayoutDocument
from sciencebeam_parser.document.semantic_document import (
    SemanticDocument,
    SemanticGraphic
)
from sciencebeam_parser.document.tei_document import get_tei_for_semantic_document
from sciencebeam_parser.processors.fulltext.models import FullTextModels
from sciencebeam_parser.resources.xslt import TEI_TO_JATS_XSLT_FILE
from sciencebeam_parser.transformers.doc_converter_wrapper import DocConverterWrapper
from sciencebeam_parser.transformers.xslt import XsltTransformerWrapper
from sciencebeam_parser.utils.download import download_with_zip_path_support
from sciencebeam_parser.utils.lazy import LazyLoaded
from sciencebeam_parser.utils.media_types import (
    MediaTypes
)
from sciencebeam_parser.utils.text import normalize_text
from sciencebeam_parser.utils.tokenizer import get_tokenized_tokens
from sciencebeam_parser.processors.fulltext.api import (
    FullTextProcessor,
    FullTextProcessorConfig,
    FullTextProcessorDocumentContext,
    load_models
)


LOGGER = logging.getLogger(__name__)


TEMP_ALTO_XML_FILENAME = 'temp.lxml'


DOC_TO_PDF_SUPPORTED_MEDIA_TYPES = {
    MediaTypes.DOCX,
    MediaTypes.DOTX,
    MediaTypes.DOC,
    MediaTypes.RTF
}


JATS_MEDIA_TYPES = {MediaTypes.JATS_XML, MediaTypes.JATS_ZIP}
ASSET_ZIP_MEDIA_TYPES = {MediaTypes.TEI_ZIP, MediaTypes.JATS_ZIP}


def normalize_and_tokenize_text(text: str) -> List[str]:
    return get_tokenized_tokens(
        normalize_text(text),
        keep_whitespace=True
    )


def normalize_layout_document(
    layout_document: LayoutDocument,
    **kwargs
) -> LayoutDocument:
    return (
        layout_document
        .retokenize(tokenize_fn=normalize_and_tokenize_text)
        .remove_empty_blocks(**kwargs)
    )


def load_app_features_context(
    config: AppConfig,
    download_manager: DownloadManager
):
    return AppFeaturesContext(
        country_lookup=load_lookup_from_config(
            config.get('lookup', {}).get('country'),
            download_manager=download_manager
        ),
        first_name_lookup=load_lookup_from_config(
            config.get('lookup', {}).get('first_name'),
            download_manager=download_manager
        ),
        last_name_lookup=load_lookup_from_config(
            config.get('lookup', {}).get('last_name'),
            download_manager=download_manager
        )
    )


def create_asset_zip_for_semantic_document(
    zip_filename: str,
    semantic_document: SemanticDocument,
    relative_xml_filename: str,
    local_xml_filename: str
):
    semantic_graphic_list = list(semantic_document.iter_by_type_recursively(
        SemanticGraphic
    ))
    LOGGER.debug('semantic_graphic_list: %r', semantic_graphic_list)
    with ZipFile(zip_filename, 'w') as zip_file:
        zip_file.write(
            local_xml_filename,
            relative_xml_filename
        )
        for semantic_graphic in semantic_graphic_list:
            assert semantic_graphic.relative_path, \
                "graphic relative_path missing, ensure extract_graphic_assets was enabled"
            layout_graphic = semantic_graphic.layout_graphic
            assert layout_graphic
            assert layout_graphic.local_file_path
            zip_file.write(
                layout_graphic.local_file_path,
                semantic_graphic.relative_path
            )
    LOGGER.debug('response_content (bytes): %d', Path(zip_filename).stat().st_size)


def get_xml_tree(xml_root: etree.ElementBase) -> etree._ElementTree:
    if isinstance(xml_root, etree._ElementTree):  # pylint: disable=protected-access
        # Note: _XSLTResultTree is extending _ElementTree
        return xml_root
    return etree.ElementTree(xml_root)


def serialize_xml_to_file(
    xml_root: etree.ElementBase,
    filename: str
):
    get_xml_tree(xml_root).write(
        filename,
        encoding='utf-8',
        pretty_print=False
    )


@dataclass
class DocumentRequestParameters:
    first_page: Optional[int] = None
    last_page: Optional[int] = None


class ScienceBeamParserError(RuntimeError):
    pass


class BadRequestScienceBeamParserError(ScienceBeamParserError):
    pass


class UnsupportedRequestMediaTypeScienceBeamParserError(BadRequestScienceBeamParserError):
    pass


class UnsupportedResponseMediaTypeScienceBeamParserError(BadRequestScienceBeamParserError):
    pass


class ScienceBeamBaseParser:
    def __init__(self, config: AppConfig):
        self.config = config
        self.download_manager = DownloadManager(
            download_dir=get_download_dir(config)
        )
        self.pdfalto_wrapper = PdfAltoWrapper(
            download_with_zip_path_support(
                self.download_manager,
                config['pdfalto'].get('path') or get_default_pdfalto_url()
            )
        )
        self.pdfalto_wrapper.ensure_executable()
        self.app_context = AppContext(
            app_config=config,
            download_manager=self.download_manager,
            lazy_wapiti_binary_wrapper=LazyWapitiBinaryWrapper(
                install_url=config.get('wapiti', {}).get('install_source'),
                download_manager=self.download_manager
            )
        )
        self.fulltext_processor_config = FullTextProcessorConfig.from_app_config(app_config=config)
        self.fulltext_models = load_models(
            config,
            app_context=self.app_context,
            fulltext_processor_config=self.fulltext_processor_config
        )
        if config.get('preload_on_startup'):
            self.fulltext_models.preload()
        self.app_features_context = load_app_features_context(
            config,
            download_manager=self.download_manager
        )
        tei_to_jats_config = config.get('xslt', {}).get('tei_to_jats', {})
        self.tei_to_jats_xslt_transformer = XsltTransformerWrapper.from_template_file(
            TEI_TO_JATS_XSLT_FILE,
            xslt_template_parameters=tei_to_jats_config.get('parameters', {})
        )
        self.doc_to_pdf_enabled = config.get('doc_to_pdf', {}).get('enabled', True)
        self.doc_to_pdf_convert_parameters = config.get('doc_to_pdf', {}).get('convert', {})
        self.doc_converter_wrapper = DocConverterWrapper(
            **config.get('doc_to_pdf', {}).get('listener', {})
        )


class ScienceBeamParserBaseSession:
    def __init__(
        self,
        parser: 'ScienceBeamParser',
        temp_dir: Optional[str] = None,
        fulltext_processor_config: Optional[FullTextProcessorConfig] = None,
        document_request_parameters: Optional[DocumentRequestParameters] = None
    ):
        self.parser = parser
        self.exit_stack = ExitStack()
        self._temp_dir: Optional[str] = temp_dir
        if fulltext_processor_config is None:
            fulltext_processor_config = parser.fulltext_processor_config
        self.fulltext_processor_config = fulltext_processor_config
        if document_request_parameters is None:
            document_request_parameters = DocumentRequestParameters()
        self.document_request_parameters = document_request_parameters

    def __enter__(self) -> 'ScienceBeamParserBaseSession':
        return self

    def close(self):
        self.exit_stack.close()

    def __exit__(self, exc, value, tb):
        self.close()

    @property
    def temp_dir(self) -> str:
        if not self._temp_dir:
            temp_dir_context = TemporaryDirectory(  # pylint: disable=consider-using-with
                suffix='-sb-parser'
            )
            self.exit_stack.push(temp_dir_context)
            self._temp_dir = temp_dir_context.__enter__()  # pylint: disable=unnecessary-dunder-call
        return self._temp_dir

    @property
    def temp_path(self) -> Path:
        return Path(self.temp_dir)


class _ScienceBeamParserSessionDerivative:
    def __init__(
        self,
        session: 'ScienceBeamParserBaseSession'
    ):
        self.session = session

    @property
    def parser(self) -> ScienceBeamBaseParser:
        return self.session.parser

    @property
    def temp_dir(self) -> str:
        return self.session.temp_dir

    @property
    def temp_path(self) -> Path:
        return self.session.temp_path


class ScienceBeamParserSessionParsedSemanticDocument(_ScienceBeamParserSessionDerivative):
    def __init__(
        self,
        session: 'ScienceBeamParserBaseSession',
        semantic_document: SemanticDocument
    ):
        super().__init__(session=session)
        self.semantic_document = semantic_document

    @property
    def tei_to_jats_xslt_transformer(self) -> XsltTransformerWrapper:
        return self.parser.tei_to_jats_xslt_transformer

    def _get_tei_to_jats_xml_root(self, xml_root: etree.ElementBase) -> etree.ElementBase:
        start = monotonic()
        xml_root = self.tei_to_jats_xslt_transformer(xml_root)
        end = monotonic()
        LOGGER.info('tei to jats, took=%.3fs', end - start)
        return xml_root

    def _serialize_xml_to_file(
        self,
        xml_root: etree.ElementBase,
        filename: str
    ) -> str:
        start = monotonic()
        serialize_xml_to_file(xml_root, filename=filename)
        end = monotonic()
        LOGGER.info('serializing xml, took=%.3fs', end - start)
        return filename

    def get_supported_response_media_type(self) -> Set[str]:
        return {
            MediaTypes.TEI_XML,
            MediaTypes.TEI_ZIP,
            MediaTypes.JATS_XML,
            MediaTypes.JATS_ZIP
        }

    def get_local_file_for_response_media_type(
        self,
        response_media_type: str
    ) -> str:
        if response_media_type not in self.get_supported_response_media_type():
            raise UnsupportedResponseMediaTypeScienceBeamParserError()
        tei_document = get_tei_for_semantic_document(
            self.semantic_document
        )
        xml_root = tei_document.root
        relative_xml_filename = 'tei.xml'
        if response_media_type in JATS_MEDIA_TYPES:
            xml_root = self._get_tei_to_jats_xml_root(xml_root)
            relative_xml_filename = 'jats.xml'
        local_xml_filename = os.path.join(self.temp_dir, relative_xml_filename)
        self._serialize_xml_to_file(xml_root, local_xml_filename)
        LOGGER.debug('local_xml_filename: %r', local_xml_filename)
        if response_media_type in ASSET_ZIP_MEDIA_TYPES:
            zip_filename = os.path.join(self.temp_dir, 'results.zip')
            create_asset_zip_for_semantic_document(
                zip_filename,
                semantic_document=self.semantic_document,
                local_xml_filename=local_xml_filename,
                relative_xml_filename=relative_xml_filename
            )
            return zip_filename
        return local_xml_filename


class ScienceBeamParserSessionParsedLayoutDocument(_ScienceBeamParserSessionDerivative):
    def __init__(
        self,
        session: 'ScienceBeamParserBaseSession',
        layout_document: LayoutDocument,
        pdf_path: str
    ):
        super().__init__(session=session)
        self.layout_document = layout_document
        self.pdf_path = pdf_path

    @property
    def fulltext_models(self) -> FullTextModels:
        return self.parser.fulltext_models

    @property
    def app_features_context(self) -> AppFeaturesContext:
        return self.parser.app_features_context

    def _get_semantic_document(
        self,
        fulltext_processor: FullTextProcessor
    ) -> SemanticDocument:
        context = FullTextProcessorDocumentContext(
            pdf_path=self.pdf_path,
            temp_dir=self.temp_dir
        )
        semantic_document = (
            fulltext_processor
            .get_semantic_document_for_layout_document(
                self.layout_document,
                context=context
            )
        )
        return semantic_document

    def get_parsed_semantic_document(
        self,
        fulltext_processor_config: Optional[FullTextProcessorConfig] = None
    ) -> ScienceBeamParserSessionParsedSemanticDocument:
        if fulltext_processor_config is None:
            fulltext_processor_config = self.session.fulltext_processor_config
        fulltext_processor = FullTextProcessor(
            self.fulltext_models,
            app_features_context=self.app_features_context,
            config=fulltext_processor_config
        )
        return ScienceBeamParserSessionParsedSemanticDocument(
            self.session,
            self._get_semantic_document(fulltext_processor)
        )

    def get_local_file_for_response_media_type(
        self,
        response_media_type: str
    ) -> str:
        if response_media_type == MediaTypes.PDF:
            return self.pdf_path
        fulltext_processor_config = self.session.fulltext_processor_config
        if response_media_type in ASSET_ZIP_MEDIA_TYPES:
            fulltext_processor_config = (
                fulltext_processor_config
                ._replace(
                    extract_graphic_assets=True,
                    extract_graphic_bounding_boxes=True
                )
            )
            assert fulltext_processor_config.extract_graphic_assets, \
                "extract_graphic_assets required for asset zip"
        return (
            self.get_parsed_semantic_document(
                fulltext_processor_config
            ).get_local_file_for_response_media_type(
                response_media_type
            )
        )


class ScienceBeamParserSessionSource(_ScienceBeamParserSessionDerivative):
    def __init__(
        self,
        session: 'ScienceBeamParserBaseSession',
        source_path: str,
        source_media_type: str
    ):
        super().__init__(session=session)
        self.source_path = source_path
        self.source_media_type = source_media_type
        self.lazy_pdf_path = LazyLoaded[str](self._get_or_convert_to_pdf_path)
        self.lazy_alto_xml_path = LazyLoaded[str](self._parse_to_alto_xml)
        self.lazy_parsed_layout_document = LazyLoaded[
            ScienceBeamParserSessionParsedLayoutDocument
        ](self._parse_to_parsed_layout_document)

    @property
    def parser(self) -> ScienceBeamBaseParser:
        return self.session.parser

    @property
    def doc_to_pdf_enabled(self) -> bool:
        return self.parser.doc_to_pdf_enabled

    @property
    def doc_converter_wrapper(self) -> DocConverterWrapper:
        return self.parser.doc_converter_wrapper

    @property
    def doc_to_pdf_convert_parameters(self) -> dict:
        return self.parser.doc_to_pdf_convert_parameters

    @property
    def pdfalto_wrapper(self) -> PdfAltoWrapper:
        return self.parser.pdfalto_wrapper

    @property
    def document_request_parameters(self) -> DocumentRequestParameters:
        return self.session.document_request_parameters

    def _get_or_convert_to_pdf_path(
        self
    ) -> str:
        LOGGER.info(
            'media_type=%r (filename=%r)',
            self.source_media_type,
            self.source_path
        )
        if self.source_media_type in DOC_TO_PDF_SUPPORTED_MEDIA_TYPES:
            if not self.doc_to_pdf_enabled:
                LOGGER.info('doc to pdf not enabled')
                raise UnsupportedRequestMediaTypeScienceBeamParserError(
                    'doc to pdf not enabled'
                )
            target_temp_file = self.doc_converter_wrapper.convert(
                self.source_path,
                **self.doc_to_pdf_convert_parameters
            )
            return target_temp_file
        if self.source_media_type != MediaTypes.PDF:
            raise UnsupportedRequestMediaTypeScienceBeamParserError(
                'unsupported media type: %r' % self.source_media_type
            )
        return self.source_path

    def _parse_to_alto_xml(self) -> str:
        output_path = os.path.join(self.temp_dir, TEMP_ALTO_XML_FILENAME)
        self.pdfalto_wrapper.convert_pdf_to_pdfalto_xml(
            str(self.lazy_pdf_path.get()),
            str(output_path),
            first_page=self.document_request_parameters.first_page,
            last_page=self.document_request_parameters.last_page
        )
        return output_path

    def _parse_to_parsed_layout_document(
        self
    ) -> ScienceBeamParserSessionParsedLayoutDocument:
        pdf_path = self.lazy_pdf_path.get()
        root = etree.parse(self.lazy_alto_xml_path.get())
        layout_document = normalize_layout_document(
            parse_alto_root(root),
            preserve_empty_pages=True
        )
        return ScienceBeamParserSessionParsedLayoutDocument(
            self.session,
            layout_document=layout_document,
            pdf_path=pdf_path
        )

    def get_parsed_layout_document(self) -> ScienceBeamParserSessionParsedLayoutDocument:
        return self.lazy_parsed_layout_document.get()

    def get_layout_document(self) -> LayoutDocument:
        return self.get_parsed_layout_document().layout_document

    def get_local_file_for_response_media_type(
        self,
        response_media_type: str
    ) -> str:
        if response_media_type == MediaTypes.PDF:
            return self.lazy_pdf_path.get()
        if response_media_type == MediaTypes.ALTO_XML:
            return self.lazy_alto_xml_path.get()
        return self.get_parsed_layout_document().get_local_file_for_response_media_type(
            response_media_type
        )


class ScienceBeamParserSession(ScienceBeamParserBaseSession):
    def __enter__(self) -> 'ScienceBeamParserSession':
        super().__enter__()
        return self

    def get_source(
        self,
        source_path: str,
        source_media_type: str
    ) -> ScienceBeamParserSessionSource:
        return ScienceBeamParserSessionSource(
            self,
            source_path=source_path,
            source_media_type=source_media_type
        )


class ScienceBeamParser(ScienceBeamBaseParser):
    @staticmethod
    def from_config(config: AppConfig) -> 'ScienceBeamParser':
        return ScienceBeamParser(config)

    def get_new_session(self, **kwargs) -> ScienceBeamParserSession:
        return ScienceBeamParserSession(self, **kwargs)
