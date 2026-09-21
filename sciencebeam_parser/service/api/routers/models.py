import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Iterable, Optional, Sequence, Type

from lxml import etree

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse

from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_crf_lines
from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    TagLabelFormats,
    iter_format_tag_result
)

from sciencebeam_parser.app.parser import (
    ScienceBeamParser,
    ScienceBeamParserSessionSource,
    normalize_layout_document
)
from sciencebeam_parser.app.profiles import ProfileBundle
from sciencebeam_parser.document.layout_document import LayoutDocument
from sciencebeam_parser.document.layout_noise_filter import (
    LayoutNoiseFilterConfig,
    get_noise_blocks,
    remove_noise_blocks,
)
from sciencebeam_parser.document.semantic_document import (
    SemanticMixedContentWrapper,
    SemanticRawAffiliationAddress,
    SemanticRawAuthors,
    SemanticRawFigure,
    SemanticRawReference,
    SemanticRawReferenceText,
    SemanticRawTable,
    SemanticReference,
    T_SemanticContentWrapper,
    iter_by_semantic_type_recursively
)
from sciencebeam_parser.external.pdfalto.parser import parse_alto_root
from sciencebeam_parser.external.pdfalto.wrapper import PdfAltoWrapper
from sciencebeam_parser.models.data import AppFeaturesContext, DocumentFeaturesContext
from sciencebeam_parser.models.model import Model
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig
from sciencebeam_parser.service.api.dependencies import (
    get_profile_bundle,
    get_sciencebeam_parser_session_source_dependency_factory
)
from sciencebeam_parser.utils.telemetry import span


LOGGER = logging.getLogger(__name__)


class ModelOutputFormats:
    RAW_DATA = 'raw_data'


DEFAULT_MODEL_OUTPUT_FORMAT = TagOutputFormats.JSON

VALID_MODEL_OUTPUT_FORMATS = [
    TagOutputFormats.JSON,
    ModelOutputFormats.RAW_DATA,
    TagOutputFormats.DATA,
    TagOutputFormats.XML
]


def get_noise_filter_config(
    fulltext_processor_config: FullTextProcessorConfig
) -> LayoutNoiseFilterConfig:
    return LayoutNoiseFilterConfig(
        enabled=fulltext_processor_config.noise_filter_enabled,
        repetition_fraction=fulltext_processor_config.noise_filter_repetition_fraction,
        preserve_first_page_head=fulltext_processor_config.noise_filter_preserve_first_page_head,
        preserve_first_page_foot=fulltext_processor_config.noise_filter_preserve_first_page_foot,
    )


class ModelResponseRouterFactory:
    """One route per sequence model, over whichever profile the request names.

    Holds the model's name rather than the model: which instance that names is
    a property of the request, and a router built around one profile's instances
    would answer every request with the deployment's models whatever was asked
    for.
    """

    def __init__(
        self,
        name: str,
        sequence_model_name: str,
        pdfalto_wrapper: PdfAltoWrapper,
        app_features_context: AppFeaturesContext,
        model_name: str = 'dummy'
    ):
        self.name = name
        self.sequence_model_name = sequence_model_name
        self.pdfalto_wrapper = pdfalto_wrapper
        self.app_features_context = app_features_context
        self.model_name = model_name

    def get_model(self, profile_bundle: ProfileBundle) -> Model:
        return profile_bundle.fulltext_models.get_sequence_model_by_name(
            self.sequence_model_name
        )

    def _register_feature_names_route(self, router: APIRouter) -> None:
        @router.get('/feature-names')
        def get_feature_names(
            profile_bundle: Annotated[ProfileBundle, Depends(get_profile_bundle)]
        ) -> dict:
            data_generator = self.get_model(profile_bundle).get_data_generator(
                DocumentFeaturesContext(
                    app_features_context=self.app_features_context
                )
            )
            return {'feature_names': data_generator.feature_names}

    def create_router(self) -> APIRouter:
        router = APIRouter()
        self._register_feature_names_route(router)

        @router.post('', description=self.model_name)
        def process_post(
            source: Annotated[
                ScienceBeamParserSessionSource,
                Depends(
                    get_sciencebeam_parser_session_source_dependency_factory()
                )
            ],
            profile_bundle: Annotated[ProfileBundle, Depends(get_profile_bundle)],
            output_format: Annotated[
                str,
                Query(json_schema_extra={
                    'enum': VALID_MODEL_OUTPUT_FORMATS
                })
            ] = DEFAULT_MODEL_OUTPUT_FORMAT,
        ) -> FileResponse:
            LOGGER.info('model_name: %r', self.model_name)
            return self.handle_post(
                source=source,
                output_format=output_format,
                profile_bundle=profile_bundle
            )
        return router

    def _apply_noise_filter(
        self,
        layout_document: LayoutDocument,
        profile_bundle: ProfileBundle
    ) -> LayoutDocument:
        noise_filter_config = get_noise_filter_config(
            profile_bundle.fulltext_processor_config
        )
        if not noise_filter_config.enabled:
            return layout_document
        noise_blocks = get_noise_blocks(layout_document, noise_filter_config)
        return remove_noise_blocks(layout_document, noise_blocks)

    def iter_filter_layout_document(
        self,
        layout_document: LayoutDocument,
        filter_params: dict,  # pylint: disable=unused-argument
        profile_bundle: ProfileBundle  # pylint: disable=unused-argument
    ) -> Iterable[LayoutDocument]:
        return [layout_document]

    def handle_post(
        self,
        source: ScienceBeamParserSessionSource,
        output_format: str,
        profile_bundle: ProfileBundle,
        filter_params: Optional[dict] = None
    ):
        # The same span name as the full pipeline, so one query finds every
        # document however the parser was entered. The model attribute is what
        # tells a single-model request apart from a whole document.
        with span('process_document', {
            'sciencebeam.document.name': source.source_name,
            'sciencebeam.document.source_media_type': source.source_media_type,
            'sciencebeam.model.name': self.model_name,
            'sciencebeam.model.output_format': output_format,
            'sciencebeam.profile.name': profile_bundle.name,
        }, tracer_name=__name__):
            return self._handle_post(source, output_format, profile_bundle, filter_params)

    def _handle_post(  # pylint: disable=too-many-locals
        self,
        source: ScienceBeamParserSessionSource,
        output_format: str,
        profile_bundle: ProfileBundle,
        filter_params: Optional[dict] = None
    ):
        model = self.get_model(profile_bundle)
        with TemporaryDirectory(suffix='-request') as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = source.source_path
            output_path = temp_path / 'test.lxml'
            first_page = source.document_request_parameters.first_page
            last_page = source.document_request_parameters.last_page
            assert output_format in VALID_MODEL_OUTPUT_FORMATS, \
                f'{output_format} not in {VALID_MODEL_OUTPUT_FORMATS}'
            self.pdfalto_wrapper.convert_pdf_to_pdfalto_xml(
                str(pdf_path),
                str(output_path),
                first_page=first_page,
                last_page=last_page
            )
            xml_content = output_path.read_bytes()
            root = etree.fromstring(xml_content)
            layout_document_iterable = self.iter_filter_layout_document(
                self._apply_noise_filter(
                    normalize_layout_document(parse_alto_root(root)),
                    profile_bundle
                ),
                filter_params=(filter_params or {}),
                profile_bundle=profile_bundle
            )
            data_generator = model.get_data_generator(
                DocumentFeaturesContext(
                    app_features_context=self.app_features_context
                )
            )
            data_lines = data_generator.iter_data_lines_for_layout_documents(
                layout_document_iterable
            )
            response_type = 'text/plain'
            if output_format == ModelOutputFormats.RAW_DATA:
                response_content = '\n'.join(data_lines) + '\n'
            else:
                texts, features = load_data_crf_lines(data_lines)
                LOGGER.info('texts length: %d', len(texts))
                if not len(texts):  # pylint: disable=len-as-condition
                    tag_result = []
                else:
                    tag_result = model.predict_labels(
                        texts=texts.tolist(), features=features.tolist(),
                        output_format=None
                    )
                LOGGER.debug('tag_result: %s', tag_result)
                formatted_tag_result_iterable = iter_format_tag_result(
                    tag_result,
                    output_format=output_format,
                    label_format=TagLabelFormats.GROBID,
                    expected_tag_result=None,
                    texts=texts,
                    features=features,
                    model_name=self.model_name
                )
                response_content = ''.join(formatted_tag_result_iterable)
                if output_format == TagOutputFormats.JSON:
                    response_type = 'application/json'
            LOGGER.debug('response_content: %r', response_content)
        return Response(
            content=response_content,
            media_type=response_type
        )


class SegmentedModelRouterFactory(ModelResponseRouterFactory):
    def __init__(
        self,
        *args,
        segmentation_labels: Sequence[str],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.segmentation_labels = segmentation_labels

    def create_router(self) -> APIRouter:
        router = APIRouter()
        self._register_feature_names_route(router)

        @router.post('', description=self.model_name)
        def process_post(
            source: Annotated[
                ScienceBeamParserSessionSource,
                Depends(
                    get_sciencebeam_parser_session_source_dependency_factory()
                )
            ],
            profile_bundle: Annotated[ProfileBundle, Depends(get_profile_bundle)],
            output_format: Annotated[
                str,
                Query(json_schema_extra={
                    'enum': VALID_MODEL_OUTPUT_FORMATS
                })
            ] = DEFAULT_MODEL_OUTPUT_FORMAT,
            no_use_segmentation: bool = False
        ) -> FileResponse:
            LOGGER.info('model_name: %r', self.model_name)
            return self.handle_post(
                source=source,
                output_format=output_format,
                profile_bundle=profile_bundle,
                filter_params={
                    'no_use_segmentation': no_use_segmentation
                }
            )
        return router

    def iter_filter_layout_document_by_segmentation_labels(
        self,
        layout_document: LayoutDocument,
        segmentation_labels: Sequence[str],
        profile_bundle: ProfileBundle
    ) -> Iterable[LayoutDocument]:
        segmentation_model = profile_bundle.fulltext_models.segmentation_model
        segmentation_label_result = (
            segmentation_model.get_label_layout_document_result(
                layout_document,
                app_features_context=self.app_features_context
            )
        )
        for segmentation_label in segmentation_labels:
            layout_document = segmentation_label_result.get_filtered_document_by_label(
                segmentation_label
            ).remove_empty_blocks()
            if not layout_document:
                LOGGER.info(
                    'empty document for segmentation label %r, available labels: %r',
                    segmentation_label,
                    segmentation_label_result.get_available_labels()
                )
                continue
            yield layout_document

    def filter_layout_document_by_segmentation_label(
        self,
        layout_document: LayoutDocument,
        segmentation_label: str,
        profile_bundle: ProfileBundle
    ) -> LayoutDocument:
        for filtered_layout_document in self.iter_filter_layout_document_by_segmentation_labels(
            layout_document,
            segmentation_labels=[segmentation_label],
            profile_bundle=profile_bundle
        ):
            return filtered_layout_document
        return LayoutDocument(pages=[])

    def iter_filter_layout_document(
        self,
        layout_document: LayoutDocument,
        filter_params: dict,
        profile_bundle: ProfileBundle
    ) -> Iterable[LayoutDocument]:
        if filter_params['no_use_segmentation']:
            return [layout_document]
        return self.iter_filter_layout_document_by_segmentation_labels(
            layout_document,
            segmentation_labels=self.segmentation_labels,
            profile_bundle=profile_bundle
        )


class NameHeaderModelRouterFactory(SegmentedModelRouterFactory):
    def iter_filter_layout_document(
        self,
        layout_document: LayoutDocument,
        filter_params: dict,
        profile_bundle: ProfileBundle
    ) -> Iterable[LayoutDocument]:
        header_model = profile_bundle.fulltext_models.header_model
        merge_raw_authors = profile_bundle.fulltext_processor_config.merge_raw_authors
        header_layout_document = self.filter_layout_document_by_segmentation_label(
            layout_document, '<header>', profile_bundle
        )
        labeled_layout_tokens = header_model.predict_labels_for_layout_document(
            header_layout_document,
            app_features_context=self.app_features_context
        )
        LOGGER.debug('labeled_layout_tokens: %r', labeled_layout_tokens)
        semantic_raw_authors_list = list(
            SemanticMixedContentWrapper(list(
                header_model.iter_semantic_content_for_labeled_layout_tokens(
                    labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawAuthors)
        )
        LOGGER.info('semantic_raw_authors_list count: %d', len(semantic_raw_authors_list))
        LOGGER.info('merge_raw_authors: %s', merge_raw_authors)
        if merge_raw_authors:
            return [
                LayoutDocument.for_blocks([
                    block
                    for semantic_raw_authors in semantic_raw_authors_list
                    for block in semantic_raw_authors.iter_blocks()
                ]).remove_empty_blocks()
            ]
        return [
            LayoutDocument.for_blocks(
                list(semantic_raw_authors.iter_blocks())
            ).remove_empty_blocks()
            for semantic_raw_authors in semantic_raw_authors_list
        ]


class AffiliationAddressModelRouterFactory(SegmentedModelRouterFactory):
    def iter_filter_layout_document(
        self,
        layout_document: LayoutDocument,
        filter_params: dict,
        profile_bundle: ProfileBundle
    ) -> Iterable[LayoutDocument]:
        header_model = profile_bundle.fulltext_models.header_model
        header_layout_document = self.filter_layout_document_by_segmentation_label(
            layout_document, '<header>', profile_bundle
        )
        labeled_layout_tokens = header_model.predict_labels_for_layout_document(
            header_layout_document,
            app_features_context=self.app_features_context
        )
        LOGGER.debug('labeled_layout_tokens: %r', labeled_layout_tokens)
        semantic_raw_aff_address_list = list(
            SemanticMixedContentWrapper(list(
                header_model.iter_semantic_content_for_labeled_layout_tokens(
                    labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawAffiliationAddress)
        )
        LOGGER.info('semantic_raw_aff_address_list count: %d', len(semantic_raw_aff_address_list))
        return [
            LayoutDocument.for_blocks(
                list(semantic_raw_aff_address.iter_blocks())
            ).remove_empty_blocks()
            for semantic_raw_aff_address in semantic_raw_aff_address_list
        ]


class CitationModelRouterFactory(SegmentedModelRouterFactory):
    def iter_filter_layout_document(
        self,
        layout_document: LayoutDocument,
        filter_params: dict,
        profile_bundle: ProfileBundle
    ) -> Iterable[LayoutDocument]:
        reference_segmenter_model = profile_bundle.fulltext_models.reference_segmenter_model
        citation_model = profile_bundle.fulltext_models.citation_model
        references_layout_document = self.filter_layout_document_by_segmentation_label(
            layout_document, '<references>', profile_bundle
        )
        labeled_layout_tokens = reference_segmenter_model.predict_labels_for_layout_document(
            references_layout_document,
            app_features_context=self.app_features_context
        )
        LOGGER.debug('labeled_layout_tokens: %r', labeled_layout_tokens)
        semantic_raw_references = list(
            SemanticMixedContentWrapper(list(
                reference_segmenter_model.iter_semantic_content_for_labeled_layout_tokens(
                    labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawReference)
        )
        LOGGER.info('semantic_raw_references count: %d', len(semantic_raw_references))
        docs = [
            LayoutDocument.for_blocks(
                [semantic_raw_reference.view_by_type(SemanticRawReferenceText).merged_block]
            ).remove_empty_blocks()
            for semantic_raw_reference in semantic_raw_references
        ]
        return citation_model.retokenize_layout_documents(docs)


class NameCitationModelRouterFactory(SegmentedModelRouterFactory):
    def iter_filter_layout_document(
        self,
        layout_document: LayoutDocument,
        filter_params: dict,
        profile_bundle: ProfileBundle
    ) -> Iterable[LayoutDocument]:
        reference_segmenter_model = profile_bundle.fulltext_models.reference_segmenter_model
        citation_model = profile_bundle.fulltext_models.citation_model
        references_layout_document = self.filter_layout_document_by_segmentation_label(
            layout_document, '<references>', profile_bundle
        )
        labeled_layout_tokens = reference_segmenter_model.predict_labels_for_layout_document(
            references_layout_document,
            app_features_context=self.app_features_context
        )
        LOGGER.debug('labeled_layout_tokens: %r', labeled_layout_tokens)
        semantic_raw_references = list(
            SemanticMixedContentWrapper(list(
                reference_segmenter_model.iter_semantic_content_for_labeled_layout_tokens(
                    labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawReference)
        )
        LOGGER.info('semantic_raw_references count: %d', len(semantic_raw_references))
        raw_reference_documents = [
            LayoutDocument.for_blocks(
                [semantic_raw_reference.view_by_type(SemanticRawReferenceText).merged_block]
            ).remove_empty_blocks()
            for semantic_raw_reference in semantic_raw_references
        ]
        citation_labeled_layout_tokens_list = (
            citation_model.predict_labels_for_layout_documents(
                raw_reference_documents,
                app_features_context=self.app_features_context
            )
        )
        raw_authors = [
            raw_author
            for citation_labeled_layout_tokens in citation_labeled_layout_tokens_list
            for ref in (
                citation_model.iter_semantic_content_for_labeled_layout_tokens(
                    citation_labeled_layout_tokens
                )
            )
            if isinstance(ref, SemanticReference)
            for raw_author in ref.iter_by_type(SemanticRawAuthors)
        ]
        return [
            LayoutDocument.for_blocks([raw_author.merged_block]).remove_empty_blocks()
            for raw_author in raw_authors
        ]


class FullTextChildModelRouterFactory(SegmentedModelRouterFactory):
    def __init__(
        self,
        *args,
        semantic_type: Type[T_SemanticContentWrapper],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.semantic_type = semantic_type

    def iter_filter_layout_document(
        self,
        layout_document: LayoutDocument,
        filter_params: dict,
        profile_bundle: ProfileBundle
    ) -> Iterable[LayoutDocument]:
        fulltext_model = profile_bundle.fulltext_models.fulltext_model
        fulltext_layout_documents = list(self.iter_filter_layout_document_by_segmentation_labels(
            layout_document, self.segmentation_labels, profile_bundle
        ))
        fulltext_labeled_layout_tokens_list = (
            fulltext_model.predict_labels_for_layout_documents(
                fulltext_layout_documents,
                app_features_context=self.app_features_context
            )
        )
        LOGGER.debug('fulltext_labeled_layout_tokens_list: %r', fulltext_labeled_layout_tokens_list)
        semanti_content_list = [
            semantic_content
            for fulltext_labeled_layout_tokens in fulltext_labeled_layout_tokens_list
            for semantic_content in iter_by_semantic_type_recursively(
                fulltext_model.iter_semantic_content_for_labeled_layout_tokens(
                    fulltext_labeled_layout_tokens
                ),
                self.semantic_type
            )
        ]
        LOGGER.debug('semanti_content_list: %s', semanti_content_list)
        return [
            LayoutDocument.for_blocks([semanti_content.merged_block]).remove_empty_blocks()
            for semanti_content in semanti_content_list
        ]


class FigureModelRouterFactory(FullTextChildModelRouterFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, semantic_type=SemanticRawFigure, **kwargs)


class TableModelRouterFactory(FullTextChildModelRouterFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, semantic_type=SemanticRawTable, **kwargs)


def create_models_router(
    sciencebeam_parser: ScienceBeamParser
) -> APIRouter:
    router = APIRouter(tags=['models'])

    pdfalto_wrapper = sciencebeam_parser.pdfalto_wrapper
    app_features_context = sciencebeam_parser.app_features_context

    router.include_router(
        ModelResponseRouterFactory(
            'Segmentation',
            sequence_model_name='segmentation',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context
        ).create_router(),
        prefix='/models/segmentation'
    )

    router.include_router(
        SegmentedModelRouterFactory(
            'Header',
            sequence_model_name='header',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=['<header>']
        ).create_router(),
        prefix='/models/header'
    )

    router.include_router(
        NameHeaderModelRouterFactory(
            'Name Header',
            sequence_model_name='name_header',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=['<header>']
        ).create_router(),
        prefix='/models/name-header'
    )

    router.include_router(
        AffiliationAddressModelRouterFactory(
            'Affiliation Address',
            sequence_model_name='affiliation_address',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=['<header>']
        ).create_router(),
        prefix='/models/affiliation-address'
    )

    fulltext_segmentation_labels = ['<body>', '<acknowledgement>', '<annex>']

    router.include_router(
        SegmentedModelRouterFactory(
            'FullText',
            sequence_model_name='fulltext',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=fulltext_segmentation_labels
        ).create_router(),
        prefix='/models/fulltext'
    )

    router.include_router(
        FigureModelRouterFactory(
            'Figure',
            sequence_model_name='figure',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=fulltext_segmentation_labels
        ).create_router(),
        prefix='/models/figure'
    )

    router.include_router(
        TableModelRouterFactory(
            'Table',
            sequence_model_name='table',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=fulltext_segmentation_labels
        ).create_router(),
        prefix='/models/table'
    )

    router.include_router(
        SegmentedModelRouterFactory(
            'Reference Segmenter',
            sequence_model_name='reference_segmenter',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=['<references>']
        ).create_router(),
        prefix='/models/reference-segmenter'
    )

    router.include_router(
        CitationModelRouterFactory(
            'Citation (Reference)',
            sequence_model_name='citation',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=['<references>']
        ).create_router(),
        prefix='/models/citation'
    )

    router.include_router(
        NameCitationModelRouterFactory(
            'Name Citaton',
            sequence_model_name='name_citation',
            pdfalto_wrapper=pdfalto_wrapper,
            app_features_context=app_features_context,
            segmentation_labels=['<references>']
        ).create_router(),
        prefix='/models/name-citation'
    )

    return router
