import logging
import multiprocessing
from typing import (
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union
)

from sciencebeam_parser.models.data import AppFeaturesContext, DEFAULT_APP_FEATURES_CONTEXT
from sciencebeam_parser.models.model import LayoutDocumentLabelResult, Model
from sciencebeam_parser.cv_models.cv_model import ComputerVisionModel
from sciencebeam_parser.processors.fulltext.models import FullTextModels
from sciencebeam_parser.utils.misc import iter_ids

from sciencebeam_parser.document.semantic_document import (
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticCitation,
    SemanticContentWrapper,
    SemanticDocument,
    SemanticEditor,
    SemanticFigure,
    SemanticFigureCitation,
    SemanticGraphic,
    SemanticInvalidReference,
    SemanticLabel,
    SemanticMixedContentWrapper,
    SemanticMixedNote,
    SemanticNote,
    SemanticRawAffiliationAddress,
    SemanticRawAuthors,
    SemanticRawEditors,
    SemanticRawFigure,
    SemanticRawReference,
    SemanticRawReferenceText,
    SemanticRawTable,
    SemanticReference,
    SemanticReferenceCitation,
    SemanticReferenceList,
    SemanticSection,
    SemanticSectionTypes,
    SemanticTable,
    SemanticTableCitation,
    T_SemanticContentWrapper,
    T_SemanticName,
    T_SemanticRawNameList
)
from sciencebeam_parser.document.tei_document import TeiDocument, get_tei_for_semantic_document
from sciencebeam_parser.document.layout_document import LayoutDocument
from sciencebeam_parser.document.layout_noise_filter import (
    LayoutNoiseFilterConfig,
    get_noise_blocks,
    remove_noise_blocks,
)
from sciencebeam_parser.models.segmentation.model import SegmentationModel
from sciencebeam_parser.models.header.model import HeaderModel
from sciencebeam_parser.models.name.model import NameModel
from sciencebeam_parser.models.affiliation_address.model import AffiliationAddressModel
from sciencebeam_parser.models.fulltext.model import FullTextModel
from sciencebeam_parser.models.figure.model import FigureModel
from sciencebeam_parser.models.table.model import TableModel
from sciencebeam_parser.models.reference_segmenter.model import ReferenceSegmenterModel
from sciencebeam_parser.models.citation.model import CitationModel
from sciencebeam_parser.processors.ref_matching import (
    ChainedContentIdMatcher,
    ContentIdMatcher,
    PartialContentIdMatcher,
    SimpleContentIdMatcher
)
from sciencebeam_parser.processors.document_page_image import (
    iter_pdf_document_page_images
)
from sciencebeam_parser.processors.graphic_matching import (
    BoundingBoxDistanceGraphicMatcher,
    ChainedGraphicMatcher,
    GraphicMatcher,
    GraphicRelatedBlockTextGraphicMatcher,
    OpticalCharacterRecognitionGraphicMatcher
)
from sciencebeam_parser.processors.graphic_provider import (
    DocumentGraphicProvider,
    SimpleDocumentGraphicProvider,
    get_graphic_matching_candidate_page_numbers_for_semantic_content_list,
    get_layout_document_with_graphics_replaced_by_graphics,
    get_layout_document_with_text_and_graphics_replaced_by_graphics,
    get_page_numbers_with_mostly_bitmap_graphics,
    get_page_numbers_with_uncommon_page_dimension
)
from sciencebeam_parser.processors.fulltext.config import (
    FullTextProcessorConfig
)


LOGGER = logging.getLogger(__name__)


def _is_initial(s: str) -> bool:
    return len(s.replace('.', '').replace(' ', '')) <= 2


def _authors_match(
    a: SemanticAuthor,
    b: SemanticAuthor,
    use_initial_fallback: bool
) -> bool:
    if not a.surname_text or a.surname_text.lower() != b.surname_text.lower():
        return False
    gn_a = a.given_name_text
    gn_b = b.given_name_text
    if not gn_a or not gn_b:
        return False
    if not _is_initial(gn_a) and not _is_initial(gn_b):
        return gn_a.lower() == gn_b.lower()
    if not use_initial_fallback:
        return False
    return gn_a[0].lower() == gn_b[0].lower()


def _iter_deduplicated_authors(
    raw_authors: List[SemanticRawAuthors],
    parsed_per_block: List[List[SemanticContentWrapper]],
    use_initial_fallback: bool
) -> Iterable[SemanticContentWrapper]:
    seen: List[SemanticAuthor] = []
    for raw_author, parsed in zip(raw_authors, parsed_per_block):
        parsed_authors = [c for c in parsed if isinstance(c, SemanticAuthor)]
        if seen and parsed_authors and all(
            any(_authors_match(a, s, use_initial_fallback) for s in seen)
            for a in parsed_authors
        ):
            yield SemanticNote(layout_block=raw_author.merged_block, note_type='raw_authors')
        else:
            for item in parsed:
                yield item
                if isinstance(item, SemanticAuthor):
                    seen.append(item)


class FullTextProcessorDocumentContext(NamedTuple):
    pdf_path: Optional[str] = None
    temp_dir: Optional[str] = None


def get_cv_document_graphic_provider(
    cv_model: ComputerVisionModel,
    context: FullTextProcessorDocumentContext,
    page_numbers: Optional[Sequence[int]],
    cv_render_dpi: float
) -> DocumentGraphicProvider:
    from sciencebeam_parser.processors.cv_graphic_provider import (  # noqa pylint: disable=import-outside-toplevel
        ComputerVisionDocumentGraphicProvider
    )
    assert context.pdf_path
    assert context.temp_dir
    return ComputerVisionDocumentGraphicProvider(
        cv_model,
        iter_pdf_document_page_images(
            pdf_path=context.pdf_path,
            output_dir=context.temp_dir,
            page_numbers=page_numbers,
            dpi=cv_render_dpi,
            thread_count=multiprocessing.cpu_count()
        ),
        temp_dir=context.temp_dir
    )


class FullTextProcessor:
    def __init__(
        self,
        fulltext_models: FullTextModels,
        config: Optional[FullTextProcessorConfig] = None,
        app_features_context: AppFeaturesContext = DEFAULT_APP_FEATURES_CONTEXT
    ) -> None:
        self.fulltext_models = fulltext_models
        self.app_features_context = app_features_context
        if not config:
            config = FullTextProcessorConfig()
        self.config = config

    @property
    def segmentation_model(self) -> SegmentationModel:
        return self.fulltext_models.segmentation_model

    @property
    def header_model(self) -> HeaderModel:
        return self.fulltext_models.header_model

    @property
    def affiliation_address_model(self) -> AffiliationAddressModel:
        return self.fulltext_models.affiliation_address_model

    @property
    def name_header_model(self) -> NameModel:
        return self.fulltext_models.name_header_model

    @property
    def name_citation_model(self) -> NameModel:
        return self.fulltext_models.name_citation_model

    @property
    def fulltext_model(self) -> FullTextModel:
        return self.fulltext_models.fulltext_model

    @property
    def figure_model(self) -> FigureModel:
        return self.fulltext_models.figure_model

    @property
    def table_model(self) -> TableModel:
        return self.fulltext_models.table_model

    @property
    def reference_segmenter_model(self) -> ReferenceSegmenterModel:
        return self.fulltext_models.reference_segmenter_model

    @property
    def citation_model(self) -> CitationModel:
        return self.fulltext_models.citation_model

    def get_semantic_document_for_layout_document(
        self,
        layout_document: LayoutDocument,
        context: Optional[FullTextProcessorDocumentContext] = None
    ) -> SemanticDocument:
        if context is None:
            context = FullTextProcessorDocumentContext()
        layout_document = self._preprocess_layout_graphics(
            layout_document,
            context=context
        )
        noise_blocks = get_noise_blocks(
            layout_document,
            LayoutNoiseFilterConfig(
                enabled=self.config.noise_filter_enabled,
                repetition_fraction=self.config.noise_filter_repetition_fraction,
                preserve_first_page_head=self.config.noise_filter_preserve_first_page_head,
                preserve_first_page_foot=self.config.noise_filter_preserve_first_page_foot,
            )
        )
        segmentation_input = remove_noise_blocks(layout_document, noise_blocks)
        segmentation_label_result = self.segmentation_model.get_label_layout_document_result(
            segmentation_input,
            app_features_context=self.app_features_context
        )
        header_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<header>'
        ).remove_empty_blocks()
        document = SemanticDocument()
        if self.config.extract_front:
            self._process_header_layout_document(
                header_layout_document=header_layout_document,
                semantic_document=document
            )

        if self.config.extract_body_sections:
            self._update_semantic_section_using_segmentation_result_and_fulltext_model(
                document.body_section,
                segmentation_label_result,
                '<body>',
                SemanticSectionTypes.OTHER
            )
        if self.config.extract_acknowledgements:
            self._update_semantic_section_using_segmentation_result_and_fulltext_model(
                document.back_section,
                segmentation_label_result,
                '<acknowledgement>',
                SemanticSectionTypes.ACKNOWLEDGEMENT
            )
        if self.config.extract_back_sections:
            self._update_semantic_section_using_segmentation_result_and_fulltext_model(
                document.back_section,
                segmentation_label_result,
                '<annex>',
                SemanticSectionTypes.OTHER
            )
        if self.config.extract_references:
            self._extract_raw_references_from_segmentation(
                semantic_document=document,
                segmentation_label_result=segmentation_label_result
            )
        if self.config.extract_citation_fields:
            self._extract_reference_fields_from_raw_references(
                semantic_document=document
            )
            if self.config.extract_citation_authors or self.config.extract_citation_editors:
                self._extract_reference_name_lists_from_raw_references(
                    semantic_document=document
                )
            references = list(document.iter_by_type_recursively(SemanticReference))
            ref_citations = list(document.iter_by_type_recursively(SemanticReferenceCitation))
            self._assign_content_ids(references, iter(iter_ids('b')))
            self._assign_target_content_ids(ref_citations, ChainedContentIdMatcher([
                SimpleContentIdMatcher(
                    self._get_semantic_content_text_by_content_id(references, SemanticLabel)
                ),
                PartialContentIdMatcher(
                    self._get_semantic_content_text_by_content_id(
                        references, SemanticRawReferenceText
                    )
                )
            ]))
        if self.config.extract_figure_fields:
            self._extract_figure_fields_from_raw_figures(semantic_document=document)
            figures = list(document.iter_by_type_recursively(SemanticFigure))
            figure_citations = list(document.iter_by_type_recursively(SemanticFigureCitation))
            self._assign_content_ids(figures, iter(iter_ids('fig_')))
            self._assign_target_content_ids(figure_citations, SimpleContentIdMatcher(
                self._get_semantic_content_text_by_content_id(figures, SemanticLabel)
            ))
        if self.config.extract_table_fields:
            self._extract_table_fields_from_raw_tables(semantic_document=document)
            tables = list(document.iter_by_type_recursively(SemanticTable))
            table_citations = list(document.iter_by_type_recursively(SemanticTableCitation))
            self._assign_content_ids(tables, iter(iter_ids('tab_')))
            self._assign_target_content_ids(table_citations, SimpleContentIdMatcher(
                self._get_semantic_content_text_by_content_id(tables, SemanticLabel)
            ))
        for nb in noise_blocks:
            document.body_section.add_content(SemanticNote(
                layout_block=nb.block,
                note_type=nb.note_type
            ))
        if self.config.extract_graphic_bounding_boxes:
            self._process_graphics(
                document=document,
                layout_document=layout_document,
                context=context
            )
        return document

    def _process_header_layout_document(
        self,
        header_layout_document: LayoutDocument,
        semantic_document: SemanticDocument
    ):
        LOGGER.debug('header_layout_document: %s', header_layout_document)
        if not header_layout_document.pages:
            return
        labeled_layout_tokens = self.header_model.predict_labels_for_layout_document(
            header_layout_document,
            app_features_context=self.app_features_context
        )
        LOGGER.debug('labeled_layout_tokens: %r', labeled_layout_tokens)
        entity_blocks = self.header_model.iter_entity_layout_blocks_for_labeled_layout_tokens(
            labeled_layout_tokens
        )
        self.header_model.update_semantic_document_with_entity_blocks(
            semantic_document, entity_blocks
        )
        if self.config.extract_authors:
            self._process_raw_authors(semantic_document.front)
        if self.config.extract_affiliations:
            self._process_raw_affiliations(semantic_document)

    def _preprocess_layout_graphics(
        self,
        layout_document: LayoutDocument,
        context: FullTextProcessorDocumentContext
    ) -> LayoutDocument:
        if not self.config.use_cv_model:
            return layout_document
        candidate_page_numbers = sorted(
            set(get_page_numbers_with_uncommon_page_dimension(layout_document))
            - set(get_page_numbers_with_mostly_bitmap_graphics(layout_document))
        )
        LOGGER.debug('candidate_page_numbers: %r', candidate_page_numbers)
        if not candidate_page_numbers:
            return layout_document
        document_graphic_provider = self._get_document_graphic_provider(
            context=context,
            page_numbers=candidate_page_numbers
        )
        semantic_graphics = list(
            document_graphic_provider.iter_semantic_graphic_for_layout_document(
                layout_document,
                extract_graphic_assets=self.config.extract_graphic_assets
            )
        )
        if not semantic_graphics:
            LOGGER.info('no semantic graphics found on pages %r', candidate_page_numbers)
            return layout_document
        if not self.config.replace_text_by_cv_graphic:
            return get_layout_document_with_graphics_replaced_by_graphics(
                layout_document,
                semantic_graphics
            )
        return get_layout_document_with_text_and_graphics_replaced_by_graphics(
            layout_document,
            semantic_graphics
        )

    def _process_graphics(
        self,
        document: SemanticDocument,
        layout_document: LayoutDocument,
        context: FullTextProcessorDocumentContext
    ):
        unmatched_graphics_container = SemanticMixedNote(note_type='unmatched_graphics')
        candidate_semantic_content_list = list(
            document.iter_by_types_recursively((SemanticFigure, SemanticTable,))
        )
        self._match_graphic_elements(
            semantic_graphic_list=list(
                self._get_document_graphic_provider(
                    context=context,
                    page_numbers=(
                        get_graphic_matching_candidate_page_numbers_for_semantic_content_list(
                            candidate_semantic_content_list,
                            layout_document=layout_document
                        )
                    )
                ).iter_semantic_graphic_for_layout_document(
                    layout_document,
                    extract_graphic_assets=self.config.extract_graphic_assets
                )
            ),
            candidate_semantic_content_list=candidate_semantic_content_list,
            unmatched_graphics_container=unmatched_graphics_container
        )
        if not unmatched_graphics_container.is_empty():
            LOGGER.debug('unmatched_graphics_container: %r', unmatched_graphics_container)
            document.back_section.add_content(unmatched_graphics_container)
        else:
            LOGGER.debug('no unmatched graphics')

    def _get_document_graphic_provider(
        self,
        context: FullTextProcessorDocumentContext,
        page_numbers: Optional[Sequence[int]]
    ) -> DocumentGraphicProvider:
        if self.config.use_cv_model:
            assert self.fulltext_models.cv_model is not None
            return get_cv_document_graphic_provider(
                cv_model=self.fulltext_models.cv_model,
                context=context,
                page_numbers=page_numbers,
                cv_render_dpi=self.config.cv_render_dpi
            )
        return SimpleDocumentGraphicProvider()

    def _match_graphic_elements(
        self,
        semantic_graphic_list: Sequence[SemanticGraphic],
        candidate_semantic_content_list: Sequence[SemanticContentWrapper],
        unmatched_graphics_container: SemanticMixedContentWrapper
    ):
        _graphic_matchers: List[GraphicMatcher] = [
            BoundingBoxDistanceGraphicMatcher(
                max_distance=self.config.max_graphic_distance
            ),
            GraphicRelatedBlockTextGraphicMatcher()
        ]
        if self.config.use_ocr_model:
            assert self.fulltext_models.ocr_model
            _graphic_matchers.append(
                OpticalCharacterRecognitionGraphicMatcher(
                    ocr_model=self.fulltext_models.ocr_model
                )
            )
        graphic_matcher = ChainedGraphicMatcher(_graphic_matchers)
        graphic_match_result = graphic_matcher.get_graphic_matches(
            semantic_graphic_list=semantic_graphic_list,
            candidate_semantic_content_list=candidate_semantic_content_list
        )
        for graphic_match in graphic_match_result.graphic_matches:
            if isinstance(graphic_match.candidate_semantic_content,  SemanticMixedContentWrapper):
                graphic_match.candidate_semantic_content.add_content(
                    graphic_match.semantic_graphic
                )
        LOGGER.info('unmatched_graphics: %r', graphic_match_result.unmatched_graphics)
        for unmatched_graphic in graphic_match_result.unmatched_graphics:
            unmatched_graphics_container.add_content(unmatched_graphic)

    def _assign_content_ids(
        self,
        semantic_content_iterable: Iterable[SemanticMixedContentWrapper],
        content_id_iterator: Iterator[str]
    ):
        for semantic_content in semantic_content_iterable:
            semantic_content.content_id = next(content_id_iterator)

    def _get_semantic_content_text_by_content_id(
        self,
        semantic_content_iterable: Iterable[SemanticMixedContentWrapper],
        type_: Type[SemanticContentWrapper]
    ) -> Mapping[str, str]:
        d = {}
        for semantic_content in semantic_content_iterable:
            if not semantic_content.content_id:
                continue
            text = semantic_content.get_text_by_type(type_)
            if not text:
                continue
            d[semantic_content.content_id] = text
        return d

    def _assign_target_content_ids(
        self,
        semantic_content_iterable: Iterable[SemanticCitation],
        content_id_matcher: ContentIdMatcher
    ):
        for citation in semantic_content_iterable:
            content_id = content_id_matcher.get_id_by_text(citation.get_text())
            if content_id:
                citation.target_content_id = content_id

    def _process_raw_authors(self, semantic_parent: SemanticMixedContentWrapper):
        result_content: List[SemanticContentWrapper] = []
        raw_authors: List[SemanticRawAuthors] = []
        for semantic_content in semantic_parent:
            if isinstance(semantic_content, SemanticRawAuthors):
                raw_authors.append(semantic_content)
                continue
            result_content.append(semantic_content)
        if raw_authors:
            if self.config.merge_raw_authors:
                raw_authors_layout_documents = [
                    LayoutDocument.for_blocks([
                        block
                        for raw_author in raw_authors
                        for block in raw_author.iter_blocks()
                    ])
                ]
            else:
                raw_authors_layout_documents = [
                    LayoutDocument.for_blocks(list(raw_author.iter_blocks()))
                    for raw_author in raw_authors
                ]
            labeled_layout_tokens_list = self.name_header_model.predict_labels_for_layout_documents(
                raw_authors_layout_documents,
                app_features_context=self.app_features_context
            )
            LOGGER.debug('labeled_layout_tokens_list (author): %r', labeled_layout_tokens_list)
            parsed_per_block = [
                list(self.name_header_model.iter_semantic_content_for_labeled_layout_tokens(llt))
                for llt in labeled_layout_tokens_list
            ]
            if self.config.deduplicate_raw_authors and not self.config.merge_raw_authors:
                result_content.extend(_iter_deduplicated_authors(
                    raw_authors,
                    parsed_per_block,
                    self.config.deduplicate_raw_authors_use_initial_fallback
                ))
            else:
                for parsed in parsed_per_block:
                    result_content.extend(parsed)
        semantic_parent.mixed_content = result_content

    def _process_raw_affiliations(self, semantic_document: SemanticDocument):
        result_content: List[SemanticContentWrapper] = []
        raw_aff_address_list: List[SemanticRawAffiliationAddress] = []
        for semantic_content in semantic_document.front:
            if isinstance(semantic_content, SemanticRawAffiliationAddress):
                raw_aff_address_list.append(semantic_content)
                continue
            result_content.append(semantic_content)
        if raw_aff_address_list:
            raw_aff_layout_documents = [
                LayoutDocument.for_blocks(list(raw_aff_or_address.iter_blocks()))
                for raw_aff_or_address in raw_aff_address_list
            ]
            labeled_layout_tokens_list = (
                self.affiliation_address_model
                .predict_labels_for_layout_documents(
                    raw_aff_layout_documents,
                    app_features_context=self.app_features_context
                )
            )
            LOGGER.debug('labeled_layout_tokens_list (aff): %r', labeled_layout_tokens_list)
            aff_iterable = (
                aff
                for labeled_layout_tokens in labeled_layout_tokens_list
                for aff in (
                    self.affiliation_address_model
                    .iter_semantic_content_for_labeled_layout_tokens(labeled_layout_tokens)
                )
            )
            for aff in aff_iterable:
                result_content.append(aff)
        semantic_document.front.mixed_content = result_content
        self._assign_content_ids(
            semantic_document.front.iter_by_type(SemanticAffiliationAddress),
            iter(iter_ids('aff'))
        )

    def _extract_raw_references_from_segmentation(
        self,
        semantic_document: SemanticDocument,
        segmentation_label_result: LayoutDocumentLabelResult
    ):
        references_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<references>'
        ).remove_empty_blocks()
        LOGGER.debug('references_layout_document: %s', references_layout_document)
        if not references_layout_document:
            return
        labeled_layout_tokens = self.reference_segmenter_model.predict_labels_for_layout_document(
            references_layout_document,
            app_features_context=self.app_features_context
        )
        LOGGER.debug('labeled_layout_tokens: %r', labeled_layout_tokens)
        semantic_content_iterable = (
            self.reference_segmenter_model
            .iter_semantic_content_for_labeled_layout_tokens(labeled_layout_tokens)
        )
        reference_list = SemanticReferenceList(list(semantic_content_iterable))
        semantic_document.back_section.add_content(reference_list)

    def _iter_parse_semantic_references(
        self,
        semantic_raw_references: List[SemanticRawReference]
    ) -> Iterable[Union[SemanticReference, SemanticInvalidReference]]:
        layout_documents = [
            LayoutDocument.for_blocks([semantic_raw_reference.merged_block])
            for semantic_raw_reference in semantic_raw_references
        ]
        labeled_layout_tokens_list = (
            self.citation_model
            .predict_labels_for_layout_documents(
                layout_documents,
                app_features_context=self.app_features_context
            )
        )
        LOGGER.debug('labeled_layout_tokens_list: %r', labeled_layout_tokens_list)
        for labeled_layout_tokens, semantic_raw_reference in zip(
            labeled_layout_tokens_list, semantic_raw_references
        ):
            semantic_content_iterable = (
                self.citation_model
                .iter_semantic_content_for_labeled_layout_tokens(
                    labeled_layout_tokens,
                    semantic_raw_reference=semantic_raw_reference
                )
            )
            ref: Optional[Union[SemanticReference, SemanticInvalidReference]] = None
            for semantic_content in semantic_content_iterable:
                if isinstance(semantic_content, (SemanticReference, SemanticInvalidReference)):
                    ref = semantic_content
            if not ref:
                raise AssertionError('no semantic reference extracted')
            yield ref

    def _extract_reference_fields_from_raw_references(
        self,
        semantic_document: SemanticDocument
    ):
        reference_lists = list(semantic_document.back_section.iter_by_type(
            SemanticReferenceList
        ))
        semantic_raw_references = [
            raw_reference
            for reference_list in reference_lists
            for raw_reference in reference_list.iter_by_type(SemanticRawReference)
        ]
        semantic_references = list(self._iter_parse_semantic_references(
            semantic_raw_references
        ))
        LOGGER.debug('semantic_references: %r', semantic_references)
        semantic_reference_by_semantic_raw_reference_id = {
            id(semantic_raw_reference): semantic_reference
            for semantic_raw_reference, semantic_reference in zip(
                semantic_raw_references, semantic_references
            )
        }
        LOGGER.debug(
            'semantic_reference_by_semantic_raw_reference_id keys: %s',
            semantic_reference_by_semantic_raw_reference_id.keys()
        )
        for reference_list in reference_lists:
            updated_content: List[SemanticContentWrapper] = []
            for semantic_content in reference_list:
                if isinstance(semantic_content, SemanticRawReference):
                    semantic_reference = semantic_reference_by_semantic_raw_reference_id[
                        id(semantic_content)
                    ]
                    updated_content.append(semantic_reference)
                    continue
                updated_content.append(semantic_content)
            reference_list.mixed_content = updated_content

    def _iter_parse_semantic_name_lists(
        self,
        semantic_raw_name_lists: Sequence[T_SemanticRawNameList],
        name_type: Type[T_SemanticName]
    ) -> Iterable[Tuple[T_SemanticRawNameList, List[SemanticContentWrapper]]]:
        layout_documents = [
            LayoutDocument.for_blocks([semantic_raw_name_list.merged_block])
            for semantic_raw_name_list in semantic_raw_name_lists
        ]
        labeled_layout_tokens_list = (
            self.name_citation_model
            .predict_labels_for_layout_documents(
                layout_documents,
                app_features_context=self.app_features_context
            )
        )
        LOGGER.debug('labeled_layout_tokens_list: %r', labeled_layout_tokens_list)
        for labeled_layout_tokens, semantic_raw_name_list in zip(
            labeled_layout_tokens_list, semantic_raw_name_lists
        ):
            semantic_content_iterable = (
                self.name_citation_model
                .iter_semantic_content_for_labeled_layout_tokens(
                    labeled_layout_tokens,
                    name_type=name_type
                )
            )
            yield semantic_raw_name_list, list(semantic_content_iterable)

    def _extract_reference_name_lists_from_raw_references(
        self,
        semantic_document: SemanticDocument
    ):
        reference_lists = list(semantic_document.back_section.iter_by_type(
            SemanticReferenceList
        ))
        ref_list = [
            ref
            for reference_list in reference_lists
            for ref in reference_list.iter_by_type(SemanticReference)
        ]
        if self.config.extract_citation_authors:
            raw_authors = [
                raw_author
                for ref in ref_list
                for raw_author in ref.iter_by_type(SemanticRawAuthors)
            ]
        else:
            raw_authors = []
        if self.config.extract_citation_editors:
            raw_editors = [
                raw_author
                for ref in ref_list
                for raw_author in ref.iter_by_type(SemanticRawEditors)
            ]
        else:
            raw_editors = []
        content_list_by_raw_author_id = {
            id(raw_author): content_list
            for raw_author, content_list in (
                self._iter_parse_semantic_name_lists(raw_authors, name_type=SemanticAuthor)
            )
        }
        content_list_by_raw_editor_id = {
            id(raw_author): content_list
            for raw_author, content_list in (
                self._iter_parse_semantic_name_lists(raw_editors, name_type=SemanticEditor)
            )
        }
        LOGGER.debug(
            'content_list_by_raw_author_id keys: %s',
            content_list_by_raw_author_id.keys()
        )
        LOGGER.debug(
            'content_list_by_raw_editor_id keys: %s',
            content_list_by_raw_editor_id.keys()
        )
        for reference_list in reference_lists:
            for semantic_content in reference_list:
                if isinstance(semantic_content, SemanticReference):
                    if self.config.extract_citation_authors:
                        semantic_content.flat_map_inplace_by_type(
                            SemanticRawAuthors,
                            lambda raw_author: content_list_by_raw_author_id[
                                id(raw_author)
                            ]
                        )
                    if self.config.extract_citation_editors:
                        semantic_content.flat_map_inplace_by_type(
                            SemanticRawEditors,
                            lambda raw_editor: content_list_by_raw_editor_id[
                                id(raw_editor)
                            ]
                        )

    def _iter_parse_semantic_content_lists(
        self,
        semantic_raw_content_lists: Sequence[T_SemanticContentWrapper],
        model: Model
    ) -> Iterable[Tuple[T_SemanticContentWrapper, List[SemanticContentWrapper]]]:
        layout_documents = [
            LayoutDocument.for_blocks([semantic_raw_name_list.merged_block])
            for semantic_raw_name_list in semantic_raw_content_lists
        ]
        labeled_layout_tokens_list = (
            model
            .predict_labels_for_layout_documents(
                layout_documents,
                app_features_context=self.app_features_context
            )
        )
        LOGGER.debug('labeled_layout_tokens_list: %r', labeled_layout_tokens_list)
        for labeled_layout_tokens, semantic_raw_name_list in zip(
            labeled_layout_tokens_list, semantic_raw_content_lists
        ):
            semantic_content_iterable = (
                model
                .iter_semantic_content_for_labeled_layout_tokens(
                    labeled_layout_tokens
                )
            )
            yield semantic_raw_name_list, list(semantic_content_iterable)

    def _extract_semantic_content_from_raw_content(
        self,
        semantic_document: SemanticDocument,
        semantic_type: Type[T_SemanticContentWrapper],
        model: Model
    ):
        parents = [
            parent
            for root in [
                semantic_document.body_section,
                semantic_document.back_section
            ]
            for parent in root.iter_parent_by_semantic_type_recursively(
                semantic_type
            )
        ]
        raw_content_lists = [
            raw_content
            for parent in parents
            for raw_content in parent.iter_by_type(semantic_type)
        ]
        content_list_by_raw_content_id = {
            id(raw_content): content_list
            for raw_content, content_list in (
                self._iter_parse_semantic_content_lists(
                    raw_content_lists,
                    model
                )
            )
        }
        LOGGER.debug(
            'content_list_by_raw_content_id keys: %s',
            content_list_by_raw_content_id.keys()
        )
        for parent in parents:
            parent.flat_map_inplace_by_type(
                semantic_type,
                lambda raw_content: content_list_by_raw_content_id[
                    id(raw_content)
                ]
            )

    def _extract_figure_fields_from_raw_figures(
        self,
        semantic_document: SemanticDocument
    ):
        self._extract_semantic_content_from_raw_content(
            semantic_document,
            SemanticRawFigure,
            self.figure_model
        )

    def _extract_table_fields_from_raw_tables(
        self,
        semantic_document: SemanticDocument
    ):
        self._extract_semantic_content_from_raw_content(
            semantic_document,
            SemanticRawTable,
            self.table_model
        )

    def _update_semantic_section_using_segmentation_result_and_fulltext_model(
        self,
        semantic_section: SemanticSection,
        segmentation_label_result: LayoutDocumentLabelResult,
        segmentation_tag: str,
        section_type: str
    ):
        layout_document = segmentation_label_result.get_filtered_document_by_label(
            segmentation_tag
        ).remove_empty_blocks()
        self._update_semantic_section_using_layout_document_and_fulltext_model(
            semantic_section,
            layout_document,
            section_name=segmentation_tag,
            section_type=section_type
        )

    def _update_semantic_section_using_layout_document_and_fulltext_model(
        self,
        semantic_section: SemanticSection,
        layout_document: LayoutDocument,
        section_name: str,
        section_type: str
    ):
        LOGGER.debug('layout_document (%r): %s', section_name, layout_document)
        if not layout_document.pages:
            return
        labeled_layout_tokens = self.fulltext_model.predict_labels_for_layout_document(
            layout_document,
            app_features_context=self.app_features_context
        )
        LOGGER.debug('labeled_layout_tokens (%r): %r', section_name, labeled_layout_tokens)
        entity_blocks = self.fulltext_model.iter_entity_layout_blocks_for_labeled_layout_tokens(
            labeled_layout_tokens
        )
        self.fulltext_model.update_section_with_entity_blocks(
            semantic_section,
            entity_blocks,
            section_type=section_type
        )

    def get_tei_document_for_layout_document(
        self,
        layout_document: LayoutDocument
    ) -> TeiDocument:
        return get_tei_for_semantic_document(
            self.get_semantic_document_for_layout_document(
                layout_document
            )
        )
