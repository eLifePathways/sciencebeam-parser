# pylint: disable=too-many-lines
from abc import ABC, abstractmethod
import argparse
import logging
import os
import multiprocessing
import time

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from lxml import etree

from sciencebeam_trainer_delft.utils.io import (
    auto_download_input_file
)

from sciencebeam_parser.utils.io import glob, makedirs, write_bytes, write_text

from sciencebeam_parser.document.layout_document import LayoutDocument
from sciencebeam_parser.document.semantic_document import (
    SemanticMixedContentWrapper,
    SemanticRawAffiliationAddress,
    SemanticRawAuthors,
    SemanticRawFigure,
    SemanticRawReference,
    SemanticRawReferenceText,
    SemanticRawTable
)
from sciencebeam_parser.models.data import (
    DocumentFeaturesContext,
    LabeledLayoutModelData,
    LayoutModelData
)
from sciencebeam_parser.models.model import (
    LabeledLayoutToken,
    LayoutDocumentLabelResult,
    LayoutModelLabel,
    Model,
    iter_data_lines_for_model_data_iterables,
    iter_labeled_layout_token_for_layout_model_label
)
from sciencebeam_parser.models.training_data import TeiTrainingDataGenerator
from sciencebeam_parser.processors.fulltext.models import FullTextModels
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.app.parser import ScienceBeamParser
from sciencebeam_parser.utils.media_types import MediaTypes
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.field_vocab import (
    AFF_LABEL_BY_SUB_FIELD,
    CITATION_LABEL_BY_SUB_FIELD,
    FULLTEXT_LABEL_BY_FIELD,
    HEADER_LABEL_BY_FIELD,
    JatsSubFieldNames,
)
from sciencebeam_parser.training.jats.field_extractor import JatsFieldExtractor
from sciencebeam_parser.training.jats.aligner import LayoutDocumentJatsAligner
from sciencebeam_parser.training.jats.segmentation import SegmentationLabelDeriver


LOGGER = logging.getLogger(__name__)


@dataclass
class ModelResultCache:
    model_data_lists_by_key_map: Dict[
        str, Sequence[Sequence[LabeledLayoutModelData]]
    ] = field(default_factory=dict)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'ScienceBeam Parser: Generate Training Data'
    )
    parser.add_argument(
        '--source-path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--limit',
        type=int,
        required=False
    )
    parser.add_argument(
        '--use-model',
        action='store_true',
        help='Use configured models to pre-annotate training data'
    )
    parser.add_argument(
        '--use-directory-structure',
        action='store_true',
        help='Output training data to a directory structure'
    )
    parser.add_argument(
        '--gzip',
        action='store_true',
        help='Enable gzip compression for output files (with .gz suffix)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--source-xml-path',
        type=str,
        required=False,
        help='Glob pattern to JATS XML files; matched to PDFs by filename stem'
    )
    parser.add_argument(
        '--required-fields',
        type=str,
        nargs='*',
        default=[],
        help='JATS field names that must be present AND aligned; skip document if any are missing'
    )
    parser.add_argument(
        '--require-matching-fields',
        type=str,
        nargs='*',
        default=[],
        help='JATS field names that must align if present in JATS XML'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel worker processes (default: 1)'
    )
    parser.add_argument(
        '--document-timeout',
        type=int,
        default=0,
        metavar='SECONDS',
        help=(
            'Per-document time limit in seconds (0 = no limit). '
            'Documents that exceed this limit are skipped with a warning. '
            'Single-worker mode uses SIGALRM; multi-worker mode uses future timeout.'
        )
    )
    return parser.parse_args(argv)


def get_labeled_model_data_list_list(
    model_data_list_list: Sequence[Sequence[LayoutModelData]],
    model: Model
) -> Sequence[Sequence[LabeledLayoutModelData]]:
    return list(
        model.iter_labeled_model_data_list_for_model_data_list_iterable(
            model_data_list_list
        )
    )


def get_labeled_model_data_list(
    model_data_list: Sequence[LayoutModelData],
    model: Model
) -> Sequence[LabeledLayoutModelData]:
    return get_labeled_model_data_list_list(
        [model_data_list],
        model=model
    )[0]


def get_labeled_model_data_list_for_layout_document(
    layout_document: LayoutDocument,
    model: Model,
    document_features_context: DocumentFeaturesContext
) -> Sequence[LabeledLayoutModelData]:
    data_generator = model.get_data_generator(
        document_features_context=document_features_context
    )
    model_data_list: Sequence[LayoutModelData] = list(
        data_generator.iter_model_data_for_layout_document(layout_document)
    )
    return get_labeled_model_data_list(
        model_data_list,
        model=model
    )


def get_layout_model_label_for_labeled_model_data(
    labeled_model_data: LabeledLayoutModelData
) -> LayoutModelLabel:
    return LayoutModelLabel(
        label=labeled_model_data.label or '',
        label_token_text=labeled_model_data.label_token_text,
        layout_line=labeled_model_data.layout_line,
        layout_token=labeled_model_data.layout_token
    )


def iter_layout_model_label_for_labeled_model_data_list(
    labeled_model_data_iterable: Iterable[LabeledLayoutModelData],
) -> Iterable[LayoutModelLabel]:
    return (
        get_layout_model_label_for_labeled_model_data(labeled_model_data)
        for labeled_model_data in labeled_model_data_iterable
    )


def get_layout_document_label_result_for_labeled_model_data_list(
    labeled_model_data_iterable: Iterable[LabeledLayoutModelData],
    layout_document: LayoutDocument
) -> LayoutDocumentLabelResult:
    return LayoutDocumentLabelResult(
        layout_document=layout_document,
        layout_model_label_iterable=iter_layout_model_label_for_labeled_model_data_list(
            labeled_model_data_iterable
        )
    )


class TrainingDataDocumentContext(NamedTuple):
    output_path: str
    source_filename: str
    document_features_context: DocumentFeaturesContext
    fulltext_models: FullTextModels
    use_model: bool
    use_directory_structure: bool
    model_result_cache: ModelResultCache
    gzip_enabled: bool
    jats_annotated_document: Optional[JatsAnnotatedLayoutDocument] = None
    jats_segmentation_labels: Optional[Dict[int, str]] = None

    @property
    def source_name(self) -> str:
        source_basename = os.path.basename(self.source_filename)
        return os.path.splitext(source_basename)[0]


def iter_unlabeled_model_data_list_for_model_and_layout_documents(
    model: Model,
    model_layout_documents: Sequence[LayoutDocument],
    document_context: TrainingDataDocumentContext
) -> Iterable[Sequence[LayoutModelData]]:
    if not model_layout_documents:
        return []
    data_generator = model.get_data_generator(
        document_features_context=document_context.document_features_context
    )
    return [
        list(
            data_generator.iter_model_data_for_layout_document(model_layout_document)
        )
        for model_layout_document in model_layout_documents
    ]


def iter_labeled_model_data_list_for_model_and_layout_documents(
    model: Model,
    model_layout_documents: Sequence[LayoutDocument],
    document_context: TrainingDataDocumentContext
) -> Iterable[Sequence[LabeledLayoutModelData]]:
    if not model_layout_documents:
        return []
    cache_key = f'{type(model).__name__}_{id(model)}'
    LOGGER.debug('cache_key: %r', cache_key)
    model_data_lists = document_context.model_result_cache.model_data_lists_by_key_map.get(
        cache_key
    )
    if model_data_lists is not None:
        return model_data_lists
    unlabeled_model_data_lists = list(
        iter_unlabeled_model_data_list_for_model_and_layout_documents(
            model=model,
            model_layout_documents=model_layout_documents,
            document_context=document_context
        )
    )
    model_data_lists = get_labeled_model_data_list_list(
        unlabeled_model_data_lists,
        model=model
    )
    document_context.model_result_cache.model_data_lists_by_key_map[cache_key] = (
        model_data_lists
    )
    return model_data_lists


def iter_model_data_list_for_model_and_layout_documents(
    model: Model,
    model_layout_documents: Sequence[LayoutDocument],
    document_context: TrainingDataDocumentContext
) -> Iterable[Sequence[LayoutModelData]]:
    if not document_context.use_model:
        return iter_unlabeled_model_data_list_for_model_and_layout_documents(
            model=model,
            model_layout_documents=model_layout_documents,
            document_context=document_context
        )
    return iter_labeled_model_data_list_for_model_and_layout_documents(
        model=model,
        model_layout_documents=model_layout_documents,
        document_context=document_context
    )


def get_labeled_layout_tokens_list_for_model_and_layout_documents(
    model: Model,
    layout_documents: Sequence[LayoutDocument],
    document_context: TrainingDataDocumentContext
) -> Sequence[Sequence[LabeledLayoutToken]]:
    model_data_lists = list(
        iter_labeled_model_data_list_for_model_and_layout_documents(
            model=model,
            model_layout_documents=layout_documents,
            document_context=document_context
        )
    )
    assert len(model_data_lists) == len(layout_documents)
    return [
        list(iter_labeled_layout_token_for_layout_model_label(
            iter_layout_model_label_for_labeled_model_data_list(
                model_data_list
            )
        ))
        for model_data_list in model_data_lists
    ]


def get_labeled_layout_tokens_for_model_and_layout_document(
    model: Model,
    layout_document: LayoutDocument,
    document_context: TrainingDataDocumentContext
) -> Sequence[LabeledLayoutToken]:
    labeled_layout_tokens_list = get_labeled_layout_tokens_list_for_model_and_layout_documents(
        model,
        [layout_document],
        document_context
    )
    assert len(labeled_layout_tokens_list) == 1
    return labeled_layout_tokens_list[0]


def _get_jats_segmentation_label_result(
    layout_document: LayoutDocument,
    jats_segmentation_labels: Dict[int, str],
) -> LayoutDocumentLabelResult:
    """Build a LayoutDocumentLabelResult from JATS-derived per-line segmentation labels."""
    layout_model_labels = [
        LayoutModelLabel(
            label=jats_segmentation_labels.get(id(line), '<body>'),
            label_token_text=line.text,
            layout_line=line,
            layout_token=None,
        )
        for block in layout_document.iter_all_blocks()
        for line in block.lines
    ]
    return LayoutDocumentLabelResult(
        layout_document=layout_document,
        layout_model_label_iterable=layout_model_labels,
    )


def get_segmentation_label_result(
    layout_document: LayoutDocument,
    document_context: TrainingDataDocumentContext
) -> LayoutDocumentLabelResult:
    if document_context.jats_segmentation_labels is not None:
        return _get_jats_segmentation_label_result(
            layout_document=layout_document,
            jats_segmentation_labels=document_context.jats_segmentation_labels,
        )
    segmentation_label_model_data_lists = list(
        iter_labeled_model_data_list_for_model_and_layout_documents(
            model=document_context.fulltext_models.segmentation_model,
            model_layout_documents=[layout_document],
            document_context=document_context
        )
    )
    assert len(segmentation_label_model_data_lists) == 1
    LOGGER.debug('segmentation_label_model_data_lists: %r', segmentation_label_model_data_lists)
    return get_layout_document_label_result_for_labeled_model_data_list(
        labeled_model_data_iterable=segmentation_label_model_data_lists[0],
        layout_document=layout_document
    )


JatsLabelFn = Callable[
    [JatsAnnotatedLayoutDocument, Dict[int, str], LayoutModelData],
    Optional[str]
]


def _apply_jats_labels_to_model_data_list(
    model_data_list: Sequence[LayoutModelData],
    annotated: JatsAnnotatedLayoutDocument,
    jats_seg_labels: Dict[int, str],
    label_fn: JatsLabelFn,
) -> Sequence[LabeledLayoutModelData]:
    return [
        LabeledLayoutModelData.from_model_data(md, label=label_fn(annotated, jats_seg_labels, md))
        for md in model_data_list
    ]


class AbstractModelTrainingDataGenerator(ABC):
    def get_pre_file_path_suffix(self) -> str:
        return ''

    def _get_file_path_with_suffix(
        self,
        suffix: Optional[str],
        document_context: TrainingDataDocumentContext,
        sub_directory: Optional[str] = None
    ) -> Optional[str]:
        if not suffix:
            return None
        output_path = document_context.output_path
        if sub_directory and document_context.use_directory_structure:
            output_path = os.path.join(output_path, sub_directory)
        if document_context.gzip_enabled:
            suffix += '.gz'
        return os.path.join(
            output_path,
            document_context.source_name + self.get_pre_file_path_suffix() + suffix
        )

    @abstractmethod
    def get_tei_training_data_generator(
        self,
        document_context: TrainingDataDocumentContext
    ) -> TeiTrainingDataGenerator:
        pass

    @abstractmethod
    def iter_model_data_list(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[Sequence[LayoutModelData]]:
        return []

    def get_default_tei_sub_directory(
        self,
        tei_training_data_generator: TeiTrainingDataGenerator
    ) -> Optional[str]:
        return tei_training_data_generator.get_default_tei_sub_directory()

    def generate_data_for_layout_document(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ):
        tei_training_data_generator = self.get_tei_training_data_generator(document_context)
        tei_file_path = self._get_file_path_with_suffix(
            tei_training_data_generator.get_default_tei_filename_suffix(),
            document_context=document_context,
            sub_directory=self.get_default_tei_sub_directory(tei_training_data_generator)
        )
        data_file_path = self._get_file_path_with_suffix(
            tei_training_data_generator.get_default_data_filename_suffix(),
            document_context=document_context,
            sub_directory=tei_training_data_generator.get_default_data_sub_directory()
        )
        assert tei_file_path
        model_data_list_list = list(self.iter_model_data_list(
            layout_document=layout_document,
            document_context=document_context
        ))
        if not model_data_list_list:
            LOGGER.info('no entities found, skipping (%r)', tei_file_path)
            return
        training_tei_root = (
            tei_training_data_generator
            .get_training_tei_xml_for_multiple_model_data_iterables(
                model_data_list_list
            )
        )
        LOGGER.info('writing training tei to: %r', tei_file_path)
        write_bytes(
            tei_file_path,
            etree.tostring(training_tei_root, pretty_print=True)
        )
        if data_file_path:
            LOGGER.info('writing training raw data to: %r', data_file_path)
            write_text(
                data_file_path,
                '\n'.join(
                    iter_data_lines_for_model_data_iterables(model_data_list_list)
                ),
                encoding='utf-8'
            )


class AbstractDocumentModelTrainingDataGenerator(AbstractModelTrainingDataGenerator):
    @abstractmethod
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        pass

    def get_tei_training_data_generator(
        self,
        document_context: TrainingDataDocumentContext
    ) -> TeiTrainingDataGenerator:
        return self.get_main_model(document_context).get_tei_training_data_generator()

    @abstractmethod
    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        pass

    def get_jats_label_fn(self) -> Optional[JatsLabelFn]:
        """Return a JATS label function, or None to skip JATS labeling for this model."""
        return None

    def iter_model_data_list(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[Sequence[LayoutModelData]]:
        model = self.get_main_model(document_context)
        model_layout_documents = list(self.iter_model_layout_documents(
            layout_document,
            document_context=document_context
        ))
        annotated = document_context.jats_annotated_document
        jats_seg_labels = document_context.jats_segmentation_labels
        jats_label_fn = self.get_jats_label_fn()  # pylint: disable=assignment-from-none
        if annotated is not None and jats_seg_labels is not None and jats_label_fn is not None:
            unlabeled_lists = list(
                iter_unlabeled_model_data_list_for_model_and_layout_documents(
                    model=model,
                    model_layout_documents=model_layout_documents,
                    document_context=document_context,
                )
            )
            return [
                _apply_jats_labels_to_model_data_list(
                    mdl, annotated, jats_seg_labels, jats_label_fn
                )
                for mdl in unlabeled_lists
            ]
        return iter_model_data_list_for_model_and_layout_documents(
            model=model,
            model_layout_documents=model_layout_documents,
            document_context=document_context
        )


class SegmentationModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.segmentation_model

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        return [layout_document]

    def get_jats_label_fn(self) -> Optional[JatsLabelFn]:
        def fn(
            _annotated: JatsAnnotatedLayoutDocument,
            seg_labels: Dict[int, str],
            md: LayoutModelData,
        ) -> Optional[str]:
            return seg_labels.get(id(md.layout_line)) if md.layout_line else None
        return fn


class HeaderModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.header_model

    def get_jats_label_fn(self) -> Optional[JatsLabelFn]:
        # Stateful closure: emit B-/I- IOB prefix so the TEI generator can create
        # separate <byline><affiliation> blocks for each JATS <aff> element.
        # Address sub-fields (city, region, postcode, country, bulk addr range) are
        # mapped to <address> instead of <affiliation>.
        _HEADER_ADDRESS_SUB_FIELDS = frozenset({
            JatsSubFieldNames.AUTHOR_AFF_ADDR,
            JatsSubFieldNames.AUTHOR_AFF_CITY,
            JatsSubFieldNames.AUTHOR_AFF_POSTCODE,
            JatsSubFieldNames.AUTHOR_AFF_REGION,
            JatsSubFieldNames.AUTHOR_AFF_COUNTRY,
        })
        prev_label_instance: Optional[Tuple[str, int]] = None

        def fn(
            annotated: JatsAnnotatedLayoutDocument,
            _seg_labels: Dict[int, str],
            md: LayoutModelData,
        ) -> Optional[str]:
            nonlocal prev_label_instance
            token = md.layout_token
            if not token:
                prev_label_instance = None
                return None
            field_name = annotated.get_token_field(token)
            if not field_name:
                prev_label_instance = None
                return None
            sub_field_name = annotated.get_token_sub_field(token)
            if sub_field_name in _HEADER_ADDRESS_SUB_FIELDS:
                label: Optional[str] = '<address>'
            else:
                label = HEADER_LABEL_BY_FIELD.get(field_name)
            if label is None:
                prev_label_instance = None
                return None
            instance_id = annotated.get_token_instance(token)
            label_instance = (label, instance_id)
            prefix = 'B' if label_instance != prev_label_instance else 'I'
            prev_label_instance = label_instance
            return f'{prefix}-{label}'

        return fn

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        LOGGER.debug('segmentation_label_result: %r', segmentation_label_result)
        header_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<header>'
        ).remove_empty_blocks()
        LOGGER.debug('header_layout_document: %r', header_layout_document)
        if not header_layout_document.pages:
            return []
        return [header_layout_document]


class AffiliationAddressModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.affiliation_address_model

    def get_jats_label_fn(self) -> Optional[JatsLabelFn]:
        def fn(
            annotated: JatsAnnotatedLayoutDocument,
            _seg_labels: Dict[int, str],
            md: LayoutModelData,
        ) -> Optional[str]:
            token = md.layout_token
            if not token:
                return None
            sub_field = annotated.get_token_sub_field(token)
            return AFF_LABEL_BY_SUB_FIELD.get(sub_field or '') if sub_field else None
        return fn

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        header_model = document_context.fulltext_models.header_model
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        header_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<header>'
        ).remove_empty_blocks()
        LOGGER.debug('header_layout_document: %r', header_layout_document)
        if not header_layout_document.pages:
            return []
        header_labeled_layout_tokens = get_labeled_layout_tokens_for_model_and_layout_document(
            model=header_model,
            layout_document=header_layout_document,
            document_context=document_context
        )
        semantic_raw_aff_address_list = list(
            SemanticMixedContentWrapper(list(
                header_model.iter_semantic_content_for_labeled_layout_tokens(
                    header_labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawAffiliationAddress)
        )
        LOGGER.info('semantic_raw_aff_address_list count: %d', len(semantic_raw_aff_address_list))
        if not semantic_raw_aff_address_list:
            return []

        return [
            LayoutDocument.for_blocks(
                list(semantic_raw_aff_address.iter_blocks())
            )
            for semantic_raw_aff_address in semantic_raw_aff_address_list
        ]


class NameHeaderModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.name_header_model

    def get_default_tei_sub_directory(
        self,
        tei_training_data_generator: TeiTrainingDataGenerator
    ) -> str:
        return 'name/header/corpus'

    def get_pre_file_path_suffix(self) -> str:
        return '.header'

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        header_model = document_context.fulltext_models.header_model
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        header_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<header>'
        ).remove_empty_blocks()
        LOGGER.debug('header_layout_document: %r', header_layout_document)
        if not header_layout_document.pages:
            return []
        header_labeled_layout_tokens = get_labeled_layout_tokens_for_model_and_layout_document(
            model=header_model,
            layout_document=header_layout_document,
            document_context=document_context
        )
        semantic_raw_author_list = list(
            SemanticMixedContentWrapper(list(
                header_model.iter_semantic_content_for_labeled_layout_tokens(
                    header_labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawAuthors)
        )
        LOGGER.info('semantic_raw_author_list count: %d', len(semantic_raw_author_list))
        if not semantic_raw_author_list:
            return []

        return [
            LayoutDocument.for_blocks([
                block
                for semantic_raw_author in semantic_raw_author_list
                for block in semantic_raw_author.iter_blocks()
            ])
        ]


class NameCitationModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.name_citation_model

    def get_default_tei_sub_directory(
        self,
        tei_training_data_generator: TeiTrainingDataGenerator
    ) -> str:
        return 'name/citation/corpus'

    def get_pre_file_path_suffix(self) -> str:
        return '.citations'

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        reference_segmenter_model = document_context.fulltext_models.reference_segmenter_model
        citation_model = document_context.fulltext_models.citation_model
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        references_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<references>'
        ).remove_empty_blocks()
        reference_segmenter_labeled_layout_tokens = (
            get_labeled_layout_tokens_for_model_and_layout_document(
                model=reference_segmenter_model,
                layout_document=references_layout_document,
                document_context=document_context
            )
        )
        raw_reference_text_list = [
            raw_reference_text
            for raw_reference in SemanticMixedContentWrapper(list(
                reference_segmenter_model.iter_semantic_content_for_labeled_layout_tokens(
                    reference_segmenter_labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawReference)
            for raw_reference_text in raw_reference.iter_by_type(SemanticRawReferenceText)
        ]
        LOGGER.info('raw_reference_text_list count: %d', len(raw_reference_text_list))
        if not raw_reference_text_list:
            return []
        citation_layout_documents = [
            LayoutDocument.for_blocks(
                list(semantic_raw_reference_text.iter_blocks())
            )
            for semantic_raw_reference_text in raw_reference_text_list
        ]
        citation_labeled_layout_tokens_list = (
            get_labeled_layout_tokens_list_for_model_and_layout_documents(
                model=citation_model,
                layout_documents=citation_layout_documents,
                document_context=document_context
            )
        )
        semantic_raw_author_list = [
            raw_author
            for citation_labeled_layout_tokens in citation_labeled_layout_tokens_list
            for raw_author in SemanticMixedContentWrapper(list(
                citation_model.iter_semantic_content_for_labeled_layout_tokens(
                    citation_labeled_layout_tokens
                )
            )).iter_by_type_recursively(SemanticRawAuthors)
        ]
        LOGGER.info('semantic_raw_author_list count: %d', len(semantic_raw_author_list))
        if not semantic_raw_author_list:
            return []

        return [
            LayoutDocument.for_blocks([
                block
                for semantic_raw_author in semantic_raw_author_list
                for block in semantic_raw_author.iter_blocks()
            ])
        ]


class FullTextModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.fulltext_model

    def get_jats_label_fn(self) -> Optional[JatsLabelFn]:
        def fn(
            annotated: JatsAnnotatedLayoutDocument,
            _seg_labels: Dict[int, str],
            md: LayoutModelData,
        ) -> Optional[str]:
            token = md.layout_token
            if not token:
                return None
            field_name = annotated.get_token_field(token)
            return FULLTEXT_LABEL_BY_FIELD.get(field_name or '') if field_name else None
        return fn

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        body_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<body>'
        ).remove_empty_blocks()
        if not body_layout_document.pages:
            return []
        return [body_layout_document]


class FigureModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.figure_model

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        fulltext_model = document_context.fulltext_models.fulltext_model
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        body_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<body>'
        ).remove_empty_blocks()
        if not body_layout_document.pages:
            return []
        fulltext_labeled_layout_tokens = get_labeled_layout_tokens_for_model_and_layout_document(
            model=fulltext_model,
            layout_document=body_layout_document,
            document_context=document_context
        )
        raw_figure_list = list(
            SemanticMixedContentWrapper(list(
                fulltext_model.iter_semantic_content_for_labeled_layout_tokens(
                    fulltext_labeled_layout_tokens
                )
            )).iter_by_type_recursively(SemanticRawFigure)
        )
        LOGGER.info('raw_figure_list count: %d', len(raw_figure_list))

        if not raw_figure_list:
            return []
        return [
            LayoutDocument.for_blocks(list(raw_figure.iter_blocks()))
            for raw_figure in raw_figure_list
        ]


class TableModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.table_model

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        fulltext_model = document_context.fulltext_models.fulltext_model
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        body_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<body>'
        ).remove_empty_blocks()
        if not body_layout_document.pages:
            return []
        fulltext_labeled_layout_tokens = get_labeled_layout_tokens_for_model_and_layout_document(
            model=fulltext_model,
            layout_document=body_layout_document,
            document_context=document_context
        )
        raw_table_list = list(
            SemanticMixedContentWrapper(list(
                fulltext_model.iter_semantic_content_for_labeled_layout_tokens(
                    fulltext_labeled_layout_tokens
                )
            )).iter_by_type_recursively(SemanticRawTable)
        )
        LOGGER.info('raw_table_list count: %d', len(raw_table_list))

        if not raw_table_list:
            return []
        return [
            LayoutDocument.for_blocks(list(raw_table.iter_blocks()))
            for raw_table in raw_table_list
        ]


class ReferenceSegmenterModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.reference_segmenter_model

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        ref_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<references>'
        ).remove_empty_blocks()
        if not ref_layout_document.pages:
            return []
        return [ref_layout_document]


class CitationModelTrainingDataGenerator(AbstractDocumentModelTrainingDataGenerator):
    def get_main_model(self, document_context: TrainingDataDocumentContext) -> Model:
        return document_context.fulltext_models.citation_model

    def get_jats_label_fn(self) -> Optional[JatsLabelFn]:
        def fn(
            annotated: JatsAnnotatedLayoutDocument,
            _seg_labels: Dict[int, str],
            md: LayoutModelData,
        ) -> Optional[str]:
            token = md.layout_token
            if not token:
                return None
            sub_field = annotated.get_token_sub_field(token)
            return CITATION_LABEL_BY_SUB_FIELD.get(sub_field or '') if sub_field else None
        return fn

    def iter_model_layout_documents(
        self,
        layout_document: LayoutDocument,
        document_context: TrainingDataDocumentContext
    ) -> Iterable[LayoutDocument]:
        reference_segmenter_model = document_context.fulltext_models.reference_segmenter_model
        segmentation_label_result = get_segmentation_label_result(
            layout_document,
            document_context=document_context
        )
        references_layout_document = segmentation_label_result.get_filtered_document_by_label(
            '<references>'
        ).remove_empty_blocks()
        reference_segmenter_labeled_layout_tokens = (
            get_labeled_layout_tokens_for_model_and_layout_document(
                model=reference_segmenter_model,
                layout_document=references_layout_document,
                document_context=document_context
            )
        )
        raw_reference_text_list = [
            raw_reference_text
            for raw_reference in SemanticMixedContentWrapper(list(
                reference_segmenter_model.iter_semantic_content_for_labeled_layout_tokens(
                    reference_segmenter_labeled_layout_tokens
                )
            )).iter_by_type(SemanticRawReference)
            for raw_reference_text in raw_reference.iter_by_type(SemanticRawReferenceText)
        ]
        LOGGER.info('raw_reference_text_list count: %d', len(raw_reference_text_list))
        if not raw_reference_text_list:
            return []
        return [
            LayoutDocument.for_blocks(
                list(semantic_raw_reference_text.iter_blocks())
            )
            for semantic_raw_reference_text in raw_reference_text_list
        ]


def _build_jats_annotations(
    layout_document: LayoutDocument,
    jats_xml_filename: str,
) -> Optional[JatsAnnotatedLayoutDocument]:
    try:
        with auto_download_input_file(jats_xml_filename, auto_decompress=True) as local_xml:
            root = etree.parse(local_xml).getroot()
    except Exception:  # pylint: disable=broad-except
        LOGGER.warning('Failed to load JATS XML: %r', jats_xml_filename, exc_info=True)
        return None
    field_values = list(JatsFieldExtractor().iter_field_values(root))
    LOGGER.debug('JATS field values count: %d', len(field_values))
    return LayoutDocumentJatsAligner().align(layout_document, field_values)


def generate_training_data_for_layout_document(
    layout_document: LayoutDocument,
    *,
    output_path: str,
    source_filename: str,
    document_features_context: DocumentFeaturesContext,
    fulltext_models: FullTextModels,
    use_model: bool,
    use_directory_structure: bool,
    gzip_enabled: bool = False,
    jats_xml_filename: Optional[str] = None,
):
    model_result_cache = ModelResultCache()
    jats_annotated: Optional[JatsAnnotatedLayoutDocument] = None
    jats_seg_labels: Optional[Dict[int, str]] = None
    if jats_xml_filename:
        jats_annotated = _build_jats_annotations(layout_document, jats_xml_filename)
        if jats_annotated:
            jats_seg_labels = SegmentationLabelDeriver().derive_labels(
                layout_document, jats_annotated
            )
            LOGGER.debug(
                'JATS coverage ratio: %.2f', jats_annotated.coverage_ratio()
            )
    document_context = TrainingDataDocumentContext(
        output_path=output_path,
        source_filename=source_filename,
        document_features_context=document_features_context,
        fulltext_models=fulltext_models,
        use_model=use_model,
        use_directory_structure=use_directory_structure,
        model_result_cache=model_result_cache,
        gzip_enabled=gzip_enabled,
        jats_annotated_document=jats_annotated,
        jats_segmentation_labels=jats_seg_labels,
    )
    training_data_generators = [
        SegmentationModelTrainingDataGenerator(),
        HeaderModelTrainingDataGenerator(),
        AffiliationAddressModelTrainingDataGenerator(),
        NameHeaderModelTrainingDataGenerator(),
        FullTextModelTrainingDataGenerator(),
        FigureModelTrainingDataGenerator(),
        TableModelTrainingDataGenerator(),
        ReferenceSegmenterModelTrainingDataGenerator(),
        CitationModelTrainingDataGenerator(),
        NameCitationModelTrainingDataGenerator()
    ]
    for training_data_generator in training_data_generators:
        training_data_generator.generate_data_for_layout_document(
            layout_document=layout_document,
            document_context=document_context
        )


def get_layout_document_for_source_filename(
    source_filename: str,
    sciencebeam_parser: ScienceBeamParser,
) -> LayoutDocument:
    with sciencebeam_parser.get_new_session() as session:
        with auto_download_input_file(
            source_filename,
            auto_decompress=True
        ) as local_source_filename:
            source = session.get_source(local_source_filename, MediaTypes.PDF)
            layout_document = source.get_layout_document()
            return layout_document


def _find_jats_xml_for_source(
    source_filename: str,
    xml_file_list: Sequence[str],
) -> Optional[str]:
    """Find the XML file whose stem matches the PDF stem, stripping compound extensions."""
    source_stem = os.path.splitext(os.path.basename(source_filename))[0]
    for xml_filename in xml_file_list:
        xml_stem = os.path.basename(xml_filename)
        while True:
            xml_stem, ext = os.path.splitext(xml_stem)
            if xml_stem == source_stem:
                return xml_filename
            if not ext:
                break
    return None


def generate_training_data_for_source_filename(
    source_filename: str,
    *,
    output_path: str,
    sciencebeam_parser: ScienceBeamParser,
    use_model: bool,
    use_directory_structure: bool,
    gzip_enabled: bool,
    xml_file_list: Optional[Sequence[str]] = None,
):
    LOGGER.debug('use_model: %r', use_model)
    layout_document = get_layout_document_for_source_filename(
        source_filename,
        sciencebeam_parser=sciencebeam_parser
    )
    jats_xml_filename: Optional[str] = None
    if xml_file_list:
        jats_xml_filename = _find_jats_xml_for_source(source_filename, xml_file_list)
        if jats_xml_filename:
            LOGGER.info('Using JATS XML: %r', jats_xml_filename)
        else:
            LOGGER.warning('No matching JATS XML found for: %r', source_filename)
    generate_training_data_for_layout_document(
        layout_document=layout_document,
        output_path=output_path,
        source_filename=source_filename,
        document_features_context=DocumentFeaturesContext(
            sciencebeam_parser.app_features_context
        ),
        fulltext_models=sciencebeam_parser.fulltext_models,
        use_model=use_model,
        use_directory_structure=use_directory_structure,
        gzip_enabled=gzip_enabled,
        jats_xml_filename=jats_xml_filename,
    )


def get_source_file_list_or_fail(
    source_path_pattern: str
) -> Sequence[str]:
    source_file_list = list(glob(source_path_pattern))
    if not source_file_list:
        raise FileNotFoundError('no files found for file pattern: %r' % source_path_pattern)
    return source_file_list


def _format_eta(seconds: float) -> str:
    if seconds >= 3600:
        return f'{seconds / 3600:.1f}h'
    if seconds >= 60:
        return f'{int(seconds) // 60}m{int(seconds) % 60:02d}s'
    return f'{seconds:.0f}s'


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.n_ok = 0
        self.n_err = 0
        self._t_start = time.monotonic()

    @property
    def completed(self) -> int:
        return self.n_ok + self.n_err

    def record(self, source_filename: str, ok: bool, elapsed_s: float) -> None:
        if ok:
            self.n_ok += 1
        else:
            self.n_err += 1
        elapsed_total = time.monotonic() - self._t_start
        done = self.completed
        rate = done / elapsed_total if elapsed_total > 0 else 0.0
        remaining = self.total - done
        eta = _format_eta(remaining / rate) if rate > 0 else '?'
        status = 'ok' if ok else 'err'
        LOGGER.info(
            '[%d/%d] %s %s %.1fs | %.2f doc/s | ~%s left',
            done, self.total,
            os.path.basename(source_filename), status, elapsed_s, rate, eta,
        )


# Module-level worker state, initialised once per worker process.
_worker_sciencebeam_parser: Optional[ScienceBeamParser] = None


def _worker_init() -> None:
    global _worker_sciencebeam_parser  # pylint: disable=global-statement
    config = AppConfig.load_yaml(DEFAULT_CONFIG_FILE)
    _worker_sciencebeam_parser = ScienceBeamParser.from_config(config)


def _worker_process(kwargs: dict) -> bool:
    assert _worker_sciencebeam_parser is not None
    try:
        generate_training_data_for_source_filename(
            kwargs['source_filename'],
            output_path=kwargs['output_path'],
            sciencebeam_parser=_worker_sciencebeam_parser,
            use_model=kwargs['use_model'],
            use_directory_structure=kwargs['use_directory_structure'],
            gzip_enabled=kwargs['gzip_enabled'],
            xml_file_list=kwargs['xml_file_list'],
        )
        return True
    except Exception:  # pylint: disable=broad-except
        LOGGER.exception('Failed to process %r', kwargs['source_filename'])
        return False


def _run_serial(
    source_file_list: Sequence[str],
    output_path: str,
    args: argparse.Namespace,
    xml_file_list: Optional[Sequence[str]],
    progress: '_Progress',
    document_timeout: int = 0,
) -> None:
    """Run documents sequentially in a single worker process.

    Uses multiprocessing.Pool so that pool.terminate() can kill a worker
    that is stuck inside a C extension (signal.alarm cannot interrupt C code).
    The pool is recreated after each timeout so subsequent documents can run.
    """
    common_kwargs = {
        'output_path': output_path,
        'use_model': args.use_model,
        'use_directory_structure': args.use_directory_structure,
        'gzip_enabled': args.gzip,
        'xml_file_list': xml_file_list,
    }
    pool = multiprocessing.Pool(1, initializer=_worker_init)  # pylint: disable=consider-using-with
    timeout_arg = document_timeout if document_timeout > 0 else None
    try:
        for source_filename in source_file_list:
            kwargs = {'source_filename': source_filename, **common_kwargs}
            t0 = time.monotonic()
            async_result = pool.apply_async(_worker_process, (kwargs,))
            try:
                ok = async_result.get(timeout=timeout_arg)
            except multiprocessing.TimeoutError:
                LOGGER.warning(
                    'Document exceeded %ds timeout, skipping: %r',
                    document_timeout, source_filename,
                )
                pool.terminate()
                pool.join()
                # pylint: disable-next=consider-using-with
                pool = multiprocessing.Pool(1, initializer=_worker_init)
                ok = False
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception('Failed to process %r', source_filename)
                ok = False
            progress.record(source_filename, ok=ok, elapsed_s=time.monotonic() - t0)
    finally:
        pool.close()
        pool.join()


def _run_parallel_workers(
    source_file_list: Sequence[str],
    output_path: str,
    args: argparse.Namespace,
    xml_file_list: Optional[Sequence[str]],
    progress: '_Progress',
    num_workers: int,
    document_timeout: int = 0,
) -> None:
    """Process documents in parallel using a multiprocessing.Pool.

    All documents are submitted upfront so workers stay busy.  Results are
    collected in submission order; pool.terminate() at the end kills any worker
    that is still stuck in a C extension after its timeout.
    """
    common_kwargs = {
        'output_path': output_path,
        'use_model': args.use_model,
        'use_directory_structure': args.use_directory_structure,
        'gzip_enabled': args.gzip,
        'xml_file_list': xml_file_list,
    }
    # pylint: disable-next=consider-using-with
    pool = multiprocessing.Pool(num_workers, initializer=_worker_init)
    timeout_arg = document_timeout if document_timeout > 0 else None
    work = [
        (sf, pool.apply_async(_worker_process, ({'source_filename': sf, **common_kwargs},)))
        for sf in source_file_list
    ]
    try:
        for source_filename, async_result in work:
            t0 = time.monotonic()
            try:
                ok = async_result.get(timeout=timeout_arg)
            except multiprocessing.TimeoutError:
                LOGGER.warning(
                    'Document exceeded %ds timeout, skipping: %r',
                    document_timeout, source_filename,
                )
                ok = False
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception('Failed to process %r', source_filename)
                ok = False
            progress.record(source_filename, ok=ok, elapsed_s=time.monotonic() - t0)
    finally:
        pool.terminate()
        pool.join()


def run(args: argparse.Namespace):
    LOGGER.info('args: %r', args)
    source_file_list = get_source_file_list_or_fail(args.source_path)
    if args.limit:
        source_file_list = source_file_list[:args.limit]
    LOGGER.info('source files: %d', len(source_file_list))
    output_path = args.output_path
    LOGGER.info('output_path: %r', output_path)
    xml_file_list: Optional[Sequence[str]] = None
    if args.source_xml_path:
        xml_file_list = list(glob(args.source_xml_path))
        LOGGER.info('JATS XML files: %d', len(xml_file_list))
    # Note: creating the directory may not be necessary, but provides early feedback
    makedirs(output_path, exist_ok=True)
    total = len(source_file_list)
    progress = _Progress(total)
    num_workers = getattr(args, 'num_workers', 1)
    document_timeout: int = getattr(args, 'document_timeout', 0)

    if num_workers > 1:
        _run_parallel_workers(
            source_file_list, output_path, args, xml_file_list, progress, num_workers,
            document_timeout=document_timeout,
        )
    else:
        _run_serial(
            source_file_list, output_path, args, xml_file_list, progress,
            document_timeout=document_timeout,
        )

    if progress.n_err:
        LOGGER.warning('%d/%d documents failed', progress.n_err, total)
    LOGGER.info('Done. Processed %d/%d documents.', progress.n_ok, total)


def main(argv: Optional[List[str]] = None):
    LOGGER.debug('argv: %r', argv)
    args = parse_args(argv)
    if args.debug:
        # Only enable DEBUG for the training CLI itself.  Library loggers
        # (model inference, aligner per-field traces) stay at INFO to avoid
        # flooding the output with thousands of internal messages.
        logging.getLogger(__name__).setLevel('DEBUG')
        logging.getLogger('sciencebeam_parser.training').setLevel('DEBUG')
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
