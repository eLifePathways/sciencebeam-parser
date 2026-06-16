import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union
)

from sciencebeam_trainer_delft.sequence_labelling.reader import load_data_crf_lines

from sciencebeam_parser.utils.labels import get_split_prefix_label, strip_tag_prefix

from sciencebeam_parser.document.layout_document import (
    LayoutToken,
    LayoutLine,
    LayoutBlock,
    LayoutPage,
    LayoutDocument
)
from sciencebeam_parser.models.data import (
    NEW_DOCUMENT_MARKER,
    AppFeaturesContext,
    DocumentFeaturesContext,
    LabeledLayoutModelData,
    LabeledLayoutToken,
    LayoutModelData,
    ModelDataGenerator,
    NewDocumentMarker
)
from sciencebeam_parser.models.extract import ModelSemanticExtractor
from sciencebeam_parser.models.training_data import TeiTrainingDataGenerator, TrainingTeiParser
from sciencebeam_parser.document.semantic_document import SemanticContentWrapper
from sciencebeam_parser.models.model_impl import ModelImpl, T_ModelImplFactory
from sciencebeam_parser.utils.lazy import LazyLoaded, Preloadable


LOGGER = logging.getLogger(__name__)


T = TypeVar('T')
U = TypeVar('U')


@dataclass
class LayoutModelLabel:
    label: str
    label_token_text: str
    layout_line: Optional[LayoutLine] = field(repr=False, default=None)
    layout_token: Optional[LayoutToken] = field(repr=False, default=None)


def iter_entities_including_other(seq: List[str]) -> Iterable[Tuple[str, int, int]]:
    """
    Similar to get_entities, but also other (`O`) tag
    """
    prev_tag = 'O'
    prev_start = 0
    for index, prefixed_tag in enumerate(seq):
        prefix, tag = get_split_prefix_label(prefixed_tag)
        if prefix == 'B' or tag != prev_tag:
            if prev_start < index:
                yield prev_tag, prev_start, index - 1
            prev_tag = tag
            prev_start = index
    if prev_start < len(seq):
        yield prev_tag, prev_start, len(seq) - 1


def get_entities_including_other(seq: List[str]) -> List[Tuple[str, int, int]]:
    return list(iter_entities_including_other(seq))


class LayoutDocumentLabelResult:
    def __init__(
        self,
        layout_document: LayoutDocument,
        layout_model_label_iterable: Iterable[LayoutModelLabel]
    ):
        self.layout_document = layout_document
        self.layout_model_label_list = list(layout_model_label_iterable)
        self.layout_document_labels_by_label: Dict[str, List[LayoutModelLabel]] = (
            defaultdict(list)
        )
        for layout_model_label in self.layout_model_label_list:
            tag_without_prefix = strip_tag_prefix(layout_model_label.label)
            self.layout_document_labels_by_label[tag_without_prefix].append(
                layout_model_label
            )

    def get_available_labels(self) -> Set[str]:
        return set(self.layout_document_labels_by_label.keys())

    def get_layout_document_labels_by_labels(self, labels: List[str]) -> List[LayoutModelLabel]:
        if not labels:
            return []
        if len(labels) == 1:
            return self.layout_document_labels_by_label.get(labels[0], [])
        result: List[LayoutModelLabel] = []
        for label in labels:
            result.extend(self.layout_document_labels_by_label.get(label, []))
        return result

    def get_filtered_document_by_label(self, label: str) -> LayoutDocument:
        return self.get_filtered_document_by_labels([label])

    def get_filtered_document_by_labels(
        self,
        labels: List[str]
    ):  # pylint: disable=too-many-branches
        layout_document = LayoutDocument(pages=[])
        layout_document_labels = self.get_layout_document_labels_by_labels(labels)
        if not layout_document_labels:
            LOGGER.warning(
                'no layout_lines_to_include found for: %r, available keys=%r',
                labels, self.layout_document_labels_by_label.keys()
            )
            return layout_document
        layout_token_ids_to_include = {
            id(layout_document_label.layout_token)
            for layout_document_label in layout_document_labels
            if layout_document_label.layout_token
        }
        LOGGER.debug('layout_tokens_to_include: %s', layout_token_ids_to_include)
        layout_line_ids_to_include: Set[int] = set()
        if not layout_token_ids_to_include:
            layout_line_ids_to_include = {
                id(layout_document_label.layout_line)
                for layout_document_label in layout_document_labels
                if layout_document_label.layout_line
            }
        LOGGER.debug('layout_line_ids_to_include: %s', layout_line_ids_to_include)
        result_page: Optional[LayoutPage] = None
        for page in self.layout_document.pages:  # pylint: disable=too-many-nested-blocks
            result_page = None
            result_block: Optional[LayoutBlock] = None
            for block in page.blocks:
                result_block = None
                for line in block.lines:
                    accepted_line: Optional[LayoutLine] = None
                    if layout_token_ids_to_include:
                        accepted_tokens: List[LayoutToken] = []
                        for token in line.tokens:
                            if id(token) in layout_token_ids_to_include:
                                accepted_tokens.append(token)
                        if not accepted_tokens:
                            continue
                        if len(line.tokens) == accepted_tokens:
                            accepted_line = line
                        else:
                            accepted_line = LayoutLine(tokens=accepted_tokens)
                    else:
                        if id(line) not in layout_line_ids_to_include:
                            continue
                        accepted_line = line
                    if result_page is None:
                        result_page = LayoutPage(blocks=[], meta=page.meta)
                        layout_document.pages.append(result_page)
                    if result_block is None:
                        result_block = LayoutBlock(lines=[])
                        result_page.blocks.append(result_block)
                    result_block.lines.append(accepted_line)
        return layout_document


def iter_entity_layout_blocks_for_labeled_layout_tokens(
    labeled_layout_tokens: Iterable[LabeledLayoutToken]
) -> Iterable[Tuple[str, LayoutBlock]]:
    layout_tokens = [result.layout_token for result in labeled_layout_tokens]
    labels = [result.label for result in labeled_layout_tokens]
    LOGGER.debug('layout_tokens: %s', layout_tokens)
    LOGGER.debug('labels: %s', labels)
    for tag, start, end in get_entities_including_other(list(labels)):
        yield tag, LayoutBlock.for_tokens(layout_tokens[start:end + 1])


def iter_entity_values_predicted_labels(
    tag_result: List[Tuple[str, str]]
) -> Iterable[Tuple[str, str]]:
    tokens, labels = zip(*tag_result)
    LOGGER.debug('tokens: %s', tokens)
    LOGGER.debug('labels: %s', labels)
    for tag, start, end in get_entities_including_other(list(labels)):
        yield tag, ' '.join(tokens[start:end + 1])


def iter_labeled_layout_token_for_layout_model_label(
    layout_model_label_iterable: Iterable[LayoutModelLabel]
) -> Iterable[LabeledLayoutToken]:
    for layout_model_label in layout_model_label_iterable:
        layout_token = layout_model_label.layout_token
        assert layout_token is not None
        yield LabeledLayoutToken(
            layout_model_label.label,
            layout_token
        )


def iter_data_lines_for_model_data_iterables(
    model_data_iterables: Iterable[Iterable[LayoutModelData]]
) -> Iterable[str]:
    for index, model_data_list in enumerate(model_data_iterables):
        if index > 0:
            yield ''
        for model_data in model_data_list:
            yield model_data.data_line


class Model(ABC, Preloadable):
    def __init__(
        self,
        model_impl_factory: Optional[T_ModelImplFactory],
        model_config: Optional[dict] = None
    ) -> None:
        self._model_impl_factory = model_impl_factory
        self._lazy_model_impl = LazyLoaded[ModelImpl](self._load_model_impl)
        self.model_config = model_config or {}

    def __repr__(self) -> str:
        return '%s(model_config=%r, loaded=%r)' % (
            type(self).__name__, self.model_config, self._lazy_model_impl.is_loaded
        )

    @abstractmethod
    def get_data_generator(
        self,
        document_features_context: DocumentFeaturesContext
    ) -> ModelDataGenerator:
        pass

    # @abstractmethod
    def get_semantic_extractor(self) -> ModelSemanticExtractor:
        raise NotImplementedError()

    # @abstractmethod
    def get_tei_training_data_generator(self) -> TeiTrainingDataGenerator:
        raise NotImplementedError()

    # @abstractmethod
    def get_training_tei_parser(self) -> TrainingTeiParser:
        raise NotImplementedError()

    def _load_model_impl(self) -> ModelImpl:
        assert self._model_impl_factory, 'model impl factory required'
        LOGGER.info('creating model impl: %r', self._model_impl_factory)
        model_impl = self._model_impl_factory()
        if not isinstance(model_impl, ModelImpl):
            raise TypeError('invalid model impl type: %r' % model_impl)
        return model_impl

    @property
    def model_impl(self) -> ModelImpl:
        was_loaded = self._lazy_model_impl.is_loaded
        model_impl = self._lazy_model_impl.get()
        if was_loaded:
            LOGGER.info('model impl already loaded: %r', model_impl)
        return model_impl

    def preload(self):
        model_impl = self.model_impl
        model_impl.preload()

    def iter_semantic_content_for_entity_blocks(
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        return self.get_semantic_extractor().iter_semantic_content_for_entity_blocks(
            entity_tokens,
            **kwargs
        )

    def predict_labels(
        self,
        texts: List[List[str]],
        features: List[List[List[str]]],
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        return self.model_impl.predict_labels(texts, features, output_format)

    def _iter_flat_label_model_data_lists_to(  # pylint: disable=too-many-locals
        self,
        model_data_list_iterable: Iterable[Sequence[LayoutModelData]],
        item_factory: Callable[[str, LayoutModelData], T]
    ) -> Iterable[Union[T, NewDocumentMarker]]:
        # Note: currently we do need a list
        model_data_lists = list(model_data_list_iterable)
        if not model_data_lists:
            return
        data_lines = list(iter_data_lines_for_model_data_iterables(
            model_data_lists
        ))
        texts, features = load_data_crf_lines(data_lines)
        texts = texts.tolist()
        features = features.tolist()
        tag_result = self.predict_labels(
            texts=texts, features=features, output_format=None
        )
        if not tag_result:
            return
        if len(tag_result) != len(model_data_lists):
            raise AssertionError('tag result does not match number of docs: %d != %d' % (
                len(tag_result), len(model_data_lists)
            ))
        for index, (doc_tag_result, model_data_list) in enumerate(
            zip(tag_result, model_data_lists)
        ):
            if index > 0:
                yield NEW_DOCUMENT_MARKER
            if len(doc_tag_result) != len(model_data_list):
                raise AssertionError('doc tag result does not match data: %d != %d' % (
                    len(doc_tag_result), len(model_data_list)
                ))
            for token_tag_result, token_model_data in zip(doc_tag_result, model_data_list):
                label_token_text, token_label = token_tag_result
                if label_token_text != token_model_data.label_token_text:
                    raise AssertionError(
                        f'actual: {repr(label_token_text)}'
                        f', expected: {repr(token_model_data.label_token_text)}'
                    )
                yield item_factory(
                    token_label,
                    token_model_data
                )

    def _iter_stacked_label_model_data_lists_to(
        self,
        model_data_list_iterable: Iterable[Sequence[LayoutModelData]],
        item_factory: Callable[[str, LayoutModelData], T]
    ) -> Iterable[Sequence[T]]:
        # Note: currently we do need a list
        model_data_lists = list(model_data_list_iterable)
        if not model_data_lists:
            return
        doc_items: List[T] = []
        result_doc_count = 0
        for item in self._iter_flat_label_model_data_lists_to(
            model_data_lists,
            item_factory=item_factory
        ):
            if isinstance(item, NewDocumentMarker):
                yield doc_items
                doc_items = []
                result_doc_count += 1
                continue
            doc_items.append(item)
        if result_doc_count < len(model_data_lists):
            yield doc_items

    def iter_label_layout_documents(
        self,
        layout_documents: List[LayoutDocument],
        app_features_context: AppFeaturesContext
    ) -> Iterable[List[LayoutModelLabel]]:
        doc_layout_model_labels: List[LayoutModelLabel] = []
        result_doc_count = 0
        for layout_model_label in self._iter_label_layout_documents(
            layout_documents,
            app_features_context=app_features_context
        ):
            if isinstance(layout_model_label, NewDocumentMarker):
                yield doc_layout_model_labels
                doc_layout_model_labels = []
                result_doc_count += 1
                continue
            doc_layout_model_labels.append(layout_model_label)
        if result_doc_count < len(layout_documents):
            yield doc_layout_model_labels

    def iter_label_layout_document(
        self,
        layout_document: LayoutDocument,
        app_features_context: AppFeaturesContext
    ) -> Iterable[LayoutModelLabel]:
        for layout_model_label in self._iter_label_layout_documents(
            [layout_document],
            app_features_context=app_features_context
        ):
            assert isinstance(layout_model_label, LayoutModelLabel)
            yield layout_model_label

    def _iter_label_layout_documents(  # pylint: disable=too-many-locals
        self,
        layout_documents: Iterable[LayoutDocument],
        app_features_context: AppFeaturesContext
    ) -> Iterable[Union[LayoutModelLabel, NewDocumentMarker]]:
        data_generator = self.get_data_generator(
            document_features_context=DocumentFeaturesContext(
                app_features_context=app_features_context
            )
        )
        model_data_lists = [
            list(data_generator.iter_model_data_for_layout_document(
                layout_document
            ))
            for layout_document in layout_documents
        ]
        return self._iter_flat_label_model_data_lists_to(
            model_data_lists,
            lambda label, model_data: LayoutModelLabel(
                label=label,
                label_token_text=model_data.label_token_text,
                layout_line=model_data.layout_line,
                layout_token=model_data.layout_token
            )
        )

    def iter_labeled_model_data_list_for_model_data_list_iterable(
        self,
        model_data_list_iterable: Iterable[Sequence[LayoutModelData]]
    ) -> Iterable[Sequence[LabeledLayoutModelData]]:
        return self._iter_stacked_label_model_data_lists_to(
            model_data_list_iterable,
            lambda label, model_data: LabeledLayoutModelData.from_model_data(
                model_data,
                label=label
            )
        )

    def get_label_layout_document_result(
        self,
        layout_document: LayoutDocument,
        app_features_context: AppFeaturesContext
    ) -> LayoutDocumentLabelResult:
        return LayoutDocumentLabelResult(
            layout_document=layout_document,
            layout_model_label_iterable=self.iter_label_layout_document(
                layout_document,
                app_features_context=app_features_context
            )
        )

    def iter_predict_labels_for_layout_document(
        self,
        layout_document: LayoutDocument,
        app_features_context: AppFeaturesContext
    ) -> Iterable[LabeledLayoutToken]:
        # Note: this should get merged with Model.iter_label_layout_document
        yield from iter_labeled_layout_token_for_layout_model_label(
            self.iter_label_layout_document(
                layout_document,
                app_features_context=app_features_context
            )
        )

    def predict_labels_for_layout_document(
        self,
        layout_document: LayoutDocument,
        app_features_context: AppFeaturesContext
    ) -> List[LabeledLayoutToken]:
        return list(self.iter_predict_labels_for_layout_document(
            layout_document,
            app_features_context=app_features_context
        ))

    def predict_labels_for_layout_documents(
        self,
        layout_documents: List[LayoutDocument],
        app_features_context: AppFeaturesContext
    ) -> List[List[LabeledLayoutToken]]:
        return [
            list(iter_labeled_layout_token_for_layout_model_label(
                layout_model_labels
            ))
            for layout_model_labels in self.iter_label_layout_documents(
                layout_documents,
                app_features_context=app_features_context
            )
        ]

    def iter_entity_layout_blocks_for_labeled_layout_tokens(
        self,
        labeled_layout_tokens: Iterable[LabeledLayoutToken]
    ) -> Iterable[Tuple[str, LayoutBlock]]:
        return iter_entity_layout_blocks_for_labeled_layout_tokens(labeled_layout_tokens)

    def iter_semantic_content_for_labeled_layout_tokens(
        self,
        labeled_layout_tokens: Iterable[LabeledLayoutToken],
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        return self.iter_semantic_content_for_entity_blocks(
            self.iter_entity_layout_blocks_for_labeled_layout_tokens(
                labeled_layout_tokens
            ),
            **kwargs
        )
