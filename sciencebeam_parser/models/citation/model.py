import logging
from typing import List
from typing_extensions import override

from sciencebeam_parser.models.citation.training_data import (
    CitationTeiTrainingDataGenerator,
    CitationTrainingTeiParser
)

from sciencebeam_parser.models.data import (
    AppFeaturesContext,
    DocumentFeaturesContext
)
from sciencebeam_parser.models.model import LabeledLayoutToken, Model
from sciencebeam_parser.models.citation.data import CitationDataGenerator
from sciencebeam_parser.models.citation.extract import CitationSemanticExtractor
from sciencebeam_parser.document.layout_document import LayoutDocument, LayoutToken
from sciencebeam_parser.utils.tokenizer import get_subdigit_tokenized_tokens


LOGGER = logging.getLogger(__name__)

# GROBID's CitationParser strips these before labelling a reference
LEADING_CHARACTERS_TO_REMOVE = '[({.,])}:'


def remove_leading_punctuation_tokens(layout_document: LayoutDocument) -> LayoutDocument:
    """Return the document without the single-character punctuation tokens it starts with."""
    is_leading = True

    def _keep_after_leading(token: LayoutToken) -> List[LayoutToken]:
        nonlocal is_leading
        if is_leading and len(token.text) == 1 and token.text in LEADING_CHARACTERS_TO_REMOVE:
            return []
        is_leading = False
        return [token]

    return layout_document.flat_map_layout_tokens(_keep_after_leading).remove_empty_blocks()


def _has_tokens(layout_document: LayoutDocument) -> bool:
    return next(iter(layout_document.iter_all_tokens()), None) is not None


class CitationModel(Model):
    def retokenize_layout_documents(
        self,
        layout_documents: List[LayoutDocument]
    ) -> List[LayoutDocument]:
        layout_documents = [
            remove_leading_punctuation_tokens(doc) for doc in layout_documents
        ]
        if self.model_config.get('retokenize_subdigits', True):
            return [
                doc.retokenize(tokenize_fn=get_subdigit_tokenized_tokens)
                for doc in layout_documents
            ]
        return layout_documents

    @override
    def predict_labels_for_layout_documents(
        self,
        layout_documents: List[LayoutDocument],
        app_features_context: AppFeaturesContext
    ) -> List[List[LabeledLayoutToken]]:
        retokenized_documents = self.retokenize_layout_documents(layout_documents)
        non_empty_documents = [
            doc for doc in retokenized_documents if _has_tokens(doc)
        ]
        labeled_iterator = iter(super().predict_labels_for_layout_documents(
            non_empty_documents,
            app_features_context=app_features_context
        ) if non_empty_documents else [])
        return [
            next(labeled_iterator) if _has_tokens(doc) else []
            for doc in retokenized_documents
        ]

    def get_data_generator(
        self,
        document_features_context: DocumentFeaturesContext
    ) -> CitationDataGenerator:
        return CitationDataGenerator(
            document_features_context=document_features_context
        )

    def get_semantic_extractor(self) -> CitationSemanticExtractor:
        return CitationSemanticExtractor()

    def get_tei_training_data_generator(self) -> CitationTeiTrainingDataGenerator:
        return CitationTeiTrainingDataGenerator()

    def get_training_tei_parser(self) -> CitationTrainingTeiParser:
        return CitationTrainingTeiParser()
