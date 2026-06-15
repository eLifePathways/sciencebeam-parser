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
from sciencebeam_parser.document.layout_document import LayoutDocument
from sciencebeam_parser.utils.tokenizer import get_subdigit_tokenized_tokens


LOGGER = logging.getLogger(__name__)


class CitationModel(Model):
    def retokenize_layout_documents(
        self,
        layout_documents: List[LayoutDocument]
    ) -> List[LayoutDocument]:
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
        return super().predict_labels_for_layout_documents(
            self.retokenize_layout_documents(layout_documents),
            app_features_context=app_features_context
        )

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
