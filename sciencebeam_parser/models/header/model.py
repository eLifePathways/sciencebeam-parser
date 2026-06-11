import logging
from typing import Iterable, Tuple

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import SemanticDocument
from sciencebeam_parser.models.data import (
    DocumentFeaturesContext
)
from sciencebeam_parser.models.header.data import HeaderDataGenerator
from sciencebeam_parser.models.header.extract import HeaderSemanticExtractor
from sciencebeam_parser.models.header.training_data import (
    HeaderTeiTrainingDataGenerator,
    HeaderTrainingTeiParser
)
from sciencebeam_parser.models.model import Model


LOGGER = logging.getLogger(__name__)


class HeaderModel(Model):
    def get_data_generator(
        self,
        document_features_context: DocumentFeaturesContext
    ) -> HeaderDataGenerator:
        return HeaderDataGenerator(
            document_features_context=document_features_context,
            persist_indentation_reference_across_blocks=self.model_config.get(
                'persist_indentation_reference_across_blocks', False
            )
        )

    def get_semantic_extractor(self) -> HeaderSemanticExtractor:
        return HeaderSemanticExtractor()

    def get_tei_training_data_generator(self) -> HeaderTeiTrainingDataGenerator:
        return HeaderTeiTrainingDataGenerator()

    def get_training_tei_parser(self) -> HeaderTrainingTeiParser:
        return HeaderTrainingTeiParser()

    def update_semantic_document_with_entity_blocks(
        self,
        document: SemanticDocument,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]]
    ):
        semantic_content_iterable = (
            self.get_semantic_extractor()
            .iter_semantic_content_for_entity_blocks(entity_tokens)
        )
        front = document.front
        for semantic_content in semantic_content_iterable:
            front.add_content(semantic_content)
