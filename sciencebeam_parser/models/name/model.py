import logging

from sciencebeam_parser.models.data import (
    DocumentFeaturesContext
)
from sciencebeam_parser.models.name.data import NameDataGenerator
from sciencebeam_parser.models.name.extract import NameSemanticExtractor
from sciencebeam_parser.models.name.training_data import (
    NameTeiTrainingDataGenerator,
    NameTrainingTeiParser
)
from sciencebeam_parser.models.model import Model


LOGGER = logging.getLogger(__name__)


class NameModel(Model):
    def get_data_generator(
        self,
        document_features_context: DocumentFeaturesContext
    ) -> NameDataGenerator:
        return NameDataGenerator(
            document_features_context=document_features_context,
            ignore_line_breaks=self.model_config.get('ignore_line_breaks', False)
        )

    def get_semantic_extractor(self) -> NameSemanticExtractor:
        return NameSemanticExtractor()

    def get_tei_training_data_generator(self) -> NameTeiTrainingDataGenerator:
        return NameTeiTrainingDataGenerator()

    def get_training_tei_parser(self) -> NameTrainingTeiParser:
        return NameTrainingTeiParser()
