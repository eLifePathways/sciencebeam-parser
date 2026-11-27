import logging

from sciencebeam_parser.models.data import (
    DEFAULT_FEATURE_FLAVOUR,
    DocumentFeaturesContext
)
from sciencebeam_parser.models.segmentation.data import SegmentationDataGenerator
from sciencebeam_parser.models.model import Model
from sciencebeam_parser.models.segmentation.training_data import (
    SegmentationTeiTrainingDataGenerator,
    SegmentationTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


class SegmentationModel(Model):
    def get_data_generator(
        self,
        document_features_context: DocumentFeaturesContext
    ) -> SegmentationDataGenerator:
        return SegmentationDataGenerator(
            document_features_context=document_features_context,
            use_first_token_of_block=(
                self.model_config.get('use_first_token_of_block', True)
            ),
            feature_flavour=self.model_config.get('feature_flavour', DEFAULT_FEATURE_FLAVOUR)
        )

    def get_tei_training_data_generator(self) -> SegmentationTeiTrainingDataGenerator:
        return SegmentationTeiTrainingDataGenerator()

    def get_training_tei_parser(self) -> SegmentationTrainingTeiParser:
        return SegmentationTrainingTeiParser()
