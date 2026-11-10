import logging
from typing import Sequence
from unittest.mock import MagicMock

import PIL.Image
import pytest
import torch

from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.utils.lazy import LazyLoaded
from sciencebeam_parser.cv_models.cv_model_factory import get_cv_model_for_config
from sciencebeam_parser.cv_models.transformers_cv_model import (
    TransformersComputerVisionModel,
    _ObjectDetectionPipeline
)


LOGGER = logging.getLogger(__name__)


PAGE_WIDTH = 850
PAGE_HEIGHT = 1100

LABEL_NAMES = {0: 'picture', 1: 'table', 2: 'text'}


def _get_model_with_detections(
    config: dict,
    scores: Sequence[float],
    labels: Sequence[int],
    boxes: Sequence[Sequence[float]]
) -> TransformersComputerVisionModel:
    image_processor = MagicMock(name='image_processor')
    image_processor.post_process_object_detection.return_value = [{
        'scores': torch.tensor(scores),
        'labels': torch.tensor(labels),
        'boxes': torch.tensor(boxes)
    }]
    model = MagicMock(name='model')
    model.config.id2label = LABEL_NAMES
    cv_model = TransformersComputerVisionModel(config)
    cv_model._lazy_pipeline = LazyLoaded(  # pylint: disable=protected-access
        lambda: _ObjectDetectionPipeline(image_processor=image_processor, model=model)
    )
    return cv_model


def _get_page_image() -> PIL.Image.Image:
    return PIL.Image.new('RGB', (PAGE_WIDTH, PAGE_HEIGHT), 'white')


class TestTransformersComputerVisionModel:
    def test_should_map_the_label_to_the_type_name_the_provider_asks_for(self):
        cv_model = _get_model_with_detections(
            {'min_area_ratio': 0.0},
            scores=[0.9, 0.9],
            labels=[0, 1],
            boxes=[[10, 10, 110, 110], [200, 10, 300, 110]]
        )
        result = cv_model.predict_single(_get_page_image())
        assert [
            instance.get_type_name()
            for instance in result.get_instances_by_type_names(['Figure', 'Table'])
        ] == ['Figure', 'Table']

    def test_should_drop_a_label_that_is_not_mapped(self):
        cv_model = _get_model_with_detections(
            {'min_area_ratio': 0.0},
            scores=[0.9],
            labels=[2],
            boxes=[[10, 10, 110, 110]]
        )
        result = cv_model.predict_single(_get_page_image())
        assert not result.get_instances_by_type_names(['Figure', 'Table'])

    def test_should_convert_the_corners_to_a_bounding_box(self):
        cv_model = _get_model_with_detections(
            {'min_area_ratio': 0.0},
            scores=[0.9],
            labels=[0],
            boxes=[[10, 20, 110, 220]]
        )
        result = cv_model.predict_single(_get_page_image())
        assert [
            instance.get_bounding_box()
            for instance in result.get_instances_by_type_name('Figure')
        ] == [BoundingBox(10, 20, 100, 200)]

    def test_should_drop_an_instance_below_the_area_floor(self):
        # 1% of the page, against a detection covering 0.4% of it
        cv_model = _get_model_with_detections(
            {'min_area_ratio': 0.01},
            scores=[0.9],
            labels=[0],
            boxes=[[10, 10, 74, 74]]
        )
        result = cv_model.predict_single(_get_page_image())
        assert not result.get_instances_by_type_name('Figure')

    def test_should_keep_the_highest_scoring_of_two_overlapping_instances(self):
        cv_model = _get_model_with_detections(
            {'min_area_ratio': 0.0},
            scores=[0.6, 0.9],
            labels=[0, 0],
            boxes=[[10, 10, 410, 110], [10, 10, 110, 110]]
        )
        result = cv_model.predict_single(_get_page_image())
        assert [
            instance.get_bounding_box()
            for instance in result.get_instances_by_type_name('Figure')
        ] == [BoundingBox(10, 10, 100, 100)]

    def test_should_pass_the_image_size_as_the_target_size(self):
        cv_model = _get_model_with_detections(
            {}, scores=[], labels=[], boxes=[]
        )
        cv_model.predict_single(_get_page_image())
        pipeline = cv_model._lazy_pipeline.get()  # pylint: disable=protected-access
        _, kwargs = pipeline.image_processor.post_process_object_detection.call_args
        assert kwargs['target_sizes'].tolist() == [[PAGE_HEIGHT, PAGE_WIDTH]]


@pytest.mark.slow
class TestShippedCvModel:
    """Covers the shipped `cv_models.default` against the checkpoint it names.

    What it asks of the model is deliberately weak -- how well a checkpoint
    finds a figure is a benchmark question, and a page drawn here is nothing
    like the pages it was trained on. What it does catch is the configuration
    being unloadable at all, which is how computer-vision extraction came to be
    broken in the first place.
    """

    @pytest.fixture(name='shipped_cv_model', scope='class')
    def _shipped_cv_model(self, sciencebeam_parser_config: dict):
        return get_cv_model_for_config(
            sciencebeam_parser_config['cv_models']['default']
        )

    def test_should_predict_using_the_shipped_model(self, shipped_cv_model):
        result = shipped_cv_model.predict_single(_get_page_image())
        LOGGER.debug('result: %r', result)
        assert result.get_instances_by_type_names(['Figure', 'Table']) is not None

    def test_should_map_labels_the_shipped_checkpoint_actually_has(
        self, shipped_cv_model, sciencebeam_parser_config: dict
    ):
        shipped_cv_model.preload()
        model = shipped_cv_model._lazy_pipeline.get().model  # pylint: disable=protected-access
        checkpoint_labels = set(model.config.id2label.values())
        configured_labels = set(
            sciencebeam_parser_config['cv_models']['default']['type_name_by_label']
        )
        assert configured_labels <= checkpoint_labels
