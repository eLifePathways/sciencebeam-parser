import logging
from typing import Any, List, Mapping, NamedTuple, Sequence

import PIL.Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.utils.lazy import LazyLoaded
from sciencebeam_parser.cv_models.cv_model import (
    DEFAULT_MAX_OVERLAP_RATIO,
    ComputerVisionModel,
    ComputerVisionModelInstance,
    ComputerVisionModelResult,
    SimpleComputerVisionModelInstance,
    SimpleComputerVisionModelResult,
    get_non_overlapping_instances
)


LOGGER = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = 'docling-project/docling-layout-heron'

DEFAULT_SCORE_THRESHOLD = 0.5

# A page of a scientific article is a few large figures; the logos, icons and
# author portraits a layout detector also reports are an order of magnitude
# smaller. Expressed against the page so it holds at any render dpi.
DEFAULT_MIN_AREA_RATIO = 0.01

# The checkpoints this engine loads are trained on DocLayNet, whose vocabulary
# is not the one `ComputerVisionDocumentGraphicProvider` asks for.
DEFAULT_TYPE_NAME_BY_LABEL: Mapping[str, str] = {
    'picture': 'Figure',
    'table': 'Table'
}


class _ObjectDetectionPipeline(NamedTuple):
    # The concrete classes depend on the checkpoint -- `AutoImageProcessor` and
    # `AutoModelForObjectDetection` are factories, not the types they return.
    image_processor: Any
    model: Any


class TransformersComputerVisionModel(ComputerVisionModel):
    def __init__(
        self,
        config: dict,
        model_path: str = DEFAULT_MODEL_PATH
    ):
        super().__init__()
        self.model_path = model_path
        self.score_threshold = float(config.get('score_threshold', DEFAULT_SCORE_THRESHOLD))
        self.min_area_ratio = float(config.get('min_area_ratio', DEFAULT_MIN_AREA_RATIO))
        self.max_overlap_ratio = float(
            config.get('max_overlap_ratio', DEFAULT_MAX_OVERLAP_RATIO)
        )
        self.type_name_by_label: Mapping[str, str] = dict(
            config.get('type_name_by_label') or DEFAULT_TYPE_NAME_BY_LABEL
        )
        self._lazy_pipeline = LazyLoaded[_ObjectDetectionPipeline](self._load_pipeline)

    def __repr__(self) -> str:
        return '%s(model_path=%r)' % (type(self).__name__, self.model_path)

    def _load_pipeline(self) -> _ObjectDetectionPipeline:
        LOGGER.info('loading model: %r', self.model_path)
        pipeline = _ObjectDetectionPipeline(
            image_processor=AutoImageProcessor.from_pretrained(self.model_path),
            model=AutoModelForObjectDetection.from_pretrained(self.model_path).eval()
        )
        LOGGER.info('loaded model: %r', self.model_path)
        return pipeline

    def preload(self):
        self._lazy_pipeline.get()

    def _iter_instances_by_descending_score(
        self,
        detections: dict,
        label_names: Mapping[int, str],
        min_area: float
    ) -> Sequence[ComputerVisionModelInstance]:
        scored_instances: List[tuple] = []
        for score, label, box in zip(
            detections['scores'], detections['labels'], detections['boxes']
        ):
            type_name = self.type_name_by_label.get(label_names[int(label)])
            if not type_name:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            bounding_box = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
            if bounding_box.area < min_area:
                LOGGER.debug(
                    'ignoring instance below the area floor: %r (score=%.3f)',
                    bounding_box, float(score)
                )
                continue
            scored_instances.append((float(score), SimpleComputerVisionModelInstance(
                bounding_box=bounding_box,
                type_name=type_name
            )))
        return [
            instance
            for _, instance in sorted(scored_instances, key=lambda t: -t[0])
        ]

    def predict_single(self, image: PIL.Image.Image) -> ComputerVisionModelResult:
        image_processor, model = self._lazy_pipeline.get()
        inputs = image_processor(images=image.convert('RGB'), return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        detections = image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[image.height, image.width]]),
            threshold=self.score_threshold
        )[0]
        return SimpleComputerVisionModelResult(get_non_overlapping_instances(
            self._iter_instances_by_descending_score(
                detections,
                label_names=model.config.id2label,
                min_area=self.min_area_ratio * image.width * image.height
            ),
            max_overlap_ratio=self.max_overlap_ratio
        ))
