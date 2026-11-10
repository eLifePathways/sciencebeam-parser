from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Sequence

import PIL.Image

from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.utils.lazy import LazyLoaded, Preloadable


DEFAULT_MAX_OVERLAP_RATIO = 0.1


class ComputerVisionModelInstance(ABC):
    @abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        pass

    @abstractmethod
    def get_type_name(self) -> str:
        pass


@dataclass
class SimpleComputerVisionModelInstance(ComputerVisionModelInstance):
    bounding_box: BoundingBox
    type_name: str

    def get_bounding_box(self) -> BoundingBox:
        return self.bounding_box

    def get_type_name(self) -> str:
        return self.type_name


def is_bounding_box_overlapping_with_any_bounding_boxes(
    bounding_box: BoundingBox,
    other_bounding_boxes: Sequence[BoundingBox],
    max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO
) -> bool:
    bounding_box_area = bounding_box.area
    for other_bounding_box in other_bounding_boxes:
        intersection_bounding_box = bounding_box.intersection(
            other_bounding_box
        )
        if not intersection_bounding_box:
            continue
        if intersection_bounding_box.area / bounding_box_area >= max_overlap_ratio:
            return True
    return False


def get_non_overlapping_instances(
    instances: Sequence[ComputerVisionModelInstance],
    max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO
) -> Sequence[ComputerVisionModelInstance]:
    """Keeps the first of any instances covering the same area.

    A detector is free to report the same region more than once -- a figure and
    the panel inside it, or the same logo at two confidences -- and the caller
    orders the instances by what should win before asking for this.
    """
    accepted_instances: List[ComputerVisionModelInstance] = []
    accepted_bounding_boxes: List[BoundingBox] = []
    for instance in instances:
        bounding_box = instance.get_bounding_box()
        if is_bounding_box_overlapping_with_any_bounding_boxes(
            bounding_box,
            accepted_bounding_boxes,
            max_overlap_ratio=max_overlap_ratio
        ):
            continue
        accepted_instances.append(instance)
        accepted_bounding_boxes.append(bounding_box)
    return accepted_instances


class ComputerVisionModelResult(ABC):
    @abstractmethod
    def get_instances_by_type_names(
        self,
        type_names: Sequence[str]
    ) -> Sequence[ComputerVisionModelInstance]:
        pass

    def get_instances_by_type_name(
        self,
        type_name: str
    ) -> Sequence[ComputerVisionModelInstance]:
        return self.get_instances_by_type_names([type_name])


class SimpleComputerVisionModelResult(ComputerVisionModelResult):
    def __init__(self, instances: Sequence[ComputerVisionModelInstance]):
        super().__init__()
        self.instances = instances

    def __repr__(self) -> str:
        return '%s(instances=%r)' % (type(self).__name__, self.instances)

    def get_instances_by_type_names(
        self,
        type_names: Sequence[str]
    ) -> Sequence[ComputerVisionModelInstance]:
        return [
            instance
            for instance in self.instances
            if instance.get_type_name() in type_names
        ]


class ComputerVisionModel(ABC, Preloadable):
    @abstractmethod
    def predict_single(self, image: PIL.Image.Image) -> ComputerVisionModelResult:
        pass


T_ComputerVisionModelFactory = Callable[[], ComputerVisionModel]


class LazyComputerVisionModel(ComputerVisionModel):
    def __init__(self, factory: T_ComputerVisionModelFactory) -> None:
        super().__init__()
        self._lazy_model = LazyLoaded[ComputerVisionModel](factory)

    def __repr__(self) -> str:
        return '%s(factory=%r, loaded=%r)' % (
            type(self).__name__, self._lazy_model.factory, self._lazy_model.is_loaded
        )

    @property
    def cv_model(self) -> ComputerVisionModel:
        return self._lazy_model.get()

    def preload(self):
        self.cv_model.preload()

    def predict_single(self, image: PIL.Image.Image) -> ComputerVisionModelResult:
        return self.cv_model.predict_single(image)
