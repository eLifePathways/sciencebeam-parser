import logging

from sciencebeam_parser.utils.bounding_box import BoundingBox
from sciencebeam_parser.cv_models.cv_model import (
    SimpleComputerVisionModelInstance,
    SimpleComputerVisionModelResult,
    get_non_overlapping_instances
)


LOGGER = logging.getLogger(__name__)


def _instance(bounding_box: BoundingBox, type_name: str = 'Figure'):
    return SimpleComputerVisionModelInstance(
        bounding_box=bounding_box, type_name=type_name
    )


class TestGetNonOverlappingInstances:
    def test_should_keep_instances_that_do_not_overlap(self):
        instances = [
            _instance(BoundingBox(10, 10, 100, 100)),
            _instance(BoundingBox(200, 10, 100, 100))
        ]
        assert list(get_non_overlapping_instances(instances)) == instances

    def test_should_keep_the_first_of_two_overlapping_instances(self):
        first = _instance(BoundingBox(10, 10, 100, 100))
        instances = get_non_overlapping_instances([
            first,
            _instance(BoundingBox(11, 10, 100, 100))
        ])
        assert list(instances) == [first]

    def test_should_keep_an_instance_overlapping_below_the_max_ratio(self):
        instances = [
            _instance(BoundingBox(10, 10, 100, 100)),
            _instance(BoundingBox(105, 10, 100, 100))
        ]
        assert list(get_non_overlapping_instances(
            instances, max_overlap_ratio=0.1
        )) == instances


class TestSimpleComputerVisionModelResult:
    def test_should_filter_by_type_name(self):
        figure = _instance(BoundingBox(10, 10, 100, 100), type_name='Figure')
        table = _instance(BoundingBox(200, 10, 100, 100), type_name='Table')
        result = SimpleComputerVisionModelResult([figure, table])
        assert list(result.get_instances_by_type_name('Table')) == [table]
        assert list(result.get_instances_by_type_names(['Figure', 'Table'])) == [figure, table]
