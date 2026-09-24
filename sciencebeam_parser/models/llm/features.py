from typing import Callable, List, Mapping

from sciencebeam_parser.models.data import (
    AppFeaturesContext,
    DocumentFeaturesContext,
    ModelDataGenerator
)


def _reference_segmenter_data_generator() -> ModelDataGenerator:
    from sciencebeam_parser.models.reference_segmenter.data import (  # noqa pylint: disable=import-outside-toplevel
        ReferenceSegmenterDataGenerator
    )
    return ReferenceSegmenterDataGenerator(DocumentFeaturesContext(
        app_features_context=AppFeaturesContext(
            country_lookup=None, first_name_lookup=None, last_name_lookup=None
        )
    ))


def _segmentation_data_generator() -> ModelDataGenerator:
    from sciencebeam_parser.models.segmentation.data import (  # noqa pylint: disable=import-outside-toplevel
        SegmentationDataGenerator
    )
    # Only `feature_names` is read here, and the flag does not change it — it
    # changes which token the text columns are taken from.
    return SegmentationDataGenerator(
        DocumentFeaturesContext(
            app_features_context=AppFeaturesContext(
                country_lookup=None, first_name_lookup=None, last_name_lookup=None
            )
        ),
        use_first_token_of_block=False
    )


DATA_GENERATOR_BY_TASK: Mapping[str, Callable[[], ModelDataGenerator]] = {
    'reference_segmenter': _reference_segmenter_data_generator,
    'segmentation': _segmentation_data_generator,
}


class UnknownLlmTaskError(ValueError):
    pass


def get_feature_names(task: str) -> List[str]:
    try:
        data_generator_factory = DATA_GENERATOR_BY_TASK[task]
    except KeyError as exc:
        raise UnknownLlmTaskError(
            f'no llm task {task!r}; known: {sorted(DATA_GENERATOR_BY_TASK)}'
        ) from exc
    return data_generator_factory().feature_names


def get_feature_column_index(task: str, feature_name: str) -> int:
    feature_names = get_feature_names(task)
    try:
        return feature_names.index(feature_name) - 1
    except ValueError as exc:
        raise UnknownLlmTaskError(
            f'{task!r} has no feature {feature_name!r}; has: {feature_names}'
        ) from exc
