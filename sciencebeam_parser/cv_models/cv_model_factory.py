import functools

from sciencebeam_parser.cv_models.cv_model import (
    ComputerVisionModel,
    LazyComputerVisionModel
)


class EngineNames:
    TRANSFORMERS = 'transformers'


def get_transformers_model_for_path(config: dict, path: str) -> ComputerVisionModel:
    from sciencebeam_parser.cv_models.transformers_cv_model import (  # noqa pylint: disable=import-outside-toplevel
        TransformersComputerVisionModel
    )
    return TransformersComputerVisionModel(config, path)


def get_engine_name_for_config(config: dict) -> str:
    engine_name = config.get('engine')
    if engine_name:
        return engine_name
    return EngineNames.TRANSFORMERS


def get_cv_model_for_config(config: dict) -> ComputerVisionModel:
    path = config['path']
    engine_name = get_engine_name_for_config(config)
    if engine_name == EngineNames.TRANSFORMERS:
        return get_transformers_model_for_path(config, path)
    raise RuntimeError('invalid engine name: %r' % engine_name)


def get_lazy_cv_model_for_config(config: dict) -> ComputerVisionModel:
    return LazyComputerVisionModel(
        functools.partial(get_cv_model_for_config, config)
    )
