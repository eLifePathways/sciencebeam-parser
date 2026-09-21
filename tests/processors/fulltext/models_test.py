import dataclasses
from unittest.mock import MagicMock

import pytest
from sciencebeam_parser.app.context import AppContext
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.processors.fulltext.config import FullTextProcessorConfig

from sciencebeam_parser.processors.fulltext.models import (
    SEQUENCE_MODEL_CLASS_BY_NAME,
    FullTextModels,
    load_models
)


@pytest.fixture(name='sciencebeam_parser_config_app_context')
def _sciencebeam_parser_config_app_context():
    return MagicMock(name='AppContext')


class TestFullTextModels:
    def test_should_preload_models(self):
        mock_models = {
            field.name: MagicMock(name=field.name)
            for field in dataclasses.fields(FullTextModels)
        }
        mock_fulltext_models = FullTextModels(**mock_models)
        mock_fulltext_models.preload()
        for model in mock_models.values():
            model.preload.assert_called()


class TestLoadModels:
    def test_should_load_cv_and_ocr_models_if_enabled(
        self,
        sciencebeam_parser_config: AppConfig,
        sciencebeam_parser_config_app_context: AppContext
    ):
        models = load_models(
            app_config=sciencebeam_parser_config,
            app_context=sciencebeam_parser_config_app_context,
            fulltext_processor_config=(
                FullTextProcessorConfig
                .from_app_config(app_config=sciencebeam_parser_config)
                ._replace(use_cv_model=True, use_ocr_model=True)
            )
        )
        assert models.cv_model is not None
        assert models.ocr_model is not None

    def test_should_not_load_cv_and_ocr_models_if_disabled(
        self,
        sciencebeam_parser_config: AppConfig,
        sciencebeam_parser_config_app_context: AppContext
    ):
        models = load_models(
            app_config=sciencebeam_parser_config,
            app_context=sciencebeam_parser_config_app_context,
            fulltext_processor_config=(
                FullTextProcessorConfig
                .from_app_config(app_config=sciencebeam_parser_config)
                ._replace(use_cv_model=False, use_ocr_model=False)
            )
        )
        assert models.cv_model is None
        assert models.ocr_model is None


class TestSequenceModelClassByName:
    """The map and `load_models` have to agree.

    The map is what tells a caller what a profile costs before any of it is
    built, so a model missing from it is a model the cap does not count.
    """

    def test_should_name_every_sequence_model_load_models_builds(
        self,
        sciencebeam_parser_config_app_context: AppContext
    ):
        built_classes = {}

        def _record(model_config: dict, model_class):
            built_classes[model_config['name']] = model_class
            return MagicMock(name='model')

        load_models(
            app_config=AppConfig({
                'models': {
                    name: {'name': name} for name in SEQUENCE_MODEL_CLASS_BY_NAME
                }
            }),
            app_context=sciencebeam_parser_config_app_context,
            fulltext_processor_config=FullTextProcessorConfig(),
            model_factory=_record
        )
        assert built_classes == dict(SEQUENCE_MODEL_CLASS_BY_NAME)
