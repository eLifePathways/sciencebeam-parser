from unittest.mock import MagicMock

from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT,
    FEATURE_FLAVOUR_GROBID,
    FEATURE_FLAVOUR_SCIENCEBEAM
)
from sciencebeam_parser.models.header.data import HeaderDataGenerator
from sciencebeam_parser.models.header.model import HeaderModel


def _get_data_generator(**model_config) -> HeaderDataGenerator:
    model = HeaderModel(MagicMock(), model_config=model_config)
    return model.get_data_generator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)


def _is_persisting_indentation(data_generator: HeaderDataGenerator) -> bool:
    # pylint: disable=protected-access
    return data_generator._persist_indentation_reference_across_blocks


class TestHeaderModelFeatureFlavour:
    def test_should_default_to_sciencebeam_features(self):
        data_generator = _get_data_generator()
        assert data_generator.feature_flavour == FEATURE_FLAVOUR_SCIENCEBEAM
        assert not _is_persisting_indentation(data_generator)

    def test_should_pass_grobid_flavour_and_persist_indentation(self):
        data_generator = _get_data_generator(feature_flavour=FEATURE_FLAVOUR_GROBID)
        assert data_generator.feature_flavour == FEATURE_FLAVOUR_GROBID
        assert _is_persisting_indentation(data_generator)

    def test_should_let_an_explicit_indentation_setting_win_over_the_flavour(self):
        data_generator = _get_data_generator(
            feature_flavour=FEATURE_FLAVOUR_GROBID,
            persist_indentation_reference_across_blocks=False
        )
        assert not _is_persisting_indentation(data_generator)
