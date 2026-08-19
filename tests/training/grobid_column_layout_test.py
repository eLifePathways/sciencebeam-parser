import dataclasses
import logging
from typing import Dict, Iterable

import pytest

from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT,
    ModelDataGenerator
)
from sciencebeam_parser.models.model import Model
from sciencebeam_parser.processors.fulltext.models import FullTextModels
from sciencebeam_parser.training.grobid_column_layout import (
    GrobidColumnLayout,
    LabelSlot,
    PLACEHOLDER_COLUMN_NAME,
    get_grobid_column_layout_for_model_name,
    get_validated_training_data_feature_indices,
    load_grobid_column_layout_by_name
)

from tests.processors.fulltext.model_mocks import MockFullTextModels


LOGGER = logging.getLogger(__name__)


# The column count of a training data line, i.e. the token, the feature columns
# and the label. Recorded independently of the layout file, so that editing the
# layout without also changing GROBID's corpus fails here.
EXPECTED_TRAINING_DATA_COLUMN_COUNT_BY_MODEL_NAME = {
    'segmentation': 34,
    'header': 33,
    'fulltext': 28,
    'figure': 28,
    'table': 28,
    'reference_segmenter': 29,
    'citation': 30,
    'affiliation_address': 22,
    'name_header': 21,
    'name_citation': 21
}


EXPECTED_TRAINING_DATA_COLUMN_COUNT_WITH_EXTRA_COLUMNS_BY_MODEL_NAME = {
    **EXPECTED_TRAINING_DATA_COLUMN_COUNT_BY_MODEL_NAME,
    'segmentation': 35
}


LAYOUT_BY_MODEL_NAME = load_grobid_column_layout_by_name()

MODEL_NAMES = sorted(LAYOUT_BY_MODEL_NAME)


def iter_sequence_model_names() -> Iterable[str]:
    for field in dataclasses.fields(FullTextModels):
        if not isinstance(field.type, type) or not issubclass(field.type, Model):
            continue
        assert field.name.endswith('_model')
        yield field.name[:-len('_model')]


def get_data_generator_by_model_name() -> Dict[str, ModelDataGenerator]:
    fulltext_models = MockFullTextModels()
    return {
        model_name: fulltext_models.get_sequence_model_by_name(
            model_name
        ).get_data_generator(
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT
        )
        for model_name in iter_sequence_model_names()
    }


DATA_GENERATOR_BY_MODEL_NAME = get_data_generator_by_model_name()


class TestGrobidColumnLayoutFile:
    def test_should_cover_every_sequence_model(self):
        assert sorted(iter_sequence_model_names()) == MODEL_NAMES

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_should_name_the_data_generator_the_model_uses(self, model_name: str):
        data_generator = DATA_GENERATOR_BY_MODEL_NAME[model_name]
        assert type(data_generator).__name__ == LAYOUT_BY_MODEL_NAME[model_name].generator

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_should_match_the_columns_the_data_generator_emits(self, model_name: str):
        layout = LAYOUT_BY_MODEL_NAME[model_name]
        data_generator = DATA_GENERATOR_BY_MODEL_NAME[model_name]
        assert (
            list(data_generator.feature_names)
            == layout.get_data_generator_column_names()
        )

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_should_start_the_columns_with_the_token(self, model_name: str):
        assert LAYOUT_BY_MODEL_NAME[model_name].columns[0] == 'token_text'

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_should_not_record_the_placeholder_as_a_grobid_column(self, model_name: str):
        layout = LAYOUT_BY_MODEL_NAME[model_name]
        assert PLACEHOLDER_COLUMN_NAME not in layout.columns
        assert PLACEHOLDER_COLUMN_NAME not in layout.extra_columns

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_should_record_a_reference_training_corpus(self, model_name: str):
        assert LAYOUT_BY_MODEL_NAME[model_name].reference_training_corpus

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_should_emit_the_expected_training_data_column_count(self, model_name: str):
        layout = LAYOUT_BY_MODEL_NAME[model_name]
        assert (
            len(layout.get_training_data_column_names()) + 1
            == EXPECTED_TRAINING_DATA_COLUMN_COUNT_BY_MODEL_NAME[model_name]
        )

    @pytest.mark.parametrize('model_name', MODEL_NAMES)
    def test_should_emit_the_expected_column_count_with_extra_columns(self, model_name: str):
        layout = LAYOUT_BY_MODEL_NAME[model_name]
        assert (
            len(layout.get_training_data_column_names(include_extra_columns=True)) + 1
            == EXPECTED_TRAINING_DATA_COLUMN_COUNT_WITH_EXTRA_COLUMNS_BY_MODEL_NAME[model_name]
        )

    def test_should_only_record_extra_columns_for_segmentation(self):
        assert {
            model_name
            for model_name, layout in LAYOUT_BY_MODEL_NAME.items()
            if layout.extra_columns
        } == {'segmentation'}


class TestGrobidColumnLayout:
    def test_should_reject_an_unknown_label_slot(self):
        with pytest.raises(ValueError):
            GrobidColumnLayout(
                name='model1',
                generator='DataGenerator1',
                label_slot='other',
                columns=('token_text', 'feature1')
            )

    def test_should_replace_the_placeholder_with_the_label_when_filled(self):
        layout = GrobidColumnLayout(
            name='model1',
            generator='DataGenerator1',
            label_slot=LabelSlot.FILLED,
            columns=('token_text', 'feature1')
        )
        assert layout.get_data_generator_column_names() == [
            'token_text', 'feature1', PLACEHOLDER_COLUMN_NAME
        ]
        assert layout.get_training_data_column_names() == ['token_text', 'feature1']
        assert layout.get_training_data_feature_indices() == [0]

    def test_should_keep_the_placeholder_before_the_label_when_unfilled(self):
        layout = GrobidColumnLayout(
            name='model1',
            generator='DataGenerator1',
            label_slot=LabelSlot.UNFILLED,
            columns=('token_text', 'feature1')
        )
        assert layout.get_data_generator_column_names() == [
            'token_text', 'feature1', PLACEHOLDER_COLUMN_NAME
        ]
        assert layout.get_training_data_column_names() == [
            'token_text', 'feature1', PLACEHOLDER_COLUMN_NAME
        ]
        assert layout.get_training_data_feature_indices() == [0, 1]

    def test_should_have_no_placeholder_when_absent(self):
        layout = GrobidColumnLayout(
            name='model1',
            generator='DataGenerator1',
            label_slot=LabelSlot.ABSENT,
            columns=('token_text', 'feature1')
        )
        assert layout.get_data_generator_column_names() == ['token_text', 'feature1']
        assert layout.get_training_data_column_names() == ['token_text', 'feature1']
        assert layout.get_training_data_feature_indices() == [0]

    def test_should_drop_extra_columns_unless_asked_for(self):
        layout = GrobidColumnLayout(
            name='model1',
            generator='DataGenerator1',
            label_slot=LabelSlot.ABSENT,
            columns=('token_text', 'feature1'),
            extra_columns=('extra1',)
        )
        assert layout.get_data_generator_column_names() == [
            'token_text', 'feature1', 'extra1'
        ]
        assert layout.get_training_data_column_names() == ['token_text', 'feature1']
        assert layout.get_training_data_feature_indices() == [0]
        assert layout.get_training_data_column_names(include_extra_columns=True) == [
            'token_text', 'feature1', 'extra1'
        ]
        assert layout.get_training_data_feature_indices(include_extra_columns=True) == [0, 1]

    def test_should_keep_the_placeholder_last_with_extra_columns(self):
        layout = GrobidColumnLayout(
            name='model1',
            generator='DataGenerator1',
            label_slot=LabelSlot.UNFILLED,
            columns=('token_text', 'feature1'),
            extra_columns=('extra1',)
        )
        assert layout.get_data_generator_column_names() == [
            'token_text', 'feature1', 'extra1', PLACEHOLDER_COLUMN_NAME
        ]
        assert layout.get_training_data_feature_indices() == [0, 2]
        assert layout.get_training_data_feature_indices(include_extra_columns=True) == [0, 1, 2]


class TestGetGrobidColumnLayoutForModelName:
    def test_should_reject_an_unknown_model(self):
        with pytest.raises(ValueError):
            get_grobid_column_layout_for_model_name('model1')

    def test_should_share_one_layout_between_the_two_name_models(self):
        assert (
            get_grobid_column_layout_for_model_name('name_header').columns
            == get_grobid_column_layout_for_model_name('name_citation').columns
        )


class TestGetValidatedTrainingDataFeatureIndices:
    @pytest.fixture(name='layout')
    def _layout(self) -> GrobidColumnLayout:
        return GrobidColumnLayout(
            name='model1',
            generator='DataGenerator1',
            label_slot=LabelSlot.FILLED,
            columns=('token_text', 'feature1')
        )

    def test_should_accept_the_recorded_columns(self, layout: GrobidColumnLayout):
        assert get_validated_training_data_feature_indices(
            layout,
            feature_column_count=2,
            data_generator_name='DataGenerator1',
            data_generator_column_names=['token_text', 'feature1', PLACEHOLDER_COLUMN_NAME]
        ) == [0]

    def test_should_reject_another_feature_column_count(self, layout: GrobidColumnLayout):
        with pytest.raises(ValueError):
            get_validated_training_data_feature_indices(layout, feature_column_count=3)

    def test_should_reject_another_data_generator(self, layout: GrobidColumnLayout):
        with pytest.raises(ValueError):
            get_validated_training_data_feature_indices(
                layout,
                feature_column_count=2,
                data_generator_name='DataGenerator2'
            )

    def test_should_reject_renamed_columns(self, layout: GrobidColumnLayout):
        with pytest.raises(ValueError):
            get_validated_training_data_feature_indices(
                layout,
                feature_column_count=2,
                data_generator_column_names=[
                    'token_text', 'feature2', PLACEHOLDER_COLUMN_NAME
                ]
            )
