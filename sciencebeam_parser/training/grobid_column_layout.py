import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml


GROBID_COLUMN_LAYOUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'resources',
    'grobid_column_layout.yml'
)


PLACEHOLDER_COLUMN_NAME = 'dummy_label'


class LabelSlot:
    FILLED = 'filled'
    UNFILLED = 'unfilled'
    ABSENT = 'absent'


VALID_LABEL_SLOTS = frozenset({LabelSlot.FILLED, LabelSlot.UNFILLED, LabelSlot.ABSENT})

LABEL_SLOTS_WITH_PLACEHOLDER_COLUMN = frozenset({LabelSlot.FILLED, LabelSlot.UNFILLED})


@dataclass(frozen=True)
class GrobidColumnLayout:
    name: str
    generator: str
    label_slot: str
    columns: Sequence[str]
    extra_columns: Sequence[str] = field(default_factory=tuple)
    reference_training_corpus: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self):
        if self.label_slot not in VALID_LABEL_SLOTS:
            raise ValueError(
                'invalid label_slot for %r: %r (expected one of %s)' % (
                    self.name, self.label_slot, sorted(VALID_LABEL_SLOTS)
                )
            )

    @property
    def has_placeholder_column(self) -> bool:
        return self.label_slot in LABEL_SLOTS_WITH_PLACEHOLDER_COLUMN

    def get_data_generator_column_names(self) -> List[str]:
        """The columns a data generator emits, which is what inference gets."""
        names = list(self.columns) + list(self.extra_columns)
        if self.has_placeholder_column:
            names.append(PLACEHOLDER_COLUMN_NAME)
        return names

    def get_training_data_column_names(
        self,
        include_extra_columns: bool = False
    ) -> List[str]:
        """The columns of a training data line, before the label."""
        names = list(self.columns)
        if include_extra_columns:
            names.extend(self.extra_columns)
        if self.label_slot == LabelSlot.UNFILLED:
            names.append(PLACEHOLDER_COLUMN_NAME)
        return names

    def get_training_data_feature_indices(
        self,
        include_extra_columns: bool = False
    ) -> List[int]:
        """Indices into a feature row, which excludes the leading token column."""
        generator_column_names = self.get_data_generator_column_names()[1:]
        return [
            generator_column_names.index(name)
            for name in self.get_training_data_column_names(include_extra_columns)[1:]
        ]


def _get_grobid_column_layout(name: str, layout_config: Mapping) -> GrobidColumnLayout:
    return GrobidColumnLayout(
        name=name,
        generator=layout_config['generator'],
        label_slot=layout_config['label_slot'],
        columns=tuple(layout_config['columns']),
        extra_columns=tuple(layout_config.get('extra_columns') or ()),
        reference_training_corpus=tuple(layout_config.get('reference_training_corpus') or ())
    )


def load_grobid_column_layout_by_name(
    filename: str = GROBID_COLUMN_LAYOUT_FILE
) -> Dict[str, GrobidColumnLayout]:
    with open(filename, 'r', encoding='utf-8') as fp:
        config = yaml.safe_load(fp)
    return {
        name: _get_grobid_column_layout(name, layout_config)
        for name, layout_config in config['models'].items()
    }


def get_grobid_column_layout_for_model_name(
    model_name: str,
    filename: str = GROBID_COLUMN_LAYOUT_FILE
) -> GrobidColumnLayout:
    layout_by_name = load_grobid_column_layout_by_name(filename)
    layout = layout_by_name.get(model_name)
    if layout is None:
        raise ValueError(
            'no GROBID column layout recorded for model %r, add one to %s (known: %s)' % (
                model_name, os.path.basename(filename), sorted(layout_by_name)
            )
        )
    return layout


def get_validated_training_data_feature_indices(
    layout: GrobidColumnLayout,
    feature_column_count: int,
    data_generator_name: Optional[str] = None,
    data_generator_column_names: Optional[Sequence[str]] = None,
    include_extra_columns: bool = False
) -> List[int]:
    if data_generator_name is not None and data_generator_name != layout.generator:
        raise ValueError(
            'model %r uses data generator %r, but the recorded layout is for %r' % (
                layout.name, data_generator_name, layout.generator
            )
        )
    expected_column_names = layout.get_data_generator_column_names()
    if (
        data_generator_column_names is not None
        and list(data_generator_column_names) != expected_column_names
    ):
        raise ValueError(
            'columns of %r do not match the recorded layout: %r != %r' % (
                layout.name, list(data_generator_column_names), expected_column_names
            )
        )
    expected_feature_column_count = len(expected_column_names) - 1
    if feature_column_count != expected_feature_column_count:
        raise ValueError(
            'expected %d feature columns for %r, but found %d' % (
                expected_feature_column_count, layout.name, feature_column_count
            )
        )
    return layout.get_training_data_feature_indices(include_extra_columns)


def select_feature_columns(
    features: np.ndarray,
    feature_indices: Sequence[int]
) -> np.ndarray:
    return np.asarray([
        [
            [token_features[index] for index in feature_indices]
            for token_features in document_features
        ]
        for document_features in features.tolist()
    ], dtype=object)
