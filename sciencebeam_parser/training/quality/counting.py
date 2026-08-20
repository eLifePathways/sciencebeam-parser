"""Counting the cardinality of generated training data at each stage.

The reference of record is what `JatsFieldExtractor` emits, and every count is
taken per document so that a corpus total cannot hide a document that lost its
references.
"""
import logging
from typing import Dict, FrozenSet, Iterable, Iterator, Mapping, Optional, Sequence

from lxml import etree

from sciencebeam_parser.models.data import LabeledLayoutToken, LayoutModelData
from sciencebeam_parser.training.jats.field_vocab import CITATION_LABEL_BY_SUB_FIELD


LOGGER = logging.getLogger(__name__)


# The element a model writes once per entity, by model name.  Counting elements
# only means something for a model whose labels mark repeated entities.
ENTITY_ELEMENT_NAME_BY_MODEL: Mapping[str, str] = {
    'reference-segmenter': 'bibl',
    'citation': 'bibl',
}

# Models whose labels mark regions rather than repeated entities: an element
# count says nothing about them, and they are listed rather than left out so
# that a model with no entry is a missing decision rather than a silent pass.
MODELS_WITHOUT_ENTITY_COUNT: FrozenSet[str] = frozenset({
    'segmentation',
    'header',
    'affiliation-address',
    'name-header',
    'name-citation',
    'fulltext',
    'figure',
    'table',
})


# The label whose starts count one entity once the TEI is parsed back to labels,
# by model.  This is the stage the training data ends up at, and the only one the
# delft conversion can see.
ENTITY_LABEL_BY_MODEL: Mapping[str, str] = {
    'reference-segmenter': '<reference>',
}

# Models whose entities are one training sequence each, so that the entity count
# cannot change at the parse and what can is which labels are marked.
MODELS_COUNTED_BY_LABEL: FrozenSet[str] = frozenset({'citation'})


def get_canonical_model_name(model_name: str) -> str:
    """One spelling for a model, since the two CLIs disagree on it.

    `generate_data` names models as its generators do, hyphenated, and
    `generate_delft_data` takes the underscored name the model registry uses.
    A lookup that missed on the spelling would report no counts at all.
    """
    return model_name.replace('_', '-')


def is_model_counted_by_label(model_name: str) -> bool:
    return get_canonical_model_name(model_name) in MODELS_COUNTED_BY_LABEL


def get_local_name(element: etree._Element) -> str:
    tag = element.tag
    if not isinstance(tag, str):
        return ''
    return tag.split('}', 1)[1] if tag.startswith('{') else tag


def count_entity_elements(model_name: str, tei_root: etree._Element) -> Optional[int]:
    """Return the number of entity elements the TEI holds, or None if the model has no count."""
    canonical_model_name = get_canonical_model_name(model_name)
    element_name = ENTITY_ELEMENT_NAME_BY_MODEL.get(canonical_model_name)
    if element_name is None:
        if canonical_model_name not in MODELS_WITHOUT_ENTITY_COUNT:
            LOGGER.warning(
                'no cardinality check defined for model %r, and it is not declared as having none',
                model_name
            )
        return None
    return sum(
        1 for element in tei_root.iter() if get_local_name(element) == element_name
    )


def get_labels_for_model_data_list(
    model_data_list: Sequence[LayoutModelData]
) -> FrozenSet[str]:
    """Return the labels marked in one entity's model data, without the B- prefix.

    Unlabeled model data carries no label attribute at all, which reads here as
    an entity that marks nothing.
    """
    labels = (
        getattr(model_data, 'label', None)
        for model_data in model_data_list
    )
    return frozenset(
        label[2:] if label.startswith('B-') else label
        for label in labels
        if label
    )


def count_citation_labels(
    reference_sub_field_names: Iterable[FrozenSet[str]],
    model_data_list_list: Sequence[Sequence[LayoutModelData]],
) -> Dict[str, Dict[str, int]]:
    """Count, per citation label, references whose JATS carries it and references marking it.

    Presence per reference rather than occurrences, since the label conventions
    differ: `<author>` covers a whole author list where `<pages>` is written once
    per page number.  The two sides are counted independently -- a reference
    legitimately does not print everything its JATS carries, so what the record
    holds is a rate to compare across regenerations, not a per-reference
    requirement.
    """
    counts: Dict[str, Dict[str, int]] = {}

    def _entry(label: str) -> Dict[str, int]:
        return counts.setdefault(label, {'jats': 0, 'marked': 0})

    for sub_field_names in reference_sub_field_names:
        for label in {
            CITATION_LABEL_BY_SUB_FIELD[sub_field_name]
            for sub_field_name in sub_field_names
            if sub_field_name in CITATION_LABEL_BY_SUB_FIELD
        }:
            _entry(label)['jats'] += 1
    for model_data_list in model_data_list_list:
        for label in get_labels_for_model_data_list(model_data_list):
            _entry(label)['marked'] += 1
    return counts


def _iter_labels(
    labeled_layout_tokens: Iterable[LabeledLayoutToken]
) -> Iterator[str]:
    for labeled_layout_token in labeled_layout_tokens:
        if labeled_layout_token.label:
            yield labeled_layout_token.label


def count_entity_starts(
    model_name: str,
    labeled_layout_tokens_list: Sequence[Sequence[LabeledLayoutToken]]
) -> Optional[int]:
    """Entities the training data ends up with, or None for a model counted by label.

    Fewer of these than the TEI holds elements is a boundary lost between
    siblings; the elements are what generation recorded.
    """
    entity_label = ENTITY_LABEL_BY_MODEL.get(get_canonical_model_name(model_name))
    if entity_label is None:
        return None
    return sum(
        1
        for labeled_layout_tokens in labeled_layout_tokens_list
        for label in _iter_labels(labeled_layout_tokens)
        if label == 'B-' + entity_label
    )


def count_label_starts_per_sequence(
    labeled_layout_tokens_list: Sequence[Sequence[LabeledLayoutToken]]
) -> Dict[str, int]:
    """Per label, the number of sequences that mark it.

    Presence per sequence, so it is comparable with what generation recorded as
    marked for the citation model, where one sequence is one reference.
    """
    counts: Dict[str, int] = {}
    for labeled_layout_tokens in labeled_layout_tokens_list:
        for label in {
            label[2:]
            for label in _iter_labels(labeled_layout_tokens)
            if label.startswith('B-')
        }:
            counts[label] = counts.get(label, 0) + 1
    return counts
