from typing import List


# The labels `processors/fulltext/processor.py` reads from the segmentation result:
# `<header>`, `<body>`, `<acknowledgement>`, `<annex>` and `<references>`. The model
# predicts twelve; the other seven reach no scored field, so asking for them spends
# output on distinctions nothing downstream can use.
CONSUMED_SEGMENTATION_LABELS = (
    '<header>',
    '<body>',
    '<acknowledgement>',
    '<annex>',
    '<references>',
)


def get_segmentation_labels() -> List[str]:
    """Checked against the model's own label map rather than only restated.

    The five are a choice about what the pipeline consumes, so they are named
    here; that they are still spelled the way the segmentation model spells them
    is not a choice, and a rename there would otherwise reach the prompt as a
    label the decoder then rejects.
    """
    from sciencebeam_parser.models.segmentation.training_data import (  # noqa pylint: disable=import-outside-toplevel
        TRAINING_XML_ELEMENT_PATH_BY_LABEL
    )
    unknown = [
        label for label in CONSUMED_SEGMENTATION_LABELS
        if label not in TRAINING_XML_ELEMENT_PATH_BY_LABEL
    ]
    if unknown:
        raise ValueError(
            f'segmentation model has no label(s) {unknown};'
            f' has: {sorted(TRAINING_XML_ELEMENT_PATH_BY_LABEL)}'
        )
    return [label.strip('<>') for label in CONSUMED_SEGMENTATION_LABELS]


def get_citation_labels() -> List[str]:
    """Read from the model's own label map rather than restated.

    A prompt built from a remembered list mislabels whatever it omits, and the
    set moves: `<idno>` and `<pubnum>` were separate labels until recently.
    """
    from sciencebeam_parser.models.citation.training_data import (  # noqa pylint: disable=import-outside-toplevel
        TRAINING_XML_ELEMENT_PATH_BY_LABEL
    )
    return sorted(
        label.strip('<>') for label in TRAINING_XML_ELEMENT_PATH_BY_LABEL
    )
