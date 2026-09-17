from typing import List


# What the prompt calls each region, and the label the pipeline reads it as.
# Five are the labels `processors/fulltext/processor.py` consumes; the sixth is
# where everything it does not consume goes.
#
# The names are the ones a publisher would use rather than GROBID's: `header`
# means a page header to most readers and the article's front matter here, and
# the model is the audience for the word.
SEGMENTATION_LABEL_BY_REGION_NAME = {
    'front_matter': '<header>',
    'body': '<body>',
    'acknowledgements': '<acknowledgement>',
    'appendix': '<annex>',
    'references': '<references>',
    # Offered so furniture can be named rather than stepped over: asking a model
    # to leave lines out is a negation, and it reads whichever region surrounds
    # them as the answer. Nothing downstream reads `<other>`.
    'other': '<other>',
}


def get_segmentation_region_names() -> List[str]:
    """The names the prompt offers, checked against the labels they map to.

    Which labels the pipeline consumes is a choice, so it is stated here; that
    they are still spelled the way the segmentation model spells them is not, and
    a rename there would otherwise reach the decoder as a label nothing reads.
    """
    from sciencebeam_parser.models.segmentation.training_data import (  # noqa pylint: disable=import-outside-toplevel
        TRAINING_XML_ELEMENT_PATH_BY_LABEL
    )
    unknown = sorted(
        label for label in SEGMENTATION_LABEL_BY_REGION_NAME.values()
        if label not in TRAINING_XML_ELEMENT_PATH_BY_LABEL
    )
    if unknown:
        raise ValueError(
            f'segmentation model has no label(s) {unknown};'
            f' has: {sorted(TRAINING_XML_ELEMENT_PATH_BY_LABEL)}'
        )
    return list(SEGMENTATION_LABEL_BY_REGION_NAME)


def get_segmentation_label(region_name: str) -> str:
    return SEGMENTATION_LABEL_BY_REGION_NAME[region_name]


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
