from typing import List


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
