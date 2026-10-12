"""Where a review left a document, read from the verdict beside its record.

The gate measures and a review judges, so the two are recorded separately: the
quality record is written by generation, the verdict by whoever read the
document, and they sit in sibling directories of the same model.

A verdict is pinned to the content it judged. Regenerating a document moves the
pin, which lapses the verdict -- but a lapsed verdict is not the same as none,
and the two are reported apart. An acceptance is about the content: regenerate
the document and nobody has approved what is now there. A refusal is about the
document, and regenerating it does not withdraw it; it stands until somebody
reads the document again and says otherwise.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Optional


LOGGER = logging.getLogger(__name__)


VERDICT_SUB_DIRECTORY = 'verdict'
VERDICT_FILENAME_SUFFIX = '.verdict.json'
QUALITY_FILENAME_SUFFIX = '.quality.json'

LABELLED_CONTENT_HASH_KEY = 'labelled_content_hash'


class ReviewStanding:
    """Where a document stands, as the gate reads it.

    A verdict the pin no longer matches is reported as lapsed rather than as
    absent, because a decision somebody took is not the same as one nobody ever
    took, and what each demand makes of the two differs.
    """

    # Nobody ever judged it.
    UNREVIEWED = 'unreviewed'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    ACCEPTANCE_LAPSED = 'acceptance-lapsed'
    REJECTION_LAPSED = 'rejection-lapsed'


LAPSED_BY_VERDICT = {
    ReviewStanding.ACCEPTED: ReviewStanding.ACCEPTANCE_LAPSED,
    ReviewStanding.REJECTED: ReviewStanding.REJECTION_LAPSED,
}

# A refusal stands until somebody reads the document again and withdraws it.
REFUSING_STANDINGS = frozenset({
    ReviewStanding.REJECTED, ReviewStanding.REJECTION_LAPSED
})


def get_verdict_file_path(record_file_path: str) -> str:
    """The verdict beside a quality record, which is its sibling directory.

    Derived from the record's path rather than searched for, the same way the
    corpus name is: a record names one document of one model, and the verdict
    about it is written under that model by the review tool.
    """
    records_directory = os.path.dirname(record_file_path)
    model_directory = os.path.dirname(records_directory)
    basename = os.path.basename(record_file_path)
    if basename.endswith(QUALITY_FILENAME_SUFFIX):
        basename = basename[:-len(QUALITY_FILENAME_SUFFIX)]
    return os.path.join(
        model_directory, VERDICT_SUB_DIRECTORY, basename + VERDICT_FILENAME_SUFFIX
    )


def get_review_standing(
    verdict_json_dict: Optional[Mapping[str, Any]],
    labelled_content_hash: Optional[str]
) -> str:
    """What a verdict says, and whether it is about the content held now.

    A decision that cannot be confirmed to be about this content reads as
    lapsed rather than as never taken -- including where there is no pin to
    confirm it against, as for a model that writes no span record.
    """
    if not verdict_json_dict:
        return ReviewStanding.UNREVIEWED
    verdict = verdict_json_dict.get('verdict')
    if verdict not in LAPSED_BY_VERDICT:
        return ReviewStanding.UNREVIEWED
    if (
        labelled_content_hash
        and verdict_json_dict.get(LABELLED_CONTENT_HASH_KEY) == labelled_content_hash
    ):
        return verdict
    return LAPSED_BY_VERDICT[verdict]


def read_review_standing(
    record_file_path: str,
    labelled_content_hash: Optional[str]
) -> str:
    """Read the verdict beside one record, or report it unreviewed.

    A corpus with no verdicts at all is the ordinary case rather than a fault,
    so a missing file is read as nobody having judged it.
    """
    verdict_file_path = get_verdict_file_path(record_file_path)
    if not os.path.exists(verdict_file_path):
        return ReviewStanding.UNREVIEWED
    try:
        with open(verdict_file_path, 'r', encoding='utf-8') as verdict_file:
            verdict_json_dict = json.load(verdict_file)
    except (OSError, ValueError):
        LOGGER.warning(
            'could not read the verdict at %r, so %s reads as unreviewed',
            verdict_file_path, os.path.basename(record_file_path)
        )
        return ReviewStanding.UNREVIEWED
    return get_review_standing(verdict_json_dict, labelled_content_hash)
