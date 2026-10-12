"""Where a review left a document, read from the verdict beside its record.

The gate measures and a review judges, so the two are recorded separately: the
quality record is written by generation, the verdict by whoever read the
document, and they sit in sibling directories of the same model.

A verdict is pinned to the content it judged. Regenerating a document moves the
pin, which lapses the verdict: it then says nothing about the document as it
stands, and is kept rather than removed so that a document once refused still
says so.
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
    """Where a document stands, as the gate reads it."""

    # Nobody has read it, or what they read is not what it holds now.
    UNREVIEWED = 'unreviewed'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'


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
    """What a verdict says about the content a document holds now.

    A model with no span record has no pin, so there is nothing a verdict could
    be about and the document reads as unreviewed however it was judged.
    """
    if not verdict_json_dict or not labelled_content_hash:
        return ReviewStanding.UNREVIEWED
    if verdict_json_dict.get(LABELLED_CONTENT_HASH_KEY) != labelled_content_hash:
        return ReviewStanding.UNREVIEWED
    verdict = verdict_json_dict.get('verdict')
    if verdict in (ReviewStanding.ACCEPTED, ReviewStanding.REJECTED):
        return verdict
    return ReviewStanding.UNREVIEWED


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
