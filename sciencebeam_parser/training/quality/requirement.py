"""What a corpus demands of a document before its training data is used.

Declared with the data rather than here. The verdicts it is about are committed
in the corpus, and how much of a corpus has been read is a fact about that
corpus rather than about the model -- one corpus's segmentation can be read
through while another's has not been opened, and a per-model setting here could
not say so. `generate` in the same file says what each pair should be; this says
what it has to be worth before anything trains on it.

The thresholds stay in `training_quality.yml`, because what the gate can measure
and where its floors sit is a property of the model and the pipeline rather than
of any corpus.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml


LOGGER = logging.getLogger(__name__)


REQUIREMENT_KEY = 'require'


class Requirement:
    """How much a corpus asks of a document, from least to most."""

    # The thresholds decide on their own, and a verdict corrects nothing.
    MEASURED = 'measured'
    # A rejection excludes; a document nobody has read still falls through to
    # the thresholds.
    NOT_REJECTED = 'not-rejected'
    # Only a document accepted at the content it holds now is used.
    ACCEPTED = 'accepted'


REQUIREMENT_VALUES = (
    Requirement.MEASURED, Requirement.NOT_REJECTED, Requirement.ACCEPTED
)


class RequirementConfigError(ValueError):
    pass


def _requirement_block(config_json: Mapping[str, Any]) -> Mapping[str, Any]:
    block = config_json.get(REQUIREMENT_KEY) or {}
    if not isinstance(block, Mapping):
        raise RequirementConfigError(
            f'`{REQUIREMENT_KEY}` must map each corpus to its models, not'
            f' {type(block).__name__}'
        )
    return block


def get_requirement_by_corpus_and_model(
    config_json: Mapping[str, Any]
) -> Dict[Tuple[str, str], str]:
    """Each declared pair and what it demands, checked as it is read.

    A value the gate does not know is refused here rather than at the document
    it would have silently passed.
    """
    requirement_by_pair: Dict[Tuple[str, str], str] = {}
    for corpus, models in _requirement_block(config_json).items():
        if not isinstance(models, Mapping):
            raise RequirementConfigError(
                f'`{REQUIREMENT_KEY}.{corpus}` must map each model to a'
                f' requirement, not {type(models).__name__}'
            )
        for model, requirement in models.items():
            if str(requirement) not in REQUIREMENT_VALUES:
                raise RequirementConfigError(
                    f'`{REQUIREMENT_KEY}.{corpus}.{model}` is {requirement!r};'
                    f' expected one of {", ".join(REQUIREMENT_VALUES)}'
                )
            requirement_by_pair[(corpus, model)] = str(requirement)
    return requirement_by_pair


def load_requirement_by_corpus_and_model(
    config_file_path: str
) -> Dict[Tuple[str, str], str]:
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        return get_requirement_by_corpus_and_model(yaml.safe_load(config_file) or {})


def get_requirement(
    requirement_by_pair: Optional[Mapping[Tuple[str, str], str]],
    corpus: Optional[str],
    model_name: str
) -> str:
    """What this pair demands, refusing a corpus that declares nothing for it.

    A corpus that carries no declaration at all is assembled as it always was,
    which is what `measured` means; one that carries a declaration has said
    which pairs it has an opinion about, so a pair missing from it is an
    omission rather than a decision and is refused rather than assumed.
    """
    if requirement_by_pair is None:
        return Requirement.MEASURED
    requirement = requirement_by_pair.get((corpus or '', model_name))
    if requirement is None:
        raise RequirementConfigError(
            f'`{REQUIREMENT_KEY}` declares nothing for {corpus or "an unnamed corpus"}'
            f'/{model_name}. State what it demands -- one of'
            f' {", ".join(REQUIREMENT_VALUES)} -- rather than leaving it to be assumed'
        )
    return requirement
