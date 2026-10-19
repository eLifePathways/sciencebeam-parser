"""Deciding what a training run may use, from the record and a stated threshold.

The gate judges; the record measures. It excludes at assembly rather than at
generation, so the documents it refuses stay generatable and alignment stays
investigable.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from sciencebeam_parser.training.quality.requirement import Requirement
from sciencebeam_parser.training.quality.review import REFUSING_STANDINGS, ReviewStanding


LOGGER = logging.getLogger(__name__)


TRAINING_QUALITY_CONFIG_FILE = os.path.join(
    os.path.dirname(  # sciencebeam_parser
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    'resources',
    'training_quality.yml'
)


NO_CARDINALITY = 'none'

# A JATS that was matched and could not be used is a defect; one that was never
# matched leaves nothing to check against, which is not the same thing -- data is
# legitimately generated with no JATS at all.
JATS_STATUSES_THAT_FAILED = frozenset({'unparsable', 'unreadable'})


class ExclusionReason:
    """Why a document's training data is not used, in the order the stages run."""
    JATS_NOT_READABLE = 'jats-not-readable'
    JATS_HAS_NO_REFERENCES = 'jats-has-no-references'
    NO_GENERATED_OUTPUT = 'no-generated-output'
    ELEMENTS_SHORT_OF_JATS = 'elements-short-of-jats'
    ENTITIES_SHORT_OF_ELEMENTS = 'entities-short-of-elements'
    NO_TRAINING_SEQUENCES = 'no-training-sequences'
    TOO_FEW_MATCHED_ROWS = 'too-few-matched-rows'
    REJECTED_BY_REVIEW = 'rejected-by-review'
    NOT_REVIEWED = 'not-reviewed'


# What a measurement found nothing of, rather than found too little of. A reader
# who accepted the document outranks a ratio, but cannot supply training data
# that was never generated, so these stand whatever the review decided.
ABSENCE_REASONS = frozenset({
    ExclusionReason.NO_GENERATED_OUTPUT,
    ExclusionReason.NO_TRAINING_SEQUENCES,
})


@dataclass(frozen=True)
class ModelQualityThresholds:
    model_name: str
    cardinality: Optional[str] = None
    reason: Optional[str] = None
    min_matched_row_ratio: Optional[float] = None
    min_jats_reference_count: Optional[int] = None
    min_element_ratio: Optional[float] = None
    min_entity_ratio: Optional[float] = None
    label_floors: Mapping[str, float] = field(default_factory=dict)

    @property
    def has_cardinality_check(self) -> bool:
        return self.cardinality != NO_CARDINALITY


@dataclass(frozen=True)
class TrainingQualityConfig:
    max_excluded_ratio: float
    max_unreviewed_ratio: float
    thresholds_by_model: Mapping[str, ModelQualityThresholds]

    def get_thresholds_for_model(self, model_name: str) -> ModelQualityThresholds:
        thresholds = self.thresholds_by_model.get(model_name)
        if thresholds is None:
            raise KeyError(
                'no quality thresholds configured for model %r; add an entry,'
                ' with cardinality: %s and a reason if it has no count to check'
                % (model_name, NO_CARDINALITY)
            )
        return thresholds


def load_training_quality_config(
    config_file_path: str = TRAINING_QUALITY_CONFIG_FILE
) -> TrainingQualityConfig:
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        config_json = yaml.safe_load(config_file)
    return TrainingQualityConfig(
        max_excluded_ratio=config_json['corpus']['max_excluded_ratio'],
        max_unreviewed_ratio=config_json['corpus']['max_unreviewed_ratio'],
        thresholds_by_model={
            model_name: ModelQualityThresholds(model_name=model_name, **(entry or {}))
            for model_name, entry in config_json['models'].items()
        },
    )


@dataclass
class QualityVerdict:
    """Whether a document's training data is used, and the numbers behind it."""
    document_id: str
    exclusion_reasons: Sequence[str] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_excluded(self) -> bool:
        return bool(self.exclusion_reasons)

    @property
    def primary_reason(self) -> Optional[str]:
        """The earliest stage that failed, which is the one to fix."""
        return self.exclusion_reasons[0] if self.exclusion_reasons else None

    def __str__(self) -> str:
        if not self.is_excluded:
            return f'{self.document_id}: kept'
        detail = ', '.join(f'{key}={value}' for key, value in sorted(self.detail.items()))
        return (
            f'{self.document_id}: excluded ({", ".join(self.exclusion_reasons)})'
            f'{" [" + detail + "]" if detail else ""}'
        )


def _get_ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _get_measurement_reasons(  # pylint: disable=too-many-arguments
    thresholds: ModelQualityThresholds,
    matched_row_ratio: Optional[float],
    jats_status: Optional[str],
    jats_reference_count: Optional[int],
    written: Optional[bool],
    entity_element_count: Optional[int],
    entity_start_count: Optional[int],
    sequence_count: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Every stage the counts failed, earliest first, and the numbers behind it.

    A count that is not available is not a failure: assembly is run over corpora
    with no record at all, and what cannot be checked has to be reported as
    unchecked rather than assumed good.
    """
    reasons: List[str] = []
    detail: Dict[str, Any] = {}
    # The pairing is checked whatever the model has to count: a raw file and a
    # TEI that stopped lining up is a different failure from a document that
    # lost its references, and it can happen to a model with no cardinality.
    if matched_row_ratio is not None:
        detail['matched_row_ratio'] = round(matched_row_ratio, 4)
        if (
            thresholds.min_matched_row_ratio is not None
            and matched_row_ratio < thresholds.min_matched_row_ratio
        ):
            reasons.append(ExclusionReason.TOO_FEW_MATCHED_ROWS)
            detail['min_matched_row_ratio'] = thresholds.min_matched_row_ratio
    if not thresholds.has_cardinality_check:
        if sequence_count == 0:
            reasons.append(ExclusionReason.NO_TRAINING_SEQUENCES)
        return reasons, detail

    if jats_status in JATS_STATUSES_THAT_FAILED:
        reasons.append(ExclusionReason.JATS_NOT_READABLE)
        detail['jats_status'] = jats_status
    elif (
        thresholds.min_jats_reference_count is not None
        and jats_reference_count is not None
        and jats_reference_count < thresholds.min_jats_reference_count
    ):
        reasons.append(ExclusionReason.JATS_HAS_NO_REFERENCES)
        detail['jats_reference_count'] = jats_reference_count
    if written is False:
        reasons.append(ExclusionReason.NO_GENERATED_OUTPUT)

    element_ratio = _get_ratio(entity_element_count, jats_reference_count)
    if (
        thresholds.min_element_ratio is not None
        and element_ratio is not None
        and element_ratio < thresholds.min_element_ratio
    ):
        reasons.append(ExclusionReason.ELEMENTS_SHORT_OF_JATS)
        detail['element_ratio'] = round(element_ratio, 3)
        detail['entity_element_count'] = entity_element_count
        detail['jats_reference_count'] = jats_reference_count

    entity_ratio = _get_ratio(entity_start_count, entity_element_count)
    if (
        thresholds.min_entity_ratio is not None
        and entity_ratio is not None
        and entity_ratio < thresholds.min_entity_ratio
    ):
        reasons.append(ExclusionReason.ENTITIES_SHORT_OF_ELEMENTS)
        detail['entity_ratio'] = round(entity_ratio, 3)
        detail['entity_start_count'] = entity_start_count
        detail['entity_element_count'] = entity_element_count

    if sequence_count == 0:
        reasons.append(ExclusionReason.NO_TRAINING_SEQUENCES)
    return reasons, detail


def _get_reviewed_reasons(
    measured_reasons: Sequence[str],
    detail: Dict[str, Any],
    requirement: str,
    review: str,
) -> List[str]:
    """What the corpus's requirement makes of a measurement, given the review.

    A refusal binds until somebody withdraws it, so a rejection a regeneration
    lapsed still excludes: the document changed, which is not the same as anyone
    saying it is now sound. An acceptance is about the content and does not
    survive the regeneration, so a document whose acceptance lapsed is one
    nobody has read, which `accepted` excludes and the weaker demands do not.

    An absence is not a threshold to overrule, so it leads whatever was decided
    and survives an acceptance. What a measurement said about an admitted
    document is kept beside the admission rather than discarded, because the
    reason it would have been excluded is still worth knowing.
    """
    absences = [reason for reason in measured_reasons if reason in ABSENCE_REASONS]
    measured = [reason for reason in measured_reasons if reason not in ABSENCE_REASONS]
    if requirement == Requirement.MEASURED:
        return list(measured_reasons)
    if review in REFUSING_STANDINGS:
        return absences + [ExclusionReason.REJECTED_BY_REVIEW] + measured
    if review == ReviewStanding.ACCEPTED:
        if measured:
            detail['overruled_by_review'] = list(measured)
        return absences
    if requirement == Requirement.ACCEPTED:
        return absences + [ExclusionReason.NOT_REVIEWED] + measured
    return list(measured_reasons)


def get_quality_verdict(
    document_id: str,
    thresholds: ModelQualityThresholds,
    jats_status: Optional[str] = None,
    jats_reference_count: Optional[int] = None,
    written: Optional[bool] = None,
    entity_element_count: Optional[int] = None,
    entity_start_count: Optional[int] = None,
    sequence_count: int = 0,
    matched_row_ratio: Optional[float] = None,
    requirement: str = Requirement.MEASURED,
    review: str = ReviewStanding.UNREVIEWED,
) -> QualityVerdict:
    """Judge one document, from what was measured and what a reader decided."""
    measured_reasons, detail = _get_measurement_reasons(
        thresholds=thresholds,
        matched_row_ratio=matched_row_ratio,
        jats_status=jats_status,
        jats_reference_count=jats_reference_count,
        written=written,
        entity_element_count=entity_element_count,
        entity_start_count=entity_start_count,
        sequence_count=sequence_count,
    )
    return QualityVerdict(
        document_id,
        _get_reviewed_reasons(measured_reasons, detail, requirement, review),
        detail
    )


class CorpusRefusedError(RuntimeError):
    """A corpus assembly refused rather than proceeded with."""


class CorpusMostlyExcludedError(CorpusRefusedError):
    """Most of a corpus failed a measurement, which is a finding about the pipeline."""


class CorpusMostlyUnreviewedError(CorpusRefusedError):
    """Most of a corpus has not been read, which is a finding about the review."""


@dataclass
class CorpusGateSummary:
    """What a corpus lost, counted by the kind of finding rather than in one total.

    Three kinds, because they are three findings needing three actions: a
    measurement excluded it, nobody has read it, or a reader refused it. Only
    the first two are held to a threshold. A measurement is inferred from
    proxies that can misfire together, and a corpus nobody has started reading
    collapses to nothing without saying so, so both want a tripwire for the
    silent case. A rejection is neither inferred nor silent -- somebody looked
    at the document and said no -- so the number is worth reporting and there is
    nothing for a threshold to catch.
    """
    corpus: Optional[str]
    kept_count: int = 0
    excluded_count: int = 0
    unreviewed_count: int = 0
    rejected_count: int = 0
    excluded_by_reason: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return (
            self.kept_count + self.excluded_count
            + self.unreviewed_count + self.rejected_count
        )

    @property
    def excluded_ratio(self) -> float:
        if not self.total_count:
            return 0.0
        return self.excluded_count / self.total_count

    @property
    def unreviewed_ratio(self) -> float:
        if not self.total_count:
            return 0.0
        return self.unreviewed_count / self.total_count

    @property
    def rejected_ratio(self) -> float:
        if not self.total_count:
            return 0.0
        return self.rejected_count / self.total_count

    def __str__(self) -> str:
        parts = [
            f'kept {self.kept_count} of {self.total_count} documents'
            f' ({self.excluded_ratio:.0%} excluded,'
            f' {self.rejected_ratio:.0%} refused by a reader,'
            f' {self.unreviewed_ratio:.0%} not reviewed)'
        ]
        for reason, document_ids in sorted(self.excluded_by_reason.items()):
            parts.append(f'{reason}: {sorted(document_ids)}')
        return '; '.join(parts)


def get_gate_summary_by_corpus(
    verdict_and_corpus_list: Sequence[Any]
) -> Dict[Optional[str], CorpusGateSummary]:
    """Summarise verdicts per corpus, from (verdict, corpus) pairs."""
    summary_by_corpus: Dict[Optional[str], CorpusGateSummary] = {}
    for verdict, corpus in verdict_and_corpus_list:
        summary = summary_by_corpus.setdefault(corpus, CorpusGateSummary(corpus=corpus))
        if not verdict.is_excluded:
            summary.kept_count += 1
            continue
        assert verdict.primary_reason is not None
        if verdict.primary_reason == ExclusionReason.NOT_REVIEWED:
            summary.unreviewed_count += 1
        elif verdict.primary_reason == ExclusionReason.REJECTED_BY_REVIEW:
            summary.rejected_count += 1
        else:
            summary.excluded_count += 1
        summary.excluded_by_reason.setdefault(verdict.primary_reason, []).append(
            verdict.document_id
        )
    return summary_by_corpus


def _get_refusal_message(
    kind: str, ratio: float, mostly: Sequence[CorpusGateSummary]
) -> str:
    return '%s more than the configured %.0f%% of a corpus: %s' % (
        kind, ratio * 100,
        '; '.join(
            f'{summary.corpus or "corpus not known"} {summary}'
            for summary in mostly
        )
    )


def check_corpus_loss_or_fail(
    summary_by_corpus: Mapping[Optional[str], CorpusGateSummary],
    max_excluded_ratio: float,
    max_unreviewed_ratio: Optional[float] = None
) -> None:
    """Refuse rather than proceed when a corpus loses more than the stated share.

    Each threshold covers one kind of loss, and the refusal says which one caused
    it: reading the rest of a corpus and fixing a pipeline that drops documents
    are different pieces of work, so one threshold for both would make whoever
    first meets a merely unread corpus raise it -- which costs the
    pipeline-breakage signal the threshold exists for.

    Documents a reader refused are held to no threshold. They are the one kind
    of loss that is neither inferred nor silent, so there is no fault for a
    tripwire to catch, and refusing over them would decline the data a reader
    approved because other data was not -- and would refuse more readily the more
    carefully a corpus was read.
    """
    mostly_excluded = [
        summary
        for summary in summary_by_corpus.values()
        if summary.excluded_ratio > max_excluded_ratio
    ]
    if mostly_excluded:
        raise CorpusMostlyExcludedError(
            _get_refusal_message('excluded', max_excluded_ratio, mostly_excluded)
        )
    if max_unreviewed_ratio is None:
        return
    mostly_unreviewed = [
        summary
        for summary in summary_by_corpus.values()
        if summary.unreviewed_ratio > max_unreviewed_ratio
    ]
    if mostly_unreviewed:
        raise CorpusMostlyUnreviewedError(
            _get_refusal_message(
                'left unreviewed', max_unreviewed_ratio, mostly_unreviewed
            )
        )
