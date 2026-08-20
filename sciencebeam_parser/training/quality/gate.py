"""Deciding what a training run may use, from the record and a stated threshold.

The gate judges; the record measures. It excludes at assembly rather than at
generation, so the documents it refuses stay generatable and alignment stays
investigable.
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml


LOGGER = logging.getLogger(__name__)


TRAINING_QUALITY_CONFIG_FILE = os.path.join(
    os.path.dirname(  # sciencebeam_parser
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
    'resources',
    'training_quality.yml'
)


NO_CARDINALITY = 'none'


class ExclusionReason:
    """Why a document's training data is not used, in the order the stages run."""
    JATS_NOT_READABLE = 'jats-not-readable'
    JATS_HAS_NO_REFERENCES = 'jats-has-no-references'
    NO_GENERATED_OUTPUT = 'no-generated-output'
    ELEMENTS_SHORT_OF_JATS = 'elements-short-of-jats'
    ENTITIES_SHORT_OF_ELEMENTS = 'entities-short-of-elements'
    NO_TRAINING_SEQUENCES = 'no-training-sequences'


@dataclass(frozen=True)
class ModelQualityThresholds:
    model_name: str
    cardinality: Optional[str] = None
    reason: Optional[str] = None
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


def get_quality_verdict(  # pylint: disable=too-many-branches
    document_id: str,
    thresholds: ModelQualityThresholds,
    jats_status: Optional[str] = None,
    jats_reference_count: Optional[int] = None,
    written: Optional[bool] = None,
    entity_element_count: Optional[int] = None,
    entity_start_count: Optional[int] = None,
    sequence_count: int = 0,
) -> QualityVerdict:
    """Judge one document, naming every stage that failed, earliest first.

    A count that is not available is not a failure: assembly is run over corpora
    with no record at all, and what cannot be checked has to be reported as
    unchecked rather than assumed good.
    """
    reasons: List[str] = []
    detail: Dict[str, Any] = {}
    if sequence_count == 0:
        reasons.append(ExclusionReason.NO_TRAINING_SEQUENCES)
    if not thresholds.has_cardinality_check:
        return QualityVerdict(document_id, reasons, detail)

    if jats_status is not None and jats_status != 'ok':
        reasons.insert(0, ExclusionReason.JATS_NOT_READABLE)
        detail['jats_status'] = jats_status
    elif (
        thresholds.min_jats_reference_count is not None
        and jats_reference_count is not None
        and jats_reference_count < thresholds.min_jats_reference_count
    ):
        reasons.insert(0, ExclusionReason.JATS_HAS_NO_REFERENCES)
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

    return QualityVerdict(document_id, reasons, detail)


class CorpusMostlyExcludedError(RuntimeError):
    pass


@dataclass
class CorpusGateSummary:
    corpus: Optional[str]
    kept_count: int = 0
    excluded_count: int = 0
    excluded_by_reason: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return self.kept_count + self.excluded_count

    @property
    def excluded_ratio(self) -> float:
        if not self.total_count:
            return 0.0
        return self.excluded_count / self.total_count

    def __str__(self) -> str:
        parts = [
            f'kept {self.kept_count} of {self.total_count} documents'
            f' ({self.excluded_ratio:.0%} excluded)'
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
        summary.excluded_count += 1
        assert verdict.primary_reason is not None
        summary.excluded_by_reason.setdefault(verdict.primary_reason, []).append(
            verdict.document_id
        )
    return summary_by_corpus


def check_corpus_loss_or_fail(
    summary_by_corpus: Mapping[Optional[str], CorpusGateSummary],
    max_excluded_ratio: float
) -> None:
    """Refuse rather than proceed when a corpus loses more than the stated share."""
    mostly_excluded = [
        summary
        for summary in summary_by_corpus.values()
        if summary.excluded_ratio > max_excluded_ratio
    ]
    if not mostly_excluded:
        return
    raise CorpusMostlyExcludedError(
        'excluded more than the configured %.0f%% of a corpus: %s' % (
            max_excluded_ratio * 100,
            '; '.join(
                f'{summary.corpus or "corpus not known"} {summary}'
                for summary in mostly_excluded
            )
        )
    )
