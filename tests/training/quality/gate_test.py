import pytest

from sciencebeam_parser.training.quality.counting import MODELS_WITHOUT_ENTITY_COUNT
from sciencebeam_parser.training.quality.gate import (
    CorpusMostlyExcludedError,
    ExclusionReason,
    ModelQualityThresholds,
    QualityVerdict,
    check_corpus_loss_or_fail,
    get_gate_summary_by_corpus,
    get_quality_verdict,
    load_training_quality_config
)


REFERENCE_SEGMENTER = 'reference-segmenter'
DOCUMENT_ID_1 = 'document1'

REFERENCE_SEGMENTER_THRESHOLDS = ModelQualityThresholds(
    model_name=REFERENCE_SEGMENTER,
    min_jats_reference_count=1,
    min_element_ratio=0.9,
    min_entity_ratio=0.8,
)


def _verdict(**kwargs) -> QualityVerdict:
    return get_quality_verdict(
        document_id=kwargs.pop('document_id', DOCUMENT_ID_1),
        thresholds=kwargs.pop('thresholds', REFERENCE_SEGMENTER_THRESHOLDS),
        **kwargs
    )


class TestLoadTrainingQualityConfig:
    def test_should_load_the_shipped_thresholds(self):
        config = load_training_quality_config()
        thresholds = config.get_thresholds_for_model(REFERENCE_SEGMENTER)
        assert thresholds.min_element_ratio == 0.9
        assert thresholds.min_entity_ratio == 0.8
        assert thresholds.has_cardinality_check
        assert 0 < config.max_excluded_ratio < 1

    def test_should_declare_a_region_model_as_having_no_cardinality(self):
        thresholds = load_training_quality_config().get_thresholds_for_model('segmentation')
        assert not thresholds.has_cardinality_check
        assert thresholds.reason

    def test_should_declare_citation_without_a_label_floor(self):
        thresholds = load_training_quality_config().get_thresholds_for_model('citation')
        assert thresholds.min_element_ratio == 0.9
        assert thresholds.min_entity_ratio is None
        assert not thresholds.label_floors

    def test_should_fail_for_a_model_with_no_entry_rather_than_pass_it(self):
        with pytest.raises(KeyError):
            load_training_quality_config().get_thresholds_for_model('not-a-model')

    def test_should_have_an_entry_for_every_model_generation_writes(self):
        # A missing entry is a decision not taken; the gate must not pass it silently.
        config = load_training_quality_config()
        for model_name in [
            'segmentation', 'header', 'affiliation-address', 'name-header',
            'name-citation', 'fulltext', 'figure', 'table', 'reference-segmenter',
            'citation',
        ]:
            assert config.get_thresholds_for_model(model_name)


class TestGetQualityVerdict:
    def test_should_keep_a_document_that_is_right_throughout(self):
        verdict = _verdict(
            jats_status='ok', jats_reference_count=40, written=True,
            entity_element_count=40, entity_start_count=40, sequence_count=1,
        )
        assert not verdict.is_excluded
        assert 'kept' in str(verdict)

    def test_should_exclude_a_document_whose_jats_declares_no_references(self):
        verdict = _verdict(
            jats_status='ok', jats_reference_count=0, written=False, sequence_count=1,
        )
        assert verdict.primary_reason == ExclusionReason.JATS_HAS_NO_REFERENCES
        assert verdict.detail['jats_reference_count'] == 0

    def test_should_exclude_a_document_whose_jats_could_not_be_read(self):
        verdict = _verdict(
            jats_status='unparsable', written=True,
            entity_element_count=0, entity_start_count=0, sequence_count=29,
        )
        assert verdict.primary_reason == ExclusionReason.JATS_NOT_READABLE
        assert verdict.detail['jats_status'] == 'unparsable'

    def test_should_exclude_a_document_short_at_the_tei_stage_naming_that_stage(self):
        # PPR459453: a 45-entry reference list truncated to a 109-word region.
        verdict = _verdict(
            jats_status='ok', jats_reference_count=45, written=True,
            entity_element_count=2, entity_start_count=2, sequence_count=1,
        )
        assert verdict.exclusion_reasons == [ExclusionReason.ELEMENTS_SHORT_OF_JATS]
        assert verdict.detail['element_ratio'] == 0.044

    def test_should_exclude_a_document_collapsed_at_the_parse_naming_that_stage(self):
        # The case that got through: correct in the TEI, one entity in the data.
        verdict = _verdict(
            jats_status='ok', jats_reference_count=40, written=True,
            entity_element_count=40, entity_start_count=1, sequence_count=1,
        )
        assert verdict.exclusion_reasons == [ExclusionReason.ENTITIES_SHORT_OF_ELEMENTS]
        assert verdict.detail['entity_ratio'] == 0.025

    def test_should_keep_a_document_holding_more_elements_than_the_jats_has(self):
        # PPR534793: 22 references, 25 elements, from references split across a page.
        verdict = _verdict(
            jats_status='ok', jats_reference_count=22, written=True,
            entity_element_count=25, entity_start_count=25, sequence_count=1,
        )
        assert not verdict.is_excluded

    def test_should_keep_a_document_losing_one_entity_of_forty(self):
        # An element holding nothing but a <label> cannot produce an entity.
        verdict = _verdict(
            jats_status='ok', jats_reference_count=37, written=True,
            entity_element_count=37, entity_start_count=36, sequence_count=1,
        )
        assert not verdict.is_excluded

    def test_should_exclude_a_document_with_no_training_sequences(self):
        verdict = _verdict(
            jats_status='ok', jats_reference_count=40, written=True,
            entity_element_count=0, entity_start_count=0, sequence_count=0,
        )
        assert ExclusionReason.NO_TRAINING_SEQUENCES in verdict.exclusion_reasons

    def test_should_name_the_earliest_stage_first(self):
        verdict = _verdict(
            jats_status='ok', jats_reference_count=0, written=False, sequence_count=0,
        )
        assert verdict.primary_reason == ExclusionReason.JATS_HAS_NO_REFERENCES

    def test_should_not_treat_an_unavailable_count_as_a_failure(self):
        # Assembly is run over corpora with no record at all.
        verdict = _verdict(entity_start_count=40, sequence_count=1)
        assert not verdict.is_excluded

    def test_should_apply_no_cardinality_check_to_a_region_model(self):
        verdict = _verdict(
            thresholds=ModelQualityThresholds(
                model_name='segmentation', cardinality='none', reason='regions'
            ),
            jats_status='ok', jats_reference_count=0, written=True, sequence_count=1,
        )
        assert not verdict.is_excluded


class TestGetGateSummaryByCorpus:
    def test_should_count_what_it_kept_and_dropped_per_corpus(self):
        summary_by_corpus = get_gate_summary_by_corpus([
            (QualityVerdict('kept1'), 'ore'),
            (QualityVerdict('kept2'), 'ore'),
            (
                QualityVerdict('dropped', [ExclusionReason.JATS_NOT_READABLE]),
                'ore'
            ),
            (QualityVerdict('kept3'), 'scielo_preprints-jats'),
        ])
        assert summary_by_corpus['ore'].kept_count == 2
        assert summary_by_corpus['ore'].excluded_count == 1
        assert summary_by_corpus['scielo_preprints-jats'].excluded_count == 0

    def test_should_report_the_dropped_documents_by_reason(self):
        summary = get_gate_summary_by_corpus([
            (
                QualityVerdict('truncated', [ExclusionReason.ELEMENTS_SHORT_OF_JATS]),
                'ore'
            ),
            (QualityVerdict('kept'), 'ore'),
        ])['ore']
        assert summary.excluded_by_reason == {
            ExclusionReason.ELEMENTS_SHORT_OF_JATS: ['truncated']
        }
        assert 'truncated' in str(summary)
        assert 'kept 1 of 2' in str(summary)


class TestCheckCorpusLossOrFail:
    def test_should_proceed_when_a_corpus_keeps_most_of_its_documents(self):
        check_corpus_loss_or_fail(
            get_gate_summary_by_corpus(
                [(QualityVerdict(f'kept{index}'), 'ore') for index in range(9)]
                + [(QualityVerdict('dropped', [ExclusionReason.JATS_NOT_READABLE]), 'ore')]
            ),
            max_excluded_ratio=0.2
        )

    def test_should_refuse_when_a_corpus_is_mostly_excluded(self):
        with pytest.raises(CorpusMostlyExcludedError) as exc_info:
            check_corpus_loss_or_fail(
                get_gate_summary_by_corpus([
                    (
                        QualityVerdict(
                            f'dropped{index}',
                            [ExclusionReason.ENTITIES_SHORT_OF_ELEMENTS]
                        ),
                        'ore'
                    )
                    for index in range(6)
                ] + [(QualityVerdict(f'kept{index}'), 'ore') for index in range(4)]),
                max_excluded_ratio=0.2
            )
        assert 'ore' in str(exc_info.value)

    def test_should_refuse_for_one_corpus_even_when_another_is_sound(self):
        with pytest.raises(CorpusMostlyExcludedError):
            check_corpus_loss_or_fail(
                get_gate_summary_by_corpus(
                    [(QualityVerdict('dropped', [ExclusionReason.JATS_NOT_READABLE]), 'ore')]
                    + [
                        (QualityVerdict(f'kept{index}'), 'scielo_preprints-jats')
                        for index in range(10)
                    ]
                ),
                max_excluded_ratio=0.2
            )


class TestGetQualityVerdictWithoutJats:
    def test_should_not_exclude_a_document_generated_without_any_jats(self):
        # Training data is legitimately generated with no --source-xml-path, which
        # records the JATS as missing. There is nothing to check against, and
        # excluding on it would refuse the whole corpus.
        verdict = get_quality_verdict(
            document_id=DOCUMENT_ID_1, thresholds=REFERENCE_SEGMENTER_THRESHOLDS,
            jats_status='missing', written=True, entity_start_count=40, sequence_count=1,
        )
        assert not verdict.is_excluded

    def test_should_still_exclude_a_jats_that_could_not_be_read(self):
        for jats_status in ['unparsable', 'unreadable']:
            verdict = get_quality_verdict(
                document_id=DOCUMENT_ID_1, thresholds=REFERENCE_SEGMENTER_THRESHOLDS,
                jats_status=jats_status, written=True, sequence_count=1,
            )
            assert verdict.primary_reason == ExclusionReason.JATS_NOT_READABLE

    def test_should_name_the_earliest_stage_first_when_several_failed(self):
        # No training sequences is the last stage, so a shortfall before it leads.
        verdict = get_quality_verdict(
            document_id=DOCUMENT_ID_1, thresholds=REFERENCE_SEGMENTER_THRESHOLDS,
            jats_status='ok', jats_reference_count=45, written=True,
            entity_element_count=2, entity_start_count=0, sequence_count=0,
        )
        assert verdict.primary_reason == ExclusionReason.ELEMENTS_SHORT_OF_JATS
        assert verdict.exclusion_reasons[-1] == ExclusionReason.NO_TRAINING_SEQUENCES


class TestConfiguredModelsAgainstCounting:
    def test_should_declare_the_same_models_as_having_no_entity_count(self):
        # Two lists of the same fact drift apart; the gate would then report a
        # cardinality it never counts, or count one it does not gate.
        config = load_training_quality_config()
        without_cardinality = {
            model_name
            for model_name, thresholds in config.thresholds_by_model.items()
            if not thresholds.has_cardinality_check
        }
        assert without_cardinality == set(MODELS_WITHOUT_ENTITY_COUNT)

    def test_should_have_a_reason_for_every_model_without_a_check(self):
        config = load_training_quality_config()
        for thresholds in config.thresholds_by_model.values():
            if not thresholds.has_cardinality_check:
                assert thresholds.reason, thresholds.model_name
