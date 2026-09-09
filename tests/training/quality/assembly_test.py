import json
from pathlib import Path

import pytest

from sciencebeam_parser.training.quality.assembly import (
    AssembledDocumentRecord,
    GeneratedDocumentRecord,
    get_assembly_summary_by_corpus,
    get_corpus_name_for_record_file_path,
    get_document_ids_without_generated_output,
    read_generated_document_records,
    write_assembly_records
)


DOCUMENT_ID_1 = 'document1'
REFERENCE_SEGMENTER = 'reference-segmenter'


def _generated(
    document_id: str = DOCUMENT_ID_1,
    corpus: str = 'ore',
    reference_count: int = 45,
    entity_element_count: int = 45,
    written: bool = True
) -> GeneratedDocumentRecord:
    return GeneratedDocumentRecord(
        document_id=document_id,
        corpus=corpus,
        json_dict={
            'document_id': document_id,
            'model': REFERENCE_SEGMENTER,
            'jats': {'status': 'ok', 'reference_count': reference_count},
            'written': written,
            'entity_element_count': entity_element_count,
        },
    )


class TestGetCorpusNameForRecordFilePath:
    def test_should_take_the_directory_above_the_model(self):
        assert get_corpus_name_for_record_file_path(
            '/data/train/ore/reference-segmenter/quality.jsonl'
        ) == 'ore'

    def test_should_be_none_without_a_directory_to_read(self):
        assert get_corpus_name_for_record_file_path('quality.jsonl') is None


class TestReadGeneratedDocumentRecords:
    def test_should_read_the_rows_and_the_corpus_they_came_from(self, tmp_path: Path):
        record_file_path = tmp_path / 'train' / 'ore' / REFERENCE_SEGMENTER / 'quality.jsonl'
        record_file_path.parent.mkdir(parents=True)
        record_file_path.write_text('\n'.join([
            json.dumps({'document_id': 'document1', 'entity_element_count': 12}),
            json.dumps({'document_id': 'document2', 'entity_element_count': 34}),
        ]) + '\n', encoding='utf-8')
        record_by_document_id = read_generated_document_records(str(record_file_path))
        assert set(record_by_document_id) == {'document1', 'document2'}
        assert record_by_document_id['document1'].corpus == 'ore'
        assert record_by_document_id['document2'].entity_element_count == 34

    def test_should_fail_when_no_record_matches(self, tmp_path: Path):
        with pytest.raises(RuntimeError):
            read_generated_document_records(str(tmp_path / 'not-there' / '*.jsonl'))


class TestAssembledDocumentRecord:
    def test_should_report_entities_lost_at_the_parse(self):
        record = AssembledDocumentRecord(
            document_id=DOCUMENT_ID_1,
            model_name=REFERENCE_SEGMENTER,
            entity_start_count=1,
            generated=_generated(entity_element_count=45),
        )
        assert record.lost_at_parse == 44

    def test_should_report_nothing_lost_when_every_element_returned_an_entity(self):
        record = AssembledDocumentRecord(
            document_id=DOCUMENT_ID_1,
            model_name=REFERENCE_SEGMENTER,
            entity_start_count=45,
            generated=_generated(entity_element_count=45),
        )
        assert record.lost_at_parse == 0

    def test_should_not_claim_a_loss_without_a_record_from_generation(self):
        record = AssembledDocumentRecord(
            document_id=DOCUMENT_ID_1,
            model_name=REFERENCE_SEGMENTER,
            entity_start_count=45,
        )
        assert record.lost_at_parse is None

    def test_should_carry_the_generated_counts_into_its_row(self):
        json_dict = AssembledDocumentRecord(
            document_id=DOCUMENT_ID_1,
            model_name=REFERENCE_SEGMENTER,
            corpus='ore',
            sequence_count=1,
            entity_start_count=45,
            generated=_generated(),
        ).to_json_dict()
        assert json_dict['corpus'] == 'ore'
        assert json_dict['entity_start_count'] == 45
        assert json_dict['generated']['jats']['reference_count'] == 45


class TestGetAssemblySummaryByCorpus:
    def test_should_total_the_counts_per_corpus(self):
        summary_by_corpus = get_assembly_summary_by_corpus([
            AssembledDocumentRecord(
                document_id='document1', model_name=REFERENCE_SEGMENTER, corpus='ore',
                sequence_count=1, entity_start_count=10,
                generated=_generated('document1', entity_element_count=10),
            ),
            AssembledDocumentRecord(
                document_id='document2', model_name=REFERENCE_SEGMENTER, corpus='ore',
                sequence_count=1, entity_start_count=20,
                generated=_generated('document2', entity_element_count=20),
            ),
        ])
        summary = summary_by_corpus['ore']
        assert summary.document_count == 2
        assert summary.entity_start_count == 30
        assert summary.entity_element_count == 30
        assert not summary.documents_losing_entities

    def test_should_name_the_documents_that_lost_entities(self):
        summary_by_corpus = get_assembly_summary_by_corpus([
            AssembledDocumentRecord(
                document_id='collapsed', model_name=REFERENCE_SEGMENTER, corpus='ore',
                sequence_count=1, entity_start_count=1,
                generated=_generated('collapsed', entity_element_count=40),
            ),
            AssembledDocumentRecord(
                document_id='intact', model_name=REFERENCE_SEGMENTER, corpus='ore',
                sequence_count=1, entity_start_count=40,
                generated=_generated('intact', entity_element_count=40),
            ),
        ])
        summary = summary_by_corpus['ore']
        assert summary.documents_losing_entities == ['collapsed']
        assert 'collapsed' in str(summary)

    def test_should_count_a_document_with_no_record_from_generation(self):
        summary_by_corpus = get_assembly_summary_by_corpus([
            AssembledDocumentRecord(
                document_id='unrecorded', model_name=REFERENCE_SEGMENTER,
                sequence_count=1, entity_start_count=3,
            ),
        ])
        assert summary_by_corpus[None].documents_without_generated_record == ['unrecorded']

    def test_should_keep_corpora_apart(self):
        summary_by_corpus = get_assembly_summary_by_corpus([
            AssembledDocumentRecord(
                document_id='document1', model_name=REFERENCE_SEGMENTER, corpus='ore',
                sequence_count=1, entity_start_count=10,
            ),
            AssembledDocumentRecord(
                document_id='document2', model_name=REFERENCE_SEGMENTER,
                corpus='scielo_preprints-jats',
                sequence_count=1, entity_start_count=20,
            ),
        ])
        assert summary_by_corpus['ore'].entity_start_count == 10
        assert summary_by_corpus['scielo_preprints-jats'].entity_start_count == 20


class TestGetDocumentIdsWithoutGeneratedOutput:
    def test_should_find_a_document_generation_wrote_no_file_for(self):
        assert get_document_ids_without_generated_output({
            'document1': _generated('document1'),
            'document2': _generated('document2', written=False),
        }) == ['document2']

    def test_should_find_a_document_that_failed_before_writing_anything(self):
        record = GeneratedDocumentRecord(
            document_id='timed-out',
            corpus='ore',
            json_dict={'document_id': 'timed-out', 'status': 'timeout'},
        )
        assert get_document_ids_without_generated_output({
            'timed-out': record
        }) == ['timed-out']

    def test_should_find_nothing_when_every_document_was_written(self):
        assert not get_document_ids_without_generated_output({
            'document1': _generated('document1')
        })

    def test_should_not_depend_on_which_documents_were_assembled(self):
        # Assembly is often pointed at part of a corpus; the record is what says
        # whether a file was written, so a narrowed run must not report the rest.
        assert not get_document_ids_without_generated_output({
            'document1': _generated('document1'),
            'document2': _generated('document2'),
        })


class TestWriteAssemblyRecords:
    def test_should_write_one_json_line_per_document(self, tmp_path: Path):
        output_file_path = tmp_path / 'reference-segmenter.data.quality.jsonl'
        write_assembly_records(str(output_file_path), [
            AssembledDocumentRecord(
                document_id='document1', model_name=REFERENCE_SEGMENTER,
                sequence_count=1, entity_start_count=10,
            ),
            AssembledDocumentRecord(
                document_id='document2', model_name=REFERENCE_SEGMENTER,
                sequence_count=1, entity_start_count=20,
            ),
        ])
        lines = output_file_path.read_text(encoding='utf-8').splitlines()
        assert [json.loads(line)['document_id'] for line in lines] == [
            'document1', 'document2'
        ]
        assert [json.loads(line)['entity_start_count'] for line in lines] == [10, 20]


class TestCorpusAssemblySummaryForLabelCountedModel:
    def test_should_not_report_a_missing_entity_count_as_zero_entities(self):
        # Every citation element is its own sequence, so it has no entity count;
        # reporting it as zero would read as every entity lost.
        summary = get_assembly_summary_by_corpus([
            AssembledDocumentRecord(
                document_id=DOCUMENT_ID_1, model_name='citation', corpus='ore',
                sequence_count=45, entity_start_count=None,
                label_start_counts={'<title>': 44},
                generated=_generated(entity_element_count=45),
            ),
        ])['ore']
        assert summary.entity_start_count is None
        assert 'entities' not in str(summary)
        assert '45 sequences' in str(summary)
