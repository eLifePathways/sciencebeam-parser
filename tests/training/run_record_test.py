import json

from sciencebeam_parser.training.quality.assembly import AssembledDocumentRecord
from sciencebeam_parser.training.quality.gate import ExclusionReason, QualityVerdict
from sciencebeam_parser.training.run_record import (
    get_run_record,
    get_run_record_path_for_delft_output_path,
    write_run_record
)


def _record(document_id: str, **kwargs) -> AssembledDocumentRecord:
    return AssembledDocumentRecord(
        document_id=document_id, model_name='segmentation', **kwargs
    )


def _run_record(records, **kwargs):
    return get_run_record(
        model_name='segmentation',
        tei_source_path='corpus/tei/*.tei.xml',
        raw_source_path='corpus/raw',
        assembled_records=records,
        **kwargs
    )


class TestGetRunRecordPathForDelftOutputPath:
    def test_should_append_the_suffix(self):
        assert get_run_record_path_for_delft_output_path(
            'out/data.train.gz'
        ) == 'out/data.train.gz.run.json'


class TestGetRunRecord:
    def test_should_name_the_source_and_its_revision(self):
        run_record = _run_record(
            [], source_name='grobid', source_revision='0.9.0 (b2251cb54)'
        )
        assert run_record['source']['name'] == 'grobid'
        assert run_record['source']['revision'] == '0.9.0 (b2251cb54)'
        assert run_record['source']['tei_source_path'] == 'corpus/tei/*.tei.xml'

    def test_should_total_the_rows_the_pairing_reported(self):
        run_record = _run_record([
            _record('d1', matched_row_count=90, unmatched_row_count=10),
            _record('d2', matched_row_count=100, uncovered_row_count=400),
        ])
        assert run_record['rows'] == {
            'matched': 190, 'unmatched': 10, 'uncovered': 400,
            'matched_ratio': 0.95
        }

    def test_should_say_nothing_of_a_ratio_where_nothing_was_paired(self):
        assert _run_record([_record('d1')])['rows']['matched_ratio'] is None

    def test_should_name_every_refused_document_with_its_reason(self):
        refused = _record(
            'd2', matched_row_ratio=0.5,
            verdict=QualityVerdict(
                'd2', [ExclusionReason.TOO_FEW_MATCHED_ROWS], {}
            )
        )
        run_record = _run_record([_record('d1'), refused])
        assert run_record['documents'] == {'read': 2, 'refused': 1}
        assert run_record['refused_documents'] == [{
            'document_id': 'd2',
            'reasons': [ExclusionReason.TOO_FEW_MATCHED_ROWS],
            'matched_row_ratio': 0.5
        }]

    def test_should_total_the_unrecognised_element_paths(self):
        run_record = _run_record([
            _record('d1', unrecognised_path_counts={'back/listBibl': 3}),
            _record('d2', unrecognised_path_counts={'back/listBibl': 2, 'other': 1}),
        ])
        assert run_record['unrecognised_element_paths'] == {
            'back/listBibl': 5, 'other': 1
        }


class TestWriteRunRecord:
    def test_should_write_readable_json(self, tmp_path):
        path = tmp_path / 'data.train.run.json'
        write_run_record(str(path), _run_record([_record('d1')], source_name='grobid'))
        assert json.loads(path.read_text(encoding='utf-8'))['source']['name'] == 'grobid'

    def test_should_write_beside_data_that_is_not_a_plain_local_file(self, tmp_path):
        """The data often goes straight to cloud storage, and the record follows it.

        A plain `open` cannot write a `gs://` path at all. Standing in for that
        here: the writer has to go through the same uploading helper the rest of
        the assembly uses, which `open` would fail on as well.
        """
        path = tmp_path / 'not' / 'yet' / 'there' / 'data.train.run.json'
        write_run_record(str(path), _run_record([_record('d1')], source_name='grobid'))
        assert json.loads(path.read_text(encoding='utf-8'))['source']['name'] == 'grobid'
