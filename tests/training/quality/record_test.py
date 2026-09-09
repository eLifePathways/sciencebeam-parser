import json
from pathlib import Path

from sciencebeam_parser.training.quality.record import (
    DocumentQualityRecord,
    DocumentStatus,
    JatsQualityRecord,
    JatsStatus,
    ModelQualityRecord,
    QualityRecordWriter,
    get_failed_document_quality_record,
    get_quality_record_file_path
)


DOCUMENT_ID_1 = 'document1'
SOURCE_FILENAME_1 = '/source/document1.pdf'
REFERENCE_SEGMENTER = 'reference-segmenter'


def _record_for_models(*models: ModelQualityRecord) -> DocumentQualityRecord:
    return DocumentQualityRecord(
        document_id=DOCUMENT_ID_1,
        source_filename=SOURCE_FILENAME_1,
        jats=JatsQualityRecord(
            status=JatsStatus.OK, reference_count=45, aligned_reference_count=2
        ),
        models=models,
    )


class TestGetQualityRecordFilePath:
    def test_should_use_a_directory_per_model(self):
        assert get_quality_record_file_path(
            '/output', REFERENCE_SEGMENTER, use_directory_structure=True
        ) == '/output/reference-segmenter/quality.jsonl'

    def test_should_stay_flat_without_the_directory_structure(self):
        assert get_quality_record_file_path(
            '/output', REFERENCE_SEGMENTER, use_directory_structure=False
        ) == '/output/reference-segmenter.quality.jsonl'


class TestDocumentQualityRecord:
    def test_should_hold_the_count_at_each_stage(self):
        json_dict = _record_for_models(
            ModelQualityRecord(
                model_name=REFERENCE_SEGMENTER, written=True, entity_element_count=2
            )
        ).to_json_dict_by_model([REFERENCE_SEGMENTER])[REFERENCE_SEGMENTER]
        assert json_dict['document_id'] == DOCUMENT_ID_1
        assert json_dict['model'] == REFERENCE_SEGMENTER
        assert json_dict['status'] == DocumentStatus.OK
        assert json_dict['jats'] == {
            'status': JatsStatus.OK, 'reference_count': 45, 'aligned_reference_count': 2
        }
        assert json_dict['written'] is True
        assert json_dict['entity_element_count'] == 2

    def test_should_return_one_row_per_model(self):
        json_dict_by_model = _record_for_models(
            ModelQualityRecord(
                model_name=REFERENCE_SEGMENTER, written=True, entity_element_count=2
            ),
            ModelQualityRecord(
                model_name='citation',
                written=True,
                entity_element_count=2,
                label_counts={'<title>': {'jats': 44, 'marked': 2}}
            ),
        ).to_json_dict_by_model([REFERENCE_SEGMENTER, 'citation'])
        assert set(json_dict_by_model) == {REFERENCE_SEGMENTER, 'citation'}
        assert 'label_counts' not in json_dict_by_model[REFERENCE_SEGMENTER]
        assert json_dict_by_model['citation']['label_counts'] == {
            '<title>': {'jats': 44, 'marked': 2}
        }

    def test_should_record_a_model_that_wrote_no_file(self):
        json_dict = _record_for_models(
            ModelQualityRecord(
                model_name=REFERENCE_SEGMENTER, written=False, entity_element_count=0
            )
        ).to_json_dict_by_model([REFERENCE_SEGMENTER])[REFERENCE_SEGMENTER]
        assert json_dict['written'] is False
        assert json_dict['entity_element_count'] == 0

    def test_should_omit_an_entity_count_a_model_does_not_have(self):
        json_dict = _record_for_models(
            ModelQualityRecord(model_name='segmentation', written=True)
        ).to_json_dict_by_model(['segmentation'])['segmentation']
        assert json_dict['written'] is True
        assert 'entity_element_count' not in json_dict

    def test_should_record_a_jats_that_could_not_be_parsed_without_counts(self):
        json_dict = DocumentQualityRecord(
            document_id=DOCUMENT_ID_1,
            source_filename=SOURCE_FILENAME_1,
            jats=JatsQualityRecord(status=JatsStatus.UNPARSABLE),
            models=[ModelQualityRecord(model_name=REFERENCE_SEGMENTER, written=False)],
        ).to_json_dict_by_model([REFERENCE_SEGMENTER])[REFERENCE_SEGMENTER]
        assert json_dict['jats'] == {'status': JatsStatus.UNPARSABLE}

    def test_should_record_a_jats_declaring_no_references_as_a_zero_count(self):
        json_dict = DocumentQualityRecord(
            document_id=DOCUMENT_ID_1,
            source_filename=SOURCE_FILENAME_1,
            jats=JatsQualityRecord(
                status=JatsStatus.OK, reference_count=0, aligned_reference_count=0
            ),
            models=[ModelQualityRecord(model_name=REFERENCE_SEGMENTER, written=False)],
        ).to_json_dict_by_model([REFERENCE_SEGMENTER])[REFERENCE_SEGMENTER]
        assert json_dict['jats']['reference_count'] == 0


class TestGetFailedDocumentQualityRecord:
    def test_should_give_a_timed_out_document_a_row_for_every_model(self):
        json_dict_by_model = get_failed_document_quality_record(
            source_filename=SOURCE_FILENAME_1,
            document_id=DOCUMENT_ID_1,
            status=DocumentStatus.TIMEOUT,
        ).to_json_dict_by_model([REFERENCE_SEGMENTER, 'citation'])
        assert set(json_dict_by_model) == {REFERENCE_SEGMENTER, 'citation'}
        for json_dict in json_dict_by_model.values():
            assert json_dict['document_id'] == DOCUMENT_ID_1
            assert json_dict['status'] == DocumentStatus.TIMEOUT
            assert 'written' not in json_dict


class TestQualityRecordWriter:
    def test_should_write_one_file_per_model(self, tmp_path: Path):
        with QualityRecordWriter(
            str(tmp_path), model_names=[REFERENCE_SEGMENTER, 'citation']
        ) as writer:
            writer.write(_record_for_models(
                ModelQualityRecord(
                    model_name=REFERENCE_SEGMENTER, written=True, entity_element_count=2
                ),
                ModelQualityRecord(
                    model_name='citation', written=True, entity_element_count=3
                ),
            ))
            assert writer.written_count == 1
        for model_name, expected_count in [(REFERENCE_SEGMENTER, 2), ('citation', 3)]:
            lines = (
                tmp_path / model_name / 'quality.jsonl'
            ).read_text(encoding='utf-8').splitlines()
            assert [json.loads(line)['entity_element_count'] for line in lines] == [
                expected_count
            ]

    def test_should_write_one_line_per_document(self, tmp_path: Path):
        with QualityRecordWriter(
            str(tmp_path), model_names=[REFERENCE_SEGMENTER]
        ) as writer:
            for document_id in ['document1', 'document2']:
                writer.write(DocumentQualityRecord(
                    document_id=document_id,
                    source_filename=f'/source/{document_id}.pdf',
                    models=[
                        ModelQualityRecord(model_name=REFERENCE_SEGMENTER, written=True)
                    ],
                ))
        lines = (
            tmp_path / REFERENCE_SEGMENTER / 'quality.jsonl'
        ).read_text(encoding='utf-8').splitlines()
        assert [json.loads(line)['document_id'] for line in lines] == [
            'document1', 'document2'
        ]

    def test_should_flush_each_record_so_an_interrupted_run_keeps_what_it_had(
        self, tmp_path: Path
    ):
        with QualityRecordWriter(
            str(tmp_path), model_names=[REFERENCE_SEGMENTER]
        ) as writer:
            writer.write(DocumentQualityRecord(
                document_id=DOCUMENT_ID_1,
                source_filename=SOURCE_FILENAME_1,
                models=[
                    ModelQualityRecord(model_name=REFERENCE_SEGMENTER, written=True)
                ],
            ))
            record_file_path = tmp_path / REFERENCE_SEGMENTER / 'quality.jsonl'
            assert len(record_file_path.read_text(encoding='utf-8').splitlines()) == 1
