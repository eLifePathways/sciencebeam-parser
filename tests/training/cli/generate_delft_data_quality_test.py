# pylint: disable=not-callable
"""The gate's end-to-end behaviour: what a measurement excludes, and what a
review admits or refuses on top of it."""
import json
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import pytest
import yaml

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_delft.sequence_labelling.reader import (
    load_data_and_labels_crf_file
)

from sciencebeam_parser.training.cli.generate_delft_data import main
from sciencebeam_parser.training.quality.gate import (
    CorpusMostlyExcludedError,
    CorpusMostlyUnreviewedError
)
from sciencebeam_parser.training.quality.requirement import RequirementConfigError

from tests.test_utils import log_on_exception


def _write_reference_segmenter_tei(
    tei_source_path: Path, document_id: str, bibl_count: int
) -> None:
    tei_source_path.mkdir(parents=True, exist_ok=True)
    (
        tei_source_path / f'{document_id}.references.referenceSegmenter.tei.xml'
    ).write_bytes(etree.tostring(E('tei', E('text', E('listBibl', *[
        child
        for index in range(bibl_count)
        for child in (E('bibl', f'reference{index}', E('lb')), '\n')
    ])))))


@log_on_exception
class TestQualityFilter:
    def _write_tei(self, tei_source_path: Path, document_id: str, bibl_count: int) -> None:
        _write_reference_segmenter_tei(tei_source_path, document_id, bibl_count)

    def _write_generated_records(
        self, records_path: Path, rows: Sequence[dict]
    ) -> str:
        records_path.mkdir(parents=True, exist_ok=True)
        for row in rows:
            (
                records_path
                / f'{row["document_id"]}.reference-segmenter.quality.json'
            ).write_text(json.dumps(row) + '\n', encoding='utf-8')
        return str(records_path / '*.quality.json')

    def _run(self, tmp_path: Path, tei_source_path: Path, record_path: str, *extra):
        output_path = tmp_path / 'output.data'
        main([
            '--model-name=reference_segmenter',
            f'--tei-source-path={tei_source_path}/*.tei.xml',
            f'--quality-record-path={record_path}',
            f'--delft-output-path={output_path}',
            '--quality-filter',
            *extra
        ])
        return output_path

    def test_should_leave_out_a_document_short_at_the_tei_stage(self, tmp_path: Path):
        tei_source_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'corpus' / 'tei'
        self._write_tei(tei_source_path, 'truncated', bibl_count=2)
        for index in range(9):
            self._write_tei(tei_source_path, f'sound{index}', bibl_count=4)
        records_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'quality'
        record_path = self._write_generated_records(records_path, [
            {
                'document_id': 'truncated', 'written': True,
                'jats': {'status': 'ok', 'reference_count': 45},
                'entity_element_count': 2,
            },
        ] + [
            {
                'document_id': f'sound{index}', 'written': True,
                'jats': {'status': 'ok', 'reference_count': 4},
                'entity_element_count': 4,
            }
            for index in range(9)
        ])
        output_path = self._run(tmp_path, tei_source_path, record_path)
        texts, _labels, _features = load_data_and_labels_crf_file(str(output_path))
        assert len(texts) == 9
        rows = {
            json.loads(line)['document_id']: json.loads(line)
            for line in Path(
                str(output_path) + '.quality.jsonl'
            ).read_text(encoding='utf-8').splitlines()
        }
        assert rows['truncated']['excluded'] is True
        assert rows['truncated']['exclusion_reasons'] == ['elements-short-of-jats']
        assert rows['truncated']['exclusion_detail']['element_ratio'] == 0.044
        assert rows['sound0']['excluded'] is False

    def test_should_keep_every_document_without_the_filter(self, tmp_path: Path):
        tei_source_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'corpus' / 'tei'
        self._write_tei(tei_source_path, 'truncated', bibl_count=2)
        records_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'quality'
        record_path = self._write_generated_records(records_path, [{
            'document_id': 'truncated', 'written': True,
            'jats': {'status': 'ok', 'reference_count': 45},
            'entity_element_count': 2,
        }])
        output_path = tmp_path / 'output.data'
        main([
            '--model-name=reference_segmenter',
            f'--tei-source-path={tei_source_path}/*.tei.xml',
            f'--quality-record-path={record_path}',
            f'--delft-output-path={output_path}'
        ])
        texts, _labels, _features = load_data_and_labels_crf_file(str(output_path))
        assert len(texts) == 1
        row = json.loads(
            Path(str(output_path) + '.quality.jsonl').read_text(encoding='utf-8')
        )
        assert 'excluded' not in row

    def test_should_leave_no_usable_data_behind_when_it_refuses(self, tmp_path: Path):
        tei_source_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'corpus' / 'tei'
        for index in range(3):
            self._write_tei(tei_source_path, f'truncated{index}', bibl_count=2)
        records_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'quality'
        record_path = self._write_generated_records(records_path, [
            {
                'document_id': f'truncated{index}', 'written': True,
                'jats': {'status': 'ok', 'reference_count': 45},
                'entity_element_count': 2,
            }
            for index in range(3)
        ])
        with pytest.raises(CorpusMostlyExcludedError):
            self._run(tmp_path, tei_source_path, record_path)
        # a training run reading the path must not find a corpus there
        assert not (tmp_path / 'output.data').exists()
        # the record stays, so the refusal can be accounted for
        assert (tmp_path / 'output.data.quality.jsonl').exists()

    def test_should_refuse_when_a_corpus_is_mostly_excluded(self, tmp_path: Path):
        tei_source_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'corpus' / 'tei'
        for index in range(3):
            self._write_tei(tei_source_path, f'truncated{index}', bibl_count=2)
        records_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'quality'
        record_path = self._write_generated_records(records_path, [
            {
                'document_id': f'truncated{index}', 'written': True,
                'jats': {'status': 'ok', 'reference_count': 45},
                'entity_element_count': 2,
            }
            for index in range(3)
        ])
        with pytest.raises(CorpusMostlyExcludedError):
            self._run(tmp_path, tei_source_path, record_path)

    def test_should_allow_a_stated_larger_loss(self, tmp_path: Path):
        tei_source_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'corpus' / 'tei'
        for index in range(3):
            self._write_tei(tei_source_path, f'truncated{index}', bibl_count=2)
        records_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'quality'
        record_path = self._write_generated_records(records_path, [
            {
                'document_id': f'truncated{index}', 'written': True,
                'jats': {'status': 'ok', 'reference_count': 45},
                'entity_element_count': 2,
            }
            for index in range(3)
        ])
        output_path = self._run(
            tmp_path, tei_source_path, record_path, '--max-excluded-ratio=1.0'
        )
        assert output_path.exists()


HASH_1 = 'v1:sha256:aaaa'
HASH_SINCE_REGENERATED = 'v1:sha256:bbbb'


class TestWhatAReviewDecided:
    """The gate reading verdicts, from the files a reviewed corpus commits."""

    def _write_corpus(
        self,
        tmp_path: Path,
        verdict_by_document_id: Mapping[str, str],
        verdict_hash: str = HASH_1
    ) -> Tuple[Path, str]:
        model_path = tmp_path / 'train' / 'ore' / 'reference-segmenter'
        tei_source_path = model_path / 'corpus' / 'tei'
        records_path = model_path / 'quality'
        records_path.mkdir(parents=True)
        (model_path / 'verdict').mkdir(parents=True)
        for document_id, verdict in verdict_by_document_id.items():
            # Two of the four references the JATS declares, so the measurement
            # excludes every document unless a review says otherwise.
            _write_reference_segmenter_tei(tei_source_path, document_id, bibl_count=2)
            (
                records_path / f'{document_id}.reference-segmenter.quality.json'
            ).write_text(json.dumps({
                'document_id': document_id, 'written': True,
                'jats': {'status': 'ok', 'reference_count': 4},
                'entity_element_count': 2,
                'labelled_content_hash': HASH_1,
            }) + '\n', encoding='utf-8')
            if verdict:
                (
                    model_path / 'verdict'
                    / f'{document_id}.reference-segmenter.verdict.json'
                ).write_text(json.dumps({
                    'document_id': document_id, 'verdict': verdict,
                    'labelled_content_hash': verdict_hash,
                }) + '\n', encoding='utf-8')
        return tei_source_path, str(records_path / '*.quality.json')

    def _write_require_config(self, tmp_path: Path, requirement: str) -> str:
        config_file_path = tmp_path / 'training-source.yml'
        config_file_path.write_text(
            yaml.safe_dump({'require': {'ore': {'reference-segmenter': requirement}}}),
            encoding='utf-8'
        )
        return str(config_file_path)

    def _run(self, tmp_path: Path, tei_source_path: Path, record_path: str, *extra):
        output_path = tmp_path / 'output.data'
        main([
            '--model-name=reference_segmenter',
            f'--tei-source-path={tei_source_path}/*.tei.xml',
            f'--quality-record-path={record_path}',
            f'--delft-output-path={output_path}',
            '--quality-filter',
            *extra
        ])
        return output_path

    def test_should_admit_a_document_the_thresholds_would_have_excluded(
        self, tmp_path: Path
    ):
        tei_source_path, record_path = self._write_corpus(
            tmp_path, {f'accepted{index}': 'accepted' for index in range(10)}
        )
        output_path = self._run(
            tmp_path, tei_source_path, record_path,
            f'--require-config-path={self._write_require_config(tmp_path, "accepted")}'
        )
        texts, _labels, _features = load_data_and_labels_crf_file(str(output_path))
        assert len(texts) == 10

    def test_should_keep_what_the_measurement_said_beside_the_admission(
        self, tmp_path: Path
    ):
        tei_source_path, record_path = self._write_corpus(
            tmp_path, {'accepted1': 'accepted'}
        )
        output_path = self._run(
            tmp_path, tei_source_path, record_path,
            f'--require-config-path={self._write_require_config(tmp_path, "accepted")}'
        )
        row = json.loads(
            Path(str(output_path) + '.quality.jsonl').read_text(encoding='utf-8')
        )
        assert row['review'] == 'accepted'
        assert not row['excluded']

    def test_should_exclude_a_rejected_document_and_name_why(self, tmp_path: Path):
        verdicts = {f'accepted{index}': 'accepted' for index in range(9)}
        verdicts['rejected1'] = 'rejected'
        tei_source_path, record_path = self._write_corpus(tmp_path, verdicts)
        output_path = self._run(
            tmp_path, tei_source_path, record_path,
            f'--require-config-path={self._write_require_config(tmp_path, "accepted")}'
        )
        rows = {
            row['document_id']: row
            for row in (
                json.loads(line)
                for line in Path(
                    str(output_path) + '.quality.jsonl'
                ).read_text(encoding='utf-8').splitlines()
            )
        }
        assert rows['rejected1']['excluded']
        assert rows['rejected1']['exclusion_reasons'][0] == 'rejected-by-review'
        texts, _labels, _features = load_data_and_labels_crf_file(str(output_path))
        assert len(texts) == 9

    def test_should_refuse_a_corpus_most_of_which_nobody_has_read(self, tmp_path: Path):
        tei_source_path, record_path = self._write_corpus(
            tmp_path, {f'unread{index}': '' for index in range(10)}
        )
        with pytest.raises(CorpusMostlyUnreviewedError):
            self._run(
                tmp_path, tei_source_path, record_path,
                '--require-config-path='
                + self._write_require_config(tmp_path, 'accepted')
            )

    def test_should_keep_an_unread_document_where_only_a_rejection_excludes(
        self, tmp_path: Path
    ):
        # A rejection a regeneration lapsed leaves the document unread, which is
        # visible rather than binding for a corpus that demands only this.
        tei_source_path, record_path = self._write_corpus(
            tmp_path, {f'unread{index}': '' for index in range(10)}
        )
        output_path = self._run(
            tmp_path, tei_source_path, record_path,
            '--require-config-path='
            + self._write_require_config(tmp_path, 'not-rejected'),
            '--max-excluded-ratio=1.0'
        )
        rows = [
            json.loads(line)
            for line in Path(
                str(output_path) + '.quality.jsonl'
            ).read_text(encoding='utf-8').splitlines()
        ]
        assert all(row['review'] == 'unreviewed' for row in rows)
        assert all(row['exclusion_reasons'] == ['elements-short-of-jats'] for row in rows)

    def test_should_refuse_a_pair_the_corpus_declares_nothing_for(self, tmp_path: Path):
        tei_source_path, record_path = self._write_corpus(
            tmp_path, {'accepted1': 'accepted'}
        )
        config_file_path = tmp_path / 'training-source.yml'
        config_file_path.write_text(
            yaml.safe_dump({'require': {'ore': {'segmentation': 'accepted'}}}),
            encoding='utf-8'
        )
        with pytest.raises(RequirementConfigError):
            self._run(
                tmp_path, tei_source_path, record_path,
                f'--require-config-path={config_file_path}'
            )

    def test_should_keep_a_refusal_binding_after_the_document_was_regenerated(
        self, tmp_path: Path
    ):
        # The document changed, which is not anybody saying it is now sound.
        tei_source_path, record_path = self._write_corpus(
            tmp_path, {'refused1': 'rejected'},
            verdict_hash=HASH_SINCE_REGENERATED
        )
        output_path = self._run(
            tmp_path, tei_source_path, record_path,
            '--require-config-path='
            + self._write_require_config(tmp_path, 'not-rejected')
        )
        row = json.loads(
            Path(str(output_path) + '.quality.jsonl').read_text(encoding='utf-8')
        )
        assert row['review'] == 'rejection-lapsed'
        assert row['exclusion_reasons'][0] == 'rejected-by-review'

    def test_should_not_let_an_acceptance_survive_the_regeneration(
        self, tmp_path: Path
    ):
        tei_source_path, record_path = self._write_corpus(
            tmp_path, {'stale1': 'accepted'}, verdict_hash=HASH_SINCE_REGENERATED
        )
        output_path = self._run(
            tmp_path, tei_source_path, record_path,
            f'--require-config-path={self._write_require_config(tmp_path, "accepted")}',
            '--max-unreviewed-ratio=1.0'
        )
        row = json.loads(
            Path(str(output_path) + '.quality.jsonl').read_text(encoding='utf-8')
        )
        assert row['review'] == 'acceptance-lapsed'
        assert row['exclusion_reasons'][0] == 'not-reviewed'

    def test_should_assemble_a_corpus_a_reader_refused_most_of(
        self, tmp_path: Path
    ):
        # The one kind of loss held to no threshold: refusing here would decline
        # the document the reader approved because the others were not.
        verdicts = {f'refused{index}': 'rejected' for index in range(9)}
        verdicts['accepted1'] = 'accepted'
        tei_source_path, record_path = self._write_corpus(tmp_path, verdicts)
        output_path = self._run(
            tmp_path, tei_source_path, record_path,
            f'--require-config-path={self._write_require_config(tmp_path, "accepted")}'
        )
        texts, _labels, _features = load_data_and_labels_crf_file(str(output_path))
        assert len(texts) == 1
