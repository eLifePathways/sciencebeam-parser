from pathlib import Path

from benchmarks.prediction_files import (
    is_prediction_file,
    iter_prediction_files,
    record_id_from_name,
)


class TestIsPredictionFile:
    def test_should_accept_tei_and_jats(self):
        assert is_prediction_file('doc-1.tei.xml')
        assert is_prediction_file('doc-1.jats.xml')

    def test_should_reject_anything_else(self):
        assert not is_prediction_file('manifest.jsonl')
        assert not is_prediction_file('doc-1.xml')
        assert not is_prediction_file('summary.json')


class TestRecordIdFromName:
    def test_should_strip_the_schema_suffix(self):
        assert record_id_from_name('doc-1.tei.xml') == 'doc-1'
        assert record_id_from_name('doc-1.jats.xml') == 'doc-1'

    def test_should_keep_an_id_that_contains_the_suffix_text(self):
        # `.stem.replace('.tei', '')` rewrote the id itself here.
        assert record_id_from_name('a.tei.b.tei.xml') == 'a.tei.b'

    def test_should_keep_dots_within_an_id(self):
        assert record_id_from_name('10.1234.5678.jats.xml') == '10.1234.5678'


class TestIterPredictionFiles:
    def test_should_find_both_schemas_ordered_by_record_id(self, tmp_path: Path):
        for name in ['b.tei.xml', 'a.jats.xml', 'c.tei.xml', 'manifest.jsonl']:
            (tmp_path / name).write_text('x')
        assert [p.name for p in iter_prediction_files(tmp_path)] == [
            'a.jats.xml', 'b.tei.xml', 'c.tei.xml',
        ]

    def test_should_be_empty_for_a_missing_directory(self, tmp_path: Path):
        assert not list(iter_prediction_files(tmp_path / 'nope'))
