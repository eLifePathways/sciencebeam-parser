import json

from sciencebeam_parser.document.layout_noise_filter import LayoutNoiseFilterConfig

from benchmarks.analyze_noise_filter._report import render_report, write_report
from benchmarks.analyze_noise_filter._types import DocumentSummary, RemovedBlock

CONFIG = LayoutNoiseFilterConfig(enabled=True)


def _removed(text: str, in_jats=None, corpus: str = 'biorxiv') -> RemovedBlock:
    return RemovedBlock(
        document_id='doc-1', corpus=corpus, page_number=2, page_count=10,
        note_type='running-foot', y_relative=0.95, height_relative=0.01,
        line_count=1, text=text, in_jats=in_jats
    )


def _summary(corpus: str = 'biorxiv', error=None) -> DocumentSummary:
    return DocumentSummary(
        document_id='doc-1', corpus=corpus, page_count=10, block_count=100,
        line_count=200, removed_block_count=1, removed_line_count=1,
        in_jats_count=0, error=error
    )


class TestRenderReport:
    def test_should_list_a_removal_found_in_the_jats(self):
        report = render_report(
            [_summary()], [_removed('a body sentence that is content', in_jats=True)],
            config=CONFIG, sample_size=10
        )
        assert '1 of 1 removed blocks' in report
        assert 'a body sentence that is content' in report

    def test_should_say_none_when_no_removal_is_in_the_jats(self):
        report = render_report(
            [_summary()], [_removed('Page 12 of 15', in_jats=None)],
            config=CONFIG, sample_size=10
        )
        assert '0 of 1 removed blocks' in report
        assert 'None.' in report

    def test_should_state_how_many_samples_were_dropped(self):
        removed = [_removed(f'block {index}') for index in range(5)]
        report = render_report([_summary()], removed, config=CONFIG, sample_size=2)
        assert 'Showing 2 of 5.' in report
        assert 'block 4' not in report

    def test_should_report_the_removed_line_percentage_per_corpus(self):
        report = render_report(
            [_summary()], [_removed('x', in_jats=False)], config=CONFIG, sample_size=10
        )
        assert '| biorxiv | 1 | 0 | 200 | 1 | 1 | 0.50% | 0 | 0 |' in report

    def test_should_count_a_removal_too_short_to_look_up_separately(self):
        report = render_report(
            [_summary()], [_removed('12', in_jats=None)], config=CONFIG, sample_size=10
        )
        assert '| biorxiv | 1 | 0 | 200 | 1 | 1 | 0.50% | 0 | 1 |' in report

    def test_should_exclude_a_failed_document_from_the_line_counts(self):
        report = render_report(
            [_summary(), _summary(error='boom')], [], config=CONFIG, sample_size=10
        )
        assert '| biorxiv | 2 | 1 | 200 |' in report
        assert 'boom' in report

    def test_should_escape_a_pipe_in_the_removed_text(self):
        report = render_report(
            [_summary()], [_removed('a | b')], config=CONFIG, sample_size=10
        )
        assert 'a \\| b' in report


class TestWriteReport:
    def test_should_write_the_report_and_both_jsonl_files(self, tmp_path):
        report_path = write_report(
            tmp_path / 'out', summaries=[_summary()],
            removed=[_removed('Page 12 of 15')], config=CONFIG, sample_size=10
        )
        assert report_path.exists()
        removed_lines = (tmp_path / 'out' / 'removed-blocks.jsonl').read_text(
            encoding='utf-8'
        ).splitlines()
        assert json.loads(removed_lines[0])['text'] == 'Page 12 of 15'
        document_lines = (tmp_path / 'out' / 'documents.jsonl').read_text(
            encoding='utf-8'
        ).splitlines()
        assert json.loads(document_lines[0])['document_id'] == 'doc-1'
