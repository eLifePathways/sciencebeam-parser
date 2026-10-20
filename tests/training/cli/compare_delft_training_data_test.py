import logging

from sciencebeam_parser.training.grobid_column_layout import (
    get_grobid_column_layout_for_model_name
)
from sciencebeam_parser.training.cli.compare_delft_training_data import (
    compare,
    get_join_key,
    get_row_without_extra_columns,
    read_sequences
)


LOGGER = logging.getLogger(__name__)

SEGMENTATION_LAYOUT = get_grobid_column_layout_for_model_name('segmentation')

# 33 feature columns and a label, as the published corpus holds a row
FEATURES = ' '.join(['f%d' % index for index in range(32)])


def _row(token: str, label: str, line_text: str = '') -> str:
    extra = ' ' + line_text if line_text else ''
    return '%s %s%s %s' % (token, FEATURES, extra, label)


def _compare(generated, published, max_reported_rows=10) -> int:
    return compare(
        generated=generated,
        published=published,
        layout=SEGMENTATION_LAYOUT,
        join_row_count=2,
        max_reported_rows=max_reported_rows
    )


class TestReadSequences:
    def test_should_split_on_a_blank_line(self, tmp_path):
        path = tmp_path / 'data.train'
        path.write_text('a 1\nb 2\n\n\nc 3\n', encoding='utf-8')
        assert read_sequences(str(path)) == [['a 1', 'b 2'], ['c 3']]


class TestGetJoinKey:
    def test_should_take_the_first_two_columns_of_the_first_rows(self):
        assert get_join_key(['a b c', 'd e f', 'g h i'], 2) == (
            ('a', 'b'), ('d', 'e')
        )


class TestGetRowWithoutExtraColumns:
    def test_should_drop_the_extra_column_before_the_label(self):
        with_extra = _row('token1', '<body>', line_text='token1 and more')
        assert get_row_without_extra_columns(with_extra, SEGMENTATION_LAYOUT) == _row(
            'token1', '<body>'
        )

    def test_should_leave_a_row_of_the_published_width_alone(self):
        row = _row('token1', '<body>')
        assert get_row_without_extra_columns(row, SEGMENTATION_LAYOUT) == row


class TestCompare:
    def test_should_report_nothing_differing_for_an_identical_corpus(self):
        sequences = [[_row('a', '<body>'), _row('b', '<body>')]]
        assert _compare(sequences, [list(sequences[0])]) == 0

    def test_should_ignore_the_extra_column_when_comparing(self):
        generated = [[
            _row('a', '<body>', line_text='a line'),
            _row('b', '<body>', line_text='b line')
        ]]
        published = [[_row('a', '<body>'), _row('b', '<body>')]]
        assert _compare(generated, published) == 0

    def test_should_count_a_row_the_published_corpus_does_not_hold(self):
        generated = [[
            _row('a', '<body>'), _row('b', '<body>'), _row('c', '<body>')
        ]]
        published = [[_row('a', '<body>'), _row('b', '<body>')]]
        assert _compare(generated, published) == 1

    def test_should_not_count_a_published_row_we_chose_not_to_emit(self):
        generated = [[_row('a', '<body>'), _row('c', '<body>')]]
        published = [[
            _row('a', '<body>'), _row('b', '<body>'), _row('c', '<body>')
        ]]
        assert _compare(generated, published) == 0

    def test_should_count_a_row_whose_label_differs(self):
        generated = [[_row('a', '<body>'), _row('b', '<annex>')]]
        published = [[_row('a', '<body>'), _row('b', '<body>')]]
        assert _compare(generated, published) == 1

    def test_should_not_fail_on_a_sequence_that_does_not_join(self):
        generated = [[_row('x', '<body>'), _row('y', '<body>')]]
        published = [[_row('a', '<body>'), _row('b', '<body>')]]
        assert _compare(generated, published) == 0
