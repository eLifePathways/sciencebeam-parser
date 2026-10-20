from sciencebeam_parser.training.tei_raw_pairing import (
    PairedRow,
    get_match_key_for_labelled_line,
    get_paired_rows
)


def _keys(*line_texts: str):
    return [
        get_match_key_for_labelled_line(
            token_text=line_text.split()[0] if line_text.split() else '',
            line_text=line_text,
            line_as_token=True
        )
        for line_text in line_texts
    ]


class TestGetMatchKeyForLabelledLine:
    def test_should_take_the_first_field_of_the_line_for_a_line_model(self):
        # the raw row leads with the whole field, where our tokenizer stops short
        assert get_match_key_for_labelled_line(
            'LIMIX', 'LIMIX: genetic analysis', line_as_token=True
        ) == 'LIMIX:'

    def test_should_take_the_token_for_a_token_model(self):
        # every token of a line is its own row, so the line's first field is only
        # right for the first of them
        assert get_match_key_for_labelled_line(
            'genetic', 'LIMIX: genetic analysis', line_as_token=False
        ) == 'genetic'

    def test_should_fall_back_to_the_token_without_a_line_text(self):
        assert get_match_key_for_labelled_line(
            'token1', None, line_as_token=True
        ) == 'token1'


class TestGetPairedRows:
    def test_should_pair_aligned_rows_positionally(self):
        result = get_paired_rows(
            ['token1', 'token2', 'token3'],
            _keys('token1 a', 'token2 b', 'token3 c')
        )
        assert list(result.paired_rows) == [
            PairedRow(0, 0), PairedRow(1, 1), PairedRow(2, 2)
        ]
        assert result.matched_ratio == 1.0
        assert result.unmatched_row_count == 0

    def test_should_match_a_raw_token_against_the_lines_first_field(self):
        result = get_paired_rows(['LIMIX:'], _keys('LIMIX: genetic analysis'))
        assert list(result.paired_rows) == [PairedRow(0, 0)]

    def test_should_skip_a_line_the_raw_file_does_not_have(self):
        result = get_paired_rows(
            ['token1', 'token3'],
            _keys('token1 a', 'token2 b', 'token3 c')
        )
        assert list(result.paired_rows) == [PairedRow(0, 0), PairedRow(1, 2)]

    def test_should_leave_the_cursor_where_it_was_for_a_row_it_could_not_place(self):
        result = get_paired_rows(
            ['token1', 'other', 'token2'],
            _keys('token1 a', 'token2 b')
        )
        assert list(result.paired_rows) == [PairedRow(0, 0), PairedRow(2, 1)]
        assert result.unmatched_row_count == 1

    def test_should_not_look_further_ahead_than_the_window(self):
        line_texts = ['token%d x' % index for index in range(20)]
        result = get_paired_rows(['token19'], _keys(*line_texts), max_lookahead=6)
        assert not result.paired_rows
        assert result.unmatched_row_count == 1

    def test_should_match_the_last_line_inside_the_window(self):
        line_texts = ['token%d x' % index for index in range(20)]
        result = get_paired_rows(['token6'], _keys(*line_texts), max_lookahead=6)
        assert list(result.paired_rows) == [PairedRow(0, 6)]

    def test_should_produce_nothing_once_the_tei_has_run_out(self):
        result = get_paired_rows(
            ['token1', 'token2', 'token3'],
            _keys('token1 a')
        )
        assert list(result.paired_rows) == [PairedRow(0, 0)]
        assert result.uncovered_row_count == 2
        assert result.unmatched_row_count == 0

    def test_should_not_count_a_tei_that_stops_short_against_the_matched_share(self):
        # a document whose TEI the annotator stopped part way through is short,
        # not badly paired, and GROBID keeps the rows it does have
        result = get_paired_rows(
            ['token1', 'token2', 'token3', 'token4'],
            _keys('token1 a')
        )
        assert result.matched_ratio == 1.0
        assert result.uncovered_ratio == 0.75

    def test_should_abandon_the_document_after_too_many_consecutive_unmatched_rows(self):
        raw_row_tokens = ['token1'] + ['other'] * 5 + ['token2']
        result = get_paired_rows(
            raw_row_tokens,
            _keys('token1 a', 'token2 b'),
            max_consecutive_unmatched=3
        )
        assert list(result.paired_rows) == [PairedRow(0, 0)]
        assert result.unmatched_row_count == 4
        assert result.uncovered_row_count == 2

    def test_should_report_counts_for_an_empty_raw_file(self):
        result = get_paired_rows([], _keys('token1 a'))
        assert not result.paired_rows
        assert result.matched_ratio == 0.0
        assert result.raw_row_count == 0
