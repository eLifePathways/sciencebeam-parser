from sciencebeam_parser.training.tei_raw_pairing import (
    PairedRow,
    get_paired_rows,
    iter_match_keys_for_labelled_line
)


def _keys(*line_texts: str):
    return [
        list(iter_match_keys_for_labelled_line(
            token_text=line_text.split()[0] if line_text.split() else '',
            line_text=line_text
        ))
        for line_text in line_texts
    ]


class TestIterMatchKeysForLabelledLine:
    def test_should_yield_the_token_text_alone_without_a_line_text(self):
        assert list(iter_match_keys_for_labelled_line('token1', None)) == ['token1']

    def test_should_yield_the_first_field_of_the_line_as_well(self):
        # our generator writes the first token our tokenizer produced; GROBID
        # writes the line's first whitespace-delimited field
        assert list(iter_match_keys_for_labelled_line(
            'LIMIX', 'LIMIX: genetic analysis'
        )) == ['LIMIX', 'LIMIX:']

    def test_should_not_repeat_a_first_field_equal_to_the_token(self):
        assert list(iter_match_keys_for_labelled_line(
            'token1', 'token1 token2'
        )) == ['token1']


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

    def test_should_report_a_document_whose_trailing_rows_all_failed(self):
        result = get_paired_rows(
            ['token1'] + ['other'] * 3,
            _keys('token1 a', 'token2 b'),
            max_trailing_unmatched=3
        )
        assert result.is_document_discarded is True

    def test_should_not_report_a_document_that_recovered_before_the_end(self):
        result = get_paired_rows(
            ['token1'] + ['other'] * 3 + ['token2'],
            _keys('token1 a', 'token2 b'),
            max_trailing_unmatched=3
        )
        assert result.is_document_discarded is False

    def test_should_report_counts_for_an_empty_raw_file(self):
        result = get_paired_rows([], _keys('token1 a'))
        assert not result.paired_rows
        assert result.matched_ratio == 0.0
        assert result.raw_row_count == 0
