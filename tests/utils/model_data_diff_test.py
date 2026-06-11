from sciencebeam_parser.utils.model_data_diff import (
    compare_token_columns,
    diff_model_data,
    format_model_data_diff,
    parse_data_lines
)

FEATURE_NAMES_2 = ['token_text', 'capitalisation']
FEATURE_NAMES_3 = ['token_text', 'capitalisation', 'digit_status']


class TestParseDataLines:
    def test_parses_simple_line(self):
        assert parse_data_lines('tok f1 f2 label') == [['tok', 'f1', 'f2', 'label']]

    def test_skips_blank_lines(self):
        result = parse_data_lines('tok1 f1 label1\n\ntok2 f2 label2')
        assert result == [['tok1', 'f1', 'label1'], ['tok2', 'f2', 'label2']]

    def test_empty_text_returns_empty(self):
        assert not parse_data_lines('')

    def test_strips_trailing_whitespace(self):
        assert parse_data_lines('tok f1 label  ') == [['tok', 'f1', 'label']]


class TestCompareTokenColumns:
    def test_no_diffs_when_identical(self):
        assert not compare_token_columns(
            ['tok', 'ALLCAP', 'NODIGIT', 'B-body'],
            ['tok', 'ALLCAP', 'NODIGIT', 'B-body'],
            FEATURE_NAMES_3
        )

    def test_feature_diff_detected(self):
        diffs = compare_token_columns(
            ['tok', 'INITCAP', 'NODIGIT', 'B-body'],
            ['tok', 'ALLCAP', 'NODIGIT', 'B-body'],
            FEATURE_NAMES_3
        )
        assert len(diffs) == 1
        assert diffs[0].feature_name == 'capitalisation'
        assert diffs[0].sbeam_value == 'INITCAP'
        assert diffs[0].grobid_value == 'ALLCAP'

    def test_label_diff_detected(self):
        diffs = compare_token_columns(
            ['tok', 'ALLCAP', 'B-body'],
            ['tok', 'ALLCAP', 'I-body'],
            FEATURE_NAMES_2
        )
        assert len(diffs) == 1
        assert diffs[0].feature_name == 'label'
        assert diffs[0].sbeam_value == 'B-body'
        assert diffs[0].grobid_value == 'I-body'

    def test_extra_sbeam_feature_not_compared(self):
        # ScienceBeam has whole_line_text as an extra column GROBID lacks
        diffs = compare_token_columns(
            ['tok', 'ALLCAP', 'whole line', 'B-body'],
            ['tok', 'ALLCAP', 'B-body'],
            ['token_text', 'capitalisation', 'whole_line_text']
        )
        assert not diffs

    def test_multiple_diffs(self):
        diffs = compare_token_columns(
            ['tok', 'INITCAP', 'CONTAINSDIGITS', 'B-body'],
            ['tok', 'ALLCAP', 'NODIGIT', 'I-body'],
            FEATURE_NAMES_3
        )
        names = [d.feature_name for d in diffs]
        assert names == ['capitalisation', 'digit_status', 'label']


class TestDiffModelData:
    def _make_line(self, token: str, cap: str = 'ALLCAP', label: str = 'B-body') -> str:
        return f'{token} {cap} {label}'

    def test_identical_data_no_diffs(self):
        data = self._make_line('tok1') + '\n' + self._make_line('tok2') + '\n'
        result = diff_model_data(data, data, FEATURE_NAMES_2)
        assert result.aligned_count == 2
        assert result.aligned_with_diffs_count == 0
        assert result.sbeam_only_count == 0
        assert result.grobid_only_count == 0

    def test_label_diff_reported(self):
        sbeam = self._make_line('tok1', label='B-body')
        grobid = self._make_line('tok1', label='I-body')
        result = diff_model_data(sbeam, grobid, FEATURE_NAMES_2)
        assert result.aligned_with_diffs_count == 1
        assert result.aligned_diffs[0].token == 'tok1'
        label_diff = result.aligned_diffs[0].feature_diffs[0]
        assert label_diff.feature_name == 'label'
        assert label_diff.sbeam_value == 'B-body'
        assert label_diff.grobid_value == 'I-body'

    def test_sbeam_only_token_reported(self):
        sbeam = self._make_line('tok1') + '\n' + self._make_line('tok2') + '\n'
        grobid = self._make_line('tok1')
        result = diff_model_data(sbeam, grobid, FEATURE_NAMES_2)
        assert result.sbeam_only_count == 1
        assert result.sbeam_only_tokens[0][0] == 'tok2'

    def test_grobid_only_token_reported(self):
        sbeam = self._make_line('tok1')
        grobid = self._make_line('tok1') + '\n' + self._make_line('tok2') + '\n'
        result = diff_model_data(sbeam, grobid, FEATURE_NAMES_2)
        assert result.grobid_only_count == 1
        assert result.grobid_only_tokens[0][0] == 'tok2'

    def test_token_counts(self):
        sbeam = self._make_line('tok1') + '\n' + self._make_line('tok2') + '\n'
        grobid = self._make_line('tok1') + '\n' + self._make_line('tok3') + '\n'
        result = diff_model_data(sbeam, grobid, FEATURE_NAMES_2)
        assert result.sbeam_token_count == 2
        assert result.grobid_token_count == 2


class TestFormatModelDataDiff:
    def _make_line(self, token: str, cap: str = 'ALLCAP', label: str = 'B-body') -> str:
        return f'{token} {cap} {label}'

    def test_summary_line_present(self):
        data = self._make_line('tok')
        result = format_model_data_diff(data, data, FEATURE_NAMES_2)
        assert 'sbeam: 1 tokens' in result
        assert 'grobid: 1 tokens' in result

    def test_feature_diff_shown_with_names(self):
        sbeam = self._make_line('tok', cap='INITCAP')
        grobid = self._make_line('tok', cap='ALLCAP')
        result = format_model_data_diff(sbeam, grobid, FEATURE_NAMES_2)
        assert 'capitalisation' in result
        assert 'INITCAP' in result
        assert 'ALLCAP' in result

    def test_direction_shown(self):
        sbeam = self._make_line('tok', label='B-body')
        grobid = self._make_line('tok', label='I-body')
        result = format_model_data_diff(sbeam, grobid, FEATURE_NAMES_2)
        assert 'grobid' in result
        assert 'sbeam' in result
        assert '→' in result

    def test_sbeam_only_token_shown(self):
        sbeam = self._make_line('tok1') + '\n' + self._make_line('tok2')
        grobid = self._make_line('tok1')
        result = format_model_data_diff(sbeam, grobid, FEATURE_NAMES_2)
        assert 'sbeam-only' in result
        assert 'tok2' in result

    def test_no_detail_section_when_identical(self):
        data = self._make_line('tok')
        result = format_model_data_diff(data, data, FEATURE_NAMES_2)
        assert '< sbeam-only' not in result
        assert '> grobid-only' not in result
        assert '→' not in result

    def test_feature_summary_shows_counts_sorted_by_frequency(self):
        # Two capitalisation diffs and one label diff → summary lists cap first, then label
        sbeam = (
            self._make_line('a', cap='INITCAP', label='B-body') + '\n'
            + self._make_line('b', cap='INITCAP', label='I-header') + '\n'
            + self._make_line('c', cap='ALLCAP', label='B-body')
        )
        grobid = (
            self._make_line('a', cap='ALLCAP', label='B-body') + '\n'
            + self._make_line('b', cap='ALLCAP', label='B-body') + '\n'
            + self._make_line('c', cap='ALLCAP', label='B-body')
        )
        result = format_model_data_diff(sbeam, grobid, FEATURE_NAMES_2)
        assert 'feature diffs by type:' in result
        cap_pos = result.index('capitalisation: 2')
        label_pos = result.index('label: 1')
        assert cap_pos < label_pos

    def test_feature_summary_always_shows_label_even_when_zero(self):
        sbeam = self._make_line('tok', cap='INITCAP')
        grobid = self._make_line('tok', cap='ALLCAP')
        result = format_model_data_diff(sbeam, grobid, FEATURE_NAMES_2)
        assert 'label: 0' in result

    def test_no_feature_summary_when_no_diffs(self):
        data = self._make_line('tok')
        result = format_model_data_diff(data, data, FEATURE_NAMES_2)
        assert 'feature diffs by type:' not in result

    def test_label_pair_breakdown_shown_sorted_by_frequency(self):
        # Three label diffs: two header→body and one header→I-body.
        # header→body appears most often and should come first.
        sbeam = (
            self._make_line('a', label='B-body') + '\n'
            + self._make_line('b', label='B-body') + '\n'
            + self._make_line('c', label='I-body')
        )
        grobid = (
            self._make_line('a', label='B-header') + '\n'
            + self._make_line('b', label='B-header') + '\n'
            + self._make_line('c', label='B-header')
        )
        result = format_model_data_diff(sbeam, grobid, FEATURE_NAMES_2)
        assert '2x B-header (grobid) → B-body (sbeam)' in result
        assert '1x B-header (grobid) → I-body (sbeam)' in result
        two_pos = result.index('2x B-header')
        one_pos = result.index('1x B-header')
        assert two_pos < one_pos
