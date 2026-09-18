from benchmarks.analyze_noise_filter._jats import (
    JatsTextIndex,
    index_jats_filenames_by_stem,
    normalize_text,
)

LONG_ENOUGH = 'the evolution of mortality in the period from January'


class TestNormalizeText:
    def test_should_collapse_whitespace_and_casefold(self):
        assert normalize_text('  INTRODUÇÃO\n ') == 'introdução'

    def test_should_treat_non_breaking_space_as_space(self):
        assert normalize_text('Page 12') == 'page 12'

    def test_should_normalise_dashes(self):
        assert normalize_text('COVID–19') == 'covid-19'


class TestJatsTextIndex:
    def test_should_find_text_that_differs_only_in_case(self):
        index = JatsTextIndex(normalize_text(f'Body text. {LONG_ENOUGH}.'))
        assert index.contains(LONG_ENOUGH.upper()) is True

    def test_should_not_find_furniture_absent_from_the_jats(self):
        index = JatsTextIndex(normalize_text(f'Body text. {LONG_ENOUGH}.'))
        assert index.contains('SciELO Preprints - Este documento é um preprint') is False

    def test_should_be_inconclusive_for_a_page_number(self):
        index = JatsTextIndex(normalize_text('Table 12 shows the result'))
        assert index.contains('12') is None

    def test_should_be_inconclusive_for_a_short_running_head(self):
        index = JatsTextIndex(normalize_text('anything'))
        assert index.contains('Page 12 of 15') is None

    def test_should_drop_a_word_broken_by_a_line_ending_hyphen(self):
        index = JatsTextIndex(normalize_text(f'{LONG_ENOUGH} January'))
        assert index.contains(f'{LONG_ENOUGH} Janu-') is True


class TestIndexJatsFilenamesByStem:
    def test_should_strip_the_compound_jats_suffix(self):
        result = index_jats_filenames_by_stem(['a/b/PPR459390.jats.xml'])
        assert result == {'PPR459390': 'a/b/PPR459390.jats.xml'}

    def test_should_strip_a_plain_xml_suffix(self):
        result = index_jats_filenames_by_stem(['a/b/PPR459390.xml'])
        assert result == {'PPR459390': 'a/b/PPR459390.xml'}
