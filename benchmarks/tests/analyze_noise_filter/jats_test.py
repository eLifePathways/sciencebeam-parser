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

    def test_should_not_flag_a_running_head_absent_from_the_jats(self):
        index = JatsTextIndex(normalize_text('anything'))
        assert index.contains('Page 12 of 15') is False

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


class TestJatsTextIndexTitles:
    def test_should_be_inconclusive_for_text_found_only_in_the_title(self):
        index = JatsTextIndex(
            normalize_text(f'{LONG_ENOUGH} and some body text'),
            normalize_text(LONG_ENOUGH)
        )
        assert index.contains(LONG_ENOUGH) is None

    def test_should_still_flag_text_found_outside_the_title(self):
        index = JatsTextIndex(
            normalize_text(f'{LONG_ENOUGH} and some body text'),
            normalize_text('a different title')
        )
        assert index.contains(LONG_ENOUGH) is True

    def test_should_read_the_article_title_from_the_file(self, tmp_path):
        jats_path = tmp_path / 'doc.jats.xml'
        jats_path.write_text(
            '<article><front><title-group>'
            f'<article-title>{LONG_ENOUGH}</article-title>'
            '</title-group></front>'
            f'<body><p>{LONG_ENOUGH}</p></body></article>',
            encoding='utf-8'
        )
        assert JatsTextIndex.from_file(str(jats_path)).contains(LONG_ENOUGH) is None

    def test_should_not_treat_a_section_heading_as_a_title(self, tmp_path):
        jats_path = tmp_path / 'doc.jats.xml'
        jats_path.write_text(
            f'<article><body><sec><title>{LONG_ENOUGH}</title></sec></body></article>',
            encoding='utf-8'
        )
        assert JatsTextIndex.from_file(str(jats_path)).contains(LONG_ENOUGH) is True


class TestJatsTextIndexShortRunningHeads:
    def test_should_look_up_three_words_below_the_length_floor(self):
        index = JatsTextIndex(normalize_text('by Carlos Barba Solano, who wrote it'))
        assert index.contains('CARLOS BARBA SOLANO') is True

    def test_should_strip_a_trailing_page_number_first(self):
        index = JatsTextIndex(normalize_text('by Carlos Barba Solano, who wrote it'))
        assert index.contains('CARLOS BARBA SOLANO 133') is True

    def test_should_stay_inconclusive_below_both_floors(self):
        index = JatsTextIndex(normalize_text('by Marc Thouvenot, who wrote it'))
        assert index.contains('marc thouvenot') is None

    def test_should_still_be_inconclusive_for_a_bare_page_number(self):
        index = JatsTextIndex(normalize_text('anything at all here'))
        assert index.contains('133') is None
