from sciencebeam_parser.training.jats.text_normalizer import (
    normalize_text,
    normalize_for_alignment,
)


class TestNormalizeText:
    def test_passthrough_ascii(self):
        assert normalize_text('hello world') == 'hello world'

    def test_ligature_fi(self):
        assert normalize_text('ﬁgure') == 'figure'

    def test_ligature_fl(self):
        assert normalize_text('ﬂow') == 'flow'

    def test_ligature_ff(self):
        assert normalize_text('oﬀ') == 'off'

    def test_em_dash_to_hyphen(self):
        assert normalize_text('foo—bar') == 'foo-bar'

    def test_en_dash_to_hyphen(self):
        assert normalize_text('foo–bar') == 'foo-bar'

    def test_curly_quotes(self):
        assert normalize_text('‘it’s') == "'it's"

    def test_double_curly_quotes(self):
        assert normalize_text('“hello”') == '"hello"'

    def test_soft_hyphen_removed(self):
        # U+00AD soft hyphen
        assert normalize_text('hyp­hen') == 'hyphen'


class TestNormalizeForAlignment:
    def test_lowercase(self):
        assert normalize_for_alignment('Hello World') == 'hello world'

    def test_collapses_whitespace(self):
        assert normalize_for_alignment('foo  \t  bar') == 'foo bar'

    def test_strips_leading_trailing(self):
        assert normalize_for_alignment('  hello  ') == 'hello'

    def test_applies_ligature_normalisation(self):
        assert normalize_for_alignment('ﬁgure') == 'figure'

    def test_applies_dash_normalisation(self):
        assert normalize_for_alignment('foo—bar') == 'foo-bar'

    def test_reads_an_accented_letter_as_the_letter(self):
        assert normalize_for_alignment('Análise') == 'analise'

    def test_reads_an_accent_set_apart_as_part_of_its_word(self):
        # The diaeresis of a name a font prints as its own character, which is
        # how both the page and the deposit made from it carry it.
        assert normalize_for_alignment('O \u0308 zer A') == 'ozer a'

    def test_keeps_the_words_either_side_of_it_apart(self):
        assert normalize_for_alignment('concepcio \u0301n de datos') == 'concepcion de datos'

    def test_an_accent_written_on_its_letter_leaves_the_words_apart(self):
        assert normalize_for_alignment('Jos\u00e9 P\u00e9rez') == 'jose perez'
