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
