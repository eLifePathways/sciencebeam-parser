from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.models.data import DEFAULT_DOCUMENT_FEATURES_CONTEXT
from sciencebeam_parser.models.citation.data import CitationDataGenerator


def _get_feature_for_tokens(
    token_texts_with_whitespace: list,
    target_token_text: str,
    feature_name: str
) -> str:
    tokens = [LayoutToken(text, whitespace=ws) for text, ws in token_texts_with_whitespace]
    line = LayoutLine(tokens)
    doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
    gen = CitationDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
    idx = gen.feature_names.index(feature_name)
    for item in gen.iter_model_data_for_layout_document(doc):
        if item.data_line and item.data_line.split()[0] == target_token_text:
            return item.data_line.split()[idx]
    raise KeyError(f'token {target_token_text!r} not found')


def _get_feature(token_text: str, feature_name: str) -> str:
    tokens = [LayoutToken(token_text)]
    line = LayoutLine(tokens)
    doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
    gen = CitationDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
    data_list = list(gen.iter_model_data_for_layout_document(doc))
    for item in data_list:
        if item.data_line and item.data_line.split()[0] == token_text:
            cols = item.data_line.split()
            idx = gen.feature_names.index(feature_name)
            return cols[idx]
    raise KeyError(f'token {token_text!r} not found')


class TestIsYear:
    def test_four_digit_year_gives_1(self):
        assert _get_feature('2020', 'is_year') == '1'

    def test_word_gives_0(self):
        assert _get_feature('Lancet', 'is_year') == '0'

    def test_partial_number_gives_0(self):
        assert _get_feature('20', 'is_year') == '0'


class TestIsHttp:
    def test_http_starting_token_gives_1(self):
        tokens = [('https', ''), ('://', ''), ('doi.org', ' '), ('other', ' ')]
        assert _get_feature_for_tokens(tokens, 'https', 'is_http') == '1'

    def test_token_within_url_span_gives_1(self):
        tokens = [('https', ''), ('://', ''), ('doi.org', '/'), ('10.1234', ' ')]
        assert _get_feature_for_tokens(tokens, 'doi.org', 'is_http') == '1'

    def test_token_outside_url_gives_0(self):
        tokens = [('https', ''), ('://', ''), ('doi.org', ' '), ('other', ' ')]
        assert _get_feature_for_tokens(tokens, 'other', 'is_http') == '0'

    def test_word_without_url_gives_0(self):
        assert _get_feature('Lancet', 'is_http') == '0'


class TestIsMonth:
    def test_month_name_gives_1(self):
        assert _get_feature('january', 'is_month') == '1'

    def test_capitalised_month_gives_1(self):
        assert _get_feature('January', 'is_month') == '1'

    def test_word_gives_0(self):
        assert _get_feature('Lancet', 'is_month') == '0'


class TestIsKnownIdentifier:
    def test_doi_start_token_gives_1(self):
        # tokens of 10.1016/S0021-9991(70)90001-X with no whitespace between
        tokens = [
            ('10', ''), ('.', ''), ('1016', ''), ('/', ''), ('S0021', ''),
            ('-', ''), ('9991', ''), ('(', ''), ('70', ''), (')', ''),
            ('90001', ''), ('-', ''), ('X', ' ')
        ]
        assert _get_feature_for_tokens(tokens, '10', 'is_known_identifier') == '1'

    def test_doi_bracket_tokens_give_1(self):
        tokens = [
            ('10', ''), ('.', ''), ('1016', ''), ('/', ''), ('S0021', ''),
            ('-', ''), ('9991', ''), ('(', ''), ('70', ''), (')', ''),
            ('90001', ''), ('-', ''), ('X', ' ')
        ]
        assert _get_feature_for_tokens(tokens, '(', 'is_known_identifier') == '1'
        assert _get_feature_for_tokens(tokens, '70', 'is_known_identifier') == '1'
        assert _get_feature_for_tokens(tokens, '90001', 'is_known_identifier') == '1'

    def test_doi_prefix_label_not_part_of_identifier(self):
        # 'doi' and ':' before the DOI are not part of the identifier span
        tokens = [
            ('doi', ''), (':', ''), ('10', ''), ('.', ''), ('1016', ''),
            ('/', ''), ('abcde', ' ')
        ]
        assert _get_feature_for_tokens(tokens, 'doi', 'is_known_identifier') == '0'
        assert _get_feature_for_tokens(tokens, ':', 'is_known_identifier') == '0'

    def test_word_gives_0(self):
        assert _get_feature('Lancet', 'is_known_identifier') == '0'


class TestSentenceTokenRelativePosition:
    def test_single_token_gives_0(self):
        # single token: sentence_token_count = 1, index = 0 → bin 0
        assert _get_feature('Word', 'sentence_token_relative_position') == '0'

    def test_last_of_three_space_separated_tokens_matches_grobid(self):
        # GROBID: tokens = ["A"," ","B"," ","C"], sentenceLenth=5
        # "C" is at n=4: floor(4/5*12) = floor(9.6) = 9
        # sbeam (pre-fix) was: floor(2/3*12) = 8 — wrong
        tokens = [('A', ' '), ('B', ' '), ('C', '')]
        assert _get_feature_for_tokens(tokens, 'C', 'sentence_token_relative_position') == '9'

    def test_middle_of_three_matches_grobid(self):
        # "B" at n=2: floor(2/5*12) = floor(4.8) = 4 (same as pre-fix sbeam: floor(1/3*12) = 4)
        tokens = [('A', ' '), ('B', ' '), ('C', '')]
        assert _get_feature_for_tokens(tokens, 'B', 'sentence_token_relative_position') == '4'

    def test_no_whitespace_tokens_unchanged(self):
        # sub-tokens of a DOI with no whitespace: same as before
        tokens = [('10', ''), ('.', ''), ('1016', ''), ('/', ''), ('abcde', '')]
        assert _get_feature_for_tokens(tokens, '10', 'sentence_token_relative_position') == '0'
