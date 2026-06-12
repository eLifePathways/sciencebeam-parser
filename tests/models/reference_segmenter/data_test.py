from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPageCoordinates,
    LayoutToken
)
from sciencebeam_parser.lookup import SimpleTextLookUp
from sciencebeam_parser.models.data import (
    AppFeaturesContext,
    DEFAULT_DOCUMENT_FEATURES_CONTEXT,
    DocumentFeaturesContext
)
from sciencebeam_parser.models.reference_segmenter.data import ReferenceSegmenterDataGenerator


def _make_features_context_with_names(first_names):
    lookup = SimpleTextLookUp(set(first_names))
    return DocumentFeaturesContext(
        app_features_context=AppFeaturesContext(first_name_lookup=lookup)
    )


def _get_feature(token_text: str, feature_name: str, line_tokens=None) -> str:
    if line_tokens is None:
        line_tokens = [LayoutToken(token_text)]
    line = LayoutLine(line_tokens)
    doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
    gen = ReferenceSegmenterDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
    data_list = list(gen.iter_model_data_for_layout_document(doc))
    # find the entry for the requested token
    for item in data_list:
        if item.data_line and item.data_line.split()[0] == token_text:
            cols = item.data_line.split()
            idx = gen.feature_names.index(feature_name)
            return cols[idx]
    raise KeyError(f'token {token_text!r} not found')


class TestLineStatus:
    def test_single_token_line_gives_linestart(self):
        feature = _get_feature('word', 'line_status')
        assert feature == 'LINESTART'

    def test_first_token_of_multi_token_line_gives_linestart(self):
        tokens = [LayoutToken('first', whitespace=' '), LayoutToken('second')]
        feature = _get_feature('first', 'line_status', line_tokens=tokens)
        assert feature == 'LINESTART'

    def test_last_token_of_multi_token_line_gives_lineend(self):
        tokens = [LayoutToken('first', whitespace=' '), LayoutToken('second')]
        feature = _get_feature('second', 'line_status', line_tokens=tokens)
        assert feature == 'LINEEND'

    def test_middle_token_gives_linein(self):
        tokens = [
            LayoutToken('a', whitespace=' '),
            LayoutToken('b', whitespace=' '),
            LayoutToken('c')
        ]
        feature = _get_feature('b', 'line_status', line_tokens=tokens)
        assert feature == 'LINEIN'


class TestBlockStatus:
    def test_single_token_single_line_block_gives_blockstart(self):
        feature = _get_feature('word', 'block_status')
        assert feature == 'BLOCKSTART'


class TestPunctuationProfile:
    def test_plain_word_gives_nopunct(self):
        assert _get_feature('hello', 'punctuation_profile') == 'NOPUNCT'

    def test_dot_gives_dot(self):
        assert _get_feature('.', 'punctuation_profile') == 'DOT'

    def test_comma_gives_comma(self):
        assert _get_feature(',', 'punctuation_profile') == 'COMMA'

    def test_hyphen_gives_hyphen(self):
        assert _get_feature('-', 'punctuation_profile') == 'HYPHEN'

    def test_open_bracket_gives_openbracket(self):
        assert _get_feature('(', 'punctuation_profile') == 'OPENBRACKET'

    def test_close_bracket_gives_endbracket(self):
        assert _get_feature(')', 'punctuation_profile') == 'ENDBRACKET'


class TestTruncatedPunctuationProfileLength:
    def test_no_punctuation_on_line_gives_no(self):
        assert _get_feature('word', 'truncated_punctuation_profile_length') == 'no'

    def test_line_with_punctuation_gives_count(self):
        tokens = [
            LayoutToken('word', whitespace=' '),
            LayoutToken('.', whitespace=' '),
            LayoutToken('end')
        ]
        assert _get_feature(
            'word', 'truncated_punctuation_profile_length', line_tokens=tokens
        ) == '1'


class TestIsYear:
    def test_four_digit_year_gives_1(self):
        assert _get_feature('2020', 'is_year') == '1'

    def test_word_gives_0(self):
        assert _get_feature('word', 'is_year') == '0'


class TestIsMonth:
    def test_january_gives_1(self):
        assert _get_feature('january', 'is_month') == '1'

    def test_word_gives_0(self):
        assert _get_feature('word', 'is_month') == '0'


class TestIsHttp:
    def test_full_url_gives_1(self):
        assert _get_feature('https://doi.org/10.1234', 'is_http') == '1'

    def test_standalone_https_gives_1(self):
        assert _get_feature('https', 'is_http') == '1'

    def test_standalone_http_gives_1(self):
        assert _get_feature('http', 'is_http') == '1'

    def test_word_gives_0(self):
        assert _get_feature('word', 'is_http') == '0'


class TestIsFirstName:
    def test_all_caps_name_gives_1_when_in_lookup(self):
        # GROBID's first-name list uses all-uppercase entries; only all-uppercase tokens match
        ctx = _make_features_context_with_names({'AL'})
        tokens = [LayoutToken('AL')]
        line = LayoutLine(tokens)
        doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
        gen = ReferenceSegmenterDataGenerator(ctx)
        item = next(gen.iter_model_data_for_layout_document(doc))
        idx = gen.feature_names.index('is_first_name')
        assert item.data_line.split()[idx] == '1'

    def test_lowercase_common_abbreviation_gives_0(self):
        # 'al' (as in 'et al.') must NOT match even though 'AL' is in the lookup
        ctx = _make_features_context_with_names({'AL'})
        tokens = [LayoutToken('al')]
        line = LayoutLine(tokens)
        doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
        gen = ReferenceSegmenterDataGenerator(ctx)
        item = next(gen.iter_model_data_for_layout_document(doc))
        idx = gen.feature_names.index('is_first_name')
        assert item.data_line.split()[idx] == '0'

    def test_mixed_case_gives_0(self):
        # 'Marine' must not match 'MARINE' in the lookup
        ctx = _make_features_context_with_names({'MARINE'})
        tokens = [LayoutToken('Marine')]
        line = LayoutLine(tokens)
        doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
        gen = ReferenceSegmenterDataGenerator(ctx)
        item = next(gen.iter_model_data_for_layout_document(doc))
        idx = gen.feature_names.index('is_first_name')
        assert item.data_line.split()[idx] == '0'


class TestAlignment:
    def test_reference_number_line_after_indented_continuation_is_alignedleft(self):
        # References are: "1 Author ..."  (number at x=10, continuation at x=50).
        # The next reference number line (x=10) should be ALIGNEDLEFT, not LINEINDENT.
        # This matches GROBID's reference segmenter which compares every line
        # against the immediately preceding line across block boundaries.
        block1_lines = [
            LayoutLine([LayoutToken(
                '1', whitespace=' ',
                coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10)
            )]),
            LayoutLine([LayoutToken(
                'Author', whitespace=' ',
                coordinates=LayoutPageCoordinates(x=50, y=20, width=30, height=10)
            )]),
        ]
        block2_lines = [
            LayoutLine([LayoutToken(
                '2', whitespace=' ',
                coordinates=LayoutPageCoordinates(x=10, y=30, width=10, height=10)
            )]),
        ]
        doc = LayoutDocument.for_blocks([
            LayoutBlock(lines=block1_lines),
            LayoutBlock(lines=block2_lines),
        ])
        gen = ReferenceSegmenterDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
        items = list(gen.iter_model_data_for_layout_document(doc))
        idx = gen.feature_names.index('alignment')
        alignments = [item.data_line.split()[idx] for item in items]
        # token '1': ALIGNEDLEFT (first-ever LINESTART, double-skip: lineStartX not set)
        assert alignments[0] == 'ALIGNEDLEFT'
        # token 'Author': ALIGNEDLEFT (second LINESTART: previous=None, no comparison)
        assert alignments[1] == 'ALIGNEDLEFT'
        # token '2': ALIGNEDLEFT (x=10 < x=50 of 'Author' → not indented)
        assert alignments[2] == 'ALIGNEDLEFT'


class TestLineTokenRelativePosition:
    def test_single_token_line_includes_token_length_in_numerator(self):
        # GROBID: nn = text.length() for LINESTART token; currentLineLength includes whitespace+\n.
        # Single token "9" with trailing space: nn=1, cl=3 ("9 \n") → floor(1/3*10) = 3
        tokens = [LayoutToken('9', whitespace=' ')]
        assert _get_feature('9', 'line_token_relative_position', line_tokens=tokens) == '3'

    def test_single_token_references_heading(self):
        # "References" (10 chars) with trailing space: nn=10, cl=12 → floor(10/12*10) = 8
        tokens = [LayoutToken('References', whitespace=' ')]
        assert _get_feature(
            'References', 'line_token_relative_position', line_tokens=tokens
        ) == '8'

    def test_first_token_of_long_line_gives_small_value(self):
        # "Razavi" at start of a line with many more tokens → near 0
        # Build a line where total grobid_cl >> 6 (len "Razavi")
        long_line = [LayoutToken('Razavi', whitespace=' ')] + [
            LayoutToken('x', whitespace=' ') for _ in range(30)
        ]
        feature = _get_feature('Razavi', 'line_token_relative_position', line_tokens=long_line)
        assert feature == '0'


class TestLineRelativeLength:
    def test_single_token_line_is_shorter_than_long_line(self):
        # Short line ("9 ") vs long line should give a small relative length
        doc_lines = [
            LayoutLine([LayoutToken('9', whitespace=' ')]),
            LayoutLine([LayoutToken('x', whitespace=' ')] * 20),
        ]
        doc = LayoutDocument.for_blocks([LayoutBlock(lines=doc_lines)])
        gen = ReferenceSegmenterDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
        items = list(gen.iter_model_data_for_layout_document(doc))
        idx = gen.feature_names.index('line_relative_length')
        short_val = int(items[0].data_line.split()[idx])
        long_val = int(items[-1].data_line.split()[idx])
        assert short_val < long_val
