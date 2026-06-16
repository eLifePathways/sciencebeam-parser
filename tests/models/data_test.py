from abc import ABC, abstractmethod

from typing import List

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    LayoutFont,
    LayoutToken
)
from sciencebeam_parser.lookup import SimpleTextLookUp
from sciencebeam_parser.models.data import (
    AppFeaturesContext,
    ContextAwareLayoutTokenFeatures,
    ContextAwareLayoutTokenModelDataGenerator,
    DocumentFeaturesContext,
    FeatureDef,
    LayoutModelData,
    RelativeFontSizeFeature,
    LineIndentationStatusFeature,
    get_block_status_with_blockend_for_single_token,
    get_block_status_with_blockstart_for_single_token,
    get_line_status_with_lineend_for_single_token,
    get_line_status_with_linestart_for_single_token,
    get_token_font_size_feature,
    get_digit_feature,
    get_capitalisation_feature,
    get_punctuation_type_feature,
    get_raw_punctuation_profile_feature,
    get_punctuation_profile_feature_for_raw_punctuation_profile_feature,
    get_punctuation_profile_length_for_raw_punctuation_profile_feature,
    get_word_shape_feature
)


def _make_features(
    token_text: str,
    app_features_context: AppFeaturesContext = AppFeaturesContext(),
    is_http: bool = False
) -> ContextAwareLayoutTokenFeatures:
    token = LayoutToken(token_text)
    line = LayoutLine.for_text(token_text)
    return ContextAwareLayoutTokenFeatures(
        token,
        layout_line=line,
        document_features_context=DocumentFeaturesContext(
            app_features_context=app_features_context
        ),
        is_http=is_http
    )


class TestRelativeFontSizeFeature:
    def test_should_return_is_smallest_largest_and_larger_than_avg(self):
        layout_tokens = [
            LayoutToken('', font=LayoutFont('font1', font_size=1)),
            LayoutToken('', font=LayoutFont('font2', font_size=2)),
            LayoutToken('', font=LayoutFont('font3', font_size=3)),
            LayoutToken('', font=LayoutFont('font4', font_size=4))
        ]
        relative_font_size_feature = RelativeFontSizeFeature(layout_tokens)
        assert [
            relative_font_size_feature.is_smallest_font_size(layout_token)
            for layout_token in layout_tokens
        ] == [True, False, False, False]
        assert [
            relative_font_size_feature.is_largest_font_size(layout_token)
            for layout_token in layout_tokens
        ] == [False, False, False, True]
        assert [
            relative_font_size_feature.is_larger_than_average_font_size(layout_token)
            for layout_token in layout_tokens
        ] == [False, False, True, True]

    def test_should_return_false_if_no_font_size_available(self):
        layout_tokens = [
            LayoutToken('', font=LayoutFont('font1', font_size=None)),
            LayoutToken('', font=LayoutFont('font2', font_size=None)),
            LayoutToken('', font=LayoutFont('font3', font_size=None)),
            LayoutToken('', font=LayoutFont('font4', font_size=None))
        ]
        relative_font_size_feature = RelativeFontSizeFeature(layout_tokens)
        assert [
            relative_font_size_feature.is_smallest_font_size(layout_token)
            for layout_token in layout_tokens
        ] == [False, False, False, False]
        assert [
            relative_font_size_feature.is_largest_font_size(layout_token)
            for layout_token in layout_tokens
        ] == [False, False, False, False]
        assert [
            relative_font_size_feature.is_larger_than_average_font_size(layout_token)
            for layout_token in layout_tokens
        ] == [False, False, False, False]


class TestLineIndentationStatusFeature:
    def test_should_detect_not_indented_blocks(self):
        line_indentation_status_feature = LineIndentationStatusFeature()
        line_indentation_status_feature.on_new_block()
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False

    def test_should_detect_indented_blocks(self):
        line_indentation_status_feature = LineIndentationStatusFeature()
        line_indentation_status_feature.on_new_block()
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=50, y=10, width=10, height=10))
        ) is True

    def test_should_reset_indentation_when_shifted_back(self):
        line_indentation_status_feature = LineIndentationStatusFeature()
        line_indentation_status_feature.on_new_block()
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=50, y=10, width=10, height=10))
        ) is True
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False

    def test_should_not_compare_indentation_across_blocks(self):
        # Matches GROBID: previousFeatures=null at block boundary suppresses cross-block comparison.
        # An indented block (x=50) followed by a new block at x=10 should not mark the new
        # block's first line as un-indented — cross-block x comparison is skipped entirely.
        line_indentation_status_feature = LineIndentationStatusFeature()
        line_indentation_status_feature.on_new_block()
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=50, y=10, width=10, height=10))
        ) is True
        # New block at x=10 — should NOT reset indented (no cross-block comparison)
        line_indentation_status_feature.on_new_block()
        line_indentation_status_feature.on_new_line()
        assert line_indentation_status_feature.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is True


class TestLineIndentationStatusFeatureWithCompareAcrossBlocks:
    """Tests for compare_across_blocks=True (GROBID ReferenceSegmenterParser behaviour).

    GROBID skips the first-ever LINESTART update (previousFeatures=null), so the second
    LINESTART also skips comparison (previousLineStartX=NaN). Only from the third LINESTART
    onward does normal comparison happen.  All block boundaries are ignored (no-op on_new_block).
    """

    def _feature(self):
        return LineIndentationStatusFeature(compare_across_blocks=True)

    def test_first_line_of_new_block_resets_indentation_when_shifted_left(self):
        # GROBID double-skip: line 1 skips update; line 2 skips comparison (stores reference).
        # Line 3 (x=50) is the first real comparison: x=50 > x=10 → LINEINDENT.
        # New block at x=10 — SHOULD reset indentation (cross-block comparison happens).
        f = self._feature()
        f.on_new_block()
        f.on_new_line()
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False  # line 1: first-ever skip, lineStartX not set
        f.on_new_line()
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False  # line 2: previous=None, no comparison, stores lineStartX=10
        f.on_new_line()
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=50, y=10, width=10, height=10))
        ) is True   # line 3: compare x=50 vs x=10 → LINEINDENT
        # New block at x=10 — SHOULD reset indented (cross-block comparison happens)
        f.on_new_block()
        f.on_new_line()
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False  # line 4 (new block): compare x=10 vs x=50 → ALIGNEDLEFT

    def test_first_line_of_new_block_sets_indentation_when_shifted_right(self):
        # After the double-skip warm-up, a new block at x=50 (right of x=10) is LINEINDENT.
        f = self._feature()
        f.on_new_block()
        f.on_new_line()
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False  # line 1: first-ever skip
        f.on_new_line()
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False  # line 2: stores lineStartX=10, no comparison
        # New block indented right of x=10
        f.on_new_block()
        f.on_new_line()
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=50, y=10, width=10, height=10))
        ) is True   # line 3 (new block): compare x=50 vs x=10 → LINEINDENT


class TestLineIndentationStatusFeatureWithPersistAcrossBlocks:
    """Tests for persist_across_blocks=True (GROBID HeaderParser behaviour).

    GROBID: lineStartX persists across blocks; the first line of each new block does NOT
    update lineStartX (previousFeatures=null suppresses it); the second line compares
    against the previous block's last lineStartX.
    """

    def _feature(self) -> LineIndentationStatusFeature:
        return LineIndentationStatusFeature(persist_across_blocks=True)

    def test_first_line_of_block_does_not_update_reference(self):
        # Block 1: line 0 skip, line 1 establishes reference.
        # Block 2: first line must not update the reference — so _line_start_x stays block-1 value.
        f = self._feature()
        f.on_new_block()
        f.on_new_line()
        f.get_is_indented_and_update(  # line 0: skip
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        )
        f.on_new_line()
        f.get_is_indented_and_update(  # line 1: establishes _line_start_x=10
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        )
        f.on_new_block()
        f.on_new_line()
        # First line of block 2 at x=50: skipped → _line_start_x still 10, _is_indented still False
        assert f.get_is_indented_and_update(
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=50, y=10, width=10, height=10))
        ) is False

    def test_second_line_of_new_block_compares_against_previous_block(self):
        # Concrete case: email block ends at x=28, abstract block is at x=375.
        # Line 1 of the abstract block (second line) must detect the large x-shift as LINEINDENT.
        f = self._feature()
        # Block 1 (email): establish _line_start_x=28
        f.on_new_block()
        f.on_new_line()
        f.get_is_indented_and_update(  # line 0: skip
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=28, y=10, width=28, height=10))
        )
        f.on_new_line()
        f.get_is_indented_and_update(  # line 1: _line_start_x=28
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=28, y=10, width=28, height=10))
        )
        # Block 2 (abstract, all lines at x=375)
        f.on_new_block()
        f.on_new_line()
        # First line: skip → _line_start_x stays 28
        assert f.get_is_indented_and_update(
            LayoutToken('Introdução:', coordinates=LayoutPageCoordinates(
                x=375, y=10, width=46, height=10
            ))
        ) is False
        f.on_new_line()
        # Second line: compare 375 against _line_start_x=28 → large shift → LINEINDENT
        assert f.get_is_indented_and_update(
            LayoutToken('um', coordinates=LayoutPageCoordinates(x=375, y=10, width=14, height=10))
        ) is True

    def test_within_block_indentation_detected_from_third_line(self):
        # Line 0 skip, line 1 establishes reference, line 2 detects indentation.
        f = self._feature()
        f.on_new_block()
        f.on_new_line()
        assert f.get_is_indented_and_update(  # line 0: skip
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False
        f.on_new_line()
        assert f.get_is_indented_and_update(  # line 1: establishes ref x=10
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=10, y=10, width=10, height=10))
        ) is False
        f.on_new_line()
        assert f.get_is_indented_and_update(  # line 2: compare 50 against 10 → True
            LayoutToken('x', coordinates=LayoutPageCoordinates(x=50, y=10, width=10, height=10))
        ) is True


class _TestBaseGetLineStatus(ABC):
    @abstractmethod
    def get_line_status(self, *args) -> str:
        pass

    def test_should_return_linestart_for_first_token(self):
        assert self.get_line_status(0, 10) == 'LINESTART'

    def test_should_return_lineend_for_last_token(self):
        assert self.get_line_status(9, 10) == 'LINEEND'

    def test_should_return_linein_for_token_not_first_or_last(self):
        assert self.get_line_status(1, 10) == 'LINEIN'
        assert self.get_line_status(8, 10) == 'LINEIN'


class TestGetLineStatusWithLineEndForSingleToken(_TestBaseGetLineStatus):
    def get_line_status(self, *args) -> str:
        return get_line_status_with_lineend_for_single_token(*args)

    def test_should_return_lineend_for_single_token(self):
        assert self.get_line_status(0, 1) == 'LINEEND'


class TestGetLineStatusWithLineStartForSingleToken(_TestBaseGetLineStatus):
    def get_line_status(self, *args) -> str:
        return get_line_status_with_linestart_for_single_token(*args)

    def test_should_return_lineend_for_single_token(self):
        assert self.get_line_status(0, 1) == 'LINESTART'


class _TestBaseGetBlockStatus(ABC):
    @abstractmethod
    def get_block_status(self, *args) -> str:
        pass

    def test_should_return_blockstart_for_first_token_on_first_line(self):
        assert self.get_block_status(0, 10, 'LINESTART') == 'BLOCKSTART'

    def test_should_return_blockend_for_last_token_on_last_line(self):
        assert self.get_block_status(9, 10, 'LINEEND') == 'BLOCKEND'

    def test_should_return_blockin_for_line_not_first_or_last(self):
        assert self.get_block_status(1, 10, 'LINESTART') == 'BLOCKIN'
        assert self.get_block_status(8, 10, 'LINEEND') == 'BLOCKIN'

    def test_should_return_blockin_for_first_line_but_not_first_token(self):
        assert self.get_block_status(0, 10, 'LINEIN') == 'BLOCKIN'

    def test_should_return_blockin_for_last_line_but_not_last_token(self):
        assert self.get_block_status(9, 10, 'LINEIN') == 'BLOCKIN'


class TestGetBlockStatusWithBlockEndForSingleToken(_TestBaseGetBlockStatus):
    def get_block_status(self, *args) -> str:
        return get_block_status_with_blockend_for_single_token(*args)

    def test_should_return_blockend_for_single_token(self):
        assert self.get_block_status(0, 1, 'LINEEND') == 'BLOCKEND'


class TestGetBlockStatusWithBlockStartForSingleToken(_TestBaseGetBlockStatus):
    def get_block_status(self, *args) -> str:
        return get_block_status_with_blockstart_for_single_token(*args)

    def test_should_return_blockend_for_single_token(self):
        assert self.get_block_status(0, 1, 'LINESTART') == 'BLOCKSTART'


class TestGetTokenFontSizeFeature:
    def test_should_return_higherfont_without_previous_token(self):
        assert get_token_font_size_feature(
            previous_token=None,
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy'
            ))
        ) == 'HIGHERFONT'

    def test_should_return_lowerfont_if_font_size_is_smaller(self):
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=2
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=1
            ))
        ) == 'LOWERFONT'

    def test_should_return_samefont_if_font_size_is_the_same(self):
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=1
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=1
            ))
        ) == 'SAMEFONTSIZE'

    def test_should_return_higherfont_if_font_size_is_larger(self):
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=1
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=2
            ))
        ) == 'HIGHERFONT'

    def test_should_return_higherfont_if_previous_font_has_no_size(self):
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=None
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=1
            ))
        ) == 'HIGHERFONT'

    def test_should_return_higherfont_if_new_font_has_no_size(self):
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=1
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=None
            ))
        ) == 'HIGHERFONT'

    def test_should_return_samefontsize_for_sub_integer_float_decrease(self):
        # GROBID uses (int) cast; 9.96 and 9.5 both truncate to 9 → SAMEFONTSIZE
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=9.96
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=9.5
            ))
        ) == 'SAMEFONTSIZE'

    def test_should_return_samefontsize_for_sub_integer_float_increase(self):
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=9.5
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=9.96
            ))
        ) == 'SAMEFONTSIZE'

    def test_should_return_lowerfont_for_cross_integer_decrease(self):
        # 9.96 → int 9, 10.5 → int 10: different integers → LOWERFONT
        assert get_token_font_size_feature(
            previous_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=10.5
            )),
            current_token=LayoutToken('', font=LayoutFont(
                font_id='dummy', font_size=9.96
            ))
        ) == 'LOWERFONT'


class TestGetDigitFeature:
    def test_should_return_nodigit(self):
        assert get_digit_feature('abc') == 'NODIGIT'

    def test_should_return_alldigit(self):
        assert get_digit_feature('123') == 'ALLDIGIT'

    def test_should_return_containsdigit(self):
        assert get_digit_feature('abc123xyz') == 'CONTAINSDIGITS'


class TestGetCapitalisationFeature:
    def test_should_return_nocaps(self):
        assert get_capitalisation_feature('abc') == 'NOCAPS'

    def test_should_return_allcap(self):
        assert get_capitalisation_feature('ABC') == 'ALLCAP'

    def test_should_return_initcap(self):
        assert get_capitalisation_feature('Abc') == 'INITCAP'

    def test_should_return_allcap_for_symbols(self):
        assert get_capitalisation_feature('*') == 'ALLCAP'


class TestGetPunctuationTypeFeature:
    def test_should_return_openbracket(self):
        assert get_punctuation_type_feature('(') == 'OPENBRACKET'
        assert get_punctuation_type_feature('[') == 'OPENBRACKET'

    def test_should_return_endbracket(self):
        assert get_punctuation_type_feature(')') == 'ENDBRACKET'
        assert get_punctuation_type_feature(']') == 'ENDBRACKET'

    def test_should_return_dot(self):
        assert get_punctuation_type_feature('.') == 'DOT'

    def test_should_return_comma(self):
        assert get_punctuation_type_feature(',') == 'COMMA'

    def test_should_return_hyphen(self):
        assert get_punctuation_type_feature('-') == 'HYPHEN'
        assert get_punctuation_type_feature('–') == 'HYPHEN'

    def test_should_return_quote(self):
        assert get_punctuation_type_feature('"') == 'QUOTE'
        assert get_punctuation_type_feature('\'') == 'QUOTE'
        assert get_punctuation_type_feature('`') == 'QUOTE'
        assert get_punctuation_type_feature('’') == 'QUOTE'

    def test_should_return_punct(self):
        assert get_punctuation_type_feature(',,') == 'PUNCT'
        assert get_punctuation_type_feature('::') == 'PUNCT'
        assert get_punctuation_type_feature(';;') == 'PUNCT'
        assert get_punctuation_type_feature('??') == 'PUNCT'
        assert get_punctuation_type_feature('..') == 'PUNCT'

    def test_should_return_nopunct(self):
        assert get_punctuation_type_feature('abc') == 'NOPUNCT'


class TestGetRawPunctuationProfileFeature:
    def test_should_return_empty_string_for_empty_string(self):
        assert get_raw_punctuation_profile_feature('') == ''

    def test_should_return_empty_string_for_letters_or_digits(self):
        assert get_raw_punctuation_profile_feature('abc123') == ''

    def test_should_return_empty_string_for_space(self):
        assert get_raw_punctuation_profile_feature(' ') == ''

    def test_should_return_punctuation_characters(self):
        assert get_raw_punctuation_profile_feature('x.,;x') == '.,;'


class TestGetPunctuationProfileForRawPunctuationPofileFeature:
    def test_should_return_no_for_empty_punctuation_profile(self):
        assert get_punctuation_profile_feature_for_raw_punctuation_profile_feature(
            ''
        ) == 'no'

    def test_should_return_passed_in_value_if_not_empty(self):
        assert get_punctuation_profile_feature_for_raw_punctuation_profile_feature(
            '.:'
        ) == '.:'


class TestGetPunctuationProfileLengthForRawPunctuationPofileFeature:
    def test_should_return_zero_for_empty_punctuation_profile(self):
        assert get_punctuation_profile_length_for_raw_punctuation_profile_feature(
            ''
        ) == '0'

    def test_should_return_length_of_punctuation_profile(self):
        assert get_punctuation_profile_length_for_raw_punctuation_profile_feature(
            '.'
        ) == '1'

    def test_should_clip_value_if_max_length_is_provided(self):
        assert get_punctuation_profile_length_for_raw_punctuation_profile_feature(
            '.' * 11,
            max_length=10
        ) == '10'

    def test_should_not_clip_value_if_max_length_is_not_provided(self):
        assert get_punctuation_profile_length_for_raw_punctuation_profile_feature(
            '.' * 11
        ) == '11'


class TestGetWordShapeFeature:
    def test_should_return_blank_if_input_is_blank(self):
        # Note: this shouldn't really happen as we don't use whitespace tokens
        assert get_word_shape_feature(' ') == ' '

    def test_should_return_capitalisation(self):
        # copied test cases from:
        # https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/test/java/org/grobid/core/utilities/TextUtilitiesTest.java#L210-L228
        assert get_word_shape_feature('This') == 'Xxxx'
        assert get_word_shape_feature('Equals') == 'Xxxx'
        assert get_word_shape_feature("O'Conor") == "X'Xxxx"
        assert get_word_shape_feature('McDonalds') == 'XxXxxx'
        assert get_word_shape_feature('any-where') == 'xx-xxx'
        assert get_word_shape_feature('1.First') == 'd.Xxxx'
        assert get_word_shape_feature('ThisIsCamelCase') == 'XxXxXxXxxx'
        assert get_word_shape_feature('This:happens') == 'Xx:xxx'
        assert get_word_shape_feature('ABC') == 'XXX'
        assert get_word_shape_feature('AC') == 'XX'
        assert get_word_shape_feature('A') == 'X'
        assert get_word_shape_feature('AbA') == 'XxX'
        assert get_word_shape_feature('uü') == 'xx'
        assert get_word_shape_feature('Üwe') == 'Xxx'
        assert get_word_shape_feature('Tes9t99') == 'Xxdxdd'
        assert get_word_shape_feature('T') == 'X'


class TestContextAwareLayoutTokenFeaturesIsYear:
    def test_should_return_1_for_four_digit_year_starting_with_2(self):
        assert _make_features('2025').get_str_is_year() == '1'

    def test_should_return_1_for_four_digit_year_starting_with_1(self):
        assert _make_features('1999').get_str_is_year() == '1'

    def test_should_return_1_when_year_pattern_found_within_longer_token(self):
        # GROBID uses .find() so matches anywhere in the token (e.g. DOI components)
        assert _make_features('12688').get_str_is_year() == '1'

    def test_should_return_0_for_short_number(self):
        assert _make_features('10').get_str_is_year() == '0'

    def test_should_return_0_for_word(self):
        assert _make_features('Innovation').get_str_is_year() == '0'


class TestContextAwareLayoutTokenFeaturesIsMonth:
    def test_should_return_1_for_full_month_name(self):
        assert _make_features('January').get_str_is_month() == '1'
        assert _make_features('August').get_str_is_month() == '1'
        assert _make_features('December').get_str_is_month() == '1'

    def test_should_return_1_for_month_abbreviation(self):
        assert _make_features('Jan').get_str_is_month() == '1'
        assert _make_features('Apr').get_str_is_month() == '1'
        assert _make_features('Aug').get_str_is_month() == '1'

    def test_should_be_case_insensitive(self):
        assert _make_features('JANUARY').get_str_is_month() == '1'
        assert _make_features('aug').get_str_is_month() == '1'

    def test_should_return_0_for_non_month(self):
        assert _make_features('Innovation').get_str_is_month() == '0'
        assert _make_features('2025').get_str_is_month() == '0'


class TestContextAwareLayoutTokenFeaturesIsHttp:
    def test_should_return_1_when_pre_computed_is_http_is_true(self):
        features = _make_features('https://doi.org/10.1234/example', is_http=True)
        assert features.get_str_is_http() == '1'

    def test_should_return_0_when_pre_computed_is_http_is_false(self):
        assert _make_features('http://example.com/path').get_str_is_http() == '0'

    def test_should_return_0_for_plain_word(self):
        assert _make_features('Innovation').get_str_is_http() == '0'


class TestContextAwareLayoutTokenFeaturesIsCommonName:
    def test_should_return_0_when_no_lookup_configured(self):
        assert _make_features('innovation').get_str_is_common_name() == '0'

    def test_should_return_1_for_word_in_lookup(self):
        lookup = SimpleTextLookUp({'innovation', 'battery', 'recycling'})
        ctx = AppFeaturesContext(common_name_lookup=lookup)
        assert _make_features('innovation', ctx).get_str_is_common_name() == '1'
        assert _make_features('Innovation', ctx).get_str_is_common_name() == '1'

    def test_should_return_0_for_word_not_in_lookup(self):
        lookup = SimpleTextLookUp({'innovation'})
        ctx = AppFeaturesContext(common_name_lookup=lookup)
        assert _make_features('Patil', ctx).get_str_is_common_name() == '0'


class TestContextAwareLayoutTokenFeaturesRelativeDocumentPosition:
    def _make(self, document_char_index: int, document_char_count: int):
        return ContextAwareLayoutTokenFeatures(
            LayoutToken('x'),
            layout_line=LayoutLine.for_text('x'),
            document_features_context=DocumentFeaturesContext(AppFeaturesContext()),
            document_char_index=document_char_index,
            document_char_count=document_char_count
        )

    def test_returns_0_at_start_of_document(self):
        assert self._make(0, 100).get_str_relative_document_position() == '0'

    def test_returns_12_at_end_of_document(self):
        assert self._make(100, 100).get_str_relative_document_position() == '12'

    def test_returns_scaled_value_for_midpoint(self):
        assert self._make(50, 100).get_str_relative_document_position() == '6'

    def test_returns_0_when_document_char_count_is_zero(self):
        # No tokens → treated as past-end, returns max bin
        assert self._make(0, 0).get_str_relative_document_position() == '12'


class TestContextAwareLayoutTokenFeaturesRelativePagePosition:
    def _make(self, page_token_y: float, page_height: float):
        return ContextAwareLayoutTokenFeatures(
            LayoutToken('x'),
            layout_line=LayoutLine.for_text('x'),
            document_features_context=DocumentFeaturesContext(AppFeaturesContext()),
            page_token_y=page_token_y,
            page_height=page_height
        )

    def test_returns_0_at_top_of_page(self):
        assert self._make(0.0, 800.0).get_str_relative_page_position() == '0'

    def test_returns_12_at_bottom_of_page(self):
        assert self._make(800.0, 800.0).get_str_relative_page_position() == '12'

    def test_returns_scaled_value_for_midpoint(self):
        assert self._make(400.0, 800.0).get_str_relative_page_position() == '6'

    def test_returns_0_when_page_height_is_zero(self):
        assert self._make(0.0, 0.0).get_str_relative_page_position() == '0'


class TestContextAwareLayoutTokenFeaturesIsProperName:
    def test_should_return_0_when_no_lookups_configured(self):
        assert _make_features('Patil').get_str_is_proper_name() == '0'

    def test_should_return_1_for_word_in_first_name_lookup(self):
        lookup = SimpleTextLookUp({'Anish', 'Willem'})
        ctx = AppFeaturesContext(first_name_lookup=lookup)
        assert _make_features('Anish', ctx).get_str_is_proper_name() == '1'

    def test_should_return_1_for_word_in_last_name_lookup(self):
        lookup = SimpleTextLookUp({'Patil', 'Vonk'})
        ctx = AppFeaturesContext(last_name_lookup=lookup)
        assert _make_features('Patil', ctx).get_str_is_proper_name() == '1'

    def test_should_return_0_for_word_not_in_either_lookup(self):
        ctx = AppFeaturesContext(
            first_name_lookup=SimpleTextLookUp({'Anish'}),
            last_name_lookup=SimpleTextLookUp({'Patil'})
        )
        assert _make_features('Innovation', ctx).get_str_is_proper_name() == '0'


class _PositionFeaturesDataGenerator(ContextAwareLayoutTokenModelDataGenerator):
    """Minimal generator that only outputs position features, for loop-level tests."""
    def __init__(self):
        super().__init__(DocumentFeaturesContext(AppFeaturesContext()))
        self._feature_defs: List[FeatureDef[ContextAwareLayoutTokenFeatures]] = [
            FeatureDef('relative_document_position',
                       lambda f: f.get_str_relative_document_position()),
            FeatureDef('relative_page_position',
                       lambda f: f.get_str_relative_page_position()),
        ]


def _make_page_with_token(
    text: str,
    y: float,
    page_number: int,
    page_height: float = 800.0
) -> LayoutPage:
    token = LayoutToken(
        text,
        coordinates=LayoutPageCoordinates(
            x=0, y=y, width=10, height=10, page_number=page_number
        )
    )
    return LayoutPage(
        blocks=[LayoutBlock(lines=[LayoutLine(tokens=[token])])],
        meta=LayoutPageMeta(
            page_number=page_number,
            coordinates=LayoutPageCoordinates(
                x=0, y=0, width=600, height=page_height, page_number=page_number
            )
        )
    )


def _pos(row: LayoutModelData, feature: str) -> int:
    gen = _PositionFeaturesDataGenerator()
    index = gen.feature_names.index(feature)
    return int(row.data_line.split()[index])


class TestIterModelDataForLayoutDocumentPositionFeatures:
    def test_document_position_does_not_reset_across_pages(self):
        gen = _PositionFeaturesDataGenerator()
        page1 = _make_page_with_token('A' * 20, y=100, page_number=1)
        page2 = _make_page_with_token('B' * 20, y=100, page_number=2)
        rows = list(gen.iter_model_data_for_layout_document(
            LayoutDocument(pages=[page1, page2])
        ))
        assert _pos(rows[1], 'relative_document_position') > _pos(
            rows[0], 'relative_document_position'
        )

    def test_page_position_resets_on_new_page(self):
        # Token near bottom of page 1 should have a higher page position than
        # a token near top of page 2.
        gen = _PositionFeaturesDataGenerator()
        page1 = _make_page_with_token('a', y=700.0, page_number=1)
        page2 = _make_page_with_token('b', y=50.0, page_number=2)
        rows = list(gen.iter_model_data_for_layout_document(
            LayoutDocument(pages=[page1, page2])
        ))
        assert _pos(rows[0], 'relative_page_position') > _pos(
            rows[1], 'relative_page_position'
        )

    def test_page_position_is_0_when_page_has_no_coordinates(self):
        gen = _PositionFeaturesDataGenerator()
        token = LayoutToken(
            'word',
            coordinates=LayoutPageCoordinates(x=0, y=400, width=10, height=10)
        )
        page = LayoutPage(
            blocks=[LayoutBlock(lines=[LayoutLine(tokens=[token])])],
            meta=LayoutPageMeta(page_number=1, coordinates=None)
        )
        rows = list(gen.iter_model_data_for_layout_document(LayoutDocument(pages=[page])))
        assert rows[0].data_line.split()[0] == '0'
