from typing import Optional

from sciencebeam_parser.document.dehyphenation import (
    DehyphenateMode,
    dehyphenate_layout_block,
    get_document_words
)
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutFont,
    LayoutLine,
    LayoutLineMeta,
    LayoutPage,
    LayoutPageCoordinates
)


def _line(text: str, y: Optional[float] = None, font: Optional[LayoutFont] = None) -> LayoutLine:
    line = LayoutLine.for_text(text, line_meta=LayoutLineMeta(line_id=id(text)))
    return LayoutLine(tokens=[
        token._replace(
            coordinates=(
                LayoutPageCoordinates(x=0, y=y, width=10, height=10) if y is not None else None
            ),
            font=font if font is not None and token.text == '-' else token.font
        )
        for token in line.tokens
    ])


def _block(*lines: str) -> LayoutBlock:
    return LayoutBlock(lines=[_line(text, y=10.0 * index) for index, text in enumerate(lines)])


def _document(*lines: str) -> LayoutDocument:
    return LayoutDocument(pages=[LayoutPage(blocks=[
        LayoutBlock(lines=[_line(text) for text in lines])
    ])])


def _by_document(block: LayoutBlock, *document_lines: str) -> str:
    return dehyphenate_layout_block(
        block, DehyphenateMode.BY_DOCUMENT, get_document_words(_document(*document_lines))
    ).text


def _as_grobid(block: LayoutBlock) -> str:
    return dehyphenate_layout_block(block, DehyphenateMode.GROBID).text


class TestGetDocumentWords:
    def test_should_collect_words_set_inside_a_line_in_lower_case(self):
        assert get_document_words(_document('Percepção of colour')).words == {
            'percepção', 'of', 'colour'
        }

    def test_should_leave_out_the_parts_of_a_word_broken_at_a_line_end(self):
        assert get_document_words(_document('the per-', 'cepção of colour')).words == {
            'the', 'of', 'colour'
        }

    def test_should_keep_a_capitalised_first_word_of_the_line_after_a_broken_word(self):
        assert get_document_words(_document('the per-', 'Heading of colour')).words == {
            'the', 'heading', 'of', 'colour'
        }

    def test_should_collect_a_word_with_combining_accents_in_composed_form(self):
        assert get_document_words(_document('percepc\u0327a\u0303o')).words == {'percepção'}

    def test_should_collect_hyphenated_words_set_inside_a_line(self):
        assert get_document_words(
            _document('an alkali-activated binder', 'a binder - activated', 'the COVID-19 wave')
        ).hyphenated_words == {'alkali-activated', 'covid-19'}


class TestDehyphenateOff:
    def test_should_return_the_block_unchanged(self):
        block = _block('the per-', 'cepção of colour')
        assert dehyphenate_layout_block(block, DehyphenateMode.OFF) is block


class TestDehyphenateByDocument:
    def test_should_remove_the_hyphen_where_the_document_sets_the_joined_word(self):
        assert _by_document(
            _block('the per-', 'cepção of colour'), 'their percepção differs'
        ) == 'the percepção of colour'

    def test_should_leave_the_hyphen_and_the_space_where_the_document_says_nothing(self):
        assert _by_document(
            _block('the per-', 'cepção of colour')
        ) == 'the per- cepção of colour'

    def test_should_leave_a_suspended_hyphen_where_the_document_says_nothing(self):
        assert _by_document(
            _block('low-', 'and high-income')
        ) == 'low- and high-income'

    def test_should_keep_the_hyphen_where_the_document_sets_the_hyphenated_word(self):
        assert _by_document(
            _block('an alkali-', 'activated binder'), 'alkaliactivated', 'alkali-activated'
        ) == 'an alkali-activated binder'

    def test_should_keep_the_hyphen_before_a_capital_the_document_sets_after_it(self):
        assert _by_document(
            _block('the SARS-', 'CoV-2 virus'), 'the SARS-CoV-2 genome'
        ) == 'the SARS-CoV-2 virus'

    def test_should_keep_the_hyphen_before_a_digit_the_document_sets_after_it(self):
        assert _by_document(
            _block('the COVID-', '19 wave'), 'during COVID-19'
        ) == 'the COVID-19 wave'

    def test_should_not_remove_the_hyphen_before_a_capital(self):
        assert _by_document(
            _block('the SARS-', 'CoV-2 virus'), 'SARSCoV'
        ) == 'the SARS- CoV-2 virus'

    def test_should_remove_a_soft_hyphen_where_the_document_sets_the_joined_word(self):
        assert _by_document(
            _block('the per\u00ad', 'cepção of colour'), 'percepção'
        ) == 'the percepção of colour'

    def test_should_remove_the_hyphen_before_a_continuation_with_combining_accents(self):
        assert _by_document(
            _block('the per-', 'cepc\u0327a\u0303o of colour'), 'percepção'
        ) == 'the percepc\u0327a\u0303o of colour'

    def test_should_write_a_kept_soft_hyphen_as_a_hyphen(self):
        assert _by_document(
            _block('an alkali\u00ad', 'activated binder'), 'alkali-activated'
        ) == 'an alkali-activated binder'

    def test_should_join_the_first_line_of_the_next_block(self):
        block = LayoutBlock.merge_blocks([_block('the per-'), _block('cepção of colour')])
        assert _by_document(block, 'percepção') == 'the percepção of colour'

    def test_should_not_touch_a_hyphen_inside_a_line(self):
        assert _by_document(
            _block('the per- cepção of', 'colour'), 'percepção'
        ) == 'the per- cepção of colour'

    def test_should_not_touch_a_dash_set_apart_from_the_word(self):
        assert _by_document(
            _block('the results -', 'which differ'), 'resultswhich'
        ) == 'the results - which differ'

    def test_should_keep_each_line_and_its_coordinates(self):
        block = dehyphenate_layout_block(
            _block('the per-', 'cepção of colour'),
            DehyphenateMode.BY_DOCUMENT,
            get_document_words(_document('percepção'))
        )
        assert [line.text for line in block.lines] == ['the per', 'cepção of colour']
        assert all(token.coordinates for token in block.iter_all_tokens())


class TestDehyphenateAsGrobid:
    def test_should_remove_the_hyphen_before_a_lower_case_word_on_the_next_line(self):
        assert _as_grobid(_block('an alkali-', 'activated binder')) == 'an alkaliactivated binder'

    def test_should_keep_the_hyphen_before_a_capital_and_drop_the_line_break(self):
        assert _as_grobid(_block('the SARS-', 'CoV-2 virus')) == 'the SARS-CoV-2 virus'

    def test_should_keep_the_hyphen_where_the_next_line_lies_higher_on_the_page(self):
        block = LayoutBlock(lines=[_line('the per-', y=500), _line('cepção of colour', y=50)])
        assert _as_grobid(block) == 'the per-cepção of colour'

    def test_should_not_touch_a_hyphen_inside_a_line(self):
        assert _as_grobid(_block('an alkali-activated binder')) == 'an alkali-activated binder'

    def test_should_not_touch_a_superscript_hyphen(self):
        block = LayoutBlock(lines=[
            _line('the per-', y=0, font=LayoutFont(font_id='sup', is_superscript=True)),
            _line('cepção of colour', y=10)
        ])
        assert _as_grobid(block) == 'the per- cepção of colour'

    def test_should_not_remove_a_soft_hyphen(self):
        assert _as_grobid(
            _block('the per\u00ad', 'cepção of colour')
        ) == 'the per\u00ad cepção of colour'
