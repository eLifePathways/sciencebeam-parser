from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.models.data import DEFAULT_DOCUMENT_FEATURES_CONTEXT
from sciencebeam_parser.models.citation.data import CitationDataGenerator


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


class TestIsMonth:
    def test_month_name_gives_1(self):
        assert _get_feature('january', 'is_month') == '1'

    def test_capitalised_month_gives_1(self):
        assert _get_feature('January', 'is_month') == '1'

    def test_word_gives_0(self):
        assert _get_feature('Lancet', 'is_month') == '0'
