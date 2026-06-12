from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.models.data import DEFAULT_DOCUMENT_FEATURES_CONTEXT
from sciencebeam_parser.models.header.data import HeaderDataGenerator


def _get_feature(token_text: str, feature_name: str) -> str:
    tokens = [LayoutToken(token_text)]
    line = LayoutLine(tokens)
    doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
    gen = HeaderDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
    data_list = list(gen.iter_model_data_for_layout_document(doc))
    for item in data_list:
        if item.data_line and item.data_line.split()[0] == token_text:
            cols = item.data_line.split()
            idx = gen.feature_names.index(feature_name)
            return cols[idx]
    raise KeyError(f'token {token_text!r} not found')


def _get_email_feature(token_texts_with_whitespace, target_token_text: str) -> str:
    tokens = [
        LayoutToken(text, whitespace=ws)
        for text, ws in token_texts_with_whitespace
    ]
    line = LayoutLine(tokens)
    doc = LayoutDocument.for_blocks([LayoutBlock(lines=[line])])
    gen = HeaderDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT)
    idx = gen.feature_names.index('is_email')
    for item in gen.iter_model_data_for_layout_document(doc):
        if item.data_line and item.data_line.split()[0] == target_token_text:
            return item.data_line.split()[idx]
    raise KeyError(f'token {target_token_text!r} not found')


class TestIsEmail:
    def test_at_token_in_full_email_gives_1(self):
        tokens = [
            ('andre', ''), ('.', ''), ('freitas', ''),
            ('@', ''), ('slmandic', ''), ('.', ''), ('edu', ''), ('.', ''), ('br', ' ')
        ]
        assert _get_email_feature(tokens, '@') == '1'

    def test_local_part_token_in_full_email_gives_1(self):
        tokens = [
            ('andre', ''), ('.', ''), ('freitas', ''),
            ('@', ''), ('slmandic', ''), ('.', ''), ('edu', ''), ('.', ''), ('br', ' ')
        ]
        assert _get_email_feature(tokens, 'andre') == '1'

    def test_domain_token_in_full_email_gives_1(self):
        tokens = [
            ('user', ''), ('@', ''), ('example', ''), ('.', ''), ('com', ' ')
        ]
        assert _get_email_feature(tokens, 'example') == '1'

    def test_word_without_at_sign_gives_0(self):
        assert _get_feature('word', 'is_email') == '0'

    def test_at_sign_alone_gives_0(self):
        assert _get_feature('@', 'is_email') == '0'
