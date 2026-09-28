from typing import List, Tuple

import pytest

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.models.data import (
    DEFAULT_DOCUMENT_FEATURES_CONTEXT,
    FEATURE_FLAVOUR_GROBID
)
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


def _get_block_and_line_status(
    doc: LayoutDocument,
    **kwargs
) -> List[Tuple[str, str, str]]:
    gen = HeaderDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT, **kwargs)
    block_idx = gen.feature_names.index('block_status')
    line_idx = gen.feature_names.index('line_status')
    result = []
    for item in gen.iter_model_data_for_layout_document(doc):
        cols = item.data_line.split()
        result.append((cols[0], cols[block_idx], cols[line_idx]))
    return result


def _get_doc_with_block_starting_with_single_token_line() -> LayoutDocument:
    return LayoutDocument.for_blocks([
        LayoutBlock(lines=[LayoutLine([LayoutToken('Jane'), LayoutToken('Doe')])]),
        LayoutBlock(lines=[
            LayoutLine([LayoutToken('Resumo')]),
            LayoutLine([LayoutToken('Text'), LayoutToken('here')])
        ])
    ])


class TestBlockAndLineStatus:
    def test_should_not_start_block_on_single_token_line_by_default(self):
        # what the biorxiv models were trained on
        doc = _get_doc_with_block_starting_with_single_token_line()
        assert _get_block_and_line_status(doc)[2:] == [
            ('Resumo', 'BLOCKIN', 'LINEEND'),
            ('Text', 'BLOCKIN', 'LINESTART'),
            ('here', 'BLOCKEND', 'LINEEND')
        ]

    def test_should_start_block_on_single_token_line_with_grobid_flavour(self):
        doc = _get_doc_with_block_starting_with_single_token_line()
        assert _get_block_and_line_status(doc, feature_flavour=FEATURE_FLAVOUR_GROBID) == [
            ('Jane', 'BLOCKSTART', 'LINESTART'),
            ('Doe', 'BLOCKEND', 'LINEEND'),
            ('Resumo', 'BLOCKSTART', 'LINESTART'),
            ('Text', 'BLOCKIN', 'LINESTART'),
            ('here', 'BLOCKEND', 'LINEEND')
        ]

    def test_should_keep_feature_names_with_grobid_flavour(self):
        assert HeaderDataGenerator(
            DEFAULT_DOCUMENT_FEATURES_CONTEXT, feature_flavour=FEATURE_FLAVOUR_GROBID
        ).feature_names == HeaderDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT).feature_names

    def test_should_reject_unknown_feature_flavour(self):
        with pytest.raises(ValueError):
            HeaderDataGenerator(DEFAULT_DOCUMENT_FEATURES_CONTEXT, feature_flavour='other')
