from typing import List, Sequence

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutLineMeta,
    LayoutToken
)
from sciencebeam_parser.models.data import DEFAULT_DOCUMENT_FEATURES_CONTEXT
from sciencebeam_parser.models.name.data import (
    NameDataGenerator,
    get_layout_document_as_single_line
)


def _get_layout_document(lines: Sequence[Sequence[str]]) -> LayoutDocument:
    return LayoutDocument.for_blocks([LayoutBlock(lines=[
        LayoutLine(tokens=[
            LayoutToken(text, line_meta=LayoutLineMeta(line_id=line_id))
            for text in line_texts
        ])
        for line_id, line_texts in enumerate(lines)
    ])])


def _get_line_status_by_token(
    layout_document: LayoutDocument,
    ignore_line_breaks: bool
) -> List[List[str]]:
    data_generator = NameDataGenerator(
        DEFAULT_DOCUMENT_FEATURES_CONTEXT,
        ignore_line_breaks=ignore_line_breaks
    )
    index = data_generator.feature_names.index('line_status')
    return [
        [model_data.data_line.split()[0], model_data.data_line.split()[index]]
        for model_data in data_generator.iter_model_data_for_layout_document(layout_document)
    ]


AUTHOR_LINES = [['Ann', 'Lee', ','], ['Bo', 'Kim']]


class TestGetLayoutDocumentAsSingleLine:
    def test_should_put_every_token_on_one_line(self):
        layout_document = get_layout_document_as_single_line(
            _get_layout_document(AUTHOR_LINES)
        )
        lines = [line for block in layout_document.iter_all_blocks() for line in block.lines]
        assert [[token.text for token in line.tokens] for line in lines] == [
            ['Ann', 'Lee', ',', 'Bo', 'Kim']
        ]

    def test_should_keep_the_tokens_and_their_own_line(self):
        layout_document = _get_layout_document(AUTHOR_LINES)
        single_line_document = get_layout_document_as_single_line(layout_document)
        tokens = list(layout_document.iter_all_tokens())
        single_line_tokens = list(single_line_document.iter_all_tokens())
        assert all(a is b for a, b in zip(tokens, single_line_tokens))
        assert [token.line_meta.line_id for token in single_line_tokens] == [0, 0, 0, 1, 1]

    def test_should_return_an_empty_document_unchanged(self):
        layout_document = LayoutDocument.for_blocks([])
        assert get_layout_document_as_single_line(layout_document) is layout_document


class TestNameDataGenerator:
    def test_should_give_line_status_per_line_by_default(self):
        assert _get_line_status_by_token(
            _get_layout_document(AUTHOR_LINES), ignore_line_breaks=False
        ) == [
            ['Ann', 'LINESTART'], ['Lee', 'LINEIN'], [',', 'LINEEND'],
            ['Bo', 'LINESTART'], ['Kim', 'LINEEND']
        ]

    def test_should_end_the_line_only_at_the_last_token_when_ignoring_line_breaks(self):
        assert _get_line_status_by_token(
            _get_layout_document(AUTHOR_LINES), ignore_line_breaks=True
        ) == [
            ['Ann', 'LINESTART'], ['Lee', 'LINEIN'], [',', 'LINEIN'],
            ['Bo', 'LINEIN'], ['Kim', 'LINEEND']
        ]
