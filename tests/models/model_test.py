from typing import List, Optional, Tuple
from unittest.mock import MagicMock

from sciencebeam_parser.document.layout_document import (
    LayoutPage,
    LayoutPageCoordinates,
    LayoutPageMeta,
    get_layout_tokens_for_text,
    join_layout_tokens,
    LayoutDocument,
    LayoutBlock,
    LayoutLine
)
from sciencebeam_parser.models.data import DocumentFeaturesContext, ModelDataGenerator
from sciencebeam_parser.models.extract import ModelSemanticExtractor
from sciencebeam_parser.models.model import (
    LayoutDocumentLabelResult,
    LayoutModelLabel,
    Model,
    iter_entity_values_predicted_labels
)
from sciencebeam_parser.models.model_impl import ModelImpl


TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'


class TestLayoutDocumentLabelResult:
    def test_should_filter_by_line_without_token(self):
        tagged_lines = [
            (TAG_1, LayoutLine.for_text('this is line 1')),
            (TAG_2, LayoutLine.for_text('this is line 2'))
        ]
        layout_model_labels = [
            LayoutModelLabel(
                label=tag,
                label_token_text=line.text,
                layout_line=line,
                layout_token=None
            )
            for tag, line in tagged_lines
            for token in line.tokens
        ]
        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock(lines=[line for _, line in tagged_lines])
        ])])
        layout_document_label_result = LayoutDocumentLabelResult(
            layout_document,
            layout_model_labels
        )
        for tag, line in tagged_lines:
            assert (
                join_layout_tokens(
                    layout_document_label_result.get_filtered_document_by_label(tag)
                    .iter_all_tokens()
                ) == join_layout_tokens(line.tokens)
            )

    def test_should_filter_by_token_label(self):
        tagged_tokens = [
            (TAG_1, get_layout_tokens_for_text('this is line 1')),
            (TAG_2, get_layout_tokens_for_text('this is line 2'))
        ]
        line = LayoutLine([
            token
            for _, tokens in tagged_tokens
            for token in tokens
        ])
        layout_model_labels = [
            LayoutModelLabel(
                label=tag,
                label_token_text=token.text,
                layout_line=line,
                layout_token=token
            )
            for tag, tokens in tagged_tokens
            for token in tokens
        ]
        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock(lines=[line])
        ])])
        layout_document_label_result = LayoutDocumentLabelResult(
            layout_document,
            layout_model_labels
        )
        for tag, tokens in tagged_tokens:
            assert (
                join_layout_tokens(
                    layout_document_label_result.get_filtered_document_by_label(tag)
                    .iter_all_tokens()
                ) == join_layout_tokens(tokens)
            )

    def test_should_filter_by_token_multiple_labels(self):
        tagged_tokens = [
            (TAG_1, get_layout_tokens_for_text('tokens tag 1')),
            (TAG_2, get_layout_tokens_for_text('tokens tag 2')),
            (TAG_3, get_layout_tokens_for_text('tokens tag 3'))
        ]
        line = LayoutLine([
            token
            for _, tokens in tagged_tokens
            for token in tokens
        ])
        layout_model_labels = [
            LayoutModelLabel(
                label=tag,
                label_token_text=token.text,
                layout_line=line,
                layout_token=token
            )
            for tag, tokens in tagged_tokens
            for token in tokens
        ]
        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock(lines=[line])
        ])])
        layout_document_label_result = LayoutDocumentLabelResult(
            layout_document,
            layout_model_labels
        )
        assert join_layout_tokens(
            layout_document_label_result.get_filtered_document_by_labels(
                [TAG_1, TAG_3]
            )
            .iter_all_tokens()
        ) == join_layout_tokens(
            tagged_tokens[0][1] + tagged_tokens[2][1]
        )

    def test_should_preserve_page_meta_in_filtered_document(self):
        page_coords = LayoutPageCoordinates(x=0, y=0, width=612, height=792, page_number=1)
        page_meta = LayoutPageMeta(page_number=1, coordinates=page_coords)
        tagged_lines = [
            (TAG_1, LayoutLine.for_text('section title'))
        ]
        layout_model_labels = [
            LayoutModelLabel(
                label=TAG_1,
                label_token_text=line.text,
                layout_line=line,
                layout_token=None
            )
            for _, line in tagged_lines
            for _ in line.tokens
        ]
        layout_document = LayoutDocument(pages=[LayoutPage(
            blocks=[LayoutBlock(lines=[line for _, line in tagged_lines])],
            meta=page_meta
        )])
        layout_document_label_result = LayoutDocumentLabelResult(
            layout_document,
            layout_model_labels
        )
        filtered = layout_document_label_result.get_filtered_document_by_label(TAG_1)
        assert len(filtered.pages) == 1
        assert filtered.pages[0].meta.coordinates is not None
        assert filtered.pages[0].meta.coordinates.height == 792


class TestIterEntityValuesPredictedLabels:
    def test_should_extract_multiple_entity_values(self):
        tag_result = [
            ('The', 'B-<title>'),
            ('Title', 'I-<title>'),
            ('Some', 'B-<abstract>'),
            ('Abstract', 'I-<abstract>')
        ]
        assert list(iter_entity_values_predicted_labels(tag_result)) == [
            ('<title>', 'The Title'),
            ('<abstract>', 'Some Abstract')
        ]

    def test_should_extract_multiple_entity_values_including_none(self):
        tag_result = [
            ('The', 'B-<title>'),
            ('Title', 'I-<title>'),
            ('Other1', 'O'),
            ('Other2', 'O'),
            ('Some', 'B-<abstract>'),
            ('Abstract', 'I-<abstract>')
        ]
        assert list(iter_entity_values_predicted_labels(tag_result)) == [
            ('<title>', 'The Title'),
            ('O', 'Other1 Other2'),
            ('<abstract>', 'Some Abstract')
        ]


class MockModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_generator = MagicMock(name='data_generator')
        self._semantic_extractor = MagicMock(name='semantic_extractor')

    def get_data_generator(
        self,
        document_features_context: DocumentFeaturesContext
    ) -> ModelDataGenerator:
        return self._data_generator

    def get_semantic_extractor(self) -> ModelSemanticExtractor:
        return self._semantic_extractor


class MockModelImpl(ModelImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_labels_mock = MagicMock(name='predict_labels')
        self.preload_mock = MagicMock(name='preload')

    def predict_labels(
        self,
        texts: List[List[str]],
        features: List[List[List[str]]],
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        return self.predict_labels_mock(texts, features, output_format=output_format)

    def preload(self):
        self.preload_mock()


class TestModel:
    def test_should_preload_model_impl(self):
        model_impl = MockModelImpl()
        model_impl_factory = MagicMock(name='model_impl_factory')
        model_impl_factory.return_value = model_impl
        mock_model = MockModel(model_impl_factory=model_impl_factory)
        mock_model.preload()
        model_impl_factory.assert_called()
        model_impl.preload_mock.assert_called()
