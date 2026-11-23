from typing import List, Optional, Tuple
from unittest.mock import MagicMock

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)
from sciencebeam_parser.document.semantic_document import (
    SemanticLabel,
    SemanticRawReference,
    SemanticRawReferenceText
)
from sciencebeam_parser.models.data import DEFAULT_APP_FEATURES_CONTEXT
from sciencebeam_parser.models.citation.model import CitationModel
from sciencebeam_parser.models.model_impl import ModelImpl


class CapturingModelImpl(ModelImpl):
    def __init__(self):
        self.received_texts: List[List[str]] = []

    def predict_labels(
        self,
        texts: List[List[str]],
        features: List[List[List[str]]],
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        self.received_texts = texts
        return [[(t, '<other>') for t in doc] for doc in texts]

    def preload(self):
        pass


def _make_citation_model(
    retokenize_subdigits=None,
    feature_flavour=None
) -> Tuple[CitationModel, CapturingModelImpl]:
    impl = CapturingModelImpl()
    impl_factory = MagicMock(return_value=impl)
    model_config: dict = {}
    if feature_flavour is not None:
        model_config['feature_flavour'] = feature_flavour
    if retokenize_subdigits is not None:
        model_config['retokenize_subdigits'] = retokenize_subdigits
    model = CitationModel(impl_factory, model_config=model_config)
    model.preload()
    return model, impl


def _doc_with_tokens(*token_texts: str) -> LayoutDocument:
    tokens = [LayoutToken(t) for t in token_texts]
    return LayoutDocument.for_blocks([LayoutBlock(lines=[LayoutLine(tokens)])])


class TestCitationModelSubdigitRetokenization:
    def test_mixed_token_split_by_default(self):
        model, impl = _make_citation_model()
        model.predict_labels_for_layout_documents(
            [_doc_with_tokens('e1006572')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['e', '1006572']]

    def test_digit_then_letter_split_by_default(self):
        model, impl = _make_citation_model()
        model.predict_labels_for_layout_documents(
            [_doc_with_tokens('295X')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['295', 'X']]

    def test_pure_alpha_token_unchanged(self):
        model, impl = _make_citation_model()
        model.predict_labels_for_layout_documents(
            [_doc_with_tokens('Lancet')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['Lancet']]

    def test_retokenize_disabled_leaves_token_unsplit(self):
        model, impl = _make_citation_model(retokenize_subdigits=False)
        model.predict_labels_for_layout_documents(
            [_doc_with_tokens('e1006572')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['e1006572']]

    def test_retokenize_explicitly_enabled(self):
        model, impl = _make_citation_model(retokenize_subdigits=True)
        model.predict_labels_for_layout_documents(
            [_doc_with_tokens('e1006572')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['e', '1006572']]


class TestCitationModelLeadingPunctuation:
    def test_leading_punctuation_is_labelled_by_default(self):
        model, impl = _make_citation_model()
        model.predict_labels_for_layout_documents(
            [_doc_with_tokens('.', 'Doe', 'J', '.')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['.', 'Doe', 'J', '.']]

    def test_leading_punctuation_is_not_labelled_with_grobid_features(self):
        model, impl = _make_citation_model(feature_flavour='grobid')
        model.predict_labels_for_layout_documents(
            [_doc_with_tokens('.', '(', 'Doe', 'J', '.')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['Doe', 'J', '.']]

    def test_document_of_punctuation_only_gets_no_labels(self):
        model, impl = _make_citation_model(feature_flavour='grobid')
        result = model.predict_labels_for_layout_documents(
            [_doc_with_tokens('Doe'), _doc_with_tokens('.', ','), _doc_with_tokens('Roe')],
            DEFAULT_APP_FEATURES_CONTEXT
        )
        assert impl.received_texts == [['Doe'], ['Roe']]
        assert [[t.layout_token.text for t in doc] for doc in result] == [['Doe'], [], ['Roe']]


def _raw_reference_with_label() -> SemanticRawReference:
    return SemanticRawReference([
        SemanticLabel(layout_block=LayoutBlock.for_text('3.')),
        SemanticRawReferenceText(layout_block=LayoutBlock.for_text('Doe J. Ferns.'))
    ])


class TestCitationModelRawReferenceInput:
    def test_includes_the_label_by_default(self):
        model, _ = _make_citation_model()
        layout_document = model.get_layout_document_for_raw_reference(
            _raw_reference_with_label()
        )
        assert [t.text for t in layout_document.iter_all_tokens()][:2] == ['3', '.']

    def test_leaves_out_the_label_with_grobid_features(self):
        model, _ = _make_citation_model(feature_flavour='grobid')
        layout_document = model.get_layout_document_for_raw_reference(
            _raw_reference_with_label()
        )
        assert [t.text for t in layout_document.iter_all_tokens()][0] == 'Doe'
