from typing import List, Optional, Tuple
from unittest.mock import MagicMock

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
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


def _make_citation_model(retokenize_subdigits=None) -> Tuple[CitationModel, CapturingModelImpl]:
    impl = CapturingModelImpl()
    impl_factory = MagicMock(return_value=impl)
    model_config: dict = {}
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
