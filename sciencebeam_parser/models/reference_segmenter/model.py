import logging
from typing import Iterable, List, Optional, Tuple

from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutToken
from sciencebeam_parser.models.data import (
    DocumentFeaturesContext,
    LabeledLayoutToken
)
from sciencebeam_parser.models.model import Model, get_entities_including_other
from sciencebeam_parser.models.reference_segmenter.data import ReferenceSegmenterDataGenerator
from sciencebeam_parser.models.reference_segmenter.extract import (
    ReferenceSegmenterSemanticExtractor
)
from sciencebeam_parser.utils.labels import OTHER_LABELS
from sciencebeam_parser.models.reference_segmenter.training_data import (
    ReferenceSegmenterTeiTrainingDataGenerator,
    ReferenceSegmenterTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


REFERENCE_LABEL = '<reference>'


def iter_reference_entity_blocks(
    labeled_layout_tokens: Iterable[LabeledLayoutToken],
    keep_other_text_in_references: bool = True
) -> Iterable[Tuple[str, LayoutBlock]]:
    """The entities, with a reference continued past other text inside it.

    Where asked to keep that other text, it is part of the reference; otherwise it follows it.

    As GROBID's ReferenceSegmenterParser, a reference starts only on a token that begins one.
    """
    labeled_layout_tokens = list(labeled_layout_tokens)
    labels = [labeled_layout_token.label for labeled_layout_token in labeled_layout_tokens]
    reference_tokens: Optional[List[LayoutToken]] = None
    other_blocks: List[Tuple[str, LayoutBlock]] = []

    def flush() -> Iterable[Tuple[str, LayoutBlock]]:
        nonlocal reference_tokens, other_blocks
        if reference_tokens is not None:
            yield REFERENCE_LABEL, LayoutBlock.for_tokens(reference_tokens)
        yield from other_blocks
        reference_tokens = None
        other_blocks = []

    for tag, start, end in get_entities_including_other(labels):
        tokens = [
            labeled_layout_token.layout_token
            for labeled_layout_token in labeled_layout_tokens[start:end + 1]
        ]
        if tag == REFERENCE_LABEL:
            if reference_tokens is not None and not labels[start].startswith('B-'):
                if keep_other_text_in_references:
                    for _, other_block in other_blocks:
                        reference_tokens.extend(other_block.iter_all_tokens())
                    other_blocks = []
                reference_tokens.extend(tokens)
                continue
            yield from flush()
            reference_tokens = tokens
            continue
        if reference_tokens is not None and tag in OTHER_LABELS:
            other_blocks.append((tag, LayoutBlock.for_tokens(tokens)))
            continue
        yield from flush()
        yield tag, LayoutBlock.for_tokens(tokens)
    yield from flush()


class ReferenceSegmenterModel(Model):
    def get_data_generator(
        self,
        document_features_context: DocumentFeaturesContext
    ) -> ReferenceSegmenterDataGenerator:
        return ReferenceSegmenterDataGenerator(
            document_features_context=document_features_context
        )

    def iter_entity_layout_blocks_for_labeled_layout_tokens(
        self,
        labeled_layout_tokens: Iterable[LabeledLayoutToken]
    ) -> Iterable[Tuple[str, LayoutBlock]]:
        return iter_reference_entity_blocks(labeled_layout_tokens)

    def get_semantic_extractor(self) -> ReferenceSegmenterSemanticExtractor:
        return ReferenceSegmenterSemanticExtractor()

    def get_tei_training_data_generator(self) -> ReferenceSegmenterTeiTrainingDataGenerator:
        return ReferenceSegmenterTeiTrainingDataGenerator()

    def get_training_tei_parser(self) -> ReferenceSegmenterTrainingTeiParser:
        return ReferenceSegmenterTrainingTeiParser()
