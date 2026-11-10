from typing import Iterator, List

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine
)
from sciencebeam_parser.models.data import (
    ContextAwareLayoutTokenFeatures,
    ContextAwareLayoutTokenModelDataGenerator,
    DocumentFeaturesContext,
    FeatureDef,
    LayoutModelData
)


def get_layout_document_as_single_line(layout_document: LayoutDocument) -> LayoutDocument:
    """The tokens of the document on one line, so that line features see no line break.

    The tokens themselves are kept, with their own line, so what is extracted from the
    labels still follows the lines of the PDF.
    """
    tokens = list(layout_document.iter_all_tokens())
    if not tokens:
        return layout_document
    return LayoutDocument.for_blocks([
        LayoutBlock(lines=[LayoutLine(tokens=tokens)])
    ])


class NameDataGenerator(ContextAwareLayoutTokenModelDataGenerator):
    def __init__(
        self,
        document_features_context: DocumentFeaturesContext,
        ignore_line_breaks: bool = False
    ):
        super().__init__(document_features_context)
        # GROBID's header name model since grobidOrg/grobid#1641 is trained, and run, with
        # the author list on one line (`AuthorParser.withoutLineBreaks`): its training data
        # never has a line break inside a sequence, so a LINEEND within the list reads to it
        # as the end of the list and the author after the break is dropped.
        self.ignore_line_breaks = ignore_line_breaks
        self._feature_defs: List[FeatureDef[ContextAwareLayoutTokenFeatures]] = [
            FeatureDef('token_text', lambda f: f.token_text),
            FeatureDef('lower_token_text', lambda f: f.get_lower_token_text()),
            FeatureDef('prefix_1', lambda f: f.get_prefix(1)),
            FeatureDef('prefix_2', lambda f: f.get_prefix(2)),
            FeatureDef('prefix_3', lambda f: f.get_prefix(3)),
            FeatureDef('prefix_4', lambda f: f.get_prefix(4)),
            FeatureDef('suffix_1', lambda f: f.get_suffix(1)),
            FeatureDef('suffix_2', lambda f: f.get_suffix(2)),
            FeatureDef('suffix_3', lambda f: f.get_suffix(3)),
            FeatureDef('suffix_4', lambda f: f.get_suffix(4)),
            FeatureDef('line_status',
                       lambda f: f.get_line_status_with_lineend_for_single_token()),
            FeatureDef('capitalisation',
                       lambda f: f.get_capitalisation_status_using_allcap()),
            FeatureDef('digit_status',
                       lambda f: f.get_digit_status_using_containsdigits()),
            FeatureDef('is_single_char', lambda f: f.get_str_is_single_char()),
            FeatureDef('is_common_name', lambda f: f.get_dummy_str_is_common_name()),
            FeatureDef('is_first_name', lambda f: f.get_str_is_first_name()),
            FeatureDef('is_last_name', lambda f: f.get_str_is_last_name()),
            FeatureDef('is_known_title', lambda f: f.get_dummy_str_is_known_title()),
            FeatureDef('is_known_suffix',
                       lambda f: f.get_dummy_str_is_known_suffix()),
            FeatureDef('punctuation_type', lambda f: f.get_punctuation_type_feature()),
            FeatureDef('dummy_label', lambda f: f.get_dummy_label()),
        ]

    def iter_model_data_for_layout_document(
        self,
        layout_document: LayoutDocument
    ) -> Iterator[LayoutModelData]:
        if self.ignore_line_breaks:
            layout_document = get_layout_document_as_single_line(layout_document)
        return super().iter_model_data_for_layout_document(layout_document)
