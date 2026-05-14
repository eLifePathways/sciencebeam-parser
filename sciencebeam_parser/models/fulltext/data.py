from typing import List

from sciencebeam_parser.models.data import (
    ContextAwareLayoutTokenFeatures,
    ContextAwareLayoutTokenModelDataGenerator,
    DocumentFeaturesContext,
    FeatureDef
)


class FullTextDataGenerator(ContextAwareLayoutTokenModelDataGenerator):
    def __init__(self, document_features_context: DocumentFeaturesContext):
        super().__init__(document_features_context)
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
            FeatureDef('block_status',
                       lambda f: f.get_block_status_with_blockstart_for_single_token()),
            FeatureDef('line_status',
                       lambda f: f.get_line_status_with_linestart_for_single_token()),
            FeatureDef('alignment', lambda f: f.get_alignment_status()),
            FeatureDef('token_font_status', lambda f: f.get_token_font_status()),
            FeatureDef('token_font_size', lambda f: f.get_token_font_size_feature()),
            FeatureDef('is_bold', lambda f: f.get_str_is_bold()),
            FeatureDef('is_italic', lambda f: f.get_str_is_italic()),
            FeatureDef('capitalisation',
                       lambda f: f.get_capitalisation_status_using_allcap()),
            FeatureDef('digit_status',
                       lambda f: f.get_digit_status_using_containsdigits()),
            FeatureDef('is_single_char', lambda f: f.get_str_is_single_char()),
            FeatureDef('punctuation_type', lambda f: f.get_punctuation_type_feature()),
            FeatureDef('relative_document_position',
                       lambda f: f.get_dummy_str_relative_document_position()),
            FeatureDef('relative_page_position',
                       lambda f: f.get_dummy_str_relative_page_position()),
            FeatureDef('is_bitmap_around',
                       lambda f: f.get_dummy_str_is_bitmap_around()),
            FeatureDef('callout_type', lambda f: f.get_dummy_callout_type()),
            FeatureDef('is_callout_known',
                       lambda f: f.get_dummy_str_is_callout_known()),
            FeatureDef('is_superscript', lambda f: f.get_str_is_superscript()),
        ]
