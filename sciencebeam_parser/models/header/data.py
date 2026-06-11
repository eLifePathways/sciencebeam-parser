from typing import List

from sciencebeam_parser.models.data import (
    ContextAwareLayoutTokenFeatures,
    ContextAwareLayoutTokenModelDataGenerator,
    DocumentFeaturesContext,
    FeatureDef
)


class HeaderDataGenerator(ContextAwareLayoutTokenModelDataGenerator):
    def __init__(
        self,
        document_features_context: DocumentFeaturesContext,
        persist_indentation_reference_across_blocks: bool = False
    ):
        super().__init__(
            document_features_context,
            persist_indentation_reference_across_blocks=persist_indentation_reference_across_blocks
        )
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
                       lambda f: f.get_block_status_with_blockend_for_single_token()),
            FeatureDef('line_status',
                       lambda f: f.get_line_status_with_lineend_for_single_token()),
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
            FeatureDef('is_proper_name', lambda f: f.get_str_is_proper_name()),
            FeatureDef('is_common_name', lambda f: f.get_str_is_common_name()),
            FeatureDef('is_year', lambda f: f.get_str_is_year()),
            FeatureDef('is_month', lambda f: f.get_str_is_month()),
            FeatureDef('is_location_name',
                       lambda f: f.get_dummy_str_is_location_name()),
            FeatureDef('is_email', lambda f: f.get_dummy_str_is_email()),
            FeatureDef('is_http', lambda f: f.get_str_is_http()),
            FeatureDef('punctuation_type', lambda f: f.get_punctuation_type_feature()),
            FeatureDef('is_largest_font', lambda f: f.get_str_is_largest_font_size()),
            # bug in GROBID #795
            FeatureDef('is_smallest_font',
                       lambda f: f.get_dummy_str_is_smallest_font_size()),
            # due to bug, usually larger than mean
            FeatureDef('is_larger_than_average_font',
                       lambda f: f.get_dummy_str_is_larger_than_average_font_size('1')),
            FeatureDef('dummy_label', lambda f: f.get_dummy_label()),
        ]
