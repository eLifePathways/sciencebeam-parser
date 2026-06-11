from typing import List

from sciencebeam_parser.models.data import (
    ContextAwareLayoutTokenFeatures,
    ContextAwareLayoutTokenModelDataGenerator,
    DocumentFeaturesContext,
    FeatureDef,
    get_punctuation_type_feature
)


class ReferenceSegmenterDataGenerator(ContextAwareLayoutTokenModelDataGenerator):
    def __init__(self, document_features_context: DocumentFeaturesContext):
        super().__init__(document_features_context, compare_indentation_across_blocks=True)
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
                       lambda f: f.get_line_status_with_linestart_for_single_token()),
            FeatureDef('alignment', lambda f: f.get_alignment_status()),
            FeatureDef('capitalisation',
                       lambda f: f.get_capitalisation_status_using_allcap()),
            FeatureDef('digit_status',
                       lambda f: f.get_digit_status_using_containsdigits()),
            FeatureDef('is_single_char', lambda f: f.get_str_is_single_char()),
            FeatureDef('is_proper_name', lambda f: f.get_str_is_proper_name()),
            FeatureDef('is_common_name', lambda f: f.get_str_is_common_name()),
            FeatureDef('is_first_name',
                       lambda f: f.get_str_is_first_name_for_uppercase_lookup()),
            FeatureDef('is_location_name',
                       lambda f: f.get_dummy_str_is_location_name()),
            FeatureDef('is_year', lambda f: f.get_str_is_year()),
            FeatureDef('is_month', lambda f: f.get_str_is_month()),
            FeatureDef('is_http', lambda f: f.get_str_is_http()),
            FeatureDef('punctuation_profile',
                       lambda f: get_punctuation_type_feature(f.token_text)),
            FeatureDef('line_token_relative_position',
                       lambda f: f.get_str_grobid_line_token_relative_position()),
            FeatureDef('line_relative_length',
                       lambda f: f.get_str_grobid_line_relative_length()),
            FeatureDef('block_status',
                       lambda f: f.get_block_status_with_blockstart_for_single_token()),
            FeatureDef('truncated_punctuation_profile_length',
                       lambda f: f.get_truncated_line_punctuation_profile_length_feature()),
            FeatureDef('dummy_label', lambda f: f.get_dummy_label()),
        ]
