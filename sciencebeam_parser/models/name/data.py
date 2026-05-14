from typing import List

from sciencebeam_parser.models.data import (
    ContextAwareLayoutTokenFeatures,
    ContextAwareLayoutTokenModelDataGenerator,
    DocumentFeaturesContext,
    FeatureDef
)


class NameDataGenerator(ContextAwareLayoutTokenModelDataGenerator):
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
