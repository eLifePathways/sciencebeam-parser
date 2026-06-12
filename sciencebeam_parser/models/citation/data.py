from typing import List

from sciencebeam_parser.models.data import (
    ContextAwareLayoutTokenFeatures,
    ContextAwareLayoutTokenModelDataGenerator,
    DocumentFeaturesContext,
    FeatureDef
)


class CitationDataGenerator(ContextAwareLayoutTokenModelDataGenerator):
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
            FeatureDef('is_proper_name', lambda f: f.get_str_is_proper_name()),
            FeatureDef('is_common_name', lambda f: f.get_str_is_common_name()),
            FeatureDef('is_first_name', lambda f: f.get_str_is_first_name()),
            FeatureDef('is_last_name', lambda f: f.get_str_is_last_name()),
            FeatureDef('is_location_name', lambda f: f.get_str_is_location_name()),
            FeatureDef('is_year', lambda f: f.get_str_is_year()),
            FeatureDef('is_month', lambda f: f.get_str_is_month()),
            FeatureDef('is_http', lambda f: f.get_str_is_http()),
            FeatureDef('is_known_collaboration',
                       lambda f: f.get_dummy_str_is_known_collaboration()),
            FeatureDef('is_known_journal_title',
                       lambda f: f.get_dummy_str_is_known_journal_title()),
            FeatureDef('is_known_conference_title',
                       lambda f: f.get_dummy_str_is_known_conference_title()),
            FeatureDef('is_known_publisher',
                       lambda f: f.get_dummy_str_is_known_publisher()),
            FeatureDef('is_known_identifier',
                       lambda f: f.get_dummy_str_is_known_identifier()),
            FeatureDef('punctuation_type', lambda f: f.get_punctuation_type_feature()),
            FeatureDef('sentence_token_relative_position',
                       lambda f: f.get_str_sentence_token_relative_position()),
            FeatureDef('dummy_label', lambda f: f.get_dummy_label()),
        ]
