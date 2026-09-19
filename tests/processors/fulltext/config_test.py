import pytest

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.document.layout_noise_filter import (
    NOISE_ACTION_DROP,
    LayoutNoiseFilterConfig,
)
from sciencebeam_parser.processors.fulltext.config import (
    FullTextProcessorConfig,
    RequestFieldNames
)


EXTRACT_ALL_FULLTEXT_CONFIG = FullTextProcessorConfig(
    extract_body_sections=True,
    extract_acknowledgements=True,
    extract_back_sections=True,
    extract_references=True,
    extract_citation_fields=True,
    extract_citation_authors=True,
    extract_citation_editors=False,
    extract_figure_fields=True,
    extract_table_fields=True
)


class TestFullTextProcessorConfig:
    @pytest.mark.parametrize("field_name,value", [
        ("merge_raw_authors", False),
        ("merge_raw_authors", True)
    ])
    def test_should_override_default_from_app_config(self, field_name: str, value: bool):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {
                'fulltext': {
                    field_name: value
                }
            }
        }))
        assert getattr(config, field_name) is value

    def test_should_ignore_empty_requested_field_names(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_requested_field_names(set())
        assert config == EXTRACT_ALL_FULLTEXT_CONFIG

    def test_should_configure_only_header_extraction(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_requested_field_names({
            RequestFieldNames.TITLE,
            RequestFieldNames.ABSTRACT,
            RequestFieldNames.AUTHORS,
            RequestFieldNames.AFFILIATIONS
        })
        assert config.extract_front
        assert config.extract_authors
        assert config.extract_affiliations
        assert not config.extract_body_sections
        assert not config.extract_acknowledgements
        assert not config.extract_back_sections
        assert not config.extract_references

    def test_should_configure_extract_references(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_requested_field_names({
            RequestFieldNames.REFERENCES
        })
        assert not config.extract_front
        assert not config.extract_authors
        assert not config.extract_affiliations
        assert not config.extract_body_sections
        assert not config.extract_acknowledgements
        assert not config.extract_back_sections
        assert config.extract_references

    def test_should_treat_unknown_field_names_as_no_change(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_requested_field_names({
            'other'
        })
        assert config == EXTRACT_ALL_FULLTEXT_CONFIG

    def test_should_configure_only_header_extraction_using_get_for_header_document(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_header_document()
        assert config.extract_front
        assert config.extract_authors
        assert config.extract_affiliations
        assert not config.extract_body_sections
        assert not config.extract_acknowledgements
        assert not config.extract_back_sections
        assert not config.extract_references
        assert not config.extract_graphic_bounding_boxes


class TestGetLayoutNoiseFilterConfig:
    def test_should_carry_every_setting_across(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {
                'noise_filter_enabled': True,
                'noise_filter_repetition_fraction': 0.3,
                'noise_filter_position_consistency_fraction': 0.7,
                'noise_filter_max_position_stddev': 0.02,
                'noise_filter_max_height_ratio': 3.0,
                'noise_filter_preserve_first_page_head': True,
                'noise_filter_preserve_first_page_foot': True,
                'noise_filter_outside_main_area': True,
                'noise_filter_min_repeating_pattern_length': 5,
                'noise_filter_max_letterless_length': 8,
                'noise_filter_action': 'relabel',
            }}
        })).get_layout_noise_filter_config()
        assert config == LayoutNoiseFilterConfig(
            enabled=True,
            repetition_fraction=0.3,
            position_consistency_fraction=0.7,
            max_position_stddev=0.02,
            max_height_ratio=3.0,
            preserve_first_page_head=True,
            preserve_first_page_foot=True,
            filter_outside_main_area=True,
            min_repeating_pattern_length=5,
            max_letterless_length=8,
            action='relabel'
        )

    def test_should_default_to_dropping(self):
        config = FullTextProcessorConfig().get_layout_noise_filter_config()
        assert config.action == NOISE_ACTION_DROP
