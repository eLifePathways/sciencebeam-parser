import pytest

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.document.layout_noise_filter import (
    NOISE_ACTION_RELABEL,
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


class TestNoiseFilterConfig:
    def test_should_default_to_the_dataclass_defaults(self):
        assert FullTextProcessorConfig().noise_filter == LayoutNoiseFilterConfig()

    def test_should_override_only_the_given_settings(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'noise_filter': {
                'enabled': True,
                'outside_main_area': True,
                'action': NOISE_ACTION_RELABEL,
            }}}
        }))
        assert config.noise_filter == LayoutNoiseFilterConfig(
            enabled=True, outside_main_area=True, action=NOISE_ACTION_RELABEL
        )

    def test_should_keep_the_other_fulltext_settings(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {
                'merge_raw_authors': True,
                'noise_filter': {'enabled': True},
            }}
        }))
        assert config.merge_raw_authors is True
        assert config.noise_filter.enabled is True

    def test_should_leave_the_default_when_no_noise_filter_is_given(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'merge_raw_authors': True}}
        }))
        assert config.noise_filter == LayoutNoiseFilterConfig()
