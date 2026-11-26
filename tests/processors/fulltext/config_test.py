import pytest
import yaml

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.document.layout_noise_filter import (
    NOISE_ACTION_RELABEL,
    LayoutNoiseFilterConfig,
)
from sciencebeam_parser.processors.fulltext.config import (
    ExtractConfig,
    FrontConfig,
    FullTextProcessorConfig,
    RequestFieldNames
)


EXTRACT_ALL_FULLTEXT_CONFIG = FullTextProcessorConfig(
    extract=ExtractConfig(
        body_sections=True,
        acknowledgements=True,
        back_sections=True,
        references=True,
        reference_fields=True,
        reference_authors=True,
        reference_editors=False,
        figure_fields=True,
        table_fields=True
    )
)


class TestFullTextProcessorConfig:
    @pytest.mark.parametrize("field_name,value", [
        ("use_cv_model", False),
        ("use_cv_model", True)
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
        assert config.extract.front
        assert config.extract.front_authors
        assert config.extract.front_affiliations
        assert not config.extract.body_sections
        assert not config.extract.acknowledgements
        assert not config.extract.back_sections
        assert not config.extract.references

    def test_should_configure_extract_references(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_requested_field_names({
            RequestFieldNames.REFERENCES
        })
        assert not config.extract.front
        assert not config.extract.front_authors
        assert not config.extract.front_affiliations
        assert not config.extract.body_sections
        assert not config.extract.acknowledgements
        assert not config.extract.back_sections
        assert config.extract.references

    def test_should_treat_unknown_field_names_as_no_change(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_requested_field_names({
            'other'
        })
        assert config == EXTRACT_ALL_FULLTEXT_CONFIG

    def test_should_configure_only_header_extraction_using_get_for_header_document(self):
        config = EXTRACT_ALL_FULLTEXT_CONFIG.get_for_header_document()
        assert config.extract.front
        assert config.extract.front_authors
        assert config.extract.front_affiliations
        assert not config.extract.body_sections
        assert not config.extract.acknowledgements
        assert not config.extract.back_sections
        assert not config.extract.references
        assert not config.extract.graphics

    def test_should_read_a_bare_yaml_off_as_dehyphenate_off(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'front': yaml.safe_load('dehyphenate: off')}}
        }))
        assert config.front.dehyphenate == 'off'

    def test_should_say_to_quote_a_bare_yaml_word_read_as_true(self):
        with pytest.raises(ValueError, match='quote it'):
            FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
                'processors': {'fulltext': {'front': yaml.safe_load('dehyphenate: on')}}
            }))

    @pytest.mark.parametrize('scope,name', [
        ('front', 'select_abstracts'),
        ('front', 'select_titles'),
        ('front', 'dehyphenate'),
        ('references', 'select_titles'),
        ('references', 'select_dois')
    ])
    def test_should_refuse_an_unknown_choice(self, scope: str, name: str):
        with pytest.raises(ValueError, match=f'{scope}.{name}.*fern'):
            FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
                'processors': {'fulltext': {scope: {name: 'fern'}}}
            }))


class TestExtractConfig:
    def test_should_read_what_is_extracted(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'extract': {'reference_editors': True}}}
        }))
        assert config.extract == ExtractConfig(reference_editors=True)

    def test_should_refuse_a_flag_outside_the_extract_scope(self):
        with pytest.raises(ValueError, match='extract_citation_editors'):
            FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
                'processors': {'fulltext': {'extract_citation_editors': True}}
            }))


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
                'use_cv_model': True,
                'noise_filter': {'enabled': True},
            }}
        }))
        assert config.use_cv_model is True
        assert config.noise_filter.enabled is True

    def test_should_leave_the_default_when_no_noise_filter_is_given(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'use_cv_model': True}}
        }))
        assert config.noise_filter == LayoutNoiseFilterConfig()


class TestFrontConfig:
    def test_should_read_how_the_front_is_written(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'front': {'select_abstracts': 'variants'}}}
        }))
        assert config.front == FrontConfig(select_abstracts='variants')

    def test_should_refuse_a_setting_outside_its_scope(self):
        with pytest.raises(ValueError, match='abstracts_mode'):
            FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
                'processors': {'fulltext': {'abstracts_mode': 'variants'}}
            }))


class TestReferencesConfig:
    def test_should_read_the_fields_that_keep_a_reference(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'references': {'kept_by_any_of': ['title', 'doi']}}}
        }))
        assert config.references.kept_by_any_of == ('title', 'doi')

    def test_should_refuse_an_unknown_field_to_keep_a_reference(self):
        with pytest.raises(ValueError, match='fern'):
            FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
                'processors': {'fulltext': {'references': {'kept_by_any_of': ['title', 'fern']}}}
            }))

    def test_should_read_where_a_link_doi_of_a_fragment_goes(self):
        config = FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
            'processors': {'fulltext': {'references': {'fragment_link_doi': 'own_reference'}}}
        }))
        assert config.references.fragment_link_doi == 'own_reference'

    def test_should_refuse_an_unknown_place_for_a_link_doi_of_a_fragment(self):
        with pytest.raises(ValueError, match='fern'):
            FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
                'processors': {'fulltext': {'references': {'fragment_link_doi': 'fern'}}}
            }))

    def test_should_refuse_own_reference_where_a_doi_keeps_no_reference(self):
        with pytest.raises(ValueError, match='own_reference'):
            FullTextProcessorConfig.from_app_config(app_config=AppConfig(props={
                'processors': {'fulltext': {'references': {
                    'fragment_link_doi': 'own_reference',
                    'kept_by_any_of': ['title']
                }}}
            }))
