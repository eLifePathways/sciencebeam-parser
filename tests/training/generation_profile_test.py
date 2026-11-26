import pytest

from sciencebeam_parser.config.config import UnknownProfileError
from sciencebeam_parser.training.generation_profile import (
    DEFAULT_GENERATION_PROFILE,
    get_generation_config,
    get_generation_model_config
)


class TestGetGenerationConfig:
    def test_should_generate_under_grobid_crf_when_none_is_named(self):
        """Not the profile the service serves, which moves for reasons of its own."""
        config = get_generation_config(None)
        assert config.get_active_profile_name() == DEFAULT_GENERATION_PROFILE
        assert (
            config.resolve_profile(None)['models']
            == config.resolve_profile('grobid_crf_0_9_0')['models']
        )

    def test_should_select_the_profile_the_parser_will_resolve(self):
        # Not a merged overlay: ScienceBeamParser resolves the default profile
        # against whatever configuration it is handed, so a `models:` written
        # here would be resolved over and never reach a generator.
        config = get_generation_config('biorxiv_elife')
        assert config.props['profile'] == 'biorxiv_elife'
        assert config.get_active_profile_name() == 'biorxiv_elife'

    def test_should_resolve_to_the_feature_configuration_the_generator_uses(self):
        assert (
            get_generation_config(None)
            .resolve_profile().props['models']['segmentation']['use_first_token_of_block']
        ) is True
        assert (
            get_generation_config('biorxiv_elife')
            .resolve_profile().props['models']['segmentation']['use_first_token_of_block']
        ) is False

    def test_should_accept_a_profile_alias(self):
        assert get_generation_config('grobid_crf').props['profile'] == 'grobid_crf'

    def test_should_fail_on_an_unknown_profile(self):
        with pytest.raises(UnknownProfileError):
            get_generation_config('not-a-profile')

    def test_should_find_a_model_config_by_its_file_name(self):
        config = get_generation_config(None)
        assert get_generation_model_config(config, 'reference-segmenter') == (
            config.resolve_profile().props['models']['reference_segmenter']
        )
