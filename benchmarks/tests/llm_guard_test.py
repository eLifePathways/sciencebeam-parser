import pytest

from benchmarks.run import check_llm_profile_corpora
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE


# Shaped like the parser's own config: a profile names a sequence-model profile,
# which may extend others, and an alias names a profile.
CONFIG = AppConfig({
    'sequence_model_profiles': {
        'grobid_crf_0_9_0': {
            'citation': {'engine': 'wapiti', 'path': 'x'},
            'header': {'engine': 'wapiti', 'path': 'y'},
        },
        'llm_citation': {
            'extends': 'grobid_crf_0_9_0',
            'citation': {'engine': 'llm', 'task': 'citation'},
        },
        'llm_references': {
            'extends': ['llm_citation'],
        },
    },
    'profiles': {
        'grobid_crf_0_9_0': {'sequence_models': 'grobid_crf_0_9_0'},
        'llm_refs': {'sequence_models': 'llm_references'},
    },
    'profile_aliases': {'grobid_crf': 'grobid_crf_0_9_0', 'refs': 'llm_refs'},
    'profile': 'grobid_crf',
})


class TestCheckLlmProfileCorpora:
    def test_should_allow_a_crf_profile_with_the_private_corpus(self):
        check_llm_profile_corpora(CONFIG, 'grobid_crf_0_9_0', ['plos-manuscripts'])

    def test_should_allow_an_llm_profile_without_the_private_corpus(self):
        check_llm_profile_corpora(CONFIG, 'llm_refs', ['biorxiv'])

    def test_should_allow_an_llm_profile_with_no_opt_in_corpora(self):
        check_llm_profile_corpora(CONFIG, 'llm_refs', None)

    def test_should_refuse_an_llm_profile_with_the_private_corpus(self):
        with pytest.raises(SystemExit, match='not redistributable'):
            check_llm_profile_corpora(CONFIG, 'llm_refs', ['plos-manuscripts'])

    def test_should_name_the_profile_model_and_corpus_when_refusing(self):
        with pytest.raises(SystemExit) as excinfo:
            check_llm_profile_corpora(
                CONFIG, 'llm_refs', ['biorxiv', 'plos-manuscripts']
            )
        message = str(excinfo.value)
        assert 'llm_refs' in message
        assert 'citation' in message
        assert 'plos-manuscripts' in message

    def test_should_see_an_llm_model_through_extends(self):
        """The llm model is two `extends` away from the profile's own entry."""
        with pytest.raises(SystemExit):
            check_llm_profile_corpora(CONFIG, 'llm_refs', ['plos-manuscripts'])

    def test_should_resolve_an_alias(self):
        with pytest.raises(SystemExit):
            check_llm_profile_corpora(CONFIG, 'refs', ['plos-manuscripts'])

    def test_should_check_the_default_profile_when_none_is_named(self):
        check_llm_profile_corpora(CONFIG, None, ['plos-manuscripts'])
        llm_default = AppConfig({**CONFIG.props, 'profile': 'refs'})
        with pytest.raises(SystemExit):
            check_llm_profile_corpora(llm_default, None, ['plos-manuscripts'])

    def test_should_refuse_a_profile_it_cannot_resolve(self):
        """Fails closed: an unknown profile could be anything."""
        with pytest.raises(SystemExit, match='does not resolve'):
            check_llm_profile_corpora(CONFIG, 'nonexistent', ['plos-manuscripts'])

    def test_should_not_resolve_anything_without_a_restricted_corpus(self):
        check_llm_profile_corpora(CONFIG, 'nonexistent', ['biorxiv'])


class TestCheckLlmProfileCorporaAgainstTheShippedConfig:
    """The guard read `eval.yml`, which declares no models, and so never refused
    anything: checked here against the config the parser image is built with."""

    @pytest.fixture(scope='class')
    def shipped_config(self) -> AppConfig:
        return AppConfig.load_yaml(DEFAULT_CONFIG_FILE)

    def test_should_refuse_every_shipped_llm_profile(self, shipped_config: AppConfig):
        llm_profiles = [
            name for name in shipped_config.get_profile_names() if name.startswith('llm_')
        ]
        assert llm_profiles, 'expected the shipped config to declare llm profiles'
        for profile in llm_profiles:
            with pytest.raises(SystemExit):
                check_llm_profile_corpora(shipped_config, profile, ['plos-manuscripts'])

    def test_should_allow_the_shipped_default(self, shipped_config: AppConfig):
        check_llm_profile_corpora(shipped_config, None, ['plos-manuscripts'])
        check_llm_profile_corpora(shipped_config, 'grobid_crf', ['plos-manuscripts'])
