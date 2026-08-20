import pytest

from benchmarks.run import check_llm_profile_corpora


CONFIG = {
    'sequence_model_profiles': {
        'grobid_crf_0_9_0': {
            'citation': {'engine': 'wapiti', 'path': 'x'},
        },
        'llm_references': {
            'citation': {'engine': 'llm', 'task': 'citation'},
        },
    },
}


class TestCheckLlmProfileCorpora:
    def test_should_allow_a_crf_profile_with_the_private_corpus(self):
        check_llm_profile_corpora(CONFIG, 'grobid_crf_0_9_0', ['plos-manuscripts'])

    def test_should_allow_an_llm_profile_without_the_private_corpus(self):
        check_llm_profile_corpora(CONFIG, 'llm_references', ['biorxiv'])

    def test_should_allow_an_llm_profile_with_no_opt_in_corpora(self):
        check_llm_profile_corpora(CONFIG, 'llm_references', None)

    def test_should_refuse_an_llm_profile_with_the_private_corpus(self):
        with pytest.raises(SystemExit, match='not redistributable'):
            check_llm_profile_corpora(CONFIG, 'llm_references', ['plos-manuscripts'])

    def test_should_name_the_profile_and_corpus_when_refusing(self):
        with pytest.raises(SystemExit) as excinfo:
            check_llm_profile_corpora(
                CONFIG, 'llm_references', ['biorxiv', 'plos-manuscripts']
            )
        assert 'llm_references' in str(excinfo.value)
        assert 'plos-manuscripts' in str(excinfo.value)

    def test_should_ignore_an_unknown_profile(self):
        check_llm_profile_corpora(CONFIG, 'nonexistent', ['plos-manuscripts'])

    def test_should_ignore_no_profile(self):
        check_llm_profile_corpora(CONFIG, None, ['plos-manuscripts'])
