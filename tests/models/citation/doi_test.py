import pytest

from sciencebeam_parser.models.citation.doi import find_printed_doi


class TestFindPrintedDoi:
    def test_should_find_a_doi_after_its_label(self):
        assert find_printed_doi('doi:10.1234/fern') == '10.1234/fern'

    def test_should_find_a_doi_in_a_doi_org_link(self):
        assert find_printed_doi('https://doi.org/10.1234/fern') == '10.1234/fern'

    def test_should_ignore_text_without_a_doi(self):
        assert find_printed_doi('https://ferns.test/map') is None

    def test_should_keep_the_first_of_two_dois(self):
        assert find_printed_doi('doi:10.1234/fern doi:10.5678/moss') == '10.1234/fern'

    def test_should_join_a_doi_broken_across_lines(self):
        assert find_printed_doi('10.1234/\nfern.2001') == '10.1234/fern.2001'

    def test_should_ignore_a_doi_prefix_without_a_suffix(self):
        assert find_printed_doi('doi: 10.1234/.') is None

    def test_should_find_an_arxiv_doi(self):
        assert find_printed_doi('10.48550/arXiv:2101.00001') == '10.48550/arXiv:2101.00001'

    def test_should_find_a_doi_after_a_label_without_a_colon(self):
        assert find_printed_doi('DOI 10.1234/fern') == '10.1234/fern'

    def test_should_find_a_doi_after_a_full_stop(self):
        assert find_printed_doi('2019. 10.1038/fern') == '10.1038/fern'


class TestFindPrintedDoiEnd:
    def test_should_keep_a_suffix_that_contains_a_doi_prefix(self):
        assert find_printed_doi('10.5281/zenodo.10.1234/fern') == '10.5281/zenodo.10.1234/fern'

    def test_should_not_take_a_doi_prefix_inside_a_word(self):
        assert find_printed_doi('https://ferns.test/v10.1234/fern') is None

    def test_should_end_the_doi_before_an_https_link(self):
        assert find_printed_doi('10.1234/fern https://ferns.test/map') == '10.1234/fern'

    def test_should_end_the_doi_before_an_http_link(self):
        assert find_printed_doi('10.1234/fern http://ferns.test/map') == '10.1234/fern'

    def test_should_drop_a_full_stop_before_a_link(self):
        assert find_printed_doi('10.1234/fern. http://ferns.test/map') == '10.1234/fern'

    def test_should_end_the_doi_before_a_pmid(self):
        assert find_printed_doi('doi: 10.1234/fern PMID: 1234567') == '10.1234/fern'

    def test_should_not_take_a_following_link_as_the_suffix(self):
        assert find_printed_doi('doi: 10.1234/ https://ferns.test/map') is None

    def test_should_end_the_doi_before_an_arxiv_id(self):
        assert find_printed_doi('10.1234/fern arXiv:2101.00001') == '10.1234/fern'

    def test_should_end_the_doi_before_a_pmid_without_a_colon(self):
        assert find_printed_doi('10.1234/fern. PMID 1234567') == '10.1234/fern'

    def test_should_keep_a_year_with_a_dot_inside_the_doi(self):
        assert find_printed_doi('10.1016/j.fern.2010.01.001') == '10.1016/j.fern.2010.01.001'

    def test_should_end_the_doi_before_a_pmcid(self):
        assert find_printed_doi('doi: 10.1234/fern PMCID: PMC1234567') == '10.1234/fern'

    def test_should_end_the_doi_before_a_pmcid_without_a_label(self):
        assert find_printed_doi('doi: 10.1234/fern PMC1234567') == '10.1234/fern'

    def test_should_keep_a_suffix_that_starts_with_pubmed(self):
        assert find_printed_doi('doi:10.1093/pubmed/fern001') == '10.1093/pubmed/fern001'

    def test_should_end_the_doi_before_a_pubmed_pmid(self):
        assert find_printed_doi('doi: 10.1234/fern. PubMed PMID: 1234567') == '10.1234/fern'

    def test_should_end_the_doi_before_a_www_link(self):
        assert find_printed_doi('10.1234/fern www.ferns.test') == '10.1234/fern'

    @pytest.mark.parametrize('text', [
        '10.1234/fern (PMID: 1234567)',
        '10.1234/fern [PubMed]',
        '10.1234/fern, [www.ferns.test]',
        '10.1234/fern <https://ferns.test/map>',
        '10.1234/fern “https://ferns.test/map”',
        '10.1234/fern – https://ferns.test/map',
        '10.1234/fern — PubMed'
    ])
    def test_should_drop_what_opens_the_text_after_the_doi(self, text: str):
        assert find_printed_doi(text) == '10.1234/fern'


class TestFindPrintedDoiTrailingCharacters:
    def test_should_drop_an_unmatched_closing_bracket(self):
        assert find_printed_doi('(doi: 10.1234/fern).') == '10.1234/fern'

    def test_should_drop_angle_brackets_around_the_doi(self):
        assert find_printed_doi('<10.1234/fern>') == '10.1234/fern'

    def test_should_drop_a_closing_quote(self):
        assert find_printed_doi('doi:10.1234/fern"') == '10.1234/fern'

    def test_should_drop_a_trailing_colon(self):
        assert find_printed_doi('doi: 10.1234/fern:') == '10.1234/fern'

    def test_should_drop_trailing_commas_semicolons_and_guillemets(self):
        assert find_printed_doi('«10.1234/fern», ;') == '10.1234/fern'

    def test_should_keep_a_closing_bracket_that_belongs_to_the_doi(self):
        doi = '10.1002/(SICI)1234(199901)12:3<45::AID-FERN6>3.0.CO;2-(Y)'
        assert find_printed_doi(doi) == doi

    def test_should_keep_a_trailing_hyphen(self):
        doi = '10.1002/(SICI)1234(199901)12:3<45::AID-FERN6>3.0.CO;2-'
        assert find_printed_doi(doi) == doi

    def test_should_drop_a_trailing_slash(self):
        assert find_printed_doi('doi: 10.1234/fern/') == '10.1234/fern'
