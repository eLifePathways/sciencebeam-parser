import pytest

from lxml import etree

from benchmarks.predict_llm import (
    check_restricted_corpora,
    merge_section_documents,
)


def _article(front='', body='', back='', attrs=''):
    return (
        f'<article xmlns:xlink="http://www.w3.org/1999/xlink"{attrs}>'
        f'<front>{front}</front><body>{body}</body><back>{back}</back>'
        f'</article>'
    )


class TestMergeSectionDocuments:
    def test_should_take_each_section_from_its_own_response(self):
        merged = etree.fromstring(merge_section_documents({
            'front': _article(front='<article-title>T</article-title>'),
            'body': _article(body='<sec><title>S</title></sec>'),
            'back': _article(back='<ref-list><ref/></ref-list>'),
        }))
        assert merged.find('front/article-title').text == 'T'
        assert merged.find('body/sec/title').text == 'S'
        assert merged.find('back/ref-list') is not None

    def test_should_not_keep_the_empty_sections_of_the_skeleton(self):
        # Appending rather than replacing would leave two of each, and score
        # the empty one.
        merged = etree.fromstring(merge_section_documents({
            'front': _article(front='<article-title>T</article-title>'),
            'body': _article(body='<sec><title>S</title></sec>'),
            'back': _article(back='<ref-list><ref/></ref-list>'),
        }))
        assert len(merged.findall('front')) == 1
        assert len(merged.findall('body')) == 1
        assert len(merged.findall('back')) == 1

    def test_should_keep_the_article_attributes_of_the_front_response(self):
        merged = etree.fromstring(merge_section_documents({
            'front': _article(attrs=' article-type="research-article"'),
            'body': _article(),
        }))
        assert merged.get('article-type') == 'research-article'

    def test_should_order_sections_as_jats_requires(self):
        merged = etree.fromstring(merge_section_documents({
            'back': _article(back='<ref-list/>'),
            'body': _article(body='<sec/>'),
        }))
        assert [child.tag for child in merged] == ['front', 'body', 'back']

    def test_should_merge_what_is_present_when_a_section_is_absent(self):
        merged = etree.fromstring(merge_section_documents({
            'body': _article(body='<sec><title>S</title></sec>'),
        }))
        assert merged.find('body/sec/title').text == 'S'

    def test_should_reject_an_empty_set_of_responses(self):
        with pytest.raises(ValueError):
            merge_section_documents({})

    def test_should_produce_an_article_rooted_document(self):
        # The scorer picks its mapping from the root element.
        merged = etree.fromstring(merge_section_documents({'front': _article()}))
        assert merged.tag == 'article'


class TestCheckRestrictedCorpora:
    def test_should_allow_the_public_corpora(self):
        check_restricted_corpora(['biorxiv', 'pkp'])

    def test_should_allow_no_opt_in_corpora(self):
        check_restricted_corpora(None)

    def test_should_refuse_the_private_manuscripts(self):
        with pytest.raises(SystemExit, match='not redistributable'):
            check_restricted_corpora(['biorxiv', 'plos-manuscripts'])
