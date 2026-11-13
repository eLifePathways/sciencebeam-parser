from __future__ import annotations

from typing import List

from lxml import etree

from benchmarks.matched_property_scoring import PROPERTY_SEPARATOR
from benchmarks.variant_xpath import (
    abstract_variants,
    abstracts_offered,
    tei_abstract_variants,
    fn_jats_abstract_language_pairs,
    title_language_pairs
)


def _article_meta(inner: str) -> etree.ElementBase:
    return etree.fromstring(f"<article-meta>{inner}</article-meta>")


def _texts(nodes: List[etree.ElementBase]) -> List[str]:
    return [node.text for node in nodes]


def _tei_abstracts(inner: str) -> List[etree.ElementBase]:
    return list(etree.fromstring(f"<profileDesc>{inner}</profileDesc>"))


class TestAbstractsOffered:
    def test_should_return_a_summary_the_variants_leave_out(self):
        article_meta = _article_meta(
            '<abstract xml:lang="en">main</abstract>'
            '<abstract abstract-type="plain-language-summary" xml:lang="en">pls</abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["main"]
        assert _texts(abstracts_offered(article_meta)) == ["main", "pls"]

    def test_should_return_the_translations_as_well(self):
        article_meta = _article_meta(
            "<abstract>pt</abstract><trans-abstract xml:lang=\"en\">en</trans-abstract>"
        )
        assert _texts(abstracts_offered(article_meta)) == ["pt", "en"]

    def test_should_return_nothing_where_there_is_no_abstract(self):
        assert abstracts_offered(_article_meta("<title-group/>")) == []


class TestTeiAbstractVariants:
    def test_should_keep_an_abstract_naming_another_language(self):
        abstracts = _tei_abstracts(
            '<abstract xml:lang="pt">pt</abstract><abstract xml:lang="en">en</abstract>'
        )
        assert _texts(tei_abstract_variants(abstracts)) == ["pt", "en"]

    def test_should_leave_out_a_second_abstract_in_the_language_already_named(self):
        abstracts = _tei_abstracts(
            '<abstract xml:lang="en">main</abstract><abstract xml:lang="en">pls</abstract>'
        )
        assert _texts(tei_abstract_variants(abstracts)) == ["main"]

    def test_should_keep_an_abstract_naming_no_language(self):
        abstracts = _tei_abstracts(
            '<abstract xml:lang="en">main</abstract><abstract>unnamed</abstract>'
        )
        assert _texts(tei_abstract_variants(abstracts)) == ["main", "unnamed"]


class TestAbstractVariants:
    def test_should_return_the_only_abstract(self):
        article_meta = _article_meta("<abstract>one</abstract>")
        assert _texts(abstract_variants(article_meta)) == ["one"]

    def test_should_return_nothing_where_there_is_no_abstract(self):
        assert abstract_variants(_article_meta("<title-group/>")) == []

    def test_should_return_a_translation_after_the_article_own_abstract(self):
        article_meta = _article_meta(
            '<abstract>pt</abstract><trans-abstract xml:lang="en">en</trans-abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["pt", "en"]

    def test_should_return_every_translation(self):
        article_meta = _article_meta(
            "<abstract>pt</abstract>"
            '<trans-abstract xml:lang="en">en</trans-abstract>'
            '<trans-abstract xml:lang="es">es</trans-abstract>'
            '<trans-abstract xml:lang="fr">fr</trans-abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["pt", "en", "es", "fr"]

    def test_should_keep_two_translations_declaring_the_same_language(self):
        article_meta = _article_meta(
            "<abstract>pt</abstract>"
            '<trans-abstract xml:lang="es">first</trans-abstract>'
            '<trans-abstract xml:lang="es">second</trans-abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["pt", "first", "second"]

    def test_should_return_a_repeated_abstract_declaring_another_language(self):
        article_meta = _article_meta(
            '<abstract xml:lang="es" abstract-type="short">es</abstract>'
            '<abstract xml:lang="en" abstract-type="short">en</abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["es", "en"]

    def test_should_drop_a_repeated_abstract_declaring_the_same_language(self):
        article_meta = _article_meta(
            '<abstract xml:lang="en" abstract-type="short">first</abstract>'
            '<abstract xml:lang="en" abstract-type="short">second</abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["first"]

    def test_should_drop_a_repeated_abstract_declaring_no_language(self):
        article_meta = _article_meta(
            "<abstract>first</abstract><abstract>second</abstract>"
        )
        assert _texts(abstract_variants(article_meta)) == ["first"]

    def test_should_drop_a_plain_language_summary(self):
        article_meta = _article_meta(
            "<abstract>the abstract</abstract>"
            '<abstract abstract-type="plain-language-summary">the summary</abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["the abstract"]

    def test_should_drop_another_kind_of_abstract_whatever_language_it_declares(self):
        article_meta = _article_meta(
            '<abstract xml:lang="en">the abstract</abstract>'
            '<abstract xml:lang="es" abstract-type="plain-language-summary">el resumen</abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["the abstract"]

    def test_should_put_the_article_own_abstract_first_whatever_the_document_order(self):
        article_meta = _article_meta(
            '<trans-abstract xml:lang="en">en</trans-abstract><abstract>pt</abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["pt", "en"]

    def test_should_fall_back_to_the_first_element_where_there_is_no_plain_abstract(self):
        article_meta = _article_meta(
            '<trans-abstract xml:lang="en">en</trans-abstract>'
            '<trans-abstract xml:lang="es">es</trans-abstract>'
        )
        assert _texts(abstract_variants(article_meta)) == ["en", "es"]


def _pair(language: str, text: str) -> str:
    return language + PROPERTY_SEPARATOR + text


def _article(inner: str, language: str = "pt") -> etree.ElementBase:
    return etree.fromstring(
        f'<article xml:lang="{language}"><front><article-meta><title-group>'
        f'{inner}</title-group></article-meta></front></article>'
    )


class TestAbstractLanguagePairs:
    def test_should_give_an_abstract_the_articles_language(self):
        article = etree.fromstring(
            '<article xml:lang="pt"><front><article-meta>'
            "<abstract>O resumo</abstract>"
            "</article-meta></front></article>"
        )
        assert fn_jats_abstract_language_pairs(None, [article]) == [_pair("pt", "O resumo")]

    def test_should_prefer_a_language_on_the_abstract_itself(self):
        article = etree.fromstring(
            '<article xml:lang="pt"><front><article-meta>'
            '<abstract xml:lang="es">El resumen</abstract>'
            "</article-meta></front></article>"
        )
        assert fn_jats_abstract_language_pairs(None, [article]) == [_pair("es", "El resumen")]


class TestTitleLanguagePairs:
    def test_should_give_the_article_its_own_language(self):
        article = _article("<article-title>O titulo</article-title>")
        assert title_language_pairs(article) == [_pair("pt", "O titulo")]

    def test_should_prefer_a_language_on_the_article_title_itself(self):
        article = _article(
            '<article-title xml:lang="es">El titulo</article-title>'
            '<article-title xml:lang="en">The title</article-title>'
        )
        assert title_language_pairs(article) == [
            _pair("es", "El titulo"), _pair("en", "The title")
        ]

    def test_should_read_a_translation_language_from_its_group(self):
        article = _article(
            "<article-title>O titulo</article-title>"
            '<trans-title-group xml:lang="en"><trans-title>The title</trans-title>'
            "</trans-title-group>"
        )
        assert title_language_pairs(article) == [_pair("pt", "O titulo"), _pair("en", "The title")]

    def test_should_prefer_a_language_on_the_translation_itself(self):
        article = _article(
            "<article-title>O titulo</article-title>"
            '<trans-title-group xml:lang="en"><trans-title xml:lang="es">El titulo'
            "</trans-title></trans-title-group>"
        )
        assert title_language_pairs(article) == [_pair("pt", "O titulo"), _pair("es", "El titulo")]

    def test_should_give_nothing_where_there_is_no_title_group(self):
        article = etree.fromstring('<article xml:lang="pt"><front/></article>')
        assert title_language_pairs(article) == []
