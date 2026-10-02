from __future__ import annotations

from typing import List

from lxml import etree

from benchmarks.variant_xpath import abstract_variants


def _article_meta(inner: str) -> etree.ElementBase:
    return etree.fromstring(f"<article-meta>{inner}</article-meta>")


def _texts(nodes: List[etree.ElementBase]) -> List[str]:
    return [node.text for node in nodes]


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
