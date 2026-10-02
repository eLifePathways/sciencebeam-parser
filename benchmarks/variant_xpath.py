"""Which elements are a field's variants, and which of them is the article's own.

Written against sciencebeam-judge's xpath function interface, and registered by
`benchmarks.judge_setup`, so it can move there once the rule has settled.
"""
from __future__ import annotations

from typing import List, Optional

from lxml import etree as ET

from sciencebeam_judge.utils.xml import get_text_content

from benchmarks.matched_property_scoring import PROPERTY_SEPARATOR

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

ABSTRACT = "abstract"
TRANS_ABSTRACT = "trans-abstract"
ABSTRACT_TYPE = "abstract-type"


def _is_same_kind(node: ET.ElementBase, main: ET.ElementBase) -> bool:
    return node.get(ABSTRACT_TYPE) == main.get(ABSTRACT_TYPE)


def _declares_another_language(node: ET.ElementBase, main: ET.ElementBase) -> bool:
    language = node.get(XML_LANG)
    return bool(language) and language != main.get(XML_LANG)


def _is_translation_variant(node: ET.ElementBase, main: ET.ElementBase) -> bool:
    if node.tag == TRANS_ABSTRACT:
        return True
    return _is_same_kind(node, main) and _declares_another_language(node, main)


def _candidates(article_meta: ET.ElementBase) -> List[ET.ElementBase]:
    return [
        node for node in article_meta
        if node.tag in (ABSTRACT, TRANS_ABSTRACT)
    ]


def main_abstract(article_meta: ET.ElementBase) -> List[ET.ElementBase]:
    """The article's own abstract, as at most one element: the first `<abstract>` in
    document order, or the first candidate where the document has none."""
    candidates = _candidates(article_meta)
    if not candidates:
        return []
    return [next(
        (node for node in candidates if node.tag == ABSTRACT),
        candidates[0],
    )]


def abstract_variants(article_meta: ET.ElementBase) -> List[ET.ElementBase]:
    """The abstracts a document offers as alternatives, `main_abstract` first.

    The scoring types read that position as the article's own, so the list is built from
    `main_abstract` rather than from a second reading of the same rule.
    """
    main = main_abstract(article_meta)
    if not main:
        return []
    return main + [
        node for node in _candidates(article_meta)
        if node is not main[0] and _is_translation_variant(node, main[0])
    ]


def fn_jats_main_abstract(_, nodes):
    return [get_text_content(variant) for node in nodes for variant in main_abstract(node)]


def fn_jats_abstract_variants(_, nodes):
    # Text rather than elements: lxml re-sorts a node-set a function returns into document
    # order, which would discard the article's own abstract being first.
    return [
        get_text_content(variant)
        for node in nodes for variant in abstract_variants(node)
    ]


def _language_pair(node: ET.ElementBase) -> str:
    """A value's language and the text that identifies it, for `matched_property`."""
    return (node.get(XML_LANG) or "") + PROPERTY_SEPARATOR + get_text_content(node)


def fn_jats_abstract_language_pairs(_, nodes):
    return [
        _language_pair(variant)
        for node in nodes for variant in abstract_variants(node)
    ]


def fn_tei_abstract_language_pairs(_, nodes):
    return [_language_pair(node) for node in nodes]


def register_variant_functions(ns: Optional[ET.FunctionNamespace] = None) -> None:
    if ns is None:
        ns = ET.FunctionNamespace(None)
    ns["jats-main-abstract"] = fn_jats_main_abstract
    ns["jats-abstract-variants"] = fn_jats_abstract_variants
    ns["jats-abstract-language-pairs"] = fn_jats_abstract_language_pairs
    ns["tei-abstract-language-pairs"] = fn_tei_abstract_language_pairs
