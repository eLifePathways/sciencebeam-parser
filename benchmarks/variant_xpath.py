"""Which elements are a field's variants, and which of them is the article's own.

Written against sciencebeam-judge's xpath function interface, and registered by
`benchmarks.judge_setup`, so it can move there once the rule has settled.
"""
from __future__ import annotations

from typing import List, Optional

from lxml import etree as ET

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


def abstract_variants(article_meta: ET.ElementBase) -> List[ET.ElementBase]:
    """The abstracts a document offers as alternatives, the article's own first."""
    candidates = [
        node for node in article_meta
        if node.tag in (ABSTRACT, TRANS_ABSTRACT)
    ]
    if not candidates:
        return []
    main = _main_variant(candidates)
    return [main] + [
        node for node in candidates
        if node is not main and _is_translation_variant(node, main)
    ]


def _main_variant(candidates: List[ET.ElementBase]) -> ET.ElementBase:
    return next(
        (node for node in candidates if node.tag == ABSTRACT),
        candidates[0],
    )


def fn_jats_abstract_variants(_, nodes):
    return [variant for node in nodes for variant in abstract_variants(node)]


def register_variant_functions(ns: Optional[ET.FunctionNamespace] = None) -> None:
    if ns is None:
        ns = ET.FunctionNamespace(None)
    ns["jats-abstract-variants"] = fn_jats_abstract_variants
