"""The affiliation as one raw string, on either side of the comparison.

A label identifies an affiliation rather than naming it, so it is dropped from both
sides. Where a gold `<aff>` holds the affiliation in an `original` institution beside
the parts a database built it from, the `original` is the affiliation and the parts
repeat it.

Written against sciencebeam-judge's xpath function interface, and registered by
`benchmarks.judge_setup`, so it can move there once the rule has settled.
"""
from __future__ import annotations

import copy
from typing import List, Optional

from lxml import etree as ET

from sciencebeam_judge.utils.xml import (
    get_normalized_text_content,
    normalized_whitespace,
)

LABEL = "label"
ORIGINAL_INSTITUTION = 'institution[@content-type="original"]'
RAW_AFFILIATION = './note[@type="raw_affiliation"]'


def _without_labels(node: ET.ElementBase) -> ET.ElementBase:
    node = copy.deepcopy(node)
    for label in node.findall(LABEL):
        previous = label.getprevious()
        tail = label.tail or ""
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            node.text = (node.text or "") + tail
        node.remove(label)
    return node


def _text_without_label(node: ET.ElementBase) -> str:
    return get_normalized_text_content(_without_labels(node))


def jats_aff_raw_text(aff: ET.ElementBase) -> str:
    """A gold affiliation's `original` institution, or its own text where it has none."""
    original = aff.findall(ORIGINAL_INSTITUTION)
    if original:
        return normalized_whitespace(
            " ".join(get_normalized_text_content(node) for node in original)
        )
    return _text_without_label(aff)


def tei_aff_raw_text(aff: ET.ElementBase) -> str:
    """A predicted affiliation's raw string, without the label that identifies it."""
    return ", ".join(
        text
        for text in (
            _text_without_label(node) for node in aff.xpath(RAW_AFFILIATION)
        )
        if text
    )


def fn_jats_aff_raw_text(_, nodes: List[ET.ElementBase]) -> List[str]:
    return [jats_aff_raw_text(node) for node in nodes]


def fn_tei_aff_raw_text(_, nodes: List[ET.ElementBase]) -> List[str]:
    return [tei_aff_raw_text(node) for node in nodes]


def register_affiliation_functions(ns: Optional[ET.FunctionNamespace] = None) -> None:
    if ns is None:
        ns = ET.FunctionNamespace(None)
    ns["jats-aff-raw-text"] = fn_jats_aff_raw_text
    ns["tei-aff-raw-text"] = fn_tei_aff_raw_text
