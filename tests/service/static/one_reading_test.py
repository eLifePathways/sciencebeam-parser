"""Showing the labels annotates the reading rather than replacing it.

What a value says has to stay the same when the names are turned on, or the
two cannot be read against each other. The renderer emits one page and the
stylesheet decides what shows, so what could differ between the two is what
the stylesheet hides -- and that is a short list on purpose: the names
themselves, and the punctuation the page adds between two authors, which
stands for nothing in the document and would be left dangling by a line each.
"""
import os
import re
from typing import Set

from sciencebeam_parser.service.assets import STATIC_DIRECTORY


STYLESHEET_FILE = os.path.join(STATIC_DIRECTORY, 'index.css')

HIDDEN_RULE_PATTERN = re.compile(r'([^{}]+)\{[^{}]*display:\s*none[^{}]*\}')

EXPECTED_HIDDEN_SELECTORS = {'.jats-tag', '.jats-labelled .jats-separator'}


def get_hidden_selectors() -> Set[str]:
    with open(STYLESHEET_FILE, 'r', encoding='utf-8') as stylesheet_fp:
        stylesheet = stylesheet_fp.read()
    return {
        selector.strip()
        for rule in HIDDEN_RULE_PATTERN.findall(stylesheet)
        for selector in rule.split(',')
        if selector.strip().startswith('.jats')
    }


class TestOneReading:
    def test_should_hide_nothing_the_document_carries(self):
        assert get_hidden_selectors() == EXPECTED_HIDDEN_SELECTORS
