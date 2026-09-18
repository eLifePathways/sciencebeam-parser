"""Presence in the publisher's JATS, used as a one-sided precision alarm.

A removed line whose text appears in the JATS is content, whatever any gold says. The
converse does not hold: reference-list headings, figure and table graphic text and
reference formatting are PDF-only, so absence is not evidence of furniture.

This reads the JATS text only, not its structure, so it does not inherit the aligner's
region assignment.
"""
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from lxml import etree

LOGGER = logging.getLogger(__name__)

# Below this many characters a normalised text is not looked up: a bare page number
# occurs in almost any article, and reporting that as content would be an invented
# alarm rather than a missed one.
MIN_LOOKUP_LENGTH = 24

_WHITESPACE_RE = re.compile(r'\s+')
_DASHES = '‐‑‒–—―−'


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = text.replace(' ', ' ')
    for dash in _DASHES:
        text = text.replace(dash, '-')
    return _WHITESPACE_RE.sub(' ', text).strip().casefold()


class JatsTextIndex:
    """The concatenated text of one JATS document, normalised for substring lookup."""

    def __init__(self, text: str):
        self.text = text

    @staticmethod
    def from_file(jats_filename: str) -> 'JatsTextIndex':
        root = etree.parse(jats_filename).getroot()
        return JatsTextIndex(normalize_text(' '.join(root.itertext())))

    def contains(self, text: str) -> Optional[bool]:
        """True if the text is in the JATS, False if not, None if it cannot be looked up."""
        needle = normalize_text(text)
        if needle.endswith('-'):
            # A line broken mid-word carries a partial word the JATS never contains.
            needle = needle[:-1].rsplit(' ', 1)[0].strip()
        if len(needle) < MIN_LOOKUP_LENGTH:
            return None
        return needle in self.text


def find_jats_for_pdf(pdf_filename: str, jats_filenames: dict) -> Optional[str]:
    """Match a PDF to its JATS by stem, allowing the compound `.jats.xml` suffix."""
    stem = Path(pdf_filename).stem
    return jats_filenames.get(stem)


def index_jats_filenames_by_stem(jats_filenames) -> dict:
    result = {}
    for filename in jats_filenames:
        stem = Path(filename).name
        for suffix in ('.jats.xml', '.xml'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        result[stem] = filename
    return result
