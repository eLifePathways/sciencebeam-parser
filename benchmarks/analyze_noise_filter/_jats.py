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

# A normalised text is looked up when it is this long, or when it carries at least
# MIN_LOOKUP_WORDS words. Below both, a match says nothing: a bare page number occurs
# in almost any article, and reporting that as content would be an invented alarm.
# The word path is what reaches a running head carrying an author's name.
MIN_LOOKUP_LENGTH = 24
MIN_LOOKUP_WORDS = 3

_WHITESPACE_RE = re.compile(r'\s+')
_TRAILING_NUMBER_RE = re.compile(r'[\s|]*\d+[\s|]*$')
_DASHES = '‐‑‒–—―−'


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = text.replace(' ', ' ')
    for dash in _DASHES:
        text = text.replace(dash, '-')
    return _WHITESPACE_RE.sub(' ', text).strip().casefold()


# A running head is usually the article's short title or the journal's name, so it
# matches the JATS while still being furniture. Text found only here decides nothing.
TITLE_TAGS = frozenset({
    'article-title', 'alt-title', 'subtitle', 'trans-title',
    'journal-title', 'journal-subtitle', 'abbrev-journal-title',
})


class JatsTextIndex:
    """The concatenated text of one JATS document, normalised for substring lookup."""

    def __init__(self, text: str, title_text: str = ''):
        self.text = text
        self.title_text = title_text

    @staticmethod
    def from_file(jats_filename: str) -> 'JatsTextIndex':
        root = etree.parse(jats_filename).getroot()
        title_texts = [
            ' '.join(element.itertext())
            for element in root.iter()
            if etree.QName(element).localname in TITLE_TAGS
        ]
        return JatsTextIndex(
            normalize_text(' '.join(root.itertext())),
            normalize_text(' '.join(title_texts))
        )

    def contains(self, text: str) -> Optional[bool]:
        """True if the text is in the JATS, False if not, None if it cannot be looked up."""
        needle = normalize_text(text)
        if needle.endswith('-'):
            # A line broken mid-word carries a partial word the JATS never contains.
            needle = needle[:-1].rsplit(' ', 1)[0].strip()
        # A running head often appends the page number to whatever it repeats.
        needle = _TRAILING_NUMBER_RE.sub('', needle).strip()
        if len(needle) < MIN_LOOKUP_LENGTH and len(needle.split()) < MIN_LOOKUP_WORDS:
            return None
        if needle not in self.text:
            return False
        return None if needle in self.title_text else True


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
