import re
import unicodedata
from typing import Optional


def _remove_combining_marks(text: str) -> str:
    return ''.join(c for c in text if not unicodedata.combining(c))


def clean_doi(doi: str) -> str:
    """Return the DOI cleaned much as GROBID's BiblioItem.cleanDOI does."""
    doi = re.sub(r'\s', '', doi)
    doi = re.sub(r'https?://(dx\.)?doi\.org/', '', doi)
    if doi.lower().startswith(('doi:', 'doi/')):
        doi = doi[4:]
    if doi.lower().startswith('doi'):
        doi = doi[3:]
    doi_prefix_index = doi.lower().find('doi:10.')
    if doi_prefix_index >= 0:
        doi = doi[doi_prefix_index + 4:]
    if doi.startswith(('file://', 'https://', 'http://')):
        doi_start_index = doi.find('/10.')
        if doi_start_index >= 0:
            doi = doi[doi_start_index + 1:]
    http_index = doi.find('http://')
    if http_index > 10:
        doi = doi[:http_index]
    doi = _remove_combining_marks(doi)
    if doi.startswith(('(', '[', '⟨')):
        doi = doi[1:]
    if doi.endswith((')', ']', '⟩')):
        doi = doi[:-1]
    # GROBID would keep a prefix such as "https://doi.org:"
    doi_start = doi.find('10.')
    return doi[doi_start:] if doi_start > 0 else doi


# https://www.doi.org/doi_handbook/2_Numbering.html
_DOI_FOLLOWER = r'https?://|www\.|doi:|pmid:?\d|pubmed(?!/)|pmcid:?|pmc\d'
PRINTED_DOI_PATTERN = re.compile(
    rf'(?:(?<![0-9a-z])|(?<=doi))(10\.\d{{4,}}(?:\.\d+)*/'
    rf'(?:(?!{_DOI_FOLLOWER}).)(?:(?!{_DOI_FOLLOWER}|arxiv:).)*)',
    re.IGNORECASE
)

_CLOSING_BRACKET_BY_OPENING = {'(': ')', '[': ']', '⟨': '⟩', '<': '>'}

_TRAILING_NON_DOI_CHARACTERS = '.,;:/–—"\'“”‘’«»([⟨<'


def _ends_with_an_unmatched_bracket(doi: str) -> bool:
    return any(
        doi.endswith(closing) and doi.count(closing) > doi.count(opening)
        for opening, closing in _CLOSING_BRACKET_BY_OPENING.items()
    )


def find_printed_doi(text: str) -> Optional[str]:
    """Return the first DOI in the text, with its whitespace removed."""
    m = PRINTED_DOI_PATTERN.search(re.sub(r'\s', '', text))
    if not m:
        return None
    doi = m.group(1)
    while doi[-1] in _TRAILING_NON_DOI_CHARACTERS or _ends_with_an_unmatched_bracket(doi):
        doi = doi[:-1]
    return doi if '/' in doi else None
