import re
import unicodedata
from typing import Optional, Sequence

from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutLink, LayoutToken
from sciencebeam_parser.document.semantic_document import (
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticReference
)


# GROBID's TextUtilities.DOIPattern
LINK_DOI_PATTERN = r'10\.\d{4,5}/\S+[^;,.\s]'


def _remove_combining_marks(text: str) -> str:
    return ''.join(c for c in text if not unicodedata.combining(c))


def get_doi_for_link_uri(uri: str) -> Optional[str]:
    """Return the DOI a link points to, cleaned as GROBID's BiblioItem.cleanDOI does."""
    if not re.search(LINK_DOI_PATTERN, uri):
        return None
    doi = re.sub(r'\s', '', uri)
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
    return doi


def set_reference_doi_from_links(
    reference: SemanticReference,
    tokens: Sequence[LayoutToken],
    links: Sequence[LayoutLink]
) -> None:
    """Give the reference the DOI of the last link to a DOI over any of its tokens."""
    covering_doi_links = [
        (link, doi)
        for link in links
        for doi in [get_doi_for_link_uri(link.uri)]
        if doi and any(link.covers(token) for token in tokens)
    ]
    if not covering_doi_links:
        return
    link, doi = covering_doi_links[-1]
    for identifier in reference.iter_by_type(SemanticExternalIdentifier):
        if identifier.external_identifier_type == SemanticExternalIdentifierTypes.DOI:
            identifier.value = doi
            return
    reference.add_content(SemanticExternalIdentifier(
        layout_block=LayoutBlock.for_tokens([token for token in tokens if link.covers(token)]),
        value=doi,
        external_identifier_type=SemanticExternalIdentifierTypes.DOI
    ))
