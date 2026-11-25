import re
from typing import Optional, Sequence

from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutLink, LayoutToken
from sciencebeam_parser.models.citation.doi import clean_doi
from sciencebeam_parser.document.semantic_document import (
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticReference
)


# GROBID's TextUtilities.DOIPattern
LINK_DOI_PATTERN = r'10\.\d{4,5}/\S+[^;,.\s]'


def get_doi_for_link_uri(uri: str) -> Optional[str]:
    """Return the DOI a link points to."""
    if not re.search(LINK_DOI_PATTERN, uri):
        return None
    return clean_doi(uri)


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
