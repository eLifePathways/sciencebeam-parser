from typing import Optional, Sequence, Tuple, Union

from sciencebeam_parser.document.layout_document import LayoutLink
from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticInvalidReference,
    SemanticLabel,
    SemanticNote,
    SemanticRawReference,
    SemanticReference
)
from sciencebeam_parser.models.citation.extract import has_doi
from sciencebeam_parser.models.citation.links import set_reference_doi_from_links


def _is_split_off_line(raw_reference: SemanticRawReference) -> bool:
    return not raw_reference.has_type(SemanticLabel) and not any(
        c.isdigit() for c in raw_reference.get_text()
    )


def give_fragment_link_dois_to_previous_references(
    reference_list_content: Sequence[Union[
        Tuple[SemanticRawReference, Union[SemanticReference, SemanticInvalidReference]],
        SemanticContentWrapper
    ]],
    links: Sequence[LayoutLink]
) -> None:
    """Give the link DOI of a dropped line with no number or digit to the reference before."""
    previous_ref: Optional[SemanticReference] = None
    for raw_and_parsed_reference in reference_list_content:
        if isinstance(raw_and_parsed_reference, SemanticNote):
            continue
        if not isinstance(raw_and_parsed_reference, tuple):
            previous_ref = None
            continue
        raw_reference, ref = raw_and_parsed_reference
        if isinstance(ref, SemanticReference):
            previous_ref = ref
        elif not _is_split_off_line(raw_reference):
            previous_ref = None
        elif previous_ref is not None and not has_doi(previous_ref):
            set_reference_doi_from_links(previous_ref, list(raw_reference.iter_tokens()), links)
