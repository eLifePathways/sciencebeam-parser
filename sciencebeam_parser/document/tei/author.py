import logging
from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Union
)

from lxml import etree

from sciencebeam_parser.document.semantic_document import (
    SemanticAddressField,
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticMarker
)
from sciencebeam_parser.document.tei.common import (
    TEI_E,
    XML_ID
)
from sciencebeam_parser.document.tei.factories import (
    TeiElementFactoryContext
)


LOGGER = logging.getLogger(__name__)


def _get_tei_raw_affiliation_element_for_semantic_affiliation_address(
    semantic_affiliation_address: SemanticAffiliationAddress,
    context: TeiElementFactoryContext
) -> etree.ElementBase:
    children: List[Union[str, dict, etree.ElementBase]] = []
    children.append({'type': 'raw_affiliation'})
    pending_whitespace: str = ''
    for semantic_content in semantic_affiliation_address:
        merged_block = semantic_content.merged_block
        if pending_whitespace:
            children.append(pending_whitespace)
        if isinstance(semantic_content, SemanticMarker):
            children.append(TEI_E(
                'label',
                *context.iter_layout_block_tei_children(merged_block, enable_coordinates=False)
            ))
            pending_whitespace = merged_block.whitespace
            continue
        children.extend(
            context.iter_layout_block_tei_children(merged_block, enable_coordinates=False)
        )
        pending_whitespace = merged_block.whitespace
    return TEI_E('note', *children)


def get_tei_affiliation_for_semantic_affiliation_address_element(
    semantic_affiliation_address: SemanticAffiliationAddress,
    context: TeiElementFactoryContext
) -> etree.ElementBase:
    LOGGER.debug('semantic_affiliation_address: %s', semantic_affiliation_address)
    raw_affiliation = _get_tei_raw_affiliation_element_for_semantic_affiliation_address(
        semantic_affiliation_address,
        context=context
    )
    attributes = context.get_default_attributes_for_semantic_content(
        semantic_affiliation_address
    )
    if semantic_affiliation_address.content_id:
        attributes = {**attributes, 'key': semantic_affiliation_address.content_id}
        if XML_ID in attributes:
            del attributes[XML_ID]
    children = [
        attributes,
        raw_affiliation
    ]
    address_semantic_content_list = []
    for semantic_content in semantic_affiliation_address:
        if isinstance(semantic_content, SemanticAddressField):
            address_semantic_content_list.append(semantic_content)
            continue
        children.extend(context.get_tei_child_elements_for_semantic_content(
            semantic_content
        ))
    LOGGER.debug('address_semantic_content_list: %r', address_semantic_content_list)
    if address_semantic_content_list:
        children.append(TEI_E('address', *[
            child
            for semantic_content in address_semantic_content_list
            for child in context.get_tei_child_elements_for_semantic_content(
                semantic_content
            )
        ]))
    return TEI_E('affiliation', *children)


def get_tei_author_for_semantic_author_element(
    semantic_author: SemanticAuthor,
    context: TeiElementFactoryContext,
    affiliations_by_marker: Optional[Mapping[str, Sequence[SemanticAffiliationAddress]]] = None
) -> etree.ElementBase:
    if affiliations_by_marker is None:
        affiliations_by_marker = {}
    LOGGER.debug('semantic_author: %s', semantic_author)
    pers_name_children = []
    for semantic_content in semantic_author:
        pers_name_children.extend(context.get_tei_child_elements_for_semantic_content(
            semantic_content
        ))
    children = [
        TEI_E(
            'persName',
            context.get_default_attributes_for_semantic_content(semantic_author),
            *pers_name_children
        )
    ]
    affiliations = []
    for marker_text in semantic_author.view_by_type(SemanticMarker).get_text_list():
        semantic_affiliations = affiliations_by_marker.get(marker_text)
        if not semantic_affiliations:
            LOGGER.warning('affiliation not found for marker: %r', marker_text)
            continue
        for semantic_affiliation in semantic_affiliations:
            affiliations.append(get_tei_affiliation_for_semantic_affiliation_address_element(
                semantic_affiliation,
                context=context
            ))
    children.extend(affiliations)
    return TEI_E('author', *children)


def get_dummy_tei_author_for_semantic_affiliations_element(
    semantic_affiliations: Sequence[SemanticAffiliationAddress],
    context: TeiElementFactoryContext
) -> etree.ElementBase:
    children = [
        TEI_E('note', {'type': 'dummy_author'}, 'Dummy author for orphan affiliations')
    ]
    children.extend([
        get_tei_affiliation_for_semantic_affiliation_address_element(
            semantic_affiliation,
            context=context
        )
        for semantic_affiliation in semantic_affiliations
    ])
    return TEI_E('author', *children)


def get_authors_affiliation_markers(authors: List[SemanticAuthor]) -> Set[str]:
    return {
        marker
        for author in authors
        for marker in author.view_by_type(SemanticMarker).get_text_list()
    }


def get_orphan_affiliations(
    affiliations_by_marker: Dict[str, List[SemanticAffiliationAddress]],
    authors: List[SemanticAuthor]
) -> List[SemanticAffiliationAddress]:
    used_affiliation_markers = get_authors_affiliation_markers(authors)
    return [
        affiliation
        for marker, affiliations in affiliations_by_marker.items()
        if marker not in used_affiliation_markers
        for affiliation in affiliations
    ]
