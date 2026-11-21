import logging
from typing import (
    List,
    Sequence,
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
    context: TeiElementFactoryContext
) -> etree.ElementBase:
    """The author with the affiliations attached to it (see
    sciencebeam_parser.processors.fulltext.attach_affiliations), each written out
    in full under the author as GROBID does, under its key."""
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
    children.extend([
        get_tei_affiliation_for_semantic_affiliation_address_element(
            semantic_affiliation,
            context=context
        )
        for semantic_affiliation in semantic_author.affiliations
    ])
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


def get_orphan_affiliations(
    affiliations: Sequence[SemanticAffiliationAddress],
    authors: Sequence[SemanticAuthor]
) -> List[SemanticAffiliationAddress]:
    """The affiliations attached to no author, which GROBID writes out under an
    author of their own ("better than nothing")."""
    attached_affiliation_ids = {
        id(affiliation)
        for author in authors
        for affiliation in author.affiliations
    }
    return [
        affiliation
        for affiliation in affiliations
        if id(affiliation) not in attached_affiliation_ids
    ]
