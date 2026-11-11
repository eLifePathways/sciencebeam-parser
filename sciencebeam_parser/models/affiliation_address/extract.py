import logging
from typing import Iterable, Mapping, Optional, Tuple

from sciencebeam_parser.utils.misc import iter_ids
from sciencebeam_parser.document.semantic_document import (
    SemanticAddressLine,
    SemanticAffiliationAddress,
    SemanticContentWrapper,
    SemanticCountry,
    SemanticDepartment,
    SemanticInstitution,
    SemanticLaboratory,
    SemanticMarker,
    SemanticNote,
    SemanticPostBox,
    SemanticPostCode,
    SemanticRegion,
    SemanticSettlement,
    T_SemanticContentFactory
)
from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.models.extract import (
    SimpleModelSemanticExtractor,
    get_regex_cleaned_layout_block_with_prefix_suffix
)


LOGGER = logging.getLogger(__name__)


SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG: Mapping[str, T_SemanticContentFactory] = {
    '<institution>': SemanticInstitution,
    '<department>': SemanticDepartment,
    '<laboratory>': SemanticLaboratory,
    '<addrLine>': SemanticAddressLine,
    '<postCode>': SemanticPostCode,
    '<postBox>': SemanticPostBox,
    '<region>': SemanticRegion,
    '<settlement>': SemanticSettlement,
    '<country>': SemanticCountry
}

CLEAN_REGEX_BY_TAG: Mapping[str, str] = {
    '<country>': r'(.*[^.]).*'
}


class AffiliationAddressSemanticExtractor(SimpleModelSemanticExtractor):
    def __init__(self):
        super().__init__(semantic_content_class_by_tag=SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG)

    def iter_semantic_content_for_entity_blocks(
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        entity_tokens = list(entity_tokens)
        LOGGER.debug('entity_tokens: %s', entity_tokens)
        ids_iterator = iter(iter_ids('aff'))
        aff: Optional[SemanticAffiliationAddress] = None
        for name, layout_block in entity_tokens:
            if name == '<marker>':
                if aff:
                    yield aff
                aff = SemanticAffiliationAddress(content_id=next(ids_iterator, '?'))
                aff.add_content(SemanticMarker(layout_block=layout_block))
                continue
            prefix_block, cleaned_block, suffix_block = (
                get_regex_cleaned_layout_block_with_prefix_suffix(
                    layout_block,
                    CLEAN_REGEX_BY_TAG.get(name)
                )
            )
            semantic_content = self.get_semantic_content_for_entity_name(
                name, cleaned_block
            )
            if (
                aff is not None
                and isinstance(semantic_content, SemanticInstitution)
                and aff.has_type(SemanticInstitution)
            ):
                yield aff
                aff = None
            if not aff:
                if isinstance(semantic_content, SemanticNote):
                    yield semantic_content
                    continue
                aff = SemanticAffiliationAddress(content_id=next(ids_iterator, '?'))
            if prefix_block:
                aff.add_content(SemanticNote(layout_block=prefix_block, note_type=f'{name}-prefix'))
            aff.add_content(semantic_content)
            if suffix_block:
                aff.add_content(SemanticNote(layout_block=suffix_block, note_type=f'{name}-suffix'))
        if aff:
            yield aff
