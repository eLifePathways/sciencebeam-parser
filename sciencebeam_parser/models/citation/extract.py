import logging
import re
from typing import Iterable, Mapping, Optional, Set, Tuple, Type, Union

from sciencebeam_parser.utils.misc import iter_ids
from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticDate,
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticExternalUrl,
    SemanticInvalidReference,
    SemanticIssue,
    SemanticJournal,
    SemanticLocation,
    SemanticPageRange,
    SemanticPublisher,
    SemanticRawAuthors,
    SemanticRawEditors,
    SemanticRawReference,
    SemanticRawReferenceText,
    SemanticReference,
    SemanticTitle,
    SemanticVolume,
    T_SemanticContentFactory
)
from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.models.citation.labels import IDENTIFIER_LABEL, NOTE_CITATION_LABELS
from sciencebeam_parser.models.extract import SimpleModelSemanticExtractor


LOGGER = logging.getLogger(__name__)


# https://en.wikipedia.org/wiki/Digital_Object_Identifier
# https://www.doi.org/doi_handbook/2_Numbering.html
DOI_PATTERN = r'\b(10\.\d{4,}(?:\.\d{1,})*/.+)'

# copied and adapted from:
# https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/main/java/org/grobid/core/utilities/TextUtilities.java#L66
PMID_PATTERN = r"(?:(?:PMID)|(?:Pub(?:\s)?Med(?:\s)?(?:ID)?))(?:\s)?(?:\:)?(?:\s)*(\d{1,8})"

PMCID_PATTERN = r"(?:PMC)(\d{1,})"

# copied and adapted from:
# https://github.com/kermitt2/grobid/blob/0.6.2/grobid-core/src/main/java/org/grobid/core/utilities/TextUtilities.java#L62-L63
ARXIV_PATTERN = (
    r"(?:arXiv\s?(?:\.org)?\s?\:\s?(\d{4}\s?\.\s?\d{4,5}(?:v\d+)?))"
    r"|(?:arXiv\s?(?:\.org)?\s?\:\s?([ a-zA-Z\-\.]*\s?/\s?\d{7}(?:v\d+)?))"
)

# https://en.wikipedia.org/wiki/Publisher_Item_Identifier
PII_PATTERN = r'\b([S,B]\W*(?:[0-9xX]\W*){15,}[0-9xX])'


SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG: Mapping[str, T_SemanticContentFactory] = {
    '<author>': SemanticRawAuthors,
    '<editor>': SemanticRawEditors,
    '<title>': SemanticTitle,
    '<journal>': SemanticJournal,
    '<volume>': SemanticVolume,
    '<issue>': SemanticIssue,
    '<publisher>': SemanticPublisher,
    '<location>': SemanticLocation
}


VALID_REFERENCE_TYPES: Set[Type[SemanticContentWrapper]] = {
    SemanticTitle,
    SemanticJournal,
    SemanticRawAuthors,
    SemanticRawEditors,
    SemanticExternalIdentifier,
    SemanticExternalUrl
}


def parse_page_range(layout_block: LayoutBlock) -> SemanticPageRange:
    page_range_text = layout_block.text
    page_parts = page_range_text.split('-')
    if len(page_parts) == 2:
        from_page = page_parts[0].strip()
        to_page = page_parts[1].strip()
        if to_page and len(to_page) < len(from_page):
            to_page = from_page[:-(len(to_page))] + to_page
        return SemanticPageRange(
            layout_block=layout_block,
            from_page=from_page,
            to_page=to_page
        )
    return SemanticPageRange(layout_block=layout_block)


def parse_web(layout_block: LayoutBlock) -> Union[SemanticExternalUrl, SemanticExternalIdentifier]:
    value = re.sub(r'\s', '', layout_block.text)
    m = re.search(DOI_PATTERN, value)
    if m:
        return SemanticExternalIdentifier(
            layout_block=layout_block,
            value=m.group(1),
            external_identifier_type=SemanticExternalIdentifierTypes.DOI
        )
    return SemanticExternalUrl(
        layout_block=layout_block,
        value=value
    )


def get_detected_external_identifier_type_and_value_for_text(
    text: str
) -> Tuple[Optional[str], str]:
    value = re.sub(r'\s', '', text)
    m = re.search(DOI_PATTERN, value)
    if m:
        value = m.group(1)
        return SemanticExternalIdentifierTypes.DOI, value
    m = re.search(PMCID_PATTERN, value)
    if m:
        value = 'PMC' + m.group(1)
        return SemanticExternalIdentifierTypes.PMCID, value
    m = re.search(ARXIV_PATTERN, value)
    if m:
        value = m.group(1) or m.group(2)
        return SemanticExternalIdentifierTypes.ARXIV, value
    m = re.match(PMID_PATTERN, value)
    if m:
        value = m.group(1)
        return SemanticExternalIdentifierTypes.PMID, value
    m = re.search(PII_PATTERN, value)
    if m:
        value = m.group(1)
        return SemanticExternalIdentifierTypes.PII, value
    return None, value


def get_detected_external_identifier_type_for_text(text: str) -> Optional[str]:
    external_identifier_type, _ = get_detected_external_identifier_type_and_value_for_text(
        text
    )
    return external_identifier_type


def parse_pubnum(layout_block: LayoutBlock) -> SemanticExternalIdentifier:
    external_identifier_type, value = get_detected_external_identifier_type_and_value_for_text(
        layout_block.text
    )
    return SemanticExternalIdentifier(
        layout_block=layout_block,
        value=value,
        external_identifier_type=external_identifier_type
    )


def parse_date(layout_block: LayoutBlock) -> SemanticDate:
    value = re.sub(r'\s', '', layout_block.text)
    year: Optional[int] = None
    m = re.search(r'(\d{4})', value)
    if m:
        year = int(m.group(1))
    return SemanticDate(
        layout_block=layout_block,
        year=year
    )


def is_reference_valid(ref: SemanticReference) -> bool:
    for semantic_content in ref:
        if type(semantic_content) in VALID_REFERENCE_TYPES:
            return True
    return False


def get_invalid_reference(ref: SemanticReference) -> SemanticInvalidReference:
    return SemanticInvalidReference(
        mixed_content=[
            semantic_content
            for semantic_content in ref
            if not isinstance(semantic_content, SemanticRawReferenceText)
        ]
    )


class CitationSemanticExtractor(SimpleModelSemanticExtractor):
    def __init__(self):
        super().__init__(
            semantic_content_class_by_tag=SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG,
            expected_note_tags=NOTE_CITATION_LABELS
        )

    def get_semantic_content_for_entity_name(  # pylint: disable=too-many-return-statements
        self,
        name: str,
        layout_block: LayoutBlock
    ) -> SemanticContentWrapper:
        if name == '<pages>':
            return parse_page_range(layout_block)
        if name == '<web>':
            return parse_web(layout_block)
        if name == IDENTIFIER_LABEL:
            return parse_pubnum(layout_block)
        if name == '<date>':
            return parse_date(layout_block)
        return super().get_semantic_content_for_entity_name(name, layout_block)

    def iter_semantic_content_for_entity_blocks(  # pylint: disable=arguments-differ
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        semantic_raw_reference: Optional[SemanticRawReference] = None,
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        entity_tokens = list(entity_tokens)
        LOGGER.debug('entity_tokens: %s', entity_tokens)
        ids_iterator = iter(iter_ids('b'))
        ref: Optional[SemanticReference] = None
        for name, layout_block in entity_tokens:
            if not ref:
                ref = SemanticReference()
                if semantic_raw_reference:
                    ref.content_id = semantic_raw_reference.content_id
                    for semantic_content in semantic_raw_reference:
                        ref.add_content(semantic_content)
                if not ref.content_id:
                    ref.content_id = next(ids_iterator, '?')
            semantic_content = self.get_semantic_content_for_entity_name(
                name, layout_block=layout_block
            )
            ref.add_content(semantic_content)
        if ref and not is_reference_valid(ref):
            yield get_invalid_reference(ref)
        elif ref:
            yield ref
