import logging
import re
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

from sciencebeam_parser.utils.misc import iter_ids
from sciencebeam_parser.document.semantic_document import (
    SemanticBookTitle,
    SemanticContentWrapper,
    SemanticDate,
    SemanticExternalIdentifier,
    SemanticExternalIdentifierTypes,
    SemanticExternalUrl,
    SemanticInvalidReference,
    SemanticIssue,
    SemanticJournal,
    SemanticLocation,
    SemanticNote,
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
from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutLink
from sciencebeam_parser.models.citation.doi import find_printed_doi
from sciencebeam_parser.models.citation.links import set_reference_doi_from_links
from sciencebeam_parser.models.citation.labels import IDENTIFIER_LABEL, NOTE_CITATION_LABELS
from sciencebeam_parser.models.extract import SimpleModelSemanticExtractor


LOGGER = logging.getLogger(__name__)


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
    '<booktitle>': SemanticBookTitle,
    '<volume>': SemanticVolume,
    '<issue>': SemanticIssue,
    '<publisher>': SemanticPublisher,
    '<location>': SemanticLocation
}


def _is_doi(semantic_content: SemanticContentWrapper) -> bool:
    return (
        isinstance(semantic_content, SemanticExternalIdentifier)
        and semantic_content.external_identifier_type == SemanticExternalIdentifierTypes.DOI
    )


def _is_collaboration(semantic_content: SemanticContentWrapper) -> bool:
    return (
        isinstance(semantic_content, SemanticNote)
        and semantic_content.note_type == '<collaboration>'
    )


IS_REFERENCE_FIELD_BY_NAME: Mapping[str, Callable[[SemanticContentWrapper], bool]] = {
    'title': lambda c: isinstance(c, SemanticTitle),
    'journal': lambda c: isinstance(c, SemanticJournal),
    'book_title': lambda c: isinstance(c, SemanticBookTitle),
    'authors': lambda c: isinstance(c, SemanticRawAuthors),
    'editors': lambda c: isinstance(c, SemanticRawEditors),
    'collaboration': _is_collaboration,
    'doi': _is_doi,
    'identifier': lambda c: isinstance(c, SemanticExternalIdentifier),
    'url': lambda c: isinstance(c, SemanticExternalUrl)
}


class FragmentLinkDoi:
    """Where the DOI of a link goes when it is all that would keep a reference."""

    PREVIOUS_REFERENCE = 'previous_reference'
    OWN_REFERENCE = 'own_reference'


FRAGMENT_LINK_DOI_VALUES = (FragmentLinkDoi.PREVIOUS_REFERENCE, FragmentLinkDoi.OWN_REFERENCE)

DEFAULT_FRAGMENT_LINK_DOI = FragmentLinkDoi.PREVIOUS_REFERENCE


class ReferenceTitlesMode:
    """Which of a reference's titles it keeps as titles."""

    ALL = 'all'
    LONGEST = 'longest'


REFERENCE_TITLES_MODES = (ReferenceTitlesMode.ALL, ReferenceTitlesMode.LONGEST)

DEFAULT_REFERENCE_TITLES_MODE = ReferenceTitlesMode.ALL


class ReferenceDoisMode:
    """Which of a reference's DOIs it keeps as DOIs."""

    ALL = 'all'
    FIRST = 'first'


REFERENCE_DOIS_MODES = (ReferenceDoisMode.ALL, ReferenceDoisMode.FIRST)

DEFAULT_REFERENCE_DOIS_MODE = ReferenceDoisMode.ALL


DEFAULT_REFERENCE_KEPT_BY_ANY_OF: Sequence[str] = (
    'title', 'journal', 'book_title', 'authors', 'editors', 'identifier', 'url'
)


def validate_reference_field_names(field_names: Sequence[str]) -> Sequence[str]:
    unknown = [name for name in field_names if name not in IS_REFERENCE_FIELD_BY_NAME]
    if unknown:
        raise ValueError(
            f'unknown reference fields: {unknown!r},'
            f' expected any of {list(IS_REFERENCE_FIELD_BY_NAME)}'
        )
    return field_names


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
    doi = find_printed_doi(layout_block.text)
    if doi:
        return SemanticExternalIdentifier(
            layout_block=layout_block,
            value=doi,
            external_identifier_type=SemanticExternalIdentifierTypes.DOI
        )
    return SemanticExternalUrl(
        layout_block=layout_block,
        value=value
    )


def get_detected_external_identifier_type_and_value_for_text(
    text: str
) -> Tuple[Optional[str], str]:
    doi = find_printed_doi(text)
    if doi:
        return SemanticExternalIdentifierTypes.DOI, doi
    value = re.sub(r'\s', '', text)
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


def has_doi(ref: SemanticReference) -> bool:
    return any(_is_doi(semantic_content) for semantic_content in ref)


def is_reference_kept_by_a_doi(kept_by_any_of: Sequence[str]) -> bool:
    return any(name in ('doi', 'identifier') for name in kept_by_any_of)


def validate_fragment_link_doi(fragment_link_doi: str, kept_by_any_of: Sequence[str]) -> None:
    if fragment_link_doi not in FRAGMENT_LINK_DOI_VALUES:
        raise ValueError(
            f'unknown fragment_link_doi: {fragment_link_doi!r},'
            f' expected one of {list(FRAGMENT_LINK_DOI_VALUES)}'
        )
    if (
        fragment_link_doi == FragmentLinkDoi.OWN_REFERENCE
        and not is_reference_kept_by_a_doi(kept_by_any_of)
    ):
        raise ValueError(
            'fragment_link_doi: own_reference needs doi or identifier in references.kept_by_any_of'
        )


def _as_note(semantic_content: SemanticContentWrapper, note_type: str) -> SemanticNote:
    return SemanticNote(
        layout_block=LayoutBlock.merge_blocks(list(semantic_content.iter_blocks())),
        note_type=note_type
    )


def keep_the_longest_title(ref: SemanticReference) -> None:
    """Keep the longest title, the first of equals, and turn the others into notes."""
    titles = list(ref.iter_by_type(SemanticTitle))
    if len(titles) < 2:
        return
    longest = max(titles, key=lambda title: len(title.get_text()))
    ref.mixed_content = [
        _as_note(c, 'title') if isinstance(c, SemanticTitle) and c is not longest else c
        for c in ref.mixed_content
    ]


def keep_the_first_doi(ref: SemanticReference) -> None:
    """Keep the first DOI and turn the others into notes."""
    dois = [c for c in ref.mixed_content if _is_doi(c)]
    ref.mixed_content = [
        _as_note(c, 'doi') if _is_doi(c) and c is not dois[0] else c
        for c in ref.mixed_content
    ]


def is_reference_valid(
    ref: SemanticReference,
    kept_by_any_of: Sequence[str] = DEFAULT_REFERENCE_KEPT_BY_ANY_OF
) -> bool:
    is_field_list = [IS_REFERENCE_FIELD_BY_NAME[name] for name in kept_by_any_of]
    return any(
        is_field(semantic_content)
        for semantic_content in ref
        for is_field in is_field_list
    )


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
        reference_kept_by_any_of: Sequence[str] = DEFAULT_REFERENCE_KEPT_BY_ANY_OF,
        links: Sequence[LayoutLink] = (),
        fragment_link_doi: str = DEFAULT_FRAGMENT_LINK_DOI,
        select_titles: str = DEFAULT_REFERENCE_TITLES_MODE,
        select_dois: str = DEFAULT_REFERENCE_DOIS_MODE,
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        entity_tokens = list(entity_tokens)
        LOGGER.debug('entity_tokens: %s', entity_tokens)
        ids_iterator = iter(iter_ids('b'))

        def _new_reference() -> SemanticReference:
            new_ref = SemanticReference()
            if semantic_raw_reference is not None:
                new_ref.content_id = semantic_raw_reference.content_id
                for semantic_content in semantic_raw_reference:
                    new_ref.add_content(semantic_content)
            if not new_ref.content_id:
                new_ref.content_id = next(ids_iterator, '?')
            return new_ref

        ref: Optional[SemanticReference] = (
            _new_reference() if semantic_raw_reference is not None else None
        )
        for name, layout_block in entity_tokens:
            if ref is None:
                ref = _new_reference()
            semantic_content = self.get_semantic_content_for_entity_name(
                name, layout_block=layout_block
            )
            ref.add_content(semantic_content)
        if ref is not None and semantic_raw_reference is not None and (
            is_reference_valid(ref, reference_kept_by_any_of)
            or (
                fragment_link_doi == FragmentLinkDoi.OWN_REFERENCE
                and is_reference_kept_by_a_doi(reference_kept_by_any_of)
            )
        ):
            set_reference_doi_from_links(ref, list(semantic_raw_reference.iter_tokens()), links)
        if ref is not None and select_titles == ReferenceTitlesMode.LONGEST:
            keep_the_longest_title(ref)
        if ref is not None and select_dois == ReferenceDoisMode.FIRST:
            keep_the_first_doi(ref)
        if ref is not None and not is_reference_valid(ref, reference_kept_by_any_of):
            yield get_invalid_reference(ref)
        elif ref is not None:
            yield ref
