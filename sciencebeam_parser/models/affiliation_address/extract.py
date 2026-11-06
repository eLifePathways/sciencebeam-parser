import logging
from typing import AbstractSet, Callable, Iterable, Iterator, Mapping, Optional, Tuple

from sciencebeam_parser.utils.misc import iter_ids
from sciencebeam_parser.document.semantic_document import (
    SemanticAddressField,
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
from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutToken
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

# a marker that starts no affiliation, which GROBID drops (Affiliation.isNotNull
# does not count the marker), kept as a note so that the text is not lost
LONE_MARKER_NOTE_TYPE = 'affiliation_marker'

# a country GROBID does not repeat: a second one with the same text is skipped
REPEATED_COUNTRY_NOTE_TYPE = 'country-repeated'


def starts_new_line(
    previous_layout_block: Optional[LayoutBlock],
    layout_block: LayoutBlock
) -> bool:
    """Whether the block starts a line of its own: GROBID sees a line break token
    at the end of the cluster before, or a change of page or of vertical position
    (and treats the first cluster as starting one)."""
    if previous_layout_block is None:
        return True
    previous_tokens = list(previous_layout_block.iter_all_tokens())
    tokens = list(layout_block.iter_all_tokens())
    if not previous_tokens or not tokens:
        return False
    return previous_tokens[-1].line_meta != tokens[0].line_meta


def get_first_token(layout_block: LayoutBlock) -> Optional[LayoutToken]:
    return next(iter(layout_block.iter_all_tokens()), None)


def has_affiliation_content(
    semantic_affiliation_address: SemanticAffiliationAddress
) -> bool:
    """GROBID's Affiliation.isNotNull: anything but a marker, text labelled as
    other included (GROBID keeps it in the raw affiliation string)."""
    return any(
        not isinstance(semantic_content, SemanticMarker)
        for semantic_content in semantic_affiliation_address
    )


class AffiliationAddressSemanticExtractor(SimpleModelSemanticExtractor):
    def __init__(self):
        super().__init__(semantic_content_class_by_tag=SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG)

    def _iter_closed(
        self,
        semantic_affiliation_address: SemanticAffiliationAddress
    ) -> Iterator[SemanticContentWrapper]:
        if has_affiliation_content(semantic_affiliation_address):
            yield semantic_affiliation_address
            return
        LOGGER.debug('dropping affiliation with a marker only: %r', semantic_affiliation_address)
        yield SemanticNote(
            layout_block=semantic_affiliation_address.merged_block,
            note_type=LONE_MARKER_NOTE_TYPE
        )

    def _get_affiliation_taking_over_content(
        self,
        closed: SemanticAffiliationAddress,
        marker_since_last_institution: bool,
        new_affiliation: Callable[[], SemanticAffiliationAddress]
    ) -> Optional[SemanticAffiliationAddress]:
        """Where a marker followed the last institution and another marker now
        closes the affiliation, GROBID gives the new affiliation the content of the
        one closed, markers aside: one institution carrying two markers."""
        if not (
            has_affiliation_content(closed)
            and marker_since_last_institution
            and closed.has_type(SemanticInstitution)
        ):
            return None
        LOGGER.debug('second marker takes over the affiliation content')
        aff = new_affiliation()
        for semantic_content in closed:
            if not isinstance(semantic_content, SemanticMarker):
                aff.add_content(semantic_content)
        return aff

    def iter_semantic_content_for_entity_blocks(  # type: ignore  # noqa pylint: disable=arguments-differ, too-many-locals, too-many-branches, too-many-statements
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        segment_start_token_ids: AbstractSet[int] = frozenset(),
        **kwargs
    ) -> Iterable[SemanticContentWrapper]:
        """Cuts the labelled sequence into affiliations the way GROBID 0.9.1's
        AffiliationAddressParser.resultExtractionLayoutTokens does. A gap in the
        sequence (`segment_start_token_ids` names the tokens that open a block of the
        header another block sat before) closes the affiliation. A marker starts a
        new affiliation where the current one has a marker already or where the
        marker starts a line; where a marker followed the last institution, the new
        affiliation takes over the content of the one closed (one institution with
        two markers). A second institution starts a new affiliation once the current
        one has an address, starts a line, or follows a marker; a second department
        or laboratory does once the current one has an address or starts a line.
        Everything else, text labelled as other included, joins the current
        affiliation."""
        entity_tokens = list(entity_tokens)
        LOGGER.debug('entity_tokens: %s', entity_tokens)
        ids_iterator = iter(iter_ids('aff'))

        def new_affiliation() -> SemanticAffiliationAddress:
            return SemanticAffiliationAddress(content_id=next(ids_iterator, '?'))

        aff: Optional[SemanticAffiliationAddress] = None
        previous_layout_block: Optional[LayoutBlock] = None
        marker_since_last_institution = False
        for name, layout_block in entity_tokens:
            on_new_line = False
            first_token = get_first_token(layout_block)
            if (
                previous_layout_block is not None
                and first_token is not None
                and id(first_token) in segment_start_token_ids
                and aff is not None
                and has_affiliation_content(aff)
            ):
                LOGGER.debug('gap before %r closes the affiliation', layout_block.text)
                yield aff
                aff = None
                on_new_line = True
            if starts_new_line(previous_layout_block, layout_block):
                on_new_line = True
            previous_layout_block = layout_block
            if name == '<marker>':
                if aff is not None and (aff.has_type(SemanticMarker) or on_new_line):
                    yield from self._iter_closed(aff)
                    aff = self._get_affiliation_taking_over_content(
                        aff, marker_since_last_institution, new_affiliation
                    )
                if aff is None:
                    aff = new_affiliation()
                aff.add_content(SemanticMarker(layout_block=layout_block))
                marker_since_last_institution = True
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
            if aff is not None and isinstance(semantic_content, SemanticInstitution):
                if aff.has_type(SemanticInstitution) and (
                    aff.has_type(SemanticAddressField)
                    or on_new_line
                    or marker_since_last_institution
                ):
                    yield from self._iter_closed(aff)
                    aff = None
            elif (
                aff is not None
                and isinstance(semantic_content, (SemanticDepartment, SemanticLaboratory))
                and aff.has_type(type(semantic_content))
                and (aff.has_type(SemanticAddressField) or on_new_line)
            ):
                yield from self._iter_closed(aff)
                aff = None
            if aff is None:
                aff = new_affiliation()
            if (
                isinstance(semantic_content, SemanticCountry)
                and aff.has_type(SemanticCountry)
                and semantic_content.get_text() in aff.view_by_type(SemanticCountry).get_text_list()
            ):
                semantic_content = SemanticNote(
                    layout_block=cleaned_block,
                    note_type=REPEATED_COUNTRY_NOTE_TYPE
                )
            if prefix_block:
                aff.add_content(SemanticNote(layout_block=prefix_block, note_type=f'{name}-prefix'))
            aff.add_content(semantic_content)
            if suffix_block:
                aff.add_content(SemanticNote(layout_block=suffix_block, note_type=f'{name}-suffix'))
            if isinstance(semantic_content, SemanticInstitution):
                marker_since_last_institution = False
        if aff is not None:
            yield from self._iter_closed(aff)
