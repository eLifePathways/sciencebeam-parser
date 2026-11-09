import logging
import re
from typing import Iterable, List, Mapping, Optional, Tuple, Type, Union, cast

from sciencebeam_parser.document.semantic_document import (
    SemanticAuthor,
    SemanticContentWrapper,
    SemanticMarker,
    SemanticMiddleName,
    SemanticMixedContentWrapper,
    SemanticNameSuffix,
    SemanticNameTitle,
    SemanticNote,
    SemanticGivenName,
    SemanticSurname,
    T_SemanticContentFactory,
    T_SemanticName
)
from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutDocument, LayoutToken
from sciencebeam_parser.models.extract import SimpleModelSemanticExtractor


LOGGER = logging.getLogger(__name__)


SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG: Mapping[str, T_SemanticContentFactory] = {
    '<title>': SemanticNameTitle,
    '<forename>': SemanticGivenName,
    '<middlename>': SemanticMiddleName,
    '<surname>': SemanticSurname,
    '<suffix>': SemanticNameSuffix
}

# The name parts GROBID's Person.notNull looks at: an author exists once it has
# one of them (a suffix or a marker alone does not make one)
NAME_PART_TYPES = (SemanticNameTitle, SemanticGivenName, SemanticMiddleName, SemanticSurname)

# GROBID's AuthorParser starts a new author on a second one of these; a second
# middle name or suffix is appended to the current author instead
NAME_PART_TYPES_STARTING_NEW_AUTHOR = (SemanticNameTitle, SemanticGivenName, SemanticSurname)

# GROBID's Person.NAME_DELIMITERS: a name part is capitalised at its start and
# after any of these characters, and lower cased elsewhere
NAME_DELIMITERS = '-.,;:/_ '

# GROBID's TEI output leaves out an author whose surname is a single character
MIN_SURNAME_LENGTH = 2

INVALID_AUTHOR_NAME_NOTE_TYPE = 'invalid_author_name'


def tokenize_individual_characters(text: str) -> List[str]:
    return list(text)


def convert_two_letter_uppercase_given_name_to_given_middle_name(
    name: T_SemanticName
):
    given_names = list(name.iter_by_type(SemanticGivenName))
    middle_names = list(name.iter_by_type(SemanticMiddleName))
    if middle_names:
        LOGGER.debug('already has a middle name: %r', middle_names)
        return
    if len(given_names) != 1:
        LOGGER.debug('no or too many given names: %r', given_names)
        return
    given_name_text = given_names[0].get_text()
    if len(given_name_text) != 2 or not given_name_text.isupper():
        LOGGER.debug('not two uppercase characters: %r', given_name_text)
        return
    layout_document = LayoutDocument.for_blocks(list(given_names[0].iter_blocks()))
    retokenized_layout_document = layout_document.retokenize(
        tokenize_fn=tokenize_individual_characters
    )
    LOGGER.debug('retokenized_layout_document: %r', retokenized_layout_document)
    split_name_parts = [
        (
            SemanticGivenName(layout_block=LayoutBlock.for_tokens([token])) if index == 0
            else SemanticMiddleName(layout_block=LayoutBlock.for_tokens([token]))
        )
        for index, token in enumerate(retokenized_layout_document.iter_all_tokens())
    ]
    LOGGER.debug('split_name_parts: %r', split_name_parts)
    name.flat_map_inplace_by_type(
        SemanticGivenName,
        lambda _: split_name_parts
    )


def capitalize_fully(text: str, delimiters: str = NAME_DELIMITERS) -> str:
    """GROBID's TextUtilities.capitalizeFully: upper case at the start and after
    a delimiter, lower case elsewhere ("jean-pierre" is "Jean-Pierre", "McDONALD"
    is "Mcdonald", "o'neil" is "O'neil")."""
    result: List[str] = []
    to_upper = True
    for ch in text:
        if ch in delimiters:
            to_upper = True
            result.append(ch)
        elif to_upper:
            result.append(ch.upper())
            to_upper = False
        else:
            result.append(ch.lower())
    return ''.join(result)


def capitalize_name_parts(name: T_SemanticName):
    # GROBID's Person.normalizeName capitalises the forename, middle name and
    # surname; a title or a suffix is left as written
    for semantic_content in name:
        if isinstance(semantic_content, (SemanticGivenName, SemanticMiddleName, SemanticSurname)):
            semantic_content.value = capitalize_fully(semantic_content.get_text())


def is_valid_name(name: T_SemanticName) -> bool:
    # GROBID's Person.sanityCheck drops a person without a surname, and its TEI
    # output one whose surname is a single character
    return len(name.surname_text.strip()) >= MIN_SURNAME_LENGTH


# based on:
# https://github.com/kermitt2/grobid/blob/0.9.0/grobid-core/src/main/java/org/grobid/core/data/Person.java
# (normalizeName, sanityCheck) and the author block of BiblioItem.toTEIAuthorBlock
def normalize_name_parts(name: T_SemanticName):
    if not is_valid_name(name):
        return SemanticNote(
            layout_block=LayoutBlock.merge_blocks(name.iter_blocks()),
            note_type=INVALID_AUTHOR_NAME_NOTE_TYPE
        )
    convert_two_letter_uppercase_given_name_to_given_middle_name(name)
    capitalize_name_parts(name)
    return name


# GROBID 0.9.1's AuthorParser.MARKER_SPLIT_PATTERN: a marker is a run of letters
# or digits, a run of two or more commas, or a run of one repeated other
# character ("**", "††"); a single comma or whitespace between markers is not one
MARKER_PATTERN = re.compile(r'[0-9a-zA-Z]+|,{2,}|([^0-9a-zA-Z,\s])\1*')

MARKER_DELIMITER_NOTE_TYPE = 'marker_delimiter'


def _get_local_block(text: str) -> LayoutBlock:
    return LayoutBlock.for_tokens([LayoutToken(text, whitespace='')])


def iter_semantic_markers_for_layout_block(
    layout_block: LayoutBlock
) -> Iterable[Union[SemanticMarker, SemanticContentWrapper]]:
    """The markers of a marker cluster, cut as GROBID's AuthorParser.splitMarkers
    cuts them; the text between them, which GROBID drops, is kept as notes."""
    text = layout_block.text
    last_end = 0
    for match in MARKER_PATTERN.finditer(text):
        if match.start() > last_end:
            yield SemanticNote(
                layout_block=_get_local_block(text[last_end:match.start()]),
                note_type=MARKER_DELIMITER_NOTE_TYPE
            )
        yield SemanticMarker(layout_block=_get_local_block(match.group()))
        last_end = match.end()
    if last_end < len(text):
        yield SemanticNote(
            layout_block=_get_local_block(text[last_end:]),
            note_type=MARKER_DELIMITER_NOTE_TYPE
        )


def append_semantic_markers_for_layout_block(
    parent_semantic_content: SemanticMixedContentWrapper,
    layout_block: LayoutBlock
) -> None:
    semantic_markers = list(iter_semantic_markers_for_layout_block(layout_block))
    for semantic_marker in semantic_markers:
        parent_semantic_content.add_content(semantic_marker)


def has_name_part(name: SemanticMixedContentWrapper) -> bool:
    """GROBID's Person.notNull: the author has a title, forename, middle name or surname."""
    return any(isinstance(content, NAME_PART_TYPES) for content in name)


class NameSemanticExtractor(SimpleModelSemanticExtractor):
    def __init__(self):
        super().__init__(semantic_content_class_by_tag=SIMPLE_SEMANTIC_CONTENT_CLASS_BY_TAG)

    def iter_semantic_content_for_entity_blocks(  # type: ignore  # pylint: disable=arguments-differ
        self,
        entity_tokens: Iterable[Tuple[str, LayoutBlock]],
        name_type: Optional[Type[T_SemanticName]] = None,
        **kwargs
    ) -> Iterable[T_SemanticName]:
        """Cuts the labelled sequence into names the way GROBID's
        AuthorParser.processing does: a marker closes the author it follows and is
        recorded on that author (GROBID notes a marker usually belongs to the author
        before it, rarely to the one after); a second title, forename or surname
        starts a new author, where middle names and suffixes accumulate on the
        current one. Text labelled as other stays with the author it falls in."""
        _name_type: Type[T_SemanticName] = cast(
            Type[T_SemanticName],
            name_type if name_type is not None else SemanticAuthor
        )
        entity_tokens = list(entity_tokens)
        LOGGER.debug('entity_tokens: %s', entity_tokens)
        semantic_name: Optional[T_SemanticName] = None
        for name, layout_block in entity_tokens:
            if not layout_block.text.strip():
                continue
            if name == '<marker>':
                if semantic_name is None:
                    semantic_name = _name_type()
                append_semantic_markers_for_layout_block(semantic_name, layout_block)
                if has_name_part(semantic_name):
                    LOGGER.debug('marker closes the author it follows')
                    yield normalize_name_parts(semantic_name)
                    semantic_name = None
                continue
            semantic_content = self.get_semantic_content_for_entity_name(
                name, layout_block
            )
            if (
                semantic_name is not None
                and isinstance(semantic_content, NAME_PART_TYPES_STARTING_NEW_AUTHOR)
                and semantic_name.has_type(type(semantic_content))
            ):
                LOGGER.debug('starting new author on a second name part, name=%r', name)
                yield normalize_name_parts(semantic_name)
                semantic_name = None
            if semantic_name is None:
                semantic_name = _name_type()
            semantic_name.add_content(semantic_content)
        if semantic_name is not None:
            yield normalize_name_parts(semantic_name)
