"""Merging the authors a header names more than once, the way GROBID does.

Based on GROBID 0.9.1's Person.deduplicate, which HeaderParser applies to the
header's authors once they are parsed:
https://github.com/kermitt2/grobid/blob/0.9.1/grobid-core/src/main/java/org/grobid/core/data/Person.java

Authors are grouped by their surname and the initial of their forename. Within a
group, an author whose forename or middle name clashes with another's ("James"
and "John") is left as it is, and the others are merged into the first of them:
it takes the most complete forename, middle name, title and suffix, and every
marker, so that the affiliations of the copies stay linked to it. The copies are
left as notes, so that their text is not lost.

Two departures from GROBID, both for what the name model gives here:

- a surname is compared without its whitespace, so that "Dal- Pizzol", a name
  broken at the end of a line, is "Dal-Pizzol";
- an author with no forename but a middle name is read as having the middle name
  for a forename: the name model labels a forename that follows a marker as a
  middle name ("1. Wanga<middlename> Zembe"), which GROBID would key without an
  initial and so never merge with "Wanga Zembe".
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type

from sciencebeam_parser.document.semantic_document import (
    SemanticAuthor,
    SemanticContentWrapper,
    SemanticGivenName,
    SemanticMarker,
    SemanticMiddleName,
    SemanticNamePart,
    SemanticNameSuffix,
    SemanticNameTitle,
    SemanticNote,
    SemanticSurname
)


LOGGER = logging.getLogger(__name__)

DUPLICATE_AUTHOR_NOTE_TYPE = 'raw_authors'

# Person.deduplicate drops these before comparing two forenames or middle names
NAME_COMPARISON_IGNORED_CHARACTERS_PATTERN = re.compile(r'[\-.\s]')

# Where a name part an author did not have goes: before the first of these
NAME_PART_TYPES_FOLLOWING: Dict[
    Type[SemanticNamePart], Tuple[Type[SemanticContentWrapper], ...]
] = {
    SemanticNameTitle: (SemanticGivenName, SemanticMiddleName, SemanticSurname),
    SemanticGivenName: (SemanticMiddleName, SemanticSurname),
    SemanticMiddleName: (SemanticSurname,),
    SemanticNameSuffix: (),
}


def _get_comparable_name_text(text: str) -> str:
    return NAME_COMPARISON_IGNORED_CHARACTERS_PATTERN.sub('', text.lower())


def _get_comparable_surname_text(text: str) -> str:
    return re.sub(r'\s', '', text.lower())


def _get_parts_text(parts: Sequence[SemanticNamePart]) -> str:
    return ' '.join(part.get_text() for part in parts)


@dataclass
class _AuthorNameView:
    """An author's name as Person.deduplicate reads it."""
    author: SemanticAuthor
    forename_parts: List[SemanticNamePart]
    middle_name_parts: List[SemanticNamePart]

    @staticmethod
    def for_author(author: SemanticAuthor) -> '_AuthorNameView':
        forename_parts: List[SemanticNamePart] = list(author.iter_by_type(SemanticGivenName))
        middle_name_parts: List[SemanticNamePart] = list(author.iter_by_type(SemanticMiddleName))
        if not forename_parts and middle_name_parts:
            forename_parts = middle_name_parts[:1]
            middle_name_parts = middle_name_parts[1:]
        return _AuthorNameView(
            author=author,
            forename_parts=forename_parts,
            middle_name_parts=middle_name_parts
        )

    @property
    def forename_text(self) -> str:
        return _get_parts_text(self.forename_parts)

    @property
    def middle_name_text(self) -> str:
        return _get_parts_text(self.middle_name_parts)

    @property
    def signature(self) -> Optional[Tuple[str, str]]:
        """Person.deduplicate's: the surname and the forename's initial; None
        without a surname, which leaves nothing to deduplicate on."""
        surname = _get_comparable_surname_text(self.author.surname_text)
        if not surname:
            return None
        forename = self.forename_text.strip()
        return surname, forename[:1].lower()


def _do_name_parts_clash(
    text: str,
    other_text: str,
    allow_abbreviated_names: bool
) -> bool:
    """Person.deduplicate's test: two names clash where both are given and
    differ, two initials by differing and anything longer by neither being the
    start of the other ("J" and "James" do not clash, "James" and "John" do)."""
    name = _get_comparable_name_text(text)
    other_name = _get_comparable_name_text(other_text)
    if not name or not other_name or name == other_name:
        return False
    if len(name) == 1 and len(other_name) == 1:
        return True
    if not allow_abbreviated_names:
        return True
    return not (name.startswith(other_name) or other_name.startswith(name))


def _do_names_clash(
    name: _AuthorNameView,
    other_name: _AuthorNameView,
    allow_abbreviated_names: bool
) -> bool:
    return _do_name_parts_clash(
        name.forename_text, other_name.forename_text, allow_abbreviated_names
    ) or _do_name_parts_clash(
        name.middle_name_text, other_name.middle_name_text, allow_abbreviated_names
    )


def is_same_author(
    author: SemanticAuthor,
    other_author: SemanticAuthor,
    allow_abbreviated_names: bool = True
) -> bool:
    """Whether the two would be merged, were they the only authors."""
    name = _AuthorNameView.for_author(author)
    other_name = _AuthorNameView.for_author(other_author)
    return (
        name.signature is not None
        and name.signature == other_name.signature
        and not _do_names_clash(name, other_name, allow_abbreviated_names)
    )


def _copy_name_part(
    part: SemanticNamePart,
    name_part_type: Type[SemanticNamePart]
) -> SemanticNamePart:
    copied = name_part_type(mixed_content=list(part.mixed_content))
    copied.value = part.value
    return copied


def _replace_name_parts(
    author: SemanticAuthor,
    old_parts: Sequence[SemanticNamePart],
    new_parts: Sequence[SemanticNamePart],
    name_part_type: Type[SemanticNamePart]
):
    """Puts copies of `new_parts` where `old_parts` were, or, where there were
    none, before the first name part that follows this type in a name."""
    replacement = [_copy_name_part(part, name_part_type) for part in new_parts]
    old_part_ids = {id(part) for part in old_parts}
    content = author.mixed_content
    positions = [index for index, item in enumerate(content) if id(item) in old_part_ids]
    if positions:
        insert_at = positions[0]
    else:
        following_types = NAME_PART_TYPES_FOLLOWING[name_part_type]
        insert_at = next(
            (
                index for index, item in enumerate(content)
                if isinstance(item, following_types)
            ),
            len(content)
        )
    remaining = [item for item in content if id(item) not in old_part_ids]
    insert_at -= sum(1 for position in positions if position < insert_at)
    author.mixed_content = remaining[:insert_at] + replacement + remaining[insert_at:]


def _take_longer_name_parts(
    author: SemanticAuthor,
    parts: Sequence[SemanticNamePart],
    other_parts: Sequence[SemanticNamePart],
    name_part_type: Type[SemanticNamePart]
) -> bool:
    if not other_parts:
        return False
    if parts and len(_get_parts_text(other_parts)) <= len(_get_parts_text(parts)):
        return False
    _replace_name_parts(author, parts, other_parts, name_part_type)
    return True


def _merge_author(author: SemanticAuthor, other_author: SemanticAuthor):
    """Enriches `author` with what `other_author` says more fully, as
    Person.deduplicate enriches the first person of a group."""
    name = _AuthorNameView.for_author(author)
    other_name = _AuthorNameView.for_author(other_author)
    _take_longer_name_parts(
        author, name.forename_parts, other_name.forename_parts, SemanticGivenName
    )
    _take_longer_name_parts(
        author, name.middle_name_parts, other_name.middle_name_parts, SemanticMiddleName
    )
    for name_part_type in (SemanticNameTitle, SemanticNameSuffix):
        _take_longer_name_parts(
            author,
            list(author.iter_by_type(name_part_type)),
            list(other_author.iter_by_type(name_part_type)),
            name_part_type
        )
    markers = set(author.view_by_type(SemanticMarker).get_text_list())
    for marker in other_author.iter_by_type(SemanticMarker):
        if marker.get_text() not in markers:
            author.add_content(marker)
            markers.add(marker.get_text())


def _remove_phantom_middle_initial(author: SemanticAuthor, had_short_forename: bool):
    """Person.deduplicate drops the middle initial a forename printed in capitals
    leaves once merged with its fuller form: "YI" is cut into "Y" and "I", and
    merged with "Yi" leaves "Yi I"."""
    name = _AuthorNameView.for_author(author)
    forename = _get_comparable_name_text(name.forename_text)
    middle_name = _get_comparable_name_text(name.middle_name_text)
    if (
        had_short_forename
        and len(middle_name) == 1
        and len(forename) == 2
        and forename.endswith(middle_name)
    ):
        _replace_name_parts(author, name.middle_name_parts, [], SemanticMiddleName)


def _merge_authors(authors: Sequence[SemanticAuthor]):
    author = authors[0]
    had_short_forename = any(
        len(_get_comparable_name_text(_AuthorNameView.for_author(a).forename_text)) == 1
        for a in authors
    )
    for other_author in authors[1:]:
        _merge_author(author, other_author)
    _remove_phantom_middle_initial(author, had_short_forename)


def get_deduplicated_authors(
    semantic_content_list: Sequence[SemanticContentWrapper],
    allow_abbreviated_names: bool = True
) -> List[SemanticContentWrapper]:
    """The content with each author it names more than once merged into its first
    mention, and the later mentions left as notes. Content other than authors
    is kept as it is."""
    names_by_signature: Dict[Tuple[str, str], List[Tuple[int, _AuthorNameView]]] = {}
    for index, semantic_content in enumerate(semantic_content_list):
        if not isinstance(semantic_content, SemanticAuthor):
            continue
        name = _AuthorNameView.for_author(semantic_content)
        signature = name.signature
        if signature is None:
            continue
        names_by_signature.setdefault(signature, []).append((index, name))
    replaced_by_index: Dict[int, SemanticContentWrapper] = {}
    for indexed_names in names_by_signature.values():
        if len(indexed_names) < 2:
            continue
        mergeable = [
            (index, name)
            for index, name in indexed_names
            if not any(
                _do_names_clash(name, other_name, allow_abbreviated_names)
                for other_index, other_name in indexed_names
                if other_index != index
            )
        ]
        if len(mergeable) < 2:
            continue
        LOGGER.debug(
            'merging duplicate authors: %r',
            [name.author.get_text() for _, name in mergeable]
        )
        _merge_authors([name.author for _, name in mergeable])
        for index, name in mergeable[1:]:
            replaced_by_index[index] = SemanticNote(
                layout_block=name.author.merged_block,
                note_type=DUPLICATE_AUTHOR_NOTE_TYPE
            )
    return [
        replaced_by_index.get(index, semantic_content)
        for index, semantic_content in enumerate(semantic_content_list)
    ]
