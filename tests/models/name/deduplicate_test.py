from typing import List, Optional, Sequence

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.semantic_document import (
    SemanticAuthor,
    SemanticContentWrapper,
    SemanticGivenName,
    SemanticMarker,
    SemanticMiddleName,
    SemanticNameTitle,
    SemanticNote,
    SemanticSurname
)
from sciencebeam_parser.models.name.deduplicate import (
    DUPLICATE_AUTHOR_NOTE_TYPE,
    get_deduplicated_authors,
    is_same_author
)


def _author(
    given_name: str = '',
    surname: str = '',
    middle_name: str = '',
    markers: Sequence[str] = (),
    title: str = ''
) -> SemanticAuthor:
    author = SemanticAuthor()
    for marker in markers:
        author.add_content(SemanticMarker(layout_block=LayoutBlock.for_text(marker)))
    if title:
        author.add_content(SemanticNameTitle(layout_block=LayoutBlock.for_text(title)))
    if given_name:
        author.add_content(SemanticGivenName(layout_block=LayoutBlock.for_text(given_name)))
    if middle_name:
        author.add_content(SemanticMiddleName(layout_block=LayoutBlock.for_text(middle_name)))
    if surname:
        author.add_content(SemanticSurname(layout_block=LayoutBlock.for_text(surname)))
    return author


def _authors(content: Sequence[SemanticContentWrapper]) -> List[SemanticAuthor]:
    return [item for item in content if isinstance(item, SemanticAuthor)]


def _middle_name_text(author: SemanticAuthor) -> Optional[str]:
    return author.get_text_by_type(SemanticMiddleName) or None


class TestIsSameAuthor:
    def test_should_match_the_same_full_name(self):
        assert is_same_author(_author('Richard', 'Coward'), _author('Richard', 'Coward'))

    def test_should_not_match_different_surnames(self):
        assert not is_same_author(_author('Richard', 'Coward'), _author('Richard', 'Smith'))

    def test_should_not_match_different_forenames_with_the_same_initial(self):
        assert not is_same_author(_author('James', 'Smith'), _author('John', 'Smith'))

    def test_should_match_an_initial_with_the_forename_it_abbreviates(self):
        assert is_same_author(_author('R', 'Coward'), _author('Richard', 'Coward'))

    def test_should_not_match_an_initial_without_abbreviated_names(self):
        assert not is_same_author(
            _author('R', 'Coward'),
            _author('Richard', 'Coward'),
            allow_abbreviated_names=False
        )

    def test_should_match_regardless_of_case(self):
        assert is_same_author(_author('richard', 'coward'), _author('Richard', 'Coward'))

    def test_should_match_two_authors_with_a_surname_only(self):
        assert is_same_author(_author('', 'Basit'), _author('', 'Basit'))

    def test_should_not_match_a_surname_only_with_a_full_name(self):
        assert not is_same_author(_author('', 'Coward'), _author('Richard', 'Coward'))

    def test_should_not_match_without_a_surname(self):
        assert not is_same_author(_author('Richard', ''), _author('Richard', 'Coward'))

    def test_should_not_match_two_different_initials(self):
        assert not is_same_author(_author('A', 'Das'), _author('B', 'Das'))

    def test_should_not_match_different_middle_names(self):
        assert not is_same_author(
            _author('John', 'Smith', middle_name='Paul'),
            _author('John', 'Smith', middle_name='Peter')
        )

    def test_should_ignore_hyphens_and_dots_in_forenames(self):
        assert is_same_author(_author('Jean-Pierre', 'Dupont'), _author('Jean Pierre', 'Dupont'))

    def test_should_match_a_surname_broken_at_the_end_of_a_line(self):
        assert is_same_author(_author('Felipe', 'Dal- Pizzol'), _author('Felipe', 'Dal-Pizzol'))

    def test_should_read_a_lone_middle_name_as_the_forename(self):
        # the name model labels a forename after a marker as a middle name
        assert is_same_author(
            _author('Wanga', 'Zembe'),
            _author(middle_name='Wanga', surname='Zembe', markers=['1'])
        )


class TestGetDeduplicatedAuthors:
    def test_should_leave_distinct_authors_as_they_are(self):
        content = [_author('Alice', 'Smith'), _author('Bob', 'Jones')]
        assert get_deduplicated_authors(content) == content

    def test_should_keep_the_first_mention_and_note_the_others(self):
        first = _author('Richard', 'Coward')
        other = _author('Bob', 'Jones')
        duplicate = _author('Richard', 'Coward')
        result = get_deduplicated_authors([first, other, duplicate])
        assert result[0] is first
        assert result[1] is other
        assert isinstance(result[2], SemanticNote)
        assert result[2].note_type == DUPLICATE_AUTHOR_NOTE_TYPE
        assert result[2].get_text() == 'Richard Coward'

    def test_should_keep_content_other_than_authors(self):
        note = SemanticNote(layout_block=LayoutBlock.for_text(','), note_type='other')
        result = get_deduplicated_authors([_author('Alice', 'Smith'), note])
        assert result[1] is note

    def test_should_take_the_fuller_forename_of_a_later_mention(self):
        result = get_deduplicated_authors([
            _author('T', 'Doherty'),
            _author(middle_name='Tanya', surname='Doherty', markers=['2'])
        ])
        authors = _authors(result)
        assert len(authors) == 1
        assert authors[0].given_name_text == 'Tanya'
        assert authors[0].surname_text == 'Doherty'
        assert not _middle_name_text(authors[0])

    def test_should_take_a_middle_name_the_first_mention_lacks(self):
        result = get_deduplicated_authors([
            _author('John', 'Smith'),
            _author('John', 'Smith', middle_name='Paul')
        ])
        authors = _authors(result)
        assert len(authors) == 1
        assert _middle_name_text(authors[0]) == 'Paul'
        assert [type(part).__name__ for part in authors[0]] == [
            'SemanticGivenName', 'SemanticMiddleName', 'SemanticSurname'
        ]

    def test_should_take_a_title_the_first_mention_lacks(self):
        result = get_deduplicated_authors([
            _author('John', 'Smith'),
            _author('John', 'Smith', title='Dr')
        ])
        authors = _authors(result)
        assert authors[0].get_text_by_type(SemanticNameTitle) == 'Dr'

    def test_should_carry_the_markers_of_every_mention(self):
        # the markers link an author to their affiliations
        result = get_deduplicated_authors([
            _author('Wanga', 'Zembe', markers=['1']),
            _author(middle_name='Wanga', surname='Zembe', markers=['1', '3'])
        ])
        authors = _authors(result)
        assert len(authors) == 1
        assert authors[0].view_by_type(SemanticMarker).get_text_list() == ['1', '3']

    def test_should_not_merge_a_group_whose_forenames_clash(self):
        # Person.deduplicate: "J" matches both, but "James" and "John" clash, so
        # neither is merged and "J" has nothing left to merge with
        content = [_author('J', 'Smith'), _author('James', 'Smith'), _author('John', 'Smith')]
        assert get_deduplicated_authors(content) == content

    def test_should_merge_the_mentions_that_clash_with_none_of_the_others(self):
        content = [
            _author('James', 'Smith'),
            _author('James', 'Smith'),
            _author('A', 'Smith')
        ]
        result = get_deduplicated_authors(content)
        assert [
            author.given_name_text for author in _authors(result)
        ] == ['James', 'A']

    def test_should_drop_the_phantom_middle_initial_of_a_forename_in_capitals(self):
        # "YI" is cut into "Y" and "I"; merged with "Yi" it would read "Yi I"
        result = get_deduplicated_authors([
            _author('Y', 'Wang', middle_name='I'),
            _author('Yi', 'Wang')
        ])
        authors = _authors(result)
        assert len(authors) == 1
        assert authors[0].given_name_text == 'Yi'
        assert not _middle_name_text(authors[0])

    def test_should_keep_a_real_middle_initial(self):
        result = get_deduplicated_authors([
            _author('Anna', 'Smith', middle_name='A'),
            _author('A', 'Smith')
        ])
        authors = _authors(result)
        assert len(authors) == 1
        assert authors[0].given_name_text == 'Anna'
        assert _middle_name_text(authors[0]) == 'A'

    def test_should_not_merge_initials_with_forenames_without_abbreviated_names(self):
        content = [_author('R', 'Coward'), _author('Richard', 'Coward')]
        assert get_deduplicated_authors(content, allow_abbreviated_names=False) == content
