from typing import List, Optional

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutPageCoordinates,
    LayoutToken
)
from sciencebeam_parser.document.semantic_document import (
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticGivenName,
    SemanticInstitution,
    SemanticMarker,
    SemanticNote,
    SemanticSurname
)
from sciencebeam_parser.processors.fulltext.attach_affiliations import (
    assign,
    assign_by_sequence,
    attach_affiliations_for_front,
    compute_centroid,
    find_author_index_nearest_to_marker,
    get_affiliation_marker_text,
    get_surname_positions,
    is_content_duplicate,
    is_marker_within_longer_run,
    match_persons_by_cluster,
    normalize_for_match,
    pre_link_by_preceding_author_cluster,
    rescue_orphan_affiliations,
    split_by_name_separators
)


def _block(
    text: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    page: int = 1
) -> LayoutBlock:
    if x is None or y is None:
        return LayoutBlock.for_text(text)
    return LayoutBlock.for_text(
        text, coordinates=LayoutPageCoordinates(x=x, y=y, width=10, height=10, page_number=page)
    )


def _author(
    surname: str,
    given_name: str = '',
    markers: str = '',
    x: Optional[float] = None,
    y: Optional[float] = None
) -> SemanticAuthor:
    author = SemanticAuthor()
    if given_name:
        author.add_content(SemanticGivenName(layout_block=_block(given_name, x, y)))
    author.add_content(SemanticSurname(layout_block=_block(surname, x, y)))
    for marker in markers.split():
        author.add_content(SemanticMarker(layout_block=LayoutBlock.for_text(marker)))
    return author


def _affiliation(  # pylint: disable=too-many-arguments
    institution: str,
    marker: str = '',
    x: Optional[float] = None,
    y: Optional[float] = None,
    page: int = 1,
    other_text: str = ''
) -> SemanticAffiliationAddress:
    affiliation = SemanticAffiliationAddress(content_id=institution)
    if marker:
        affiliation.add_content(SemanticMarker(layout_block=LayoutBlock.for_text(marker)))
    affiliation.add_content(SemanticInstitution(layout_block=_block(institution, x, y, page)))
    if other_text:
        affiliation.add_content(SemanticNote(layout_block=LayoutBlock.for_text(other_text)))
    return affiliation


def _attached(authors: List[SemanticAuthor]) -> List[List[str]]:
    return [[aff.content_id or '' for aff in author.affiliations] for author in authors]


class TestGetAffiliationMarkerText:
    def test_should_keep_several_markers_as_one_string_as_grobid_does(self):
        assert get_affiliation_marker_text(_affiliation('Inst', marker='1, 2')) == '1, 2'

    def test_should_return_empty_string_without_marker(self):
        assert get_affiliation_marker_text(_affiliation('Inst')) == ''


class TestIsMarkerWithinLongerRun:
    def test_should_reject_a_digit_within_a_longer_number(self):
        assert is_marker_within_longer_run('Smith 12', '1', 6)
        assert is_marker_within_longer_run('Smith 21', '1', 7)
        assert not is_marker_within_longer_run('Smith 1,2', '1', 6)

    def test_should_reject_a_letter_within_a_word(self):
        assert is_marker_within_longer_run('Smith ab', 'a', 6)
        assert not is_marker_within_longer_run('Smith a,b', 'a', 6)

    def test_should_reject_an_asterisk_within_a_longer_run(self):
        assert is_marker_within_longer_run('Smith **', '*', 6)
        assert is_marker_within_longer_run('Smith ***', '**', 6)
        assert not is_marker_within_longer_run('Smith **', '**', 6)

    def test_should_check_the_character_right_before_a_marker_at_the_second_position(self):
        assert is_marker_within_longer_run('12 Smith', '2', 1)

    def test_should_accept_a_longer_marker_wherever_it_is(self):
        assert not is_marker_within_longer_run('Smith 112', '11', 6)


class TestGetSurnamePositions:
    def test_should_find_each_surname_after_the_one_before(self):
        text = 'l. wang 1 , j. wang 2'
        assert get_surname_positions([_author('Wang'), _author('Wang')], text) == [3, 15]

    def test_should_mark_a_surname_not_found(self):
        assert get_surname_positions([_author('Other'), _author('Smith')], 'john smith') == [-1, 5]


class TestFindAuthorIndexNearestToMarker:
    def test_should_pick_the_author_whose_surname_ends_nearest(self):
        text = 'John Smith 1 , Maria Madison 2'
        authors = [_author('Smith', 'John'), _author('Madison', 'Maria')]
        positions = get_surname_positions(authors, text.lower())
        assert find_author_index_nearest_to_marker(authors, positions, text.index('2'), set()) == 1
        assert find_author_index_nearest_to_marker(authors, positions, text.index('1'), set()) == 0

    def test_should_skip_an_author_already_given_the_affiliation(self):
        text = 'John Smith 1 , Maria Madison 1'
        authors = [_author('Smith', 'John'), _author('Madison', 'Maria')]
        positions = get_surname_positions(authors, text.lower())
        assert find_author_index_nearest_to_marker(authors, positions, text.index('1'), {0}) == 1

    def test_should_leave_out_a_surname_missing_from_the_text(self):
        text = 'John Smith 1'
        authors = [_author('Other'), _author('Smith', 'John')]
        positions = get_surname_positions(authors, text.lower())
        assert find_author_index_nearest_to_marker(authors, positions, text.index('1'), set()) == 1
        assert find_author_index_nearest_to_marker([_author('Other')], [-1], 11, set()) is None


class TestAssign:
    def test_should_give_a_lone_affiliation_to_every_author(self):
        authors = [_author('Smith'), _author('Madison')]
        assign(authors, [_affiliation('Inst1')], 'Smith Madison')
        assert _attached(authors) == [['Inst1'], ['Inst1']]

    def test_should_give_a_lone_author_every_affiliation(self):
        authors = [_author('Smith')]
        assign(authors, [_affiliation('Inst1', '1'), _affiliation('Inst2', '2')], 'Smith 1,2')
        assert _attached(authors) == [['Inst1', 'Inst2']]

    def test_should_attach_by_the_markers_recorded_on_the_authors_first(self):
        # the leading markers are recorded on the author before them, which the
        # direct marker matching reads as they are
        authors = [_author('Smith', 'John', markers='1'), _author('Madison', 'Maria', markers='2')]
        assign(
            authors,
            [_affiliation('Inst1', '1'), _affiliation('Inst2', '2')],
            'John Smith 1 , Maria Madison 2'
        )
        assert _attached(authors) == [['Inst1'], ['Inst2']]

    def test_should_attach_by_the_position_of_the_marker_in_the_author_text(self):
        # no author carries the marker the name model missed, so the text decides
        authors = [_author('Smith', 'John'), _author('Madison', 'Maria')]
        assign(
            authors,
            [_affiliation('Inst1', '1'), _affiliation('Inst2', '2')],
            'John Smith 2 , Maria Madison 1'
        )
        assert _attached(authors) == [['Inst2'], ['Inst1']]

    def test_should_attach_an_affiliation_to_each_author_carrying_its_marker(self):
        authors = [_author('Smith', 'John'), _author('Madison', 'Maria'), _author('Lee', 'Kim')]
        assign(
            authors,
            [_affiliation('Inst1', '1'), _affiliation('Inst2', '2')],
            'John Smith 1,2 , Maria Madison 1 , Kim Lee 2'
        )
        assert _attached(authors) == [['Inst1', 'Inst2'], ['Inst1'], ['Inst2']]

    def test_should_not_take_a_digit_of_a_longer_number_for_a_marker(self):
        authors = [_author('Smith', 'John'), _author('Madison', 'Maria')]
        assign(
            authors,
            [_affiliation('Inst1', '1'), _affiliation('Inst2', '12')],
            'John Smith 12 , Maria Madison 1'
        )
        assert _attached(authors) == [['Inst2'], ['Inst1']]

    def test_should_not_attach_the_same_affiliation_to_an_author_twice(self):
        authors = [_author('Smith', 'John', markers='1'), _author('Madison', 'Maria', markers='2')]
        assign(
            authors,
            [_affiliation('Inst1', '1'), _affiliation('Inst2', '2')],
            'John Smith 1 , Maria Madison 2'
        )
        assert _attached(authors) == [['Inst1'], ['Inst2']]

    def test_should_give_a_floating_author_the_nearest_unattached_affiliation_on_the_page(self):
        authors = [
            _author('Smith', 'John', markers='1', x=10, y=10),
            _author('Madison', 'Maria', x=200, y=10)
        ]
        assign(
            authors,
            [
                _affiliation('Inst1', '1', x=10, y=50),
                _affiliation('Inst2', x=200, y=50),
                _affiliation('Inst3', x=200, y=400)
            ],
            'John Smith 1 , Maria Madison'
        )
        # the affiliation left over is then rescued onto the author nearest to it
        assert _attached(authors) == [['Inst1'], ['Inst2', 'Inst3']]

    def test_should_prefer_an_affiliation_on_the_same_page(self):
        authors = [
            _author('Smith', 'John', markers='1', x=10, y=10),
            _author('Madison', x=10, y=20)
        ]
        assign(
            authors,
            [
                _affiliation('Inst1', '1', x=10, y=50),
                _affiliation('Inst2', x=10, y=25, page=2),
                _affiliation('Inst3', x=500, y=700)
            ],
            'John Smith 1 , Madison'
        )
        assert _attached(authors)[1][0] == 'Inst3'

    def test_should_deal_the_remaining_affiliations_out_in_sequence_without_positions(self):
        authors = [_author('Smith'), _author('Madison')]
        assign(authors, [_affiliation('Inst1'), _affiliation('Inst2')], 'Smith Madison')
        assert _attached(authors) == [['Inst1'], ['Inst2']]

    def test_should_do_nothing_without_authors_or_affiliations(self):
        authors = [_author('Smith')]
        assign(authors, [], 'Smith')
        assign([], [_affiliation('Inst1')], '')
        assert _attached(authors) == [[]]


class TestAssignBySequence:
    def test_should_give_every_floating_author_all_affiliations_where_there_are_fewer(self):
        authors = [_author('Smith'), _author('Madison'), _author('Lee')]
        assign_by_sequence(authors, [_affiliation('Inst1'), _affiliation('Inst2')])
        assert _attached(authors) == [['Inst1', 'Inst2']] * 3

    def test_should_pair_them_in_order_otherwise(self):
        authors = [_author('Smith'), _author('Madison')]
        assign_by_sequence(
            authors, [_affiliation('Inst1'), _affiliation('Inst2'), _affiliation('Inst3')]
        )
        assert _attached(authors) == [['Inst1'], ['Inst2']]


class TestRescueOrphanAffiliations:
    def test_should_give_an_orphan_to_the_nearest_author(self):
        authors = [_author('Smith', x=10, y=10), _author('Madison', x=300, y=10)]
        orphan = _affiliation('Inst2', x=290, y=40)
        authors[0].affiliations.append(_affiliation('Inst1'))
        authors[1].affiliations.append(_affiliation('Inst1'))
        rescue_orphan_affiliations(authors, [authors[0].affiliations[0], orphan])
        assert _attached(authors) == [['Inst1'], ['Inst1', 'Inst2']]

    def test_should_give_an_orphan_to_the_author_with_the_fewest_affiliations(self):
        authors = [_author('Smith'), _author('Madison')]
        authors[0].affiliations.append(_affiliation('Inst1'))
        orphan = _affiliation('Inst2')
        rescue_orphan_affiliations(authors, [authors[0].affiliations[0], orphan])
        assert _attached(authors) == [['Inst1'], ['Inst2']]

    def test_should_skip_an_orphan_saying_the_same_as_one_the_author_has(self):
        authors = [_author('Smith')]
        authors[0].affiliations.append(_affiliation('Dept of Physics, MIT', '1'))
        orphan = _affiliation('Dept of Physics, MIT')
        rescue_orphan_affiliations(authors, [authors[0].affiliations[0], orphan])
        assert _attached(authors) == [['Dept of Physics, MIT']]


class TestNormalizeForMatch:
    def test_should_lower_case_and_drop_parentheticals_markers_and_punctuation(self):
        assert normalize_for_match('1 Dept. of Physics (C2SM), MIT') == 'dept of physics mit'
        assert normalize_for_match('† Dept of Physics') == 'dept of physics'


class TestIsContentDuplicate:
    def test_should_take_a_contained_text_of_close_length_as_the_same(self):
        assert is_content_duplicate(
            _affiliation('Dept of Physics MIT'), [_affiliation('1 Dept of Physics, MIT.')]
        )

    def test_should_not_take_a_much_shorter_text_as_the_same(self):
        assert not is_content_duplicate(
            _affiliation('Dept of Physics'), [_affiliation('Dept of Physics and Astronomy, MIT')]
        )


class TestComputeCentroid:
    def test_should_average_the_positions_and_keep_the_first_page(self):
        tokens = [
            LayoutToken('a', coordinates=LayoutPageCoordinates(0, 10, 1, 1, page_number=2)),
            LayoutToken('b', coordinates=LayoutPageCoordinates(20, 30, 1, 1, page_number=3)),
            LayoutToken('c')
        ]
        assert compute_centroid(tokens) == (10.0, 20.0, 2)

    def test_should_return_none_without_positions(self):
        assert compute_centroid([LayoutToken('a')]) is None


class TestSplitByNameSeparators:
    def test_should_cut_at_a_comma_or_an_and(self):
        spans = split_by_name_separators(LayoutBlock.for_text('A. Smith, B. Jones and C. Lee'))
        assert [LayoutBlock.for_tokens(span).text for span in spans] == [
            'A. Smith', 'B. Jones', 'C. Lee'
        ]


class TestMatchPersonsByCluster:
    def test_should_match_the_author_whose_surname_runs_through_the_span(self):
        authors = [_author('Smith', 'John'), _author('Lin', 'Kim'), _author('Li', 'Wei')]
        tokens = list(LayoutBlock.for_text('Kim Lin').iter_all_tokens())
        assert match_persons_by_cluster(tokens, authors) == [authors[1]]

    def test_should_match_several_authors_named_without_a_separator_by_their_initials(self):
        authors = [_author('Atkinson', 'Mark'), _author('van Willigenburg', 'Stephanie')]
        tokens = list(LayoutBlock.for_text('M.D. Atkinson S.J. van Willigenburg').iter_all_tokens())
        assert match_persons_by_cluster(tokens, authors) == authors

    def test_should_tell_authors_sharing_a_surname_apart_by_the_leading_initial(self):
        authors = [_author('Wang', 'Li'), _author('Wang', 'Jun')]
        tokens = list(LayoutBlock.for_text('J. Wang').iter_all_tokens())
        assert match_persons_by_cluster(tokens, authors) == [authors[1]]

    def test_should_match_nobody_without_a_capitalised_word(self):
        assert match_persons_by_cluster(
            list(LayoutBlock.for_text('smith').iter_all_tokens()), [_author('Smith')]
        ) == []


class TestPreLinkByPrecedingAuthorCluster:
    def test_should_link_an_affiliation_to_the_authors_of_the_blocks_before_it(self):
        authors = [_author('Smith', 'John'), _author('Madison', 'Maria'), _author('Lee', 'Kim')]
        aff_1 = _affiliation('Inst1')
        aff_2 = _affiliation('Inst2')
        pre_link_by_preceding_author_cluster(authors, [aff_1, aff_2], [
            ('<author>', LayoutBlock.for_text('John Smith')),
            ('<author>', LayoutBlock.for_text('Maria Madison')),
            ('<affiliation>', aff_1.merged_block),
            ('other', LayoutBlock.for_text('Footnote')),
            ('<author>', LayoutBlock.for_text('Kim Lee')),
            ('<affiliation>', aff_2.merged_block)
        ])
        assert _attached(authors) == [['Inst1'], ['Inst1'], ['Inst2']]

    def test_should_stop_walking_back_at_a_cluster_that_is_not_an_author(self):
        authors = [_author('Smith', 'John')]
        aff_1 = _affiliation('Inst1')
        pre_link_by_preceding_author_cluster(authors, [aff_1], [
            ('<author>', LayoutBlock.for_text('John Smith')),
            ('other', LayoutBlock.for_text('Footnote')),
            ('<affiliation>', aff_1.merged_block)
        ])
        assert _attached(authors) == [[]]

    def test_should_do_nothing_where_any_marker_exists(self):
        authors = [_author('Smith', 'John', markers='1')]
        aff_1 = _affiliation('Inst1')
        pre_link_by_preceding_author_cluster(authors, [aff_1], [
            ('<author>', LayoutBlock.for_text('John Smith')),
            ('<affiliation>', aff_1.merged_block)
        ])
        assert _attached(authors) == [[]]


class TestAttachAffiliationsForFront:
    def test_should_attach_by_markers_before_the_pre_link(self):
        # with markers the pre-link stands aside, even where the header's blocks
        # would pair the affiliations the other way round
        authors = [_author('Smith', 'John', markers='2'), _author('Madison', 'Maria', markers='1')]
        aff_1 = _affiliation('Inst1', '1')
        aff_2 = _affiliation('Inst2', '2')
        attach_affiliations_for_front(authors, [aff_1, aff_2], 'John Smith 2\tMaria Madison 1', [
            ('<author>', LayoutBlock.for_text('John Smith')),
            ('<affiliation>', aff_1.merged_block),
            ('<author>', LayoutBlock.for_text('Maria Madison')),
            ('<affiliation>', aff_2.merged_block)
        ])
        assert _attached(authors) == [['Inst2'], ['Inst1']]

    def test_should_pre_link_without_markers(self):
        authors = [_author('Smith', 'John'), _author('Madison', 'Maria')]
        aff_1 = _affiliation('Inst1')
        aff_2 = _affiliation('Inst2')
        attach_affiliations_for_front(authors, [aff_1, aff_2], 'John Smith\tMaria Madison', [
            ('<author>', LayoutBlock.for_text('John Smith')),
            ('<affiliation>', aff_1.merged_block),
            ('<author>', LayoutBlock.for_text('Maria Madison')),
            ('<affiliation>', aff_2.merged_block)
        ])
        assert _attached(authors) == [['Inst1'], ['Inst2']]
