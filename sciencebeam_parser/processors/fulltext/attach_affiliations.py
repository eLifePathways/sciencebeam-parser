"""Attaching the affiliations of a document's front to its authors, as GROBID does.

A port of GROBID 0.9.1's AuthorAffiliationAssigner, with the markers first. In
priority order: an affiliation goes to the authors carrying its marker; an
affiliation's marker is looked for in the text of the author blocks and the
affiliation goes to the author whose surname ends nearest each occurrence; with no
marker anywhere, an affiliation goes to the authors of the author blocks right before
its block in the header (pre-link); a lone affiliation goes to every author and a lone
author takes every affiliation; an author still without one takes the nearest
unattached affiliation on the page; failing coordinates, the remaining affiliations
are dealt out in sequence; and an affiliation still attached to nothing is rescued
onto the nearest author, unless that author already has one saying the same.

GROBID's HeaderParser runs the pre-link before the rest. Running it after the markers
links the same: the pre-link does nothing where there is any marker, and the marker
steps do nothing where there is none.

An affiliation is attached to an author once at most (GROBID's Person.addAffiliation).
"""
import logging
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sciencebeam_parser.document.layout_document import LayoutBlock, LayoutToken
from sciencebeam_parser.document.semantic_document import (
    SemanticAffiliationAddress,
    SemanticAuthor,
    SemanticMarker,
    SemanticMixedContentWrapper,
    SemanticNamePart,
    SemanticNote,
    SemanticRawAffiliation,
    SemanticRawAffiliationAddress,
    SemanticRawAuthors
)


LOGGER = logging.getLogger(__name__)


# the labels of the header clusters the pre-link reads
HEADER_AUTHOR_LABEL = '<author>'
HEADER_AFFILIATION_LABEL = '<affiliation>'
HEADER_ADDRESS_LABEL = '<address>'
HEADER_OTHER_LABEL = 'other'

# a header cluster: its label and its tokens, in document order
HeaderCluster = Tuple[str, LayoutBlock]

# GROBID's circuit breaker on the affiliations scanned for markers
MAX_AFFILIATIONS_SCANNED_FOR_MARKERS = 60

# GROBID's penalty on the distance between two centroids on different pages
OTHER_PAGE_DISTANCE_PENALTY = 10000.0

# GROBID's minimum shorter/longer length ratio for one affiliation text contained in
# another to count as the same affiliation, rather than a narrower one
CONTENT_DUPLICATE_MIN_RATIO = 0.8

# a centroid: x, y and the page number
Centroid = Tuple[float, float, int]


def get_header_clusters(semantic_front: SemanticMixedContentWrapper) -> List[HeaderCluster]:
    """The header's labelled clusters in document order, before the raw author and
    affiliation blocks of the front are parsed: the author blocks, the affiliation
    and address blocks, and anything else in between."""
    header_clusters: List[HeaderCluster] = []
    for semantic_content in semantic_front:
        if isinstance(semantic_content, SemanticRawAuthors):
            header_clusters.append((HEADER_AUTHOR_LABEL, semantic_content.merged_block))
            continue
        if isinstance(semantic_content, SemanticRawAffiliationAddress):
            for raw_content in semantic_content:
                label = (
                    HEADER_AFFILIATION_LABEL if isinstance(raw_content, SemanticRawAffiliation)
                    else HEADER_ADDRESS_LABEL
                )
                header_clusters.append((label, raw_content.merged_block))
            continue
        header_clusters.append((HEADER_OTHER_LABEL, semantic_content.merged_block))
    return header_clusters


def add_affiliation(
    author: SemanticAuthor,
    affiliation: SemanticAffiliationAddress
) -> None:
    if any(existing is affiliation for existing in author.affiliations):
        return
    author.affiliations.append(affiliation)


def is_attached(
    affiliation: SemanticAffiliationAddress,
    authors: Sequence[SemanticAuthor]
) -> bool:
    return any(
        existing is affiliation
        for author in authors
        for existing in author.affiliations
    )


def get_floating_authors(authors: Sequence[SemanticAuthor]) -> List[SemanticAuthor]:
    return [author for author in authors if not author.affiliations]


def get_floating_affiliations(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress]
) -> List[SemanticAffiliationAddress]:
    return [
        affiliation for affiliation in affiliations
        if not is_attached(affiliation, authors)
    ]


def get_affiliation_marker_text(
    semantic_affiliation_address: SemanticAffiliationAddress
) -> str:
    """The marker as GROBID keeps it on an affiliation: the labelled text with its
    whitespace normalised (several markers are one string, e.g. "1,2")."""
    return ' '.join(
        semantic_affiliation_address.view_by_type(SemanticMarker).get_text().split()
    )


def get_author_markers(semantic_author: SemanticAuthor) -> List[str]:
    return [
        marker.strip()
        for marker in semantic_author.view_by_type(SemanticMarker).get_text_list()
        if marker.strip()
    ]


def get_raw_affiliation_text(
    semantic_affiliation_address: SemanticAffiliationAddress
) -> str:
    """GROBID's raw affiliation string: everything but the marker."""
    return ' '.join(' '.join(
        semantic_content.get_text()
        for semantic_content in semantic_affiliation_address
        if not isinstance(semantic_content, SemanticMarker)
    ).split())


def iter_author_tokens(semantic_author: SemanticAuthor) -> Iterable[LayoutToken]:
    # GROBID's Person keeps the tokens of its name parts, not of its markers
    for semantic_content in semantic_author:
        if isinstance(semantic_content, SemanticNamePart):
            yield from semantic_content.iter_tokens()


def iter_affiliation_tokens(
    semantic_affiliation_address: SemanticAffiliationAddress
) -> Iterable[LayoutToken]:
    # GROBID's Affiliation keeps the tokens of its labelled clusters, not of the
    # text labelled as other
    for semantic_content in semantic_affiliation_address:
        if not isinstance(semantic_content, SemanticNote):
            yield from semantic_content.iter_tokens()


def compute_centroid(tokens: Iterable[LayoutToken]) -> Optional[Centroid]:
    """The mean position of the tokens that have one, with the page of the first."""
    sum_x = 0.0
    sum_y = 0.0
    count = 0
    page_number = -1
    for token in tokens:
        coordinates = token.coordinates
        if coordinates is None or (coordinates.x <= 0 and coordinates.y <= 0):
            continue
        sum_x += coordinates.x
        sum_y += coordinates.y
        count += 1
        if page_number == -1:
            page_number = coordinates.page_number
    if not count:
        return None
    return (sum_x / count, sum_y / count, page_number)


def get_distance(centroid_1: Centroid, centroid_2: Centroid) -> float:
    distance = (
        (centroid_1[0] - centroid_2[0]) ** 2 + (centroid_1[1] - centroid_2[1]) ** 2
    ) ** 0.5
    if centroid_1[2] != centroid_2[2]:
        distance += OTHER_PAGE_DISTANCE_PENALTY
    return distance


def normalize_for_match(text: str) -> str:
    """GROBID's normalisation of a raw affiliation text before comparing two: lower
    case, parentheticals and a leading marker label dropped, punctuation and
    whitespace runs collapsed to one space."""
    normalized = text.lower()
    normalized = re.sub(r'\([^)]*\)', ' ', normalized)
    normalized = re.sub(r'^\s*[\d*†‡§¶a-z]{1,3}\s+', '', normalized)
    normalized = re.sub(r'[!-/:-@\[-`{-~\s]+', ' ', normalized).strip()
    return normalized


def is_content_duplicate(
    candidate: SemanticAffiliationAddress,
    existing_affiliations: Sequence[SemanticAffiliationAddress]
) -> bool:
    """Whether the candidate says the same as one of the affiliations: the same
    normalised text, or one contained in the other at close to the same length."""
    candidate_text = normalize_for_match(get_raw_affiliation_text(candidate))
    if not candidate_text:
        return False
    for existing in existing_affiliations:
        if existing is candidate:
            continue
        existing_text = normalize_for_match(get_raw_affiliation_text(existing))
        if not existing_text:
            continue
        if existing_text == candidate_text:
            return True
        if existing_text in candidate_text or candidate_text in existing_text:
            shorter = min(len(candidate_text), len(existing_text))
            longer = max(len(candidate_text), len(existing_text))
            if shorter / longer >= CONTENT_DUPLICATE_MIN_RATIO:
                return True
    return False


def is_marker_within_longer_run(text: str, marker: str, index: int) -> bool:
    """GROBID's check that a one character marker found in the author text is not
    part of a longer number ("1" in "12"), word or run of asterisks, and that "**"
    is not part of a longer run of asterisks."""
    before = text[index - 1] if index > 0 else ''
    after_index = index + len(marker)
    after = text[after_index] if after_index < len(text) else ''
    if len(marker) == 1:
        if marker.isdigit():
            return before.isdigit() or after.isdigit()
        if marker.isalpha():
            return before.isalpha() or after.isalpha()
        if marker == '*':
            return before == '*' or after == '*'
        return False
    if marker == '**':
        return before == '*' or after == '*'
    return False


def get_surname_positions(
    authors: Sequence[SemanticAuthor],
    lower_authors_text: str
) -> List[int]:
    """Where each author's surname sits in the author text, each looked for after
    the one before so that two authors sharing a surname get distinct positions;
    -1 where it is not found."""
    positions: List[int] = []
    search_from = 0
    for author in authors:
        surname = author.surname_text.lower()
        if not surname:
            positions.append(-1)
            continue
        position = lower_authors_text.find(surname, search_from)
        positions.append(position)
        if position != -1:
            search_from = position + len(surname)
    return positions


def find_author_index_nearest_to_marker(
    authors: Sequence[SemanticAuthor],
    surname_positions: Sequence[int],
    marker_index: int,
    taken_author_indices: Set[int]
) -> Optional[int]:
    """The author whose surname ends nearest the marker occurrence."""
    best_index: Optional[int] = None
    best_distance: Optional[int] = None
    for author_index, author in enumerate(authors):
        if author_index in taken_author_indices:
            continue
        surname_position = surname_positions[author_index]
        if surname_position == -1:
            continue
        distance = abs(marker_index - (surname_position + len(author.surname_text)))
        if best_distance is None or distance < best_distance:
            best_index = author_index
            best_distance = distance
    return best_index


def assign_by_direct_markers(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress]
) -> None:
    """An affiliation goes to each author carrying its marker, the markers compared
    as strings once trimmed."""
    affiliations_by_marker: Dict[str, List[SemanticAffiliationAddress]] = {}
    for affiliation in affiliations:
        marker = get_affiliation_marker_text(affiliation)
        if marker:
            affiliations_by_marker.setdefault(marker, []).append(affiliation)
    if not affiliations_by_marker:
        return
    for author in authors:
        for marker in get_author_markers(author):
            for affiliation in affiliations_by_marker.get(marker, []):
                LOGGER.debug('direct marker %r: %r', marker, author.surname_text)
                add_affiliation(author, affiliation)


def assign_by_markers(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress],
    original_authors_text: str
) -> None:
    """Each affiliation with a marker, not attached yet, goes to the author whose
    surname ends nearest each occurrence of the marker in the author text, no author
    taking the same affiliation twice."""
    if not original_authors_text.strip():
        return
    if not any(get_affiliation_marker_text(affiliation) for affiliation in affiliations):
        return
    surname_positions = get_surname_positions(authors, original_authors_text.lower())
    for affiliation_index, affiliation in enumerate(affiliations):
        if affiliation_index > MAX_AFFILIATIONS_SCANNED_FOR_MARKERS:
            break
        if is_attached(affiliation, authors):
            continue
        marker = get_affiliation_marker_text(affiliation)
        if not marker:
            continue
        taken_author_indices: Set[int] = set()
        from_index = 0
        while True:
            marker_index = original_authors_text.find(marker, from_index)
            if marker_index == -1:
                break
            if not is_marker_within_longer_run(original_authors_text, marker, marker_index):
                author_index = find_author_index_nearest_to_marker(
                    authors, surname_positions, marker_index, taken_author_indices
                )
                if author_index is not None:
                    LOGGER.debug(
                        'marker %r in text: %r', marker, authors[author_index].surname_text
                    )
                    add_affiliation(authors[author_index], affiliation)
                    taken_author_indices.add(author_index)
            from_index = marker_index + 1


def find_nearest_affiliation(
    author_centroid: Centroid,
    affiliation_centroids: Sequence[Tuple[SemanticAffiliationAddress, Centroid]]
) -> SemanticAffiliationAddress:
    nearest_affiliation, _ = min(
        affiliation_centroids,
        key=lambda item: get_distance(author_centroid, item[1])
    )
    return nearest_affiliation


def assign_by_proximity(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress]
) -> None:
    """An author still without an affiliation takes the nearest affiliation not
    attached yet, by the positions on the page; several authors may take the same."""
    author_centroids = [
        (author, centroid)
        for author in get_floating_authors(authors)
        for centroid in [compute_centroid(iter_author_tokens(author))]
        if centroid is not None
    ]
    affiliation_centroids = [
        (affiliation, centroid)
        for affiliation in get_floating_affiliations(authors, affiliations)
        for centroid in [compute_centroid(iter_affiliation_tokens(affiliation))]
        if centroid is not None
    ]
    if not author_centroids or not affiliation_centroids:
        return
    for author, author_centroid in author_centroids:
        nearest_affiliation = find_nearest_affiliation(author_centroid, affiliation_centroids)
        LOGGER.debug('proximity: %r', author.surname_text)
        add_affiliation(author, nearest_affiliation)


def assign_by_sequence(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress]
) -> None:
    """The remaining authors and affiliations, without positions to go by: fewer
    affiliations than authors and every author takes them all, otherwise one each
    in order."""
    floating_authors = get_floating_authors(authors)
    floating_affiliations = get_floating_affiliations(authors, affiliations)
    if not floating_authors or not floating_affiliations:
        return
    if len(floating_affiliations) < len(floating_authors):
        for author in floating_authors:
            for affiliation in floating_affiliations:
                add_affiliation(author, affiliation)
        return
    for author, affiliation in zip(floating_authors, floating_affiliations):
        add_affiliation(author, affiliation)


def rescue_orphan_affiliations(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress]
) -> None:
    """An affiliation still attached to nothing goes to the nearest author on the
    page, or without positions to the author with the fewest affiliations, unless
    that author already has an affiliation saying the same."""
    orphans = get_floating_affiliations(authors, affiliations)
    if not orphans or not authors:
        return
    author_centroids = {
        id(author): centroid
        for author in authors
        for centroid in [compute_centroid(iter_author_tokens(author))]
        if centroid is not None
    }
    for orphan in orphans:
        best_author: Optional[SemanticAuthor] = None
        orphan_centroid = compute_centroid(iter_affiliation_tokens(orphan))
        if orphan_centroid is not None:
            best_distance: Optional[float] = None
            for author in authors:
                author_centroid = author_centroids.get(id(author))
                if author_centroid is None:
                    continue
                distance = get_distance(author_centroid, orphan_centroid)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_author = author
        if best_author is None:
            best_author = min(authors, key=lambda author: len(author.affiliations))
        if is_content_duplicate(orphan, best_author.affiliations):
            LOGGER.debug('orphan says the same as an affiliation of %r', best_author.surname_text)
            continue
        LOGGER.debug('orphan rescued onto %r', best_author.surname_text)
        add_affiliation(best_author, orphan)


def assign(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress],
    original_authors_text: str,
    header_clusters: Sequence[HeaderCluster] = ()
) -> None:
    """GROBID's AuthorAffiliationAssigner.assign, with the markers first and the
    pre-link by the header's clusters right after them."""
    if not authors or not affiliations:
        return
    assign_by_direct_markers(authors, affiliations)
    assign_by_markers(authors, affiliations, original_authors_text)
    pre_link_by_preceding_author_cluster(authors, affiliations, header_clusters)
    if len(affiliations) == 1:
        for author in authors:
            add_affiliation(author, affiliations[0])
        return
    if len(authors) == 1:
        for affiliation in affiliations:
            add_affiliation(authors[0], affiliation)
        return
    assign_by_proximity(authors, affiliations)
    assign_by_sequence(authors, affiliations)
    rescue_orphan_affiliations(authors, affiliations)


def get_words(text: str) -> List[str]:
    """The lower case runs of letters of a text."""
    return re.findall(r'[^\W\d_]+', text.lower())


def get_block_words(layout_block: LayoutBlock) -> List[str]:
    # per token, so that two names the layout joined without a space stay apart
    return [
        word
        for token in layout_block.iter_all_tokens()
        for word in get_words(token.text)
    ]


def contains_word_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        list(haystack[index:index + len(needle)]) == list(needle)
        for index in range(len(haystack) - len(needle) + 1)
    )


def split_by_name_separators(layout_block: LayoutBlock) -> List[List[LayoutToken]]:
    """The tokens of an author cluster cut into one span per name, at a comma or
    an "and"."""
    spans: List[List[LayoutToken]] = []
    current: List[LayoutToken] = []
    for token in layout_block.iter_all_tokens():
        text = token.text.strip()
        if not text:
            continue
        if text == ',' or text.lower() == 'and':
            if current:
                spans.append(current)
                current = []
            continue
        current.append(token)
    if current:
        spans.append(current)
    return spans


def match_persons_by_cluster(
    span_tokens: Sequence[LayoutToken],
    authors: Sequence[SemanticAuthor]
) -> List[SemanticAuthor]:
    """The authors a span of an author cluster names: those whose surname words run
    through the span's words; where several do, those whose forename initial is a
    capitalised word of the span (several names without a separator), else the one
    whose initial leads the span."""
    if not span_tokens or not authors:
        return []
    span_words = [word for token in span_tokens for word in get_words(token.text)]
    capitalised_words = [
        text for text in (token.text.strip() for token in span_tokens)
        if text and text[0].isalpha() and text[0].isupper()
    ]
    if not capitalised_words:
        return []
    candidates = [
        author for author in authors
        if author.surname_text.strip()
        and contains_word_run(span_words, get_words(author.surname_text))
    ]
    if len(candidates) <= 1:
        return candidates
    capitalised_initials = {word[0].upper() for word in capitalised_words}
    candidates_by_initial: Dict[str, List[SemanticAuthor]] = {}
    for author in candidates:
        given_name = author.given_name_text.strip()
        if given_name:
            candidates_by_initial.setdefault(given_name[0].upper(), []).append(author)
    matched = [
        candidates_for_initial[0]
        for initial, candidates_for_initial in candidates_by_initial.items()
        if initial in capitalised_initials and len(candidates_for_initial) == 1
    ]
    if matched:
        return matched
    if len(capitalised_words) > 1:
        leading_initial = capitalised_words[0][0].lower()
        filtered = [
            author for author in candidates
            if author.given_name_text.strip()
            and author.given_name_text.strip()[0].lower() == leading_initial
        ]
        if len(filtered) == 1:
            return filtered
    return []


def get_affiliations_of_cluster(
    layout_block: LayoutBlock,
    affiliation_by_token_id: Dict[int, SemanticAffiliationAddress]
) -> List[SemanticAffiliationAddress]:
    """The affiliations cut from the tokens of a header cluster (one cluster may
    give several), each once, in order."""
    targets: List[SemanticAffiliationAddress] = []
    for token in layout_block.iter_all_tokens():
        affiliation = affiliation_by_token_id.get(id(token))
        if affiliation is not None and not any(target is affiliation for target in targets):
            targets.append(affiliation)
    return targets


def iter_authors_of_preceding_author_clusters(
    header_clusters: Sequence[HeaderCluster],
    cluster_index: int,
    authors: Sequence[SemanticAuthor]
) -> Iterable[SemanticAuthor]:
    """The authors named in the author clusters right before the cluster, walking
    back while the clusters are author clusters."""
    for previous_index in range(cluster_index - 1, -1, -1):
        previous_label, previous_block = header_clusters[previous_index]
        if previous_label != HEADER_AUTHOR_LABEL:
            break
        for span_tokens in split_by_name_separators(previous_block):
            yield from match_persons_by_cluster(span_tokens, authors)


def pre_link_by_preceding_author_cluster(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress],
    header_clusters: Sequence[HeaderCluster]
) -> None:
    """With no marker on any author or affiliation, an affiliation goes to the
    authors named in the author clusters right before its cluster in the header."""
    if not authors or not affiliations or not header_clusters:
        return
    if any(get_author_markers(author) for author in authors):
        return
    if any(get_affiliation_marker_text(affiliation) for affiliation in affiliations):
        return
    affiliation_by_token_id: Dict[int, SemanticAffiliationAddress] = {}
    for affiliation in affiliations:
        for token in iter_affiliation_tokens(affiliation):
            affiliation_by_token_id.setdefault(id(token), affiliation)
    for cluster_index, (label, layout_block) in enumerate(header_clusters):
        if label != HEADER_AFFILIATION_LABEL:
            continue
        targets = get_affiliations_of_cluster(layout_block, affiliation_by_token_id)
        if not targets:
            continue
        for author in iter_authors_of_preceding_author_clusters(
            header_clusters, cluster_index, authors
        ):
            for affiliation in targets:
                LOGGER.debug('pre-link by preceding author cluster: %r', author.surname_text)
                add_affiliation(author, affiliation)


def attach_affiliations_for_front(
    authors: Sequence[SemanticAuthor],
    affiliations: Sequence[SemanticAffiliationAddress],
    original_authors_text: str,
    header_clusters: Sequence[HeaderCluster] = ()
) -> None:
    """The assignment, markers first, with the pre-link by the header's clusters. The
    authors are expected deduplicated, as GROBID deduplicates them before this."""
    assign(authors, affiliations, original_authors_text, header_clusters)
