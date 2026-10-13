"""Whether an affiliation is attached to the author it belongs to.

`affiliation_text` compares a document's affiliations as one flat list, so it says how
well they were extracted and nothing about whose they are: every affiliation attached to
the wrong author scores the same as every one attached to the right author. This pairs
each gold author with a predicted author and compares the affiliations attached to each,
one author-affiliation link at a time.

It follows GROBID's `affiliation_linked` (`EndToEndEvaluation.evaluateLinkedAffiliations`):
authors are paired by normalised surname, with the forename initial as the tie-break, and
only an author whose gold link is explicit is scored. A gold that encodes the link by
position alone (no `xref/@rid`, no nested `aff`) says nothing a parser could be held to,
so such an author is left out rather than counted as missed.

It differs from GROBID in three places:

- A predicted author paired with a gold author who is out of scope is left out as well.
  GROBID pairs only the gold authors it scores, so the counterpart of an unscored author
  stays unpaired and its affiliations become false positives, which on a gold that has
  not been completed by hand turns every document without explicit links into nothing
  but false positives.
- A predicted author without a name is not a claim about anyone, see
  `_counts_as_wrong_links`.
- An affiliation is compared by the benchmark's own scoring methods, on the raw
  affiliation text where the prediction has it, rather than by GROBID's four variants
  with their substring fallback. That fallback is there because GROBID compares a
  structured `orgName` with a gold that also holds the address.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from lxml import etree

from sciencebeam_judge.evaluation.normalization import (
    normalize_whitespace,
    strip_punctuation_and_whitespace,
)
from sciencebeam_judge.evaluation.scoring_methods.scoring_methods import (
    ScoringMethod,
    get_scoring_methods,
)
from sciencebeam_judge.parsing.xml import parse_ignore_namespace
from sciencebeam_judge.parsing.xpath.jats_xpath_functions import fn_jats_authors
from sciencebeam_judge.parsing.xpath.tei_xpath_functions import fn_tei_authors
from sciencebeam_utils.utils.xml import get_text_content

AFFILIATION_LINKED_FIELD = "affiliation_linked"
# Not one of the judge's scoring types: the field is not a value or a list of them.
LINKED_SCORING_TYPE = "linked"

TEI_ROOT_TAG = "TEI"

# The gold and the predicted affiliations of one scored author. Either may be empty.
T_AffiliationPair = Tuple[List[str], List[str]]


@dataclass
class AuthorAffiliations:
    """An author reduced to what pairs them with another, and the affiliations linked
    to them. `surname` is empty for a collaboration and for an entry that is not an
    author at all."""
    surname: str
    given_initial: str
    affiliations: List[str] = field(default_factory=list)


def _normalize_name(name: str) -> str:
    """Accents folded too: a PDF often spells one as a letter and a combining mark,
    where the publisher's XML has the single character."""
    folded = "".join(
        char for char in unicodedata.normalize("NFKD", name)
        if not unicodedata.combining(char)
    )
    return strip_punctuation_and_whitespace(folded).lower()


def _text(node: etree.ElementBase) -> str:
    """A label is how the document points at the affiliation, not part of it, and a
    gold that records one would otherwise differ from a prediction that does not."""
    return normalize_whitespace(
        get_text_content(node, exclude=node.xpath(".//label"))
    ).strip()


def _first_text(node: etree.ElementBase, xpaths: Sequence[str]) -> str:
    for xpath in xpaths:
        for child in node.xpath(xpath):
            return _text(child)
    return ""


def _author(
    surname: str,
    given_name: str,
    affiliations: List[str],
) -> AuthorAffiliations:
    return AuthorAffiliations(
        surname=_normalize_name(surname),
        given_initial=_normalize_name(given_name)[:1],
        affiliations=[affiliation for affiliation in affiliations if affiliation],
    )


def _jats_affiliation_nodes(
    contrib: etree.ElementBase,
    aff_by_id: Dict[str, etree.ElementBase],
) -> List[etree.ElementBase]:
    rids = [
        rid
        for value in contrib.xpath('xref[@ref-type="aff"]/@rid')
        for rid in value.split()
    ]
    if not rids:
        return contrib.xpath(".//aff")
    return [aff_by_id[rid] for rid in dict.fromkeys(rids) if rid in aff_by_id]


def _jats_authors(root: etree.ElementBase) -> List[AuthorAffiliations]:
    aff_by_id = {aff.get("id"): aff for aff in root.xpath("front//aff[@id]")}
    return [
        _author(
            surname=_first_text(contrib, ("name/surname", "string-name/surname")),
            given_name=_first_text(
                contrib, ("name/given-names", "string-name/given-names")
            ),
            affiliations=[
                _text(aff) for aff in _jats_affiliation_nodes(contrib, aff_by_id)
            ],
        )
        for contrib in fn_jats_authors(None, [root])
    ]


def _tei_affiliation_text(affiliation: etree.ElementBase) -> str:
    """The raw text where the parser was asked for it, which is what a gold affiliation
    is. Otherwise the fields it was parsed into, which is less than the gold holds."""
    raw_affiliations = affiliation.xpath('note[@type="raw_affiliation"]')
    return " ".join(
        _text(node)
        for node in raw_affiliations or affiliation.xpath("orgName | address/*")
    )


def _tei_authors(root: etree.ElementBase) -> List[AuthorAffiliations]:
    return [
        _author(
            surname=_first_text(author, ("persName/surname",)),
            given_name=_first_text(
                author, ('persName/forename[@type="first"]', "persName/forename")
            ),
            affiliations=[
                _tei_affiliation_text(affiliation)
                for affiliation in author.xpath("affiliation")
                # GROBID writes a collaboration as the affiliation of its members.
                if not affiliation.xpath('orgName[@type="collaboration"]')
            ],
        )
        for author in fn_tei_authors(None, [root])
    ]


def extract_author_affiliations(xml: bytes) -> List[AuthorAffiliations]:
    """The authors of a document, each with the affiliations linked to them. Reads the
    authors the judge reads for `author_full_names`, from TEI or from JATS."""
    root = parse_ignore_namespace(BytesIO(xml))
    if root.tag == TEI_ROOT_TAG:  # pylint: disable=no-member
        return _tei_authors(root)
    return _jats_authors(root)


def _find_predicted_author(
    gold: AuthorAffiliations,
    predicted_authors: Sequence[AuthorAffiliations],
    consumed: Set[int],
) -> Optional[int]:
    """The first predicted author nobody was paired with yet that has the surname,
    preferring one whose forename initial matches where several share it."""
    first_surname_match: Optional[int] = None
    for index, candidate in enumerate(predicted_authors):
        if index in consumed or candidate.surname != gold.surname:
            continue
        if gold.given_initial and candidate.given_initial == gold.given_initial:
            return index
        if first_surname_match is None:
            first_surname_match = index
    return first_surname_match


def _counts_as_wrong_links(predicted: AuthorAffiliations) -> bool:
    """Whether the affiliations of a predicted author that no gold author was paired
    with are links the parser got wrong.

    A named author the gold does not have is a claim about whose affiliations these
    are, and a wrong one. An entry without a name claims nothing: it is where a parser
    leaves the affiliations it could not attach (the dummy author here, an empty
    `<author>` in GROBID), and the links it did not make are already counted as missed.
    """
    return bool(predicted.surname)


def iter_affiliation_pairs(
    gold_authors: Sequence[AuthorAffiliations],
    predicted_authors: Sequence[AuthorAffiliations],
) -> Iterator[T_AffiliationPair]:
    """The gold and predicted affiliations of each author that is scored."""
    consumed: Set[int] = set()
    for gold in gold_authors:
        if not gold.surname:
            continue
        index = _find_predicted_author(gold, predicted_authors, consumed)
        if index is not None:
            consumed.add(index)
        if not gold.affiliations:
            # Out of scope. Paired all the same, so that its counterpart is neither
            # scored against nothing nor left for another gold author to take.
            continue
        yield (
            gold.affiliations,
            predicted_authors[index].affiliations if index is not None else [],
        )
    for index, predicted in enumerate(predicted_authors):
        if index in consumed or not predicted.affiliations:
            continue
        if _counts_as_wrong_links(predicted):
            yield [], predicted.affiliations


def _matched_similarities(
    gold_affiliations: Sequence[str],
    predicted_affiliations: Sequence[str],
    similarity: Callable[[str, str], float],
) -> List[float]:
    """The similarity of each affiliation matched one to one, most similar first, so
    that the order an author's affiliations are listed in does not matter."""
    candidates = sorted(
        (
            (similarity(gold, predicted), gold_index, predicted_index)
            for gold_index, gold in enumerate(gold_affiliations)
            for predicted_index, predicted in enumerate(predicted_affiliations)
        ),
        key=lambda candidate: -candidate[0],
    )
    used_gold: Set[int] = set()
    used_predicted: Set[int] = set()
    similarities: List[float] = []
    for score, gold_index, predicted_index in candidates:
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
        similarities.append(score)
    return similarities


def _match_score(
    pairs: Sequence[T_AffiliationPair],
    scoring_method: ScoringMethod,
) -> Dict[str, Any]:
    """One document's links, in the counts the judge sums a field's documents by."""
    similarity = scoring_method.wrap_with_preprocessing(scoring_method.scoring_fn)
    n_expected = 0
    n_predicted = 0
    n_matched = 0
    similarity_sum = 0.0
    for gold_affiliations, predicted_affiliations in pairs:
        similarities = _matched_similarities(
            gold_affiliations, predicted_affiliations, similarity
        )
        n_expected += len(gold_affiliations)
        n_predicted += len(predicted_affiliations)
        n_matched += sum(1 for score in similarities if score >= scoring_method.threshold)
        similarity_sum += sum(similarities)
    if scoring_method.continuous:
        return {
            "sim_sum": similarity_sum,
            "expected_count": n_expected,
            "predicted_count": n_predicted,
        }
    return {
        "expected_something": n_expected > 0,
        "actual_something": n_predicted > 0,
        "true_positive": n_matched,
        "false_positive": n_predicted - n_matched,
        "false_negative": n_expected - n_matched,
        "true_negative": 0,
    }


def score_linked_affiliations(
    gold_xml: bytes,
    predicted_xml: bytes,
    measures: Sequence[str],
) -> List[dict]:
    """One document's score per method, shaped like the judge's own field scores."""
    pairs = list(iter_affiliation_pairs(
        extract_author_affiliations(gold_xml),
        extract_author_affiliations(predicted_xml),
    ))
    return [
        {
            "field_name": AFFILIATION_LINKED_FIELD,
            "scoring_type": LINKED_SCORING_TYPE,
            "scoring_method": scoring_method.name,
            "match_score": _match_score(pairs, scoring_method),
        }
        for scoring_method in get_scoring_methods(list(measures))
    ]
