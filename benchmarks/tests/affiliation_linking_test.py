from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.affiliation_linking import (
    AFFILIATION_LINKED_FIELD,
    LINKED_SCORING_TYPE,
    AuthorAffiliations,
    extract_author_affiliations,
    iter_affiliation_pairs,
    score_linked_affiliations,
)


INSTITUTE_1 = "Institute of Biology, University One, City One, Country One"
INSTITUTE_2 = "Department of Physics, University Two, City Two, Country Two"


def _jats(contribs: str, affs: str = "") -> bytes:
    return f"""<article>
  <front>
    <article-meta>
      <contrib-group>{contribs}{affs}</contrib-group>
    </article-meta>
  </front>
</article>""".encode("utf-8")


def _jats_contrib(surname: str, given_names: str, rids: Sequence[str] = ()) -> str:
    xrefs = "".join(f'<xref ref-type="aff" rid="{rid}"/>' for rid in rids)
    return (
        '<contrib contrib-type="author">'
        f"<name><surname>{surname}</surname><given-names>{given_names}</given-names></name>"
        f"{xrefs}</contrib>"
    )


def _jats_aff(aff_id: str, text: str, label: Optional[str] = None) -> str:
    label_xml = f"<label>{label}</label>" if label else ""
    return f'<aff id="{aff_id}">{label_xml}{text}</aff>'


def _tei(authors: str) -> bytes:
    return f"""<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><sourceDesc><biblStruct><analytic>
    {authors}
  </analytic></biblStruct></sourceDesc></fileDesc></teiHeader>
</TEI>""".encode("utf-8")


def _tei_raw_affiliation(text: str, label: Optional[str] = None) -> str:
    label_xml = f"<label>{label}</label> " if label else ""
    return (
        f'<affiliation><note type="raw_affiliation">{label_xml}{text}</note>'
        '<orgName type="institution">Parsed institution</orgName></affiliation>'
    )


def _tei_author(surname: str, forename: str, affiliations: Sequence[str] = ()) -> str:
    return (
        f'<author><persName><forename type="first">{forename}</forename>'
        f"<surname>{surname}</surname></persName>"
        + "".join(_tei_raw_affiliation(text) for text in affiliations)
        + "</author>"
    )


def _extracted(xml: bytes) -> List[Tuple[str, str, List[str]]]:
    return [
        (author.surname, author.given_initial, author.affiliations)
        for author in extract_author_affiliations(xml)
    ]


def _scores(gold_xml: bytes, predicted_xml: bytes, method: str) -> dict:
    scores = score_linked_affiliations(gold_xml, predicted_xml, [method])
    assert len(scores) == 1
    return scores[0]["match_score"]


def _counts(match_score: dict) -> Dict[str, int]:
    return {
        key: match_score[key]
        for key in ("true_positive", "false_positive", "false_negative")
    }


GOLD_TWO_AUTHORS = _jats(
    _jats_contrib("Smith", "Jo", ["aff1"]) + _jats_contrib("Jones", "Al", ["aff2"]),
    _jats_aff("aff1", INSTITUTE_1) + _jats_aff("aff2", INSTITUTE_2),
)


class TestExtractAuthorAffiliationsFromJats:
    def test_should_follow_the_xref_of_an_author_to_its_affiliation(self):
        assert _extracted(GOLD_TWO_AUTHORS) == [
            ("smith", "j", [INSTITUTE_1]),
            ("jones", "a", [INSTITUTE_2]),
        ]

    def test_should_follow_several_ids_in_one_xref(self):
        xml = _jats(
            '<contrib contrib-type="author"><name><surname>Smith</surname></name>'
            '<xref ref-type="aff" rid="aff1 aff2"/></contrib>',
            _jats_aff("aff1", INSTITUTE_1) + _jats_aff("aff2", INSTITUTE_2),
        )
        assert _extracted(xml) == [("smith", "", [INSTITUTE_1, INSTITUTE_2])]

    def test_should_find_an_affiliation_outside_the_contrib_group(self):
        xml = b"""<article><front><article-meta>
          <contrib-group>
            <contrib contrib-type="author"><name><surname>Smith</surname></name>
              <xref ref-type="aff" rid="aff1"/></contrib>
          </contrib-group>
          <aff id="aff1">Institute 1</aff>
        </article-meta></front></article>"""
        assert _extracted(xml) == [("smith", "", ["Institute 1"])]

    def test_should_fall_back_to_an_affiliation_nested_in_the_contrib(self):
        xml = _jats(
            '<contrib contrib-type="author"><name><surname>Smith</surname></name>'
            f"<aff>{INSTITUTE_1}</aff></contrib>"
        )
        assert _extracted(xml) == [("smith", "", [INSTITUTE_1])]

    def test_should_leave_the_label_out_of_the_affiliation_text(self):
        xml = _jats(
            _jats_contrib("Smith", "Jo", ["aff1"]),
            _jats_aff("aff1", INSTITUTE_1, label="1"),
        )
        assert _extracted(xml) == [("smith", "j", [INSTITUTE_1])]

    def test_should_record_no_link_for_an_author_linked_by_position_only(self):
        xml = _jats(_jats_contrib("Smith", "Jo"), _jats_aff("aff1", INSTITUTE_1))
        assert _extracted(xml) == [("smith", "j", [])]

    def test_should_ignore_an_xref_to_an_affiliation_that_is_not_there(self):
        xml = _jats(_jats_contrib("Smith", "Jo", ["missing"]))
        assert _extracted(xml) == [("smith", "j", [])]

    def test_should_give_a_collaboration_no_surname(self):
        xml = _jats(
            '<contrib contrib-type="author"><collab>The Consortium</collab>'
            '<xref ref-type="aff" rid="aff1"/></contrib>',
            _jats_aff("aff1", INSTITUTE_1),
        )
        assert _extracted(xml) == [("", "", [INSTITUTE_1])]


class TestExtractAuthorAffiliationsFromTei:
    def test_should_read_the_affiliations_nested_in_each_author(self):
        xml = _tei(
            _tei_author("Smith", "Jo", [INSTITUTE_1, INSTITUTE_2])
            + _tei_author("Jones", "Al")
        )
        assert _extracted(xml) == [
            ("smith", "j", [INSTITUTE_1, INSTITUTE_2]),
            ("jones", "a", []),
        ]

    def test_should_leave_the_label_out_of_the_raw_affiliation(self):
        xml = _tei(
            "<author><persName><surname>Smith</surname></persName>"
            + _tei_raw_affiliation(INSTITUTE_1, label="1")
            + "</author>"
        )
        assert _extracted(xml) == [("smith", "", [INSTITUTE_1])]

    def test_should_fall_back_to_the_parsed_fields_without_a_raw_affiliation(self):
        xml = _tei(
            "<author><persName><surname>Smith</surname></persName><affiliation>"
            '<orgName type="department">Department 1</orgName>'
            '<orgName type="institution">Institute 1</orgName>'
            "<address><settlement>City 1</settlement><country>Country 1</country></address>"
            "</affiliation></author>"
        )
        assert _extracted(xml) == [
            ("smith", "", ["Department 1 Institute 1 City 1 Country 1"])
        ]

    def test_should_give_a_dummy_author_no_surname(self):
        xml = _tei(
            '<author><note type="dummy_author">Dummy author</note>'
            + _tei_raw_affiliation(INSTITUTE_1)
            + "</author>"
        )
        assert _extracted(xml) == [("", "", [INSTITUTE_1])]

    def test_should_not_read_a_collaboration_as_an_affiliation(self):
        xml = _tei(
            "<author><persName><surname>Smith</surname></persName><affiliation>"
            '<orgName type="collaboration">The Consortium</orgName>'
            "</affiliation></author>"
        )
        assert _extracted(xml) == [("smith", "", [])]

    def test_should_fold_accents_and_punctuation_out_of_a_name(self):
        # "u" followed by a combining diaeresis, as a PDF tends to spell it
        xml = _tei(_tei_author("Müller-Stöhr", "Éric"))
        assert _extracted(xml) == [("mullerstohr", "e", [])]


def _author(
    surname: str,
    given_initial: str = "",
    affiliations: Sequence[str] = (),
) -> AuthorAffiliations:
    return AuthorAffiliations(surname, given_initial, list(affiliations))


class TestIterAffiliationPairs:
    def test_should_pair_authors_by_surname_whatever_their_order(self):
        pairs = list(iter_affiliation_pairs(
            [_author("smith", "j", ["a"]), _author("jones", "a", ["b"])],
            [_author("jones", "a", ["b2"]), _author("smith", "j", ["a2"])],
        ))
        assert pairs == [(["a"], ["a2"]), (["b"], ["b2"])]

    def test_should_prefer_the_matching_initial_where_authors_share_a_surname(self):
        pairs = list(iter_affiliation_pairs(
            [_author("smith", "j", ["a"]), _author("smith", "k", ["b"])],
            [_author("smith", "k", ["b2"]), _author("smith", "j", ["a2"])],
        ))
        assert pairs == [(["a"], ["a2"]), (["b"], ["b2"])]

    def test_should_pair_on_surname_alone_where_no_initial_matches(self):
        pairs = list(iter_affiliation_pairs(
            [_author("smith", "j", ["a"])],
            [_author("smith", "", ["a2"])],
        ))
        assert pairs == [(["a"], ["a2"])]

    def test_should_pair_a_predicted_author_once(self):
        pairs = list(iter_affiliation_pairs(
            [_author("smith", "j", ["a"]), _author("smith", "j", ["b"])],
            [_author("smith", "j", ["a2"])],
        ))
        assert pairs == [(["a"], ["a2"]), (["b"], [])]

    def test_should_expect_every_affiliation_of_an_author_that_was_not_predicted(self):
        pairs = list(iter_affiliation_pairs([_author("smith", "j", ["a", "b"])], []))
        assert pairs == [(["a", "b"], [])]

    def test_should_leave_out_an_author_whose_gold_link_is_not_explicit(self):
        pairs = list(iter_affiliation_pairs(
            [_author("smith", "j"), _author("jones", "a", ["b"])],
            [_author("smith", "j", ["a2"]), _author("jones", "a", ["b2"])],
        ))
        assert pairs == [(["b"], ["b2"])]

    def test_should_leave_out_a_collaboration(self):
        pairs = list(iter_affiliation_pairs([_author("", "", ["a"])], []))
        assert not pairs

    def test_should_count_the_affiliations_of_an_author_the_gold_does_not_have(self):
        pairs = list(iter_affiliation_pairs(
            [_author("smith", "j", ["a"])],
            [_author("smith", "j", ["a2"]), _author("brown", "b", ["c2", "d2"])],
        ))
        assert pairs == [(["a"], ["a2"]), ([], ["c2", "d2"])]

    def test_should_not_count_affiliations_that_were_attached_to_nobody(self):
        pairs = list(iter_affiliation_pairs(
            [_author("smith", "j", ["a"])],
            [_author("smith", "j"), _author("", "", ["a2"])],
        ))
        assert pairs == [(["a"], [])]


class TestScoreLinkedAffiliations:
    def test_should_name_the_field_type_and_each_method_asked_for(self):
        predicted = _tei(
            _tei_author("Smith", "Jo", [INSTITUTE_1]) + _tei_author("Jones", "Al", [INSTITUTE_2])
        )
        scores = score_linked_affiliations(
            GOLD_TWO_AUTHORS, predicted, ["exact", "levenshtein", "edit_sim"]
        )
        assert [
            (score["field_name"], score["scoring_type"], score["scoring_method"])
            for score in scores
        ] == [
            (AFFILIATION_LINKED_FIELD, LINKED_SCORING_TYPE, "exact"),
            (AFFILIATION_LINKED_FIELD, LINKED_SCORING_TYPE, "levenshtein"),
            (AFFILIATION_LINKED_FIELD, LINKED_SCORING_TYPE, "edit_sim"),
        ]

    def test_should_match_every_link_attached_to_the_right_author(self):
        predicted = _tei(
            _tei_author("Smith", "Jo", [INSTITUTE_1]) + _tei_author("Jones", "Al", [INSTITUTE_2])
        )
        assert _counts(_scores(GOLD_TWO_AUTHORS, predicted, "exact")) == {
            "true_positive": 2, "false_positive": 0, "false_negative": 0,
        }

    def test_should_match_no_link_where_the_affiliations_are_swapped(self):
        # Every affiliation is extracted, so `affiliation_text` cannot tell this apart
        # from the document above.
        predicted = _tei(
            _tei_author("Smith", "Jo", [INSTITUTE_2]) + _tei_author("Jones", "Al", [INSTITUTE_1])
        )
        assert _counts(_scores(GOLD_TWO_AUTHORS, predicted, "levenshtein")) == {
            "true_positive": 0, "false_positive": 2, "false_negative": 2,
        }

    def test_should_miss_every_link_where_the_affiliations_were_attached_to_nobody(self):
        predicted = _tei(
            _tei_author("Smith", "Jo") + _tei_author("Jones", "Al")
            + '<author><note type="dummy_author">Dummy author</note>'
            + _tei_raw_affiliation(INSTITUTE_1) + _tei_raw_affiliation(INSTITUTE_2)
            + "</author>"
        )
        assert _counts(_scores(GOLD_TWO_AUTHORS, predicted, "levenshtein")) == {
            "true_positive": 0, "false_positive": 0, "false_negative": 2,
        }

    def test_should_count_an_extra_and_a_missing_affiliation_of_one_author(self):
        gold = _jats(
            _jats_contrib("Smith", "Jo", ["aff1", "aff2"]),
            _jats_aff("aff1", INSTITUTE_1) + _jats_aff("aff2", INSTITUTE_2),
        )
        predicted = _tei(_tei_author("Smith", "Jo", [INSTITUTE_1, "Somewhere else entirely"]))
        assert _counts(_scores(gold, predicted, "levenshtein")) == {
            "true_positive": 1, "false_positive": 1, "false_negative": 1,
        }

    def test_should_not_depend_on_the_order_of_an_authors_affiliations(self):
        gold = _jats(
            _jats_contrib("Smith", "Jo", ["aff1", "aff2"]),
            _jats_aff("aff1", INSTITUTE_1) + _jats_aff("aff2", INSTITUTE_2),
        )
        predicted = _tei(_tei_author("Smith", "Jo", [INSTITUTE_2, INSTITUTE_1]))
        assert _counts(_scores(gold, predicted, "exact")) == {
            "true_positive": 2, "false_positive": 0, "false_negative": 0,
        }

    def test_should_tolerate_a_small_difference_only_by_a_method_that_does(self):
        predicted = _tei(
            _tei_author("Smith", "Jo", [INSTITUTE_1 + "."])
            + _tei_author("Jones", "Al", [INSTITUTE_2])
        )
        assert _counts(_scores(GOLD_TWO_AUTHORS, predicted, "exact"))["true_positive"] == 1
        assert _counts(_scores(GOLD_TWO_AUTHORS, predicted, "levenshtein"))["true_positive"] == 2

    def test_should_score_a_jats_prediction_like_a_tei_one(self):
        assert _counts(_scores(GOLD_TWO_AUTHORS, GOLD_TWO_AUTHORS, "exact")) == {
            "true_positive": 2, "false_positive": 0, "false_negative": 0,
        }

    def test_should_count_links_for_a_continuous_method(self):
        predicted = _tei(
            _tei_author("Smith", "Jo", [INSTITUTE_1])
            + _tei_author("Jones", "Al")
            + _tei_author("Brown", "Bo", [INSTITUTE_2])
        )
        assert _scores(GOLD_TWO_AUTHORS, predicted, "edit_sim") == {
            "sim_sum": 1.0, "expected_count": 2, "predicted_count": 2,
        }

    def test_should_expect_nothing_where_no_gold_link_is_explicit(self):
        gold = _jats(
            _jats_contrib("Smith", "Jo") + _jats_contrib("Jones", "Al"),
            _jats_aff("aff1", INSTITUTE_1),
        )
        predicted = _tei(
            _tei_author("Smith", "Jo", [INSTITUTE_1]) + _tei_author("Jones", "Al", [INSTITUTE_1])
        )
        assert _scores(gold, predicted, "edit_sim") == {
            "sim_sum": 0.0, "expected_count": 0, "predicted_count": 0,
        }
        match_score = _scores(gold, predicted, "levenshtein")
        assert _counts(match_score) == {
            "true_positive": 0, "false_positive": 0, "false_negative": 0,
        }
        assert not match_score["expected_something"]
