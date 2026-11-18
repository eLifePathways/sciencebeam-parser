from __future__ import annotations

from lxml import etree

from benchmarks.affiliation_xpath import jats_aff_raw_text, tei_aff_raw_text


def _aff(inner: str) -> etree.ElementBase:
    return etree.fromstring(f"<aff>{inner}</aff>")


def _affiliation(inner: str) -> etree.ElementBase:
    return etree.fromstring(f"<affiliation>{inner}</affiliation>")


class TestJatsAffRawText:
    def test_should_read_the_printed_string_where_there_is_one(self):
        aff = _aff(
            '<institution content-type="original">'
            "University of Nowhere, Department of Imaginary Studies, Nowhere"
            "</institution>"
            '<institution content-type="orgname">University of Nowhere</institution>'
            '<institution content-type="orgdiv1">Department of Imaginary Studies</institution>'
            "<country>Nowhere</country>"
        )
        assert jats_aff_raw_text(aff) == (
            "University of Nowhere, Department of Imaginary Studies, Nowhere"
        )

    def test_should_read_the_whole_text_where_there_is_no_printed_string(self):
        aff = _aff(
            '<institution content-type="orgname">University of Nowhere</institution>'
            "<country>Nowhere</country>"
        )
        assert jats_aff_raw_text(aff) == "University of Nowhere Nowhere"

    def test_should_drop_the_label(self):
        assert jats_aff_raw_text(
            _aff("<label>1</label>University of Nowhere")
        ) == "University of Nowhere"

    def test_should_drop_a_trailing_label(self):
        assert jats_aff_raw_text(
            _aff("University of Nowhere<label>1</label>")
        ) == "University of Nowhere"

    def test_should_keep_the_words_of_elements_written_without_a_space_between(self):
        assert jats_aff_raw_text(
            _aff(
                "<label>1</label>"
                "<institution>University of Nowhere</institution>"
                "<country>Nowhere</country>"
            )
        ) == "University of Nowhere Nowhere"

    def test_should_keep_the_text_around_a_dropped_label(self):
        assert jats_aff_raw_text(
            _aff("<label>1</label>University of Nowhere<italic>, Nowhere</italic>")
        ) == "University of Nowhere, Nowhere"


class TestTeiAffRawText:
    def test_should_drop_the_label(self):
        affiliation = _affiliation(
            '<note type="raw_affiliation"><label>1</label>'
            "University of Nowhere, Nowhere</note>"
        )
        assert tei_aff_raw_text(affiliation) == "University of Nowhere, Nowhere"

    def test_should_keep_the_words_of_elements_written_without_a_space_between(self):
        affiliation = _affiliation(
            '<note type="raw_affiliation"><label>1</label>'
            "<orgName>University of Nowhere</orgName><country>Nowhere</country></note>"
        )
        assert tei_aff_raw_text(affiliation) == "University of Nowhere Nowhere"

    def test_should_join_several_raw_affiliations(self):
        affiliation = _affiliation(
            '<note type="raw_affiliation">University of Nowhere</note>'
            '<note type="raw_affiliation">Institute of Elsewhere</note>'
        )
        assert tei_aff_raw_text(affiliation) == (
            "University of Nowhere, Institute of Elsewhere"
        )

    def test_should_return_nothing_where_there_is_no_raw_affiliation(self):
        assert tei_aff_raw_text(_affiliation("<orgName>Nowhere</orgName>")) == ""
