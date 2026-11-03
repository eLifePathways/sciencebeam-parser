"""Tests for the reviewed deposit changes the fetch applies."""
from __future__ import annotations

import pytest  # pyright: ignore[reportMissingImports]
from lxml import etree  # pyright: ignore[reportMissingImports]

from benchmarks.corrections import (
    CorrectionError,
    apply_correction,
    corrections_by_document,
    overlay_corpus,
    overlay_deposit,
    read_correction,
    summarise,
)

DEPOSIT = (
    "<article><front><article-meta>"
    "<title-group><article-title>A Title</article-title></title-group>"
    "<abstract><p>A paragraph.</p></abstract>"
    "<kwd-group><kwd>One</kwd></kwd-group>"
    "</article-meta></front></article>"
)

ENTRY = """
reason: The article prints a second language the deposit does not declare.
kind: correction
changes:
  - under: front/article-meta
    after: abstract
    xml: |
      <trans-abstract xml:lang="pt"><p>Um parágrafo.</p></trans-abstract>
"""


def _entry(tmp_path, text: str = ENTRY, document_id: str = "PPR1"):
    corpus_dir = tmp_path / "source-corrections/a-corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    path = corpus_dir / f"{document_id}.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_an_entry_says_what_it_adds_and_why(tmp_path):
    correction = read_correction(_entry(tmp_path), "a-corpus")
    assert correction.document_id == "PPR1"
    assert correction.kind == "correction"
    assert correction.reason.startswith("The article prints")
    assert len(correction.changes) == 1


def test_an_entry_of_no_stated_kind_is_refused(tmp_path):
    path = _entry(tmp_path, ENTRY.replace("kind: correction", "kind: guess"))
    with pytest.raises(CorrectionError, match="kind"):
        read_correction(path, "a-corpus")


def test_an_entry_without_a_reason_is_refused(tmp_path):
    without_reason = ENTRY.replace(
        "reason: The article prints a second language the deposit does not declare.",
        "reason: ''",
    )
    path = _entry(tmp_path, without_reason)
    with pytest.raises(CorrectionError, match="why"):
        read_correction(path, "a-corpus")


def test_a_change_naming_both_an_element_and_a_parent_is_refused(tmp_path):
    path = _entry(tmp_path, ENTRY.replace(
        "  - under: front/article-meta",
        "  - under: front/article-meta\n    replace: front/article-meta/abstract",
    ))
    with pytest.raises(CorrectionError, match="either"):
        read_correction(path, "a-corpus")


def test_the_addition_lands_where_the_entry_says(tmp_path):
    correction = read_correction(_entry(tmp_path), "a-corpus")
    overlaid = apply_correction(DEPOSIT, correction)
    assert overlaid is not None
    meta = etree.fromstring(overlaid.encode("utf-8")).find(".//article-meta")
    assert [child.tag for child in meta] == [
        "title-group", "abstract", "trans-abstract", "kwd-group",
    ]


def test_an_entry_the_deposit_has_caught_up_with_makes_no_difference(tmp_path):
    correction = read_correction(_entry(tmp_path), "a-corpus")
    carried = DEPOSIT.replace(
        "<kwd-group>",
        '<trans-abstract xml:lang="pt"><p>Um parágrafo.</p></trans-abstract><kwd-group>',
    )
    assert apply_correction(carried, correction) is None


def test_a_second_group_is_told_from_the_one_the_deposit_carries(tmp_path):
    path = _entry(tmp_path, """
reason: The article prints keywords in both languages.
kind: correction
changes:
  - under: front/article-meta
    after: kwd-group
    unless: kwd-group[@xml:lang="pt"]
    xml: |
      <kwd-group xml:lang="pt"><kwd>Um</kwd></kwd-group>
""")
    overlaid = apply_correction(DEPOSIT, read_correction(path, "a-corpus"))
    assert overlaid is not None
    groups = etree.fromstring(overlaid.encode("utf-8")).findall(".//kwd-group")
    assert len(groups) == 2


def test_an_element_is_replaced_whole(tmp_path):
    path = _entry(tmp_path, """
reason: The deposit holds as prose what the article structures.
kind: enrichment
changes:
  - replace: front/article-meta/abstract
    xml: |
      <abstract><p>A structured paragraph.</p></abstract>
""")
    overlaid = apply_correction(DEPOSIT, read_correction(path, "a-corpus"))
    assert overlaid is not None
    assert "A structured paragraph." in overlaid
    assert "A paragraph." not in overlaid


def test_a_replacement_the_deposit_already_says_makes_no_difference(tmp_path):
    path = _entry(tmp_path, """
reason: The deposit holds as prose what the article structures.
kind: enrichment
changes:
  - replace: front/article-meta/abstract
    xml: |
      <abstract><p>A paragraph.</p></abstract>
""")
    assert apply_correction(DEPOSIT, read_correction(path, "a-corpus")) is None


def test_a_change_naming_no_element_of_the_deposit_is_refused(tmp_path):
    path = _entry(tmp_path, ENTRY.replace(
        "under: front/article-meta", "under: front/article-meta/nowhere"
    ))
    with pytest.raises(CorrectionError, match="matches 0 elements"):
        apply_correction(DEPOSIT, read_correction(path, "a-corpus"))


def test_the_deposit_as_published_is_kept_beside_the_overlaid_one(tmp_path):
    _entry(tmp_path)
    deposits = tmp_path / "deposits/a-corpus"
    deposits.mkdir(parents=True)
    (deposits / "PPR1.jats.xml").write_text(DEPOSIT, encoding="utf-8")

    applied = overlay_corpus(tmp_path / "source-corrections", "a-corpus", deposits)

    assert [one.overlaid for one in applied] == [True]
    assert (deposits / "PPR1.as-deposited.xml").read_text(encoding="utf-8") == DEPOSIT
    assert "trans-abstract" in (deposits / "PPR1.jats.xml").read_text(encoding="utf-8")


def test_applying_twice_adds_the_element_once(tmp_path):
    _entry(tmp_path)
    deposits = tmp_path / "deposits/a-corpus"
    deposits.mkdir(parents=True)
    (deposits / "PPR1.jats.xml").write_text(DEPOSIT, encoding="utf-8")
    root = tmp_path / "source-corrections"

    overlay_corpus(root, "a-corpus", deposits)
    overlay_corpus(root, "a-corpus", deposits)

    overlaid = etree.parse(str(deposits / "PPR1.jats.xml"))
    assert len(overlaid.findall(".//trans-abstract")) == 1


def test_the_deposit_is_put_back_once_a_release_carries_the_change(tmp_path):
    path = _entry(tmp_path)
    deposits = tmp_path / "deposits/a-corpus"
    deposits.mkdir(parents=True)
    (deposits / "PPR1.jats.xml").write_text(DEPOSIT, encoding="utf-8")
    root = tmp_path / "source-corrections"
    overlay_corpus(root, "a-corpus", deposits)

    carried = DEPOSIT.replace(
        "<kwd-group>",
        '<trans-abstract xml:lang="pt"><p>Um parágrafo.</p></trans-abstract><kwd-group>',
    )
    (deposits / "PPR1.as-deposited.xml").write_text(carried, encoding="utf-8")

    applied = overlay_corpus(root, "a-corpus", deposits)

    assert [one.retired for one in applied] == [True]
    assert (deposits / "PPR1.jats.xml").read_text(encoding="utf-8") == carried
    assert not (deposits / "PPR1.as-deposited.xml").exists()
    assert path.exists(), "the entry is deleted by hand, once a run has said so"


def test_a_document_the_fetch_did_not_write_is_left_alone(tmp_path):
    _entry(tmp_path)
    deposits = tmp_path / "deposits/a-corpus"
    deposits.mkdir(parents=True)
    assert not overlay_corpus(tmp_path / "source-corrections", "a-corpus", deposits)


def test_the_entries_are_read_by_corpus_and_document(tmp_path):
    _entry(tmp_path, document_id="PPR1")
    _entry(tmp_path, document_id="PPR2")
    by_document = corrections_by_document(tmp_path / "source-corrections")
    assert sorted(by_document) == ["PPR1", "PPR2"]
    assert all(one.corpus == "a-corpus" for one in by_document.values())


def test_the_summary_counts_the_documents_overlaid_of_each_kind(tmp_path):
    _entry(tmp_path)
    deposits = tmp_path / "deposits/a-corpus"
    deposits.mkdir(parents=True)
    (deposits / "PPR1.jats.xml").write_text(DEPOSIT, encoding="utf-8")
    applied = overlay_corpus(tmp_path / "source-corrections", "a-corpus", deposits)
    assert summarise(applied) == {"a-corpus": {"correction": 1}}


def test_one_deposit_can_be_overlaid_on_its_own(tmp_path):
    correction = read_correction(_entry(tmp_path), "a-corpus")
    deposit_path = tmp_path / "PPR1.jats.xml"
    deposit_path.write_text(DEPOSIT, encoding="utf-8")
    assert overlay_deposit(deposit_path, correction).overlaid


def test_an_unless_that_cannot_tell_the_addition_apart_is_refused(tmp_path):
    # The deposit already carries a <kwd-group>, so the tag on its own would say
    # the entry is caught up with and it would never be applied.
    path = _entry(tmp_path, """
reason: The article prints keywords in both languages.
kind: correction
changes:
  - under: front/article-meta
    after: kwd-group
    xml: |
      <kwd-group xml:lang="pt"><kwd>Um</kwd></kwd-group>
""")
    with pytest.raises(CorrectionError, match="never be applied"):
        apply_correction(DEPOSIT, read_correction(path, "a-corpus"))


def test_an_unless_the_addition_does_not_satisfy_is_refused(tmp_path):
    path = _entry(tmp_path, """
reason: The article prints a second language the deposit does not declare.
kind: correction
changes:
  - under: front/article-meta
    after: abstract
    unless: trans-abstract[@xml:lang="es"]
    xml: |
      <trans-abstract xml:lang="pt"><p>Um parágrafo.</p></trans-abstract>
""")
    with pytest.raises(CorrectionError, match="could never retire"):
        apply_correction(DEPOSIT, read_correction(path, "a-corpus"))


def test_a_deposit_that_says_the_same_thing_retires_the_entry(tmp_path):
    correction = read_correction(_entry(tmp_path), "a-corpus")
    carried = DEPOSIT.replace(
        "<kwd-group>",
        '<trans-abstract xml:lang="pt"><p>Um parágrafo.</p></trans-abstract><kwd-group>',
    )
    assert apply_correction(carried, correction) is None
