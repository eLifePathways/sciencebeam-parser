"""Tests for reading a document from the corrected deposit its corpus declares."""
from __future__ import annotations

from pathlib import Path

from benchmarks.corrections import substitute_corrected_deposits

PUBLISHED = "<article><front><article-meta/></front></article>"
CORRECTED = (
    "<article><front><article-meta>"
    "<trans-abstract xml:lang='pt'><p>Um parágrafo.</p></trans-abstract>"
    "</article-meta></front></article>"
)


def _corrections(tmp_path: Path, document_id: str = "PPR1") -> Path:
    corpus_dir = tmp_path / "source-corrections/a-corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"{document_id}.jats.xml").write_text(CORRECTED, encoding="utf-8")
    return tmp_path / "source-corrections"


def _deposits(tmp_path: Path, *document_ids: str) -> Path:
    deposits = tmp_path / "deposits/a-corpus"
    deposits.mkdir(parents=True, exist_ok=True)
    for document_id in document_ids or ("PPR1",):
        (deposits / f"{document_id}.jats.xml").write_text(
            PUBLISHED, encoding="utf-8"
        )
    return deposits


def test_a_document_is_read_from_its_corrected_deposit(tmp_path: Path):
    root = _corrections(tmp_path)
    deposits = _deposits(tmp_path)
    assert substitute_corrected_deposits(root, "a-corpus", deposits) == ["PPR1"]
    assert (deposits / "PPR1.jats.xml").read_text(encoding="utf-8") == CORRECTED


def test_the_published_deposit_is_kept_beside_it(tmp_path: Path):
    root = _corrections(tmp_path)
    deposits = _deposits(tmp_path)
    substitute_corrected_deposits(root, "a-corpus", deposits)
    assert (deposits / "PPR1.as-deposited.xml").read_text(
        encoding="utf-8"
    ) == PUBLISHED


def test_substituting_twice_says_the_same_thing(tmp_path: Path):
    root = _corrections(tmp_path)
    deposits = _deposits(tmp_path)
    substitute_corrected_deposits(root, "a-corpus", deposits)
    substitute_corrected_deposits(root, "a-corpus", deposits)
    assert (deposits / "PPR1.jats.xml").read_text(encoding="utf-8") == CORRECTED
    assert (deposits / "PPR1.as-deposited.xml").read_text(
        encoding="utf-8"
    ) == PUBLISHED


def test_a_document_the_corpus_says_nothing_about_is_left_alone(tmp_path: Path):
    root = _corrections(tmp_path)
    deposits = _deposits(tmp_path, "PPR1", "PPR2")
    assert substitute_corrected_deposits(root, "a-corpus", deposits) == ["PPR1"]
    assert (deposits / "PPR2.jats.xml").read_text(encoding="utf-8") == PUBLISHED
    assert not (deposits / "PPR2.as-deposited.xml").exists()


def test_a_document_the_fetch_did_not_write_is_not_invented(tmp_path: Path):
    root = _corrections(tmp_path, "PPR9")
    deposits = _deposits(tmp_path)
    assert not substitute_corrected_deposits(root, "a-corpus", deposits)
    assert not (deposits / "PPR9.jats.xml").exists()


def test_the_published_deposit_comes_back_once_the_correction_goes(tmp_path: Path):
    root = _corrections(tmp_path)
    deposits = _deposits(tmp_path)
    substitute_corrected_deposits(root, "a-corpus", deposits)

    (root / "a-corpus/PPR1.jats.xml").unlink()
    assert not substitute_corrected_deposits(root, "a-corpus", deposits)

    assert (deposits / "PPR1.jats.xml").read_text(encoding="utf-8") == PUBLISHED
    assert not (deposits / "PPR1.as-deposited.xml").exists()


def test_a_corpus_with_nothing_declared_is_left_alone(tmp_path: Path):
    deposits = _deposits(tmp_path)
    assert not substitute_corrected_deposits(
        tmp_path / "source-corrections", "a-corpus", deposits
    )
    assert not list(deposits.glob("*.as-deposited.xml"))
