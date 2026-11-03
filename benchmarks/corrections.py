"""Reviewed changes to a deposit, applied as the source documents are fetched.

A deposit is sometimes thinner than the article it describes: a document whose
page prints two languages may declare one, and a correspondence block may hold
as prose what GROBID's annotators label as an institution, an address and a
telephone number. Neither is the aligner's to repair, and waiting for a release
before the corpus can improve is the thing to avoid, so the change is declared
beside the training-source config, reviewed as a file, and applied here.

An entry adds only what the article itself carries. It retires itself: once a
release carries the change, applying it makes no difference and the entry is
deleted. `unless` is what says so, and it is checked rather than trusted -- it
must not match the deposit before the change and must match it after, so an
entry that could never apply is refused rather than silently doing nothing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import yaml
from lxml import etree

LOGGER = logging.getLogger(__name__)

CORRECTION = "correction"
ENRICHMENT = "enrichment"
_KINDS = (CORRECTION, ENRICHMENT)


class CorrectionError(Exception):
    """An entry that cannot be read or cannot be applied."""


@dataclass(frozen=True)
class Change:
    """One element to add under a parent, or one to replace.

    `under` names the parent an element is added to, `after` where among its
    children it goes, so that the result is still in the order JATS declares,
    and `unless` what the deposit would carry if a release had caught up --
    which is the element's own tag where the entry does not say otherwise.
    `replace` names an element instead, and is caught up with once that element
    already says what the fragment says.
    """
    xml: str
    under: Optional[str] = None
    after: Optional[str] = None
    unless: Optional[str] = None
    replace: Optional[str] = None


@dataclass(frozen=True)
class Correction:
    """What one document's deposit is missing, and why."""
    corpus: str
    document_id: str
    reason: str
    kind: str
    changes: Sequence[Change]
    path: Path


def _read_change(raw: Dict[str, object], path: Path) -> Change:
    under = raw.get("under")
    replace = raw.get("replace")
    if bool(under) == bool(replace):
        raise CorrectionError(
            f"{path}: a change names either `under` or `replace`, not both and not"
            f" neither"
        )
    if replace and (raw.get("after") or raw.get("unless")):
        raise CorrectionError(
            f"{path}: `after` and `unless` belong to a change that adds an element,"
            f" and a `replace` is caught up with once the deposit says the same"
        )
    xml = raw.get("xml")
    if not isinstance(xml, str) or not xml.strip():
        raise CorrectionError(f"{path}: a change carries no `xml`")
    return Change(
        xml=xml,
        under=str(under) if under else None,
        after=str(raw["after"]) if raw.get("after") else None,
        unless=str(raw["unless"]) if raw.get("unless") else None,
        replace=str(replace) if replace else None,
    )


def read_correction(path: Path, corpus: str) -> Correction:
    """One entry, as the file declares it."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kind = raw.get("kind")
    if kind not in _KINDS:
        raise CorrectionError(
            f"{path}: `kind` is {kind!r}, and an entry is a {CORRECTION} where the"
            f" deposit is thinner than the article or an {ENRICHMENT} where the"
            f" deposit is valid and the change is ours to keep"
        )
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise CorrectionError(f"{path}: an entry says why it is there")
    changes = raw.get("changes")
    if not isinstance(changes, list) or not changes:
        raise CorrectionError(f"{path}: an entry declares at least one change")
    return Correction(
        corpus=corpus,
        document_id=path.stem,
        reason=reason,
        kind=str(kind),
        changes=[_read_change(change, path) for change in changes],
        path=path,
    )


def iter_corrections(root: Path, corpus: Optional[str] = None) -> Iterator[Correction]:
    """Every entry under the overlay directory, by corpus and document id."""
    if not root.is_dir():
        return
    for corpus_dir in sorted(root.iterdir()):
        if not corpus_dir.is_dir() or (corpus and corpus_dir.name != corpus):
            continue
        for path in sorted(corpus_dir.glob("*.yml")):
            yield read_correction(path, corpus_dir.name)


def corrections_by_document(
    root: Path, corpus: Optional[str] = None
) -> Dict[str, Correction]:
    """The entries for one corpus, keyed by the document id each names."""
    return {
        correction.document_id: correction
        for correction in iter_corrections(root, corpus)
    }


def _fragment(xml: str) -> List[etree._Element]:
    try:
        wrapped = etree.fromstring(f"<changes>{xml}</changes>")
    except etree.XMLSyntaxError as exc:
        raise CorrectionError(f"a change's `xml` does not parse: {exc}") from exc
    return list(wrapped)


def _one(tree: etree._Element, xpath: str, path: Path) -> etree._Element:
    found = tree.xpath(xpath)
    if len(found) != 1:
        raise CorrectionError(
            f"{path}: {xpath!r} matches {len(found)} elements of the deposit, and a"
            f" change names one"
        )
    return found[0]


def _same(left: etree._Element, right: etree._Element) -> bool:
    return etree.canonicalize(
        etree.tostring(left, encoding="unicode"), strip_text=True
    ) == etree.canonicalize(
        etree.tostring(right, encoding="unicode"), strip_text=True
    )


def _apply_change(
    tree: etree._Element, change: Change, path: Path
) -> bool:
    """Make one change, and say whether it made a difference."""
    elements = _fragment(change.xml)
    if change.replace:
        target = _one(tree, change.replace, path)
        if len(elements) != 1:
            raise CorrectionError(
                f"{path}: a `replace` carries one element, not {len(elements)}"
            )
        if _same(target, elements[0]):
            return False
        parent = target.getparent()
        elements[0].tail = target.tail
        parent.replace(target, elements[0])
        return True
    parent = _one(tree, str(change.under), path)
    unless = change.unless or elements[0].tag
    carried = parent.xpath(unless)
    if carried:
        if any(_same(one, elements[0]) for one in carried):
            return False
        raise CorrectionError(
            f"{path}: {unless!r} already matches {len(carried)} element(s) of"
            f" {change.under!r} that say something else, so this change would never"
            f" be applied. Name an `unless` that tells what it adds from what the"
            f" deposit carries."
        )
    at = len(parent)
    if change.after:
        preceding = parent.xpath(change.after)
        if not preceding:
            raise CorrectionError(
                f"{path}: {change.after!r} names no child of {change.under!r} to"
                f" place this after"
            )
        at = parent.index(preceding[-1]) + 1
    for offset, element in enumerate(elements):
        parent.insert(at + offset, element)
    if not parent.xpath(unless):
        raise CorrectionError(
            f"{path}: {unless!r} does not match what this change adds, so a release"
            f" carrying it could never retire the entry"
        )
    return True


def apply_correction(xml_text: str, correction: Correction) -> Optional[str]:
    """The deposit with the entry applied, or None where it made no difference.

    None is what retires an entry: the release carries the change, so the file
    beside `training-source.yml` has nothing left to say.
    """
    tree = etree.fromstring(xml_text.encode("utf-8"))
    changed = False
    for change in correction.changes:
        changed |= _apply_change(tree, change, correction.path)
    if not changed:
        return None
    return etree.tostring(
        tree.getroottree(), encoding="unicode", xml_declaration=False
    )


_DEPOSIT_SUFFIX = ".jats.xml"
_AS_DEPOSITED_SUFFIX = ".as-deposited.xml"


@dataclass(frozen=True)
class Applied:
    """What one entry did to one document."""
    corpus: str
    document_id: str
    kind: str
    overlaid: bool

    @property
    def retired(self) -> bool:
        """Whether the deposit now says what the entry was there to add."""
        return not self.overlaid


def _as_deposited_path(deposit_path: Path) -> Path:
    name = deposit_path.name
    if name.endswith(_DEPOSIT_SUFFIX):
        name = name[: -len(_DEPOSIT_SUFFIX)]
    return deposit_path.with_name(name + _AS_DEPOSITED_SUFFIX)


def overlay_deposit(deposit_path: Path, correction: Correction) -> Applied:
    """Apply one entry to one deposit, from the deposit as published."""
    as_deposited = _as_deposited_path(deposit_path)
    published = (as_deposited if as_deposited.exists() else deposit_path).read_text(
        encoding="utf-8"
    )
    overlaid = apply_correction(published, correction)
    if overlaid is None:
        if as_deposited.exists():
            deposit_path.write_text(published, encoding="utf-8")
            as_deposited.unlink()
        LOGGER.info(
            "%s/%s: the deposit carries what %s adds, so the entry can be deleted",
            correction.corpus, correction.document_id, correction.path,
        )
        return Applied(
            corpus=correction.corpus,
            document_id=correction.document_id,
            kind=correction.kind,
            overlaid=False,
        )
    if not as_deposited.exists():
        as_deposited.write_text(published, encoding="utf-8")
    deposit_path.write_text(overlaid, encoding="utf-8")
    return Applied(
        corpus=correction.corpus,
        document_id=correction.document_id,
        kind=correction.kind,
        overlaid=True,
    )


def overlay_corpus(
    root: Path, corpus: str, deposits_dir: Path
) -> List[Applied]:
    """Apply every entry this corpus declares, to the deposits that are there.

    A document the entry names and the fetch did not write is left alone: the
    selection decides which documents are on disk, and an entry outliving one
    of them is not an error.
    """
    applied: List[Applied] = []
    for document_id, correction in corrections_by_document(root, corpus).items():
        deposit_path = deposits_dir / f"{document_id}{_DEPOSIT_SUFFIX}"
        if not deposit_path.exists():
            continue
        applied.append(overlay_deposit(deposit_path, correction))
    return applied


def summarise(applied: Sequence[Applied]) -> Dict[str, Dict[str, int]]:
    """How many documents of each corpus were overlaid, and of which kind."""
    summary: Dict[str, Dict[str, int]] = {}
    for one in applied:
        if not one.overlaid:
            continue
        counts = summary.setdefault(one.corpus, {})
        counts[one.kind] = counts.get(one.kind, 0) + 1
    return summary
