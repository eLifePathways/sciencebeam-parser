# Abstracts

What the parser emits where a document carries more than one abstract, and what it says about each.

## How many abstracts reach the output

The header model labels every region it reads as an abstract, and `abstracts_mode` decides which of them the output carries:

| mode | what it emits |
| --- | --- |
| `first` | the first abstract, with the rest left as notes |
| `variants` | every abstract that is at least 0.15 of the first one's length, each declaring its own language |
| `merged_by_language` | the same, with abstracts sharing a language joined into one |

The `recommended` profile asks for `variants`; a profile serving the GROBID CRF models asks for `first`. A profile sets it under `processors.fulltext`.

An abstract the page prints in two pieces — the second half carried over a page break — is one abstract in every mode. The second block is joined to the first where it opens mid-sentence under an abstract that ends unfinished, carries no heading of its own, and begins on the page the abstract ended on or the next.

## What each abstract declares

Each abstract carries the language of its own text in `xml:lang`, under `variants` and above, where the text is long enough and consistent enough to name one. A document whose abstracts are in Portuguese and English therefore says which is which, in the TEI and in the JATS.

## The heading printed above an abstract

A page prints `Abstract`, `Resumo` or `Plain language summary` above the abstract it introduces. TEI gives `profileDesc/abstract` nowhere to carry that heading, so it is emitted as a note immediately before the abstract it belongs to:

```xml
<profileDesc>
  <note type="abstract-heading">Plain language summary</note>
  <abstract xml:lang="en">
    <p>Human induced pluripotent stem cells …</p>
  </abstract>
</profileDesc>
```

A note of that type is the heading of the abstract that follows it, and an abstract with no heading above it has no note. The heading is not part of the abstract's own text.

The JATS route puts it where JATS carries it, inside the abstract:

```xml
<abstract xml:lang="en">
  <title>Plain language summary</title>
  <p>Human induced pluripotent stem cells …</p>
</abstract>
```

The reading view shows it above the abstract it belongs to, in place of the word `Abstract` it falls back to otherwise.

Headings printed inside a structured abstract — `Background`, `Methods`, `Results` — are part of the abstract's text and are not separated from it.
