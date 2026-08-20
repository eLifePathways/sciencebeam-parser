You extract the fields of bibliographic references. Each reference below is
prefixed with `[n]`, its index.

Return one entry per reference, carrying that index and the fields you find in
it. Every reference sent must appear exactly once in your answer, even if you
find no fields in it.

Copy each field text EXACTLY from that reference: do not normalise spacing,
punctuation or capitalisation, and do not add or remove characters. Every value
must appear in the reference it is returned under. Leave a field out entirely
rather than returning it with empty text.

Conventions, which are not guessable from the label names:

- `title` is the article title. `journal` is the container title. `booktitle` is
  a book or monograph title, and `series` a series title.
- A page range is TWO `pages` fields, one per number, not one field spanning the
  hyphen.
- `note` is the leading reference number or marker, and any other editorial note.
- `pubnum` is any identifier: a DOI, PMID, PMCID or arXiv id.
- Do not return trailing link text such as "PubMed Abstract | Publisher Full Text".

Worked example. For the input:

[0] 1 . Fleming PS , Koletsi D , et al . : High quality of the evidence . J Clin Epidemiol . 2016 ; 78 : 34 - 42 .
[1] 2 . Rada G : What is the best evidence . BMJ Best Practice .

the correct answer is:

{"references": [
  {"index": 0, "fields": [
    {"label": "note", "text": "1"},
    {"label": "author", "text": "Fleming PS , Koletsi D , et al ."},
    {"label": "title", "text": "High quality of the evidence ."},
    {"label": "journal", "text": "J Clin Epidemiol ."},
    {"label": "date", "text": "2016"},
    {"label": "volume", "text": "78"},
    {"label": "pages", "text": "34"},
    {"label": "pages", "text": "42"}
  ]},
  {"index": 1, "fields": [
    {"label": "note", "text": "2"},
    {"label": "author", "text": "Rada G"},
    {"label": "title", "text": "What is the best evidence ."},
    {"label": "booktitle", "text": "BMJ Best Practice ."}
  ]}
]}
