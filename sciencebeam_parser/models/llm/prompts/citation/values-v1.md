You extract the fields of a single bibliographic reference.

Return an ordered list of the fields you find, each with the field text copied
EXACTLY from the input. Copy verbatim: do not normalise spacing, punctuation or
capitalisation, and do not add or remove characters. Every value you return must
appear in the input. Leave a field out entirely rather than returning it with
empty text.

Conventions, which are not guessable from the label names:

- `title` is the article title. `journal` is the container title. `booktitle` is
  a book or monograph title, and `series` a series title.
- A page range is TWO `pages` fields, one per number, not one field spanning the
  hyphen.
- `note` is the leading reference number or marker, and any other editorial note.
- `pubnum` is any identifier: a DOI, PMID, PMCID or arXiv id.
- Do not return trailing link text such as "PubMed Abstract | Publisher Full Text".

Worked example. For the input:

1 . Fleming PS , Koletsi D , Ioannidis JP , et al . : High quality of the
evidence for medical and other health - related interventions was uncommon in
Cochrane systematic reviews . J Clin Epidemiol . 2016 ; 78 : 34 - 42 .

the correct answer is:

{"fields": [
  {"label": "note", "text": "1"},
  {"label": "author", "text": "Fleming PS , Koletsi D , Ioannidis JP , et al ."},
  {"label": "title", "text": "High quality of the evidence for medical and other health - related interventions was uncommon in Cochrane systematic reviews ."},
  {"label": "journal", "text": "J Clin Epidemiol ."},
  {"label": "date", "text": "2016"},
  {"label": "volume", "text": "78"},
  {"label": "pages", "text": "34"},
  {"label": "pages", "text": "42"}
]}
