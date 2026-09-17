The numbered lines below are one scientific article, in reading order. Each line is prefixed with
its line number and a tab.

Split the article into regions. For each region, give the line it starts on, the line it ends on,
and its label:

- `front_matter` — title, authors, affiliations, abstract and keywords
- `body` — the article text, including section headings, figures, tables and their captions
- `acknowledgements` — funding, thanks and contribution statements
- `appendix` — appendices and supplementary material
- `references` — the bibliography, including its heading

Regions follow one another in reading order, and a label may appear more than once where the
document returns to that kind of content.

Leave out any line that belongs to none of them — a running head, a footer, a page number. A region
ends on the line before such a line and the next region starts after it.

Answer in this shape:

```
{"regions": [
  {"start": <0..{{last_line}}>, "end": <0..{{last_line}}>,
   "label": <front_matter|body|acknowledgements|appendix|references>}
]}
```
