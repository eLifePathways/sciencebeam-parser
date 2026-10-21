import re
from datetime import date
from itertools import chain
from typing import (
    Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple
)
from dataclasses import dataclass

from lxml import etree

from sciencebeam_parser.training.jats.back_matter import (
    LabelledSections,
    NOTE_PARAGRAPH_FIELD_BY_KIND,
    PARAGRAPH_FIELD_BY_KIND,
    TITLE_FIELD_BY_KIND,
    element_text,
    get_fn_kind,
    is_bold_heading,
    match_fn_keywords,
    statement_text,
)
from sciencebeam_parser.training.jats.field_vocab import (
    JatsFieldNames,
    JatsSubFieldNames,
)


@dataclass
class JatsFieldValue:
    text: str
    field_name: str
    sub_field_name: Optional[str] = None
    fallback_text: Optional[str] = None
    exact_only: bool = False


def _iter_sub_article_stub_texts(
    sub_article: etree._Element,
) -> Iterator[Tuple[etree._Element, str]]:
    """Yield the front-stub text a peer-review report prints above its body.

    A report's heading block -- its licence sentence, reviewer name and
    affiliation -- is in `<front-stub>` rather than in a `<title>` or a `<p>`,
    so without it the region starts at the report's first paragraph and the
    heading above falls back to `<body>`.

    These are matched verbatim or not at all.  The renderer prints them from
    these very elements, and there are enough of them per report that letting
    them slide over the region instead would spend the wide-search budget on the
    headings and lose the report prose that budget is for.

    The report's own DOI is left out: the PDF breaks it into tokens around every
    dot and slash, so it is never found verbatim, and letting that one value
    slide costs more region elsewhere than the heading line it wins.
    """
    for el in sub_article.xpath('.//front-stub//contrib/name'):
        given_names = _element_text_of_first(el, 'given-names')
        surname = _element_text_of_first(el, 'surname')
        text = ' '.join(part for part in (given_names, surname) if part)
        if text:
            yield el, text
    for el in sub_article.xpath('.//front-stub//aff'):
        text = _affiliation_text(el)
        if text:
            yield el, text
    for el in sub_article.xpath('.//front-stub//license-p'):
        text = element_text(el)
        if text:
            yield el, text


def _element_text_of_first(el: etree._Element, tag: str) -> str:
    found = el.find(tag)
    return element_text(found) if found is not None else ''


def _affiliation_text(el: etree._Element) -> str:
    """The affiliation without its `<label>`, which prints as the author's marker."""
    parts = [el.text or '']
    for child in el:
        if child.tag != 'label':
            parts.append(element_text(child))
        parts.append(child.tail or '')
    return ' '.join(' '.join(parts).split())


def _reference_parent_text(ref_el: etree._Element) -> str:
    """Extract parent-match text for a <ref>, skipping <pub-id> subtrees.

    PMID/PMCID values appear in JATS <pub-id> elements but rarely in PDF
    reference lists.  Including them in the SW needle inflates its length
    without adding matched characters, pushing match quality below the 0.8
    threshold and causing the entire reference to go unmatched.  DOI text is
    also excluded here — it is handled by the dedicated REFERENCE_DOI
    sub-field match, so the parent only needs the bibliographic prose.
    """
    parts: List[str] = []
    for child in ref_el.iter():
        tag = child.tag
        local = tag.split('}', 1)[1] if isinstance(tag, str) and tag.startswith('{') else tag
        if local == 'pub-id':
            if child.tail:
                parts.append(child.tail)
            continue
        if child.text:
            parts.append(child.text)
        if child is not ref_el and child.tail:
            parts.append(child.tail)
    return ' '.join(' '.join(parts).split())


def _iter_sub_field_values(
    parent_el: etree._Element,
    field_name: str,
    sub_xpath_by_sub_field: Sequence[tuple],
) -> Iterator[JatsFieldValue]:
    """Yield one JatsFieldValue per sub-field span found in parent_el."""
    for sub_field_name, xpath in sub_xpath_by_sub_field:
        for child in parent_el.xpath(xpath):
            text = element_text(child)
            if text:
                yield JatsFieldValue(
                    text=text,
                    field_name=field_name,
                    sub_field_name=sub_field_name,
                )


# Sub-field XPaths for references (relative to each <ref> element)
_REFERENCE_SUB_FIELDS = [
    (JatsSubFieldNames.REFERENCE_LABEL,           './label'),
    (JatsSubFieldNames.REFERENCE_ARTICLE_TITLE,   './/article-title | .//chapter-title'),
    (JatsSubFieldNames.REFERENCE_SOURCE,          './/source'),
    (JatsSubFieldNames.REFERENCE_YEAR,            './/year'),
    (JatsSubFieldNames.REFERENCE_VOLUME,          './/volume'),
    (JatsSubFieldNames.REFERENCE_ISSUE,           './/issue'),
    (JatsSubFieldNames.REFERENCE_FPAGE,           './/fpage'),
    (JatsSubFieldNames.REFERENCE_LPAGE,           './/lpage'),
    (JatsSubFieldNames.REFERENCE_PUBLISHER_NAME,  './/publisher-name'),
    (JatsSubFieldNames.REFERENCE_PUBLISHER_LOC,   './/publisher-loc'),
    (JatsSubFieldNames.REFERENCE_DOI,             './/pub-id[@pub-id-type="doi"]'),
    (JatsSubFieldNames.REFERENCE_PMID,            './/pub-id[@pub-id-type="pmid"]'),
    (JatsSubFieldNames.REFERENCE_PMCID,           './/pub-id[@pub-id-type="pmcid"]'),
    (JatsSubFieldNames.REFERENCE_WEB,
     './/ext-link[@ext-link-type="uri"][starts-with(normalize-space(.), "http")]'),
]

# Sub-field XPaths for affiliations (relative to each aff element)
_AFF_SUB_FIELDS = [
    (JatsSubFieldNames.AUTHOR_AFF_LABEL,       './label'),
    (JatsSubFieldNames.AUTHOR_AFF_INSTITUTION, './institution'),
    (JatsSubFieldNames.AUTHOR_AFF_DEPARTMENT,
     './addr-line/named-content[@content-type="department"]'),
    (JatsSubFieldNames.AUTHOR_AFF_CITY,
     './addr-line/named-content[@content-type="city"]'),
    (JatsSubFieldNames.AUTHOR_AFF_POSTCODE,
     './addr-line/named-content[@content-type="postcode"]'),
    (JatsSubFieldNames.AUTHOR_AFF_REGION,      './addr-line/named-content[@content-type="state"]'),
    (JatsSubFieldNames.AUTHOR_AFF_COUNTRY,     './country'),
]


def _iter_reference_author_values(
    ref_el: etree._Element,
    field_name: str,
) -> Iterator[JatsFieldValue]:
    """Yield one JatsFieldValue per <name> in each author person-group, then
    'et al.' if <etal/> is present.  Per-name emission lets the aligner
    match each author independently, so a JATS/PDF format mismatch on one
    name (e.g. given-names abbreviation style) cannot cut off the next name.

    Each name value carries a fallback_text set to just the surname, so the
    aligner can retry with surname-only matching when the full JATS name text
    (e.g. 'Maier BE') does not match the PDF representation ('MAIER, B. F.').
    """
    for pg_el in ref_el.xpath('.//person-group[@person-group-type="author"]'):
        for name_el in pg_el.xpath('name'):
            text = element_text(name_el)
            if not text:
                continue
            surname_el = name_el.find('surname')
            raw_surname = (surname_el.text or '').strip() if surname_el is not None else ''
            fallback = raw_surname if raw_surname and raw_surname != text else None
            yield JatsFieldValue(
                text=text,
                field_name=field_name,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
                fallback_text=fallback,
            )
        if pg_el.xpath('etal'):
            yield JatsFieldValue(
                text='et al.',
                field_name=field_name,
                sub_field_name=JatsSubFieldNames.REFERENCE_AUTHOR,
            )


def iter_reference_sub_field_names(
    root: etree._Element,
) -> Iterator[FrozenSet[str]]:
    """Yield the sub-field names the JATS carries for each reference, in JATS order.

    The reference set is the one iter_field_values emits a REFERENCE value for,
    so a `<ref-list>` inside a `<sub-article>` is not in it: those entries are
    never offered to the aligner and cannot be a shortfall.
    """
    for ref_el in root.xpath('back/ref-list/ref'):
        if not _reference_parent_text(ref_el):
            continue
        yield frozenset(
            field_value.sub_field_name
            for field_value in chain(
                _iter_reference_author_values(ref_el, JatsFieldNames.REFERENCE),
                _iter_sub_field_values(
                    ref_el, JatsFieldNames.REFERENCE, _REFERENCE_SUB_FIELDS
                ),
            )
            if field_value.sub_field_name
        )


def _local_tag(el: etree._Element) -> str:
    tag = el.tag
    if isinstance(tag, str) and tag.startswith('{'):
        return tag.split('}', 1)[1]
    return tag if isinstance(tag, str) else ''


def _aff_addr_parts(aff_el: etree._Element) -> List[str]:
    """Collect address text from an <aff> in document order.

    Covers three JATS patterns:
    - Structured: <addr-line> and/or <country> elements
    - Semi-structured: <institution> present but city/postcode sit in its tail text
      (no <addr-line>), e.g. '<institution>UCL</institution>, London WC1N 1EH,
      <country>UK</country>'
    - Unstructured (label-only affs): returns nothing; address cannot be determined

    Institution tail text is only included when the aff also has a <country> or
    <addr-line> element.  Without that anchor the tail may be continuation of the
    institution name rather than a geographic address (e.g. a department name split
    across two <institution> tags).
    """
    has_structured_addr = bool(aff_el.xpath('./country | ./addr-line'))
    parts: List[str] = []
    for child in aff_el:
        tag = _local_tag(child)
        if tag in ('addr-line', 'country'):
            text = element_text(child)
            if text:
                parts.append(text)
        elif tag == 'institution' and has_structured_addr:
            # Tail text after </institution> is city/postcode when no <addr-line> is present.
            # Only collected when a <country> or <addr-line> confirms this aff has structured
            # address content, to avoid misclassifying department-name continuations.
            tail = ' '.join((child.tail or '').split()).strip(', ')
            if tail:
                parts.append(tail)
    return parts


def _contrib_group_author_texts(contrib_group: etree._Element) -> List[str]:
    """Each author of the group, as the byline prints them.

    JATS stores names in Surname-Given order; PDFs display Given-Surname, so each
    name part is reversed.  Each carries its own affiliation, footnote and
    correspondence markers, because GROBID labels the whole byline `<author>`,
    the markers between the names included.
    """
    author_parts: List[str] = []
    for contrib in contrib_group.xpath(
        'contrib[not(@contrib-type) or @contrib-type="author"]'
    ):
        name_el = contrib.find('name')
        if name_el is None:
            continue
        given = (name_el.findtext('given-names') or '').strip()
        surname = (name_el.findtext('surname') or '').strip()
        name_text = (
            ' '.join(part for part in [given, surname] if part) or element_text(name_el)
        )
        if not name_text:
            continue
        markers = [
            x.text.strip()
            for x in contrib.xpath(
                'xref[@ref-type="aff" or @ref-type="fn" or @ref-type="corresp"]'
            )
            if x.text and x.text.strip()
        ]
        author_parts.append(' '.join([name_text] + markers))
    return author_parts


# How many authors the citation a renderer composes names before "et al.", and
# how many it names in full rather than reaching for one.
_SELF_CITATION_NAMES = 3
_SELF_CITATION_NAMES_IN_FULL = 4


def _cited_names(names: Sequence[str]) -> Sequence[str]:
    if len(names) <= _SELF_CITATION_NAMES_IN_FULL:
        return names
    return names[:_SELF_CITATION_NAMES]


def element_text_of_xpath(root: etree._Element, xpath: str) -> str:
    """The text of the first element the xpath finds, or nothing."""
    for el in root.xpath(xpath):
        text = element_text(el)
        if text:
            return text
    return ''


# The statements whose value is written without the words introducing it, which
# is what the header model's labels are read from.
_LEAD_IN_FIELDS: FrozenSet[str] = frozenset({
    JatsFieldNames.CONFLICT_SECTION_PARAGRAPH,
    JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH,
})


def _name_initials(name_el: etree._Element) -> str:
    given = (name_el.findtext('given-names') or '').strip()
    return ''.join(part[0] for part in re.split(r'[^\w]+', given) if part)


def _contrib_group_role_text(contrib_group: etree._Element) -> str:
    """Who did what, as the front matter prints it.

    The JATS files each author's roles under their own `<contrib>`; the page
    prints them as one statement, surname and initials before each list.

    Only a CRediT role.  A publisher also writes what an author does for a
    living in a `<role>`, which prints inside the affiliation rather than as a
    statement of who did what.
    """
    statements: List[str] = []
    for contrib in contrib_group.xpath(
        'contrib[not(@contrib-type) or @contrib-type="author"]'
    ):
        name_el = contrib.find('name')
        roles = [
            element_text(role_el)
            for role_el in contrib.xpath('role[contains(@content-type, "credit")]')
            if element_text(role_el)
        ]
        if name_el is None or not roles:
            continue
        surname = (name_el.findtext('surname') or '').strip()
        name_text = ' '.join(
            part for part in (surname, _name_initials(name_el)) if part
        )
        statements.append(f'{name_text}: ' + ', '.join(roles))
    return '; '.join(statements)


def _iter_aff_elements(root: etree._Element) -> Iterator[etree._Element]:
    yield from root.xpath(
        'front/article-meta/contrib-group/aff'
        '| front/article-meta/contrib-group/contrib/aff'
        '| front/article-meta/aff'
    )


class JatsFieldExtractor:
    """Extract (text, field_name, sub_field_name) triples from a JATS <article> root."""

    def iter_field_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        yield from self._iter_front_values(root)
        yield from self._iter_body_values(root)
        yield from self._iter_back_values(root)
        yield from self._iter_floats_group_values(root)
        yield from self._iter_sub_article_values(root)

    def _emit(
        self,
        elements: List[etree._Element],
        field_name: str,
    ) -> Iterator[JatsFieldValue]:
        for el in elements:
            text = element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=field_name)

    # ── Front matter ──────────────────────────────────────────────────────────

    def _iter_front_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        yield from self._iter_front_metadata_values(root)
        yield from self._iter_front_contrib_values(root)

    def _iter_front_metadata_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        # A document written in two languages prints both titles, one under the
        # other, and declares the second as <trans-title>.  Read only the first
        # and the other is left unlabelled, directly above a byline that is not.
        for el in root.xpath(
            'front/article-meta/title-group/article-title'
            ' | front/article-meta/title-group/trans-title-group/trans-title'
        ):
            text = element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.TITLE)

        # A document that prints its abstract twice declares the second as
        # <trans-abstract>.  Both are printed, so both are the header the page
        # shows; read only the first and the other is left unlabelled, and which
        # region it ends up in is decided by whatever happens to border it.
        for el in root.xpath(
            'front/article-meta/abstract | front/article-meta/trans-abstract'
        ):
            text = element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.ABSTRACT)

        # Per GROBID annotation guidelines, the "Keywords" heading is not annotated
        # in the header model.  It is still emitted as KEYWORDS_TITLE so that the
        # segmentation model can label it as <header>.  Combine all <kwd> children
        # of a <kwd-group> into a single KEYWORDS field value.
        for kwd_group in root.xpath('front/article-meta/kwd-group'):
            for title_el in kwd_group.xpath('./title'):
                text = element_text(title_el)
                if text:
                    yield JatsFieldValue(text=text, field_name=JatsFieldNames.KEYWORDS_TITLE)
            kwd_texts = [
                element_text(kwd_el)
                for kwd_el in kwd_group.xpath('./kwd')
                if element_text(kwd_el)
            ]
            if kwd_texts:
                yield JatsFieldValue(
                    text=', '.join(kwd_texts),
                    field_name=JatsFieldNames.KEYWORDS,
                )

        # What the article calls itself, printed above the title.  A publisher
        # says it either on the subject or on the group holding it.
        #
        # Verbatim or not at all: one publisher's whole answer is "Article", and
        # matched approximately that reaches any word of the page that looks
        # like it -- "particular" on one preprint.
        for el in root.xpath(
            'front/article-meta/article-categories'
            '/subj-group/subject[@subj-group-type="display-channel"]'
            ' | front/article-meta/article-categories'
            '/subj-group[@subj-group-type="heading"]/subject'
        ):
            text = element_text(el)
            if text:
                yield JatsFieldValue(
                    text=text,
                    field_name=JatsFieldNames.MANUSCRIPT_TYPE,
                    exact_only=True,
                )

        yield from self._iter_front_publication_values(root)

    def _iter_front_publication_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """Funding statements and copyright / licence text from front matter."""
        for el in root.xpath('front/article-meta/funding-group/funding-statement'):
            text = element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.FUNDING)

        for el in root.xpath(
            'front/article-meta/permissions/copyright-statement'
            ' | front/article-meta/permissions/license/license-p'
        ):
            text = element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.COPYRIGHT)

    def _iter_front_notes_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """`<front>/<notes>` is what a versioned article says changed since the
        previous version, printed in the front matter, boxed."""
        for el in root.xpath('front/notes/*'):
            text = element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.ARTICLE_NOTES)

    def _iter_self_citation_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """How the article asks to be cited, as the renderer composes it.

        No element carries it: the page sets the authors, the title and where it
        appeared, and the renderer builds the line from the metadata.  The
        identifier at the end of it is left out, because GROBID writes that as
        the `<idno>` beside the reference rather than as part of it.

        Every author up to four, and three where there are more: that is what
        the renderer prints, four names joined with "and" on the papers that
        have four and three with "et al." on the ones with five and with
        eleven.  Listing the rest would add names the page does not print, and
        the match pays for those where it does not pay for the words it skips.

        In two values, because the renderer sets a phrase between them that no
        element carries -- which version this is and how its reviewers voted --
        and one value covering both halves pays for that phrase: on the document
        whose reviewers disagreed at length it is 75 characters and the match
        falls under the threshold.  Where it appeared is the second of them, as
        a sub-field, so that it is looked for inside the citation: the journal,
        the year and the volume are what the running foot of every page carries
        as well.
        """
        names = [
            name
            for contrib in root.xpath(
                'front/article-meta/contrib-group'
                '/contrib[not(@contrib-type) or @contrib-type="author"]'
            )
            for name_el in [contrib.find('name')]
            if name_el is not None
            for name in [' '.join(part for part in (
                (name_el.findtext('surname') or '').strip(),
                _name_initials(name_el),
            ) if part)]
            if name
        ]
        title = element_text_of_xpath(root, 'front/article-meta/title-group/article-title')
        journal = element_text_of_xpath(
            root, 'front/journal-meta/journal-title-group/journal-title'
            ' | front/journal-meta/journal-title'
        )
        year = element_text_of_xpath(
            root, 'front/article-meta/pub-date[@pub-type="epub"]/year'
        )
        volume = element_text_of_xpath(root, 'front/article-meta/volume')
        elocation = element_text_of_xpath(root, 'front/article-meta/elocation-id')
        if not (names and title and journal and year and volume and elocation):
            return
        yield JatsFieldValue(
            text=f'{", ".join(_cited_names(names))} {title}',
            field_name=JatsFieldNames.SELF_CITATION,
        )
        yield JatsFieldValue(
            text=f'{journal} {year}, {volume}:{elocation}',
            field_name=JatsFieldNames.SELF_CITATION,
            sub_field_name=JatsSubFieldNames.SELF_CITATION_SOURCE,
        )

    def _iter_publication_date_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """When this version was published, as the version history prints it.

        The JATS carries the day, the month and the year apart, and the page
        sets them as "06 Sep 2023": a zero-padded day, the month in three
        letters, which is the form both of these publishers print.

        Verbatim or not at all.  A date is eleven characters and a version
        history prints one per version, so approximately matched it reaches the
        neighbouring one -- "23 Jun 2022" and "12 Jul 2022" share nine of their
        characters in the same places.
        """
        for el in root.xpath('front/article-meta/pub-date[@pub-type="epub"]'):
            day, month, year = (
                (el.findtext('day') or '').strip(),
                (el.findtext('month') or '').strip(),
                (el.findtext('year') or '').strip(),
            )
            if not (day and month and year):
                continue
            try:
                printed = date(int(year), int(month), int(day)).strftime('%d %b %Y')
            except ValueError:
                continue
            yield JatsFieldValue(
                text=printed,
                field_name=JatsFieldNames.PUBLICATION_DATE,
                exact_only=True,
            )

    def _iter_front_identifier_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """The article's own DOI, as the page prints it.

        A preprint server prints it as the resolver URL rather than as the bare
        identifier the JATS carries, and prints it more than once: in the version
        history and in the citation the renderer composes from the metadata.
        """
        for el in root.xpath('front/article-meta/article-id[@pub-id-type="doi"]'):
            text = element_text(el)
            if text:
                yield JatsFieldValue(
                    text=f'https://doi.org/{text}',
                    field_name=JatsFieldNames.DOI,
                )

    def _iter_front_contact_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """An author's email and ORCID, which print beside the name they belong to.

        The JATS carries each exactly as the page prints it, an ORCID as the full
        URL, so neither needs building from parts.
        """
        for xpath, field_name in (
            ('front/article-meta//email', JatsFieldNames.EMAIL),
            (
                'front/article-meta//contrib-id[@contrib-id-type="orcid"]',
                JatsFieldNames.ORCID,
            ),
        ):
            for el in root.xpath(xpath):
                text = element_text(el)
                if text:
                    yield JatsFieldValue(text=text, field_name=field_name)

    def _iter_author_note_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """The notes printed under the byline, and the declarations among them.

        A publisher files the conflict and contribution statements here as well
        as in the back matter, and prints them with the front matter either way,
        which is the header's to label rather than a region of its own.  The
        rest -- who to write to, who contributed equally -- has no label, and is
        emitted so that the segmentation model reads the lines as front matter.
        """
        for el in root.xpath('front/article-meta/author-notes/*'):
            text = element_text(el)
            if not text:
                continue
            note_field = (
                NOTE_PARAGRAPH_FIELD_BY_KIND.get(get_fn_kind(el) or '')
                if el.tag == 'fn' else None
            )
            if note_field is None:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.AUTHOR_NOTES)
                continue
            statement = statement_text(el)
            if statement:
                yield JatsFieldValue(text=statement, field_name=note_field)

    def _iter_front_contrib_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        # Per GROBID annotation guidelines, all author tokens in the byline (including
        # affiliation markers and separating punctuation) are labelled <author>.
        # Emit one merged field value per contrib-group so the aligner covers the
        # whole byline span, including commas, "&", etc. between individual names.
        # JATS stores names in Surname-Given order; PDFs display Given-Surname, so
        # each name part is reversed.  Affiliation/fn/corresp xref markers are
        # appended to each name so the combined needle matches the PDF author line.
        for contrib_group in root.xpath('front/article-meta/contrib-group'):
            author_parts = _contrib_group_author_texts(contrib_group)
            # The merged value matches a byline printed as one run, which is what
            # it is for.  Where each author is printed against their own
            # affiliation instead, several hundred characters separate one name
            # from the next and the run matches nothing like itself, so each
            # author is offered on their own as well.
            #
            # They come first because a sub-field whose parent has matched is
            # searched inside that match, and the run matches a few of its own
            # words somewhere even where it does not match the byline -- which
            # would confine the names to wherever those words happened to land.
            for author_text in author_parts:
                yield JatsFieldValue(
                    text=author_text,
                    field_name=JatsFieldNames.AUTHOR,
                    sub_field_name=JatsSubFieldNames.AUTHOR_NAME,
                )
            if author_parts:
                yield JatsFieldValue(
                    text=' '.join(author_parts),
                    field_name=JatsFieldNames.AUTHOR,
                )
            role_text = _contrib_group_role_text(contrib_group)
            if role_text:
                yield JatsFieldValue(
                    text=role_text,
                    field_name=JatsFieldNames.CONTRIBUTION_NOTE,
                )

        for aff_el in _iter_aff_elements(root):
            text = element_text(aff_el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.AUTHOR_AFF)
            # Emit a bulk address value BEFORE individual sub-fields so that commas
            # between city and country also get the AUTHOR_AFF_ADDR sub-field label
            # (individual city/country sub-fields overwrite their own tokens afterward).
            addr_texts = _aff_addr_parts(aff_el)
            if addr_texts:
                yield JatsFieldValue(
                    text=' '.join(addr_texts),
                    field_name=JatsFieldNames.AUTHOR_AFF,
                    sub_field_name=JatsSubFieldNames.AUTHOR_AFF_ADDR,
                )
            yield from _iter_sub_field_values(
                aff_el, JatsFieldNames.AUTHOR_AFF, _AFF_SUB_FIELDS
            )

        # The notes first, and the contacts after them: a corresponding-author
        # note is the email with a label in front of it, and whichever is
        # matched last is the label the tokens keep.  The note has none of its
        # own, so matched last it takes the email's away.
        yield from self._iter_author_note_values(root)

        yield from self._iter_front_contact_values(root)

        yield from self._iter_self_citation_values(root)

        yield from self._iter_front_identifier_values(root)

        yield from self._iter_publication_date_values(root)

        yield from self._iter_front_notes_values(root)

        for el in root.xpath('front/article-meta/fpage | front/article-meta/lpage'):
            text = element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.PAGE_NO)

    # ── Body ──────────────────────────────────────────────────────────────────

    def _iter_body_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        body = root.find('body')
        if body is None:
            return
        # Build document-order index so that section titles, paragraphs, figures,
        # and tables are yielded interleaved as they appear in the XML, not grouped
        # by type.  If section titles are all emitted first the aligner's
        # body_content_end advances past the early paragraphs before they are matched.
        position: Dict[etree._Element, int] = {el: i for i, el in enumerate(root.iter())}
        entries: List[Tuple[int, JatsFieldValue]] = []
        labelled = LabelledSections.of(root)

        for el in body.xpath('.//sec/title'):
            text = element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text,
                    field_name=labelled.title_field(
                        el, JatsFieldNames.BODY_SECTION_TITLE
                    ))))

        for el in body.xpath('.//p[not(ancestor::fig) and not(ancestor::table-wrap)]'):
            text = element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text,
                    field_name=labelled.paragraph_field(
                        el, JatsFieldNames.BODY_SECTION_PARAGRAPH
                    ))))

        for el in body.xpath('.//fig'):
            children = el.xpath('./label') + el.xpath('./caption')
            text = (element_text(el) if not children
                    else ' '.join(element_text(c) for c in children if element_text(c)))
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BODY_FIGURE)))

        for el in body.xpath('.//table-wrap'):
            children = el.xpath('./label') + el.xpath('./caption')
            text = (element_text(el) if not children
                    else ' '.join(element_text(c) for c in children if element_text(c)))
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BODY_TABLE)))

        for _, fv in sorted(entries):
            yield fv

    def _iter_floats_group_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """Yield the captions of floats JATS holds at the end of the document.

        `<floats-group>` is where a publisher puts tables and figures that print
        after the reference list rather than beside the text that cites them.
        Nothing else reads it, so its pages -- seventeen of them on `PPR458717`,
        and some share of 49 of 50 scielo documents -- had no field to align
        against and fell to the `<body>` sink.

        Only the label and caption are taken, as the body path does, because the
        cells of a table do not align as running text.  That is enough: the
        caption anchors the float, and the deriver's gap merge carries `<annex>`
        over the rest of it and on to the end of the document.
        """
        floats_group = root.find('floats-group')
        if floats_group is None:
            return
        position: Dict[etree._Element, int] = {el: i for i, el in enumerate(root.iter())}
        entries: List[Tuple[int, JatsFieldValue]] = []
        for xpath, field_name in (
            ('.//fig', JatsFieldNames.FLOAT_FIGURE),
            ('.//table-wrap', JatsFieldNames.FLOAT_TABLE),
        ):
            for el in floats_group.xpath(xpath):
                children = el.xpath('./label') + el.xpath('./caption')
                text = (element_text(el) if not children
                        else ' '.join(element_text(c) for c in children if element_text(c)))
                if text:
                    entries.append((position[el], JatsFieldValue(
                        text=text, field_name=field_name)))
        for _, fv in sorted(entries):
            yield fv

    # ── Back matter ───────────────────────────────────────────────────────────

    def _iter_back_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        yield from self._iter_back_narrative_values(root)
        yield from self._iter_back_reference_values(root)

    def _iter_back_narrative_values(  # pylint: disable=too-many-branches
        self, root: etree._Element
    ) -> Iterator[JatsFieldValue]:
        position: Dict[etree._Element, int] = {el: i for i, el in enumerate(root.iter())}
        entries: List[Tuple[int, JatsFieldValue]] = []

        # A publisher files a funding statement under `<ack>` and lets its title
        # say so, and the title is the better evidence of the two.
        for ack in root.xpath('//ack'):
            title_el = ack.find('title')
            kind = (
                match_fn_keywords(element_text(title_el))
                if title_el is not None else None
            ) or 'acknowledgement'
            for el in ack.xpath('.//title'):
                text = element_text(el)
                if text:
                    entries.append((position[el], JatsFieldValue(
                        text=text, field_name=TITLE_FIELD_BY_KIND[kind])))
            for el in ack.xpath('.//p'):
                text = element_text(el)
                if text:
                    entries.append((position[el], JatsFieldValue(
                        text=text, field_name=PARAGRAPH_FIELD_BY_KIND[kind])))

        for el in root.xpath('//app-group/title'):
            text = element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.APPENDIX_GROUP_TITLE)))

        for el in root.xpath('//app'):
            text = element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.APPENDIX)))

        # A data-availability section is back matter the model has its own label
        # for, so it is taken out of the generic sweep rather than labelled
        # `<annex>` with everything else that follows the body.
        labelled = LabelledSections.of(root)

        for el in root.xpath('back//sec//title | back//fn//bold'):
            text = element_text(el)
            if text and labelled.claims(el) and (el.tag != 'bold' or is_bold_heading(el)):
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=labelled.title_field(el, ''))))

        for el in root.xpath('back//sec//p | back//fn//p'):
            if not labelled.claims(el) or is_bold_heading(el):
                continue
            field_name = labelled.paragraph_field(el, '')
            # A footnote states its kind in bold at the head of the paragraph as
            # often as in a paragraph of its own, and either way the page prints
            # it as a lead-in rather than as part of the statement -- for the two
            # statements the header model has a label for.  For the rest the
            # whole paragraph is the value, because what the value does there is
            # place a region: without its lead-in, one ethics statement is left
            # as its approval number and matches the methods sentence quoting it.
            text = (
                statement_text(el) if field_name in _LEAD_IN_FIELDS
                else element_text(el)
            )
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=field_name)))

        for el in root.xpath('back//sec[not(ancestor::ack)]/title'):
            text = element_text(el)
            if text and not labelled.claims(el):
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BACK_SECTION_TITLE)))

        for el in root.xpath(
            'back//sec[not(ancestor::ack)]/p[not(ancestor::ack)]'
            ' | back//p[not(ancestor::sec) and not(ancestor::ack)]'
        ):
            text = element_text(el)
            if text and not labelled.claims(el):
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BACK_SECTION_PARAGRAPH)))

        for _, fv in sorted(entries):
            yield fv

    def _iter_back_reference_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        for el in root.xpath('back/ref-list/title'):
            text = element_text(el)
            if text:
                yield JatsFieldValue(
                    text=text, field_name=JatsFieldNames.REFERENCE_LIST_TITLE
                )

        for ref_el in root.xpath('back/ref-list/ref'):
            text = _reference_parent_text(ref_el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.REFERENCE)
            yield from _iter_reference_author_values(ref_el, JatsFieldNames.REFERENCE)
            yield from _iter_sub_field_values(
                ref_el, JatsFieldNames.REFERENCE, _REFERENCE_SUB_FIELDS
            )

    # ── Sub-articles (ORE peer-review reports, etc.) ──────────────────────────

    def _iter_sub_article_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """Yield paragraph/title text from <sub-article> elements as SUB_ARTICLE values.

        ORE papers embed peer-review reports as sub-articles.  Extracting their
        content in document order lets the aligner map those PDF pages to the
        SUB_ARTICLE field, which the segmentation model labels as <other> rather
        than <body>.
        """
        position: Dict[etree._Element, int] = {el: i for i, el in enumerate(root.iter())}
        entries: List[Tuple[int, JatsFieldValue]] = []

        for sub_article in root.xpath('.//sub-article'):
            for el in sub_article.xpath('.//title | .//p'):
                text = element_text(el)
                if text:
                    entries.append((position[el], JatsFieldValue(
                        text=text, field_name=JatsFieldNames.SUB_ARTICLE)))
            for el, text in _iter_sub_article_stub_texts(sub_article):
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.SUB_ARTICLE, exact_only=True)))

        for _, fv in sorted(entries, key=lambda entry: entry[0]):
            yield fv
