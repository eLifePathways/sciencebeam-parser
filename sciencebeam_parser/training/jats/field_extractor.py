from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from lxml import etree

from sciencebeam_parser.training.jats.field_vocab import (
    JatsFieldNames,
    JatsSubFieldNames,
)


@dataclass
class JatsFieldValue:
    text: str
    field_name: str
    sub_field_name: Optional[str] = None


def _element_text(el: etree._Element) -> str:
    return ' '.join(' '.join(el.itertext()).split())


def _iter_sub_field_values(
    parent_el: etree._Element,
    field_name: str,
    sub_xpath_by_sub_field: Sequence[tuple],
) -> Iterator[JatsFieldValue]:
    """Yield one JatsFieldValue per sub-field span found in parent_el."""
    for sub_field_name, xpath in sub_xpath_by_sub_field:
        for child in parent_el.xpath(xpath):
            text = _element_text(child)
            if text:
                yield JatsFieldValue(
                    text=text,
                    field_name=field_name,
                    sub_field_name=sub_field_name,
                )


# Sub-field XPaths for references (relative to each <ref> element)
_REFERENCE_SUB_FIELDS = [
    (JatsSubFieldNames.REFERENCE_LABEL,           './label'),
    (JatsSubFieldNames.REFERENCE_AUTHOR,          './/person-group[@person-group-type="author"]'),
    (JatsSubFieldNames.REFERENCE_ARTICLE_TITLE,   './/article-title'),
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
            text = _element_text(child)
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
        yield from self._iter_sub_article_values(root)

    def _emit(
        self,
        elements: List[etree._Element],
        field_name: str,
    ) -> Iterator[JatsFieldValue]:
        for el in elements:
            text = _element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=field_name)

    # ── Front matter ──────────────────────────────────────────────────────────

    def _iter_front_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        yield from self._iter_front_metadata_values(root)
        yield from self._iter_front_contrib_values(root)

    def _iter_front_metadata_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        for el in root.xpath('front/article-meta/title-group/article-title'):
            text = _element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.TITLE)

        for el in root.xpath('front/article-meta/abstract'):
            text = _element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.ABSTRACT)

        # Per GROBID annotation guidelines, the "Keywords" heading is not annotated
        # in the header model.  It is still emitted as KEYWORDS_TITLE so that the
        # segmentation model can label it as <header>.  Combine all <kwd> children
        # of a <kwd-group> into a single KEYWORDS field value.
        for kwd_group in root.xpath('front/article-meta/kwd-group'):
            for title_el in kwd_group.xpath('./title'):
                text = _element_text(title_el)
                if text:
                    yield JatsFieldValue(text=text, field_name=JatsFieldNames.KEYWORDS_TITLE)
            kwd_texts = [
                _element_text(kwd_el)
                for kwd_el in kwd_group.xpath('./kwd')
                if _element_text(kwd_el)
            ]
            if kwd_texts:
                yield JatsFieldValue(
                    text=', '.join(kwd_texts),
                    field_name=JatsFieldNames.KEYWORDS,
                )

        for el in root.xpath(
            'front/article-meta/article-categories'
            '/subj-group/subject[@subj-group-type="display-channel"]'
        ):
            text = _element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.MANUSCRIPT_TYPE)

        yield from self._iter_front_publication_values(root)

    def _iter_front_publication_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        """Funding statements and copyright / licence text from front matter."""
        for el in root.xpath('front/article-meta/funding-group/funding-statement'):
            text = _element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.FUNDING)

        for el in root.xpath(
            'front/article-meta/permissions/copyright-statement'
            ' | front/article-meta/permissions/license/license-p'
        ):
            text = _element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.COPYRIGHT)

    def _iter_front_contrib_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        # Per GROBID annotation guidelines, all author tokens in the byline (including
        # affiliation markers and separating punctuation) are labelled <author>.
        # Emit one merged field value per contrib-group so the aligner covers the
        # whole byline span, including commas, "&", etc. between individual names.
        # JATS stores names in Surname-Given order; PDFs display Given-Surname, so
        # each name part is reversed.  Affiliation/fn/corresp xref markers are
        # appended to each name so the combined needle matches the PDF author line.
        for contrib_group in root.xpath('front/article-meta/contrib-group'):
            author_parts = []
            for contrib in contrib_group.xpath(
                'contrib[not(@contrib-type) or @contrib-type="author"]'
            ):
                name_el = contrib.find('name')
                if name_el is None:
                    continue
                given = (name_el.findtext('given-names') or '').strip()
                surname = (name_el.findtext('surname') or '').strip()
                name_text = (
                    ' '.join(p for p in [given, surname] if p) or _element_text(name_el)
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
            if author_parts:
                yield JatsFieldValue(
                    text=' '.join(author_parts),
                    field_name=JatsFieldNames.AUTHOR,
                )

        for aff_el in _iter_aff_elements(root):
            text = _element_text(aff_el)
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

        for el in root.xpath('front/article-meta/author-notes/*'):
            text = _element_text(el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.AUTHOR_NOTES)

        for el in root.xpath('front/article-meta/fpage | front/article-meta/lpage'):
            text = _element_text(el)
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

        for el in body.xpath('.//sec/title'):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BODY_SECTION_TITLE)))

        for el in body.xpath('.//p[not(ancestor::fig) and not(ancestor::table-wrap)]'):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BODY_SECTION_PARAGRAPH)))

        for el in body.xpath('.//fig'):
            children = el.xpath('./label') + el.xpath('./caption')
            text = (_element_text(el) if not children
                    else ' '.join(_element_text(c) for c in children if _element_text(c)))
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BODY_FIGURE)))

        for el in body.xpath('.//table-wrap'):
            children = el.xpath('./label') + el.xpath('./caption')
            text = (_element_text(el) if not children
                    else ' '.join(_element_text(c) for c in children if _element_text(c)))
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BODY_TABLE)))

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

        for el in root.xpath('//ack//title'):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.ACK_SECTION_TITLE)))

        for el in root.xpath('//ack//p'):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.ACK_SECTION_PARAGRAPH)))

        for el in root.xpath('//app-group/title'):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.APPENDIX_GROUP_TITLE)))

        for el in root.xpath('//app'):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.APPENDIX)))

        for el in root.xpath('back//sec[not(ancestor::ack)]/title'):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BACK_SECTION_TITLE)))

        for el in root.xpath(
            'back//sec[not(ancestor::ack)]/p[not(ancestor::ack)]'
            ' | back//p[not(ancestor::sec) and not(ancestor::ack)]'
        ):
            text = _element_text(el)
            if text:
                entries.append((position[el], JatsFieldValue(
                    text=text, field_name=JatsFieldNames.BACK_SECTION_PARAGRAPH)))

        for _, fv in sorted(entries):
            yield fv

    def _iter_back_reference_values(self, root: etree._Element) -> Iterator[JatsFieldValue]:
        for el in root.xpath('back/ref-list/title'):
            text = _element_text(el)
            if text:
                yield JatsFieldValue(
                    text=text, field_name=JatsFieldNames.REFERENCE_LIST_TITLE
                )

        for ref_el in root.xpath('back/ref-list/ref'):
            text = _element_text(ref_el)
            if text:
                yield JatsFieldValue(text=text, field_name=JatsFieldNames.REFERENCE)
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
                text = _element_text(el)
                if text:
                    entries.append((position[el], JatsFieldValue(
                        text=text, field_name=JatsFieldNames.SUB_ARTICLE)))

        for _, fv in sorted(entries):
            yield fv
