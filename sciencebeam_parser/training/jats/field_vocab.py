from typing import Dict

from sciencebeam_parser.models.citation.labels import IDENTIFIER_LABEL


class JatsFieldNames:
    TITLE = 'title'
    ABSTRACT = 'abstract'
    KEYWORDS_TITLE = 'keywords_title'
    KEYWORDS = 'keywords'
    MANUSCRIPT_TYPE = 'manuscript_type'
    AUTHOR = 'author'
    AUTHOR_AFF = 'author_aff'
    AUTHOR_NOTES = 'author_notes'
    ARTICLE_NOTES = 'article_notes'
    FUNDING = 'funding'
    COPYRIGHT = 'copyright'
    SUB_ARTICLE = 'sub_article'
    BODY_SECTION_TITLE = 'body_section_title'
    BODY_SECTION_PARAGRAPH = 'body_section_paragraph'
    BODY_FIGURE = 'body_figure'
    BODY_TABLE = 'body_table'
    FLOAT_FIGURE = 'float_figure'
    FLOAT_TABLE = 'float_table'
    BACK_SECTION_TITLE = 'back_section_title'
    BACK_SECTION_PARAGRAPH = 'back_section_paragraph'
    AVAILABILITY_SECTION_TITLE = 'availability_section_title'
    AVAILABILITY_SECTION_PARAGRAPH = 'availability_section_paragraph'
    FUNDING_SECTION_TITLE = 'funding_section_title'
    FUNDING_SECTION_PARAGRAPH = 'funding_section_paragraph'
    BACK_FOOTNOTE = 'back_footnote'
    CONFLICT_SECTION_TITLE = 'conflict_section_title'
    CONFLICT_SECTION_PARAGRAPH = 'conflict_section_paragraph'
    CONTRIBUTION_SECTION_TITLE = 'contribution_section_title'
    CONTRIBUTION_SECTION_PARAGRAPH = 'contribution_section_paragraph'
    # The same two declarations, filed as a footnote rather than a section, which
    # is how a publisher says it prints them with the front matter rather than
    # after the body.
    CONFLICT_NOTE = 'conflict_note'
    CONTRIBUTION_NOTE = 'contribution_note'
    ACK_SECTION_TITLE = 'acknowledgment_section_title'
    ACK_SECTION_PARAGRAPH = 'acknowledgment_section_paragraph'
    APPENDIX_GROUP_TITLE = 'appendix_group_title'
    APPENDIX = 'appendix'
    REFERENCE_LIST_TITLE = 'reference_list_title'
    REFERENCE = 'reference'
    PAGE_NO = 'page_no'
    EMAIL = 'email'
    ORCID = 'orcid'


class JatsSubFieldNames:
    AUTHOR_AFF_ADDR = 'author_aff_addr'
    AUTHOR_NAME = 'author-name'
    REFERENCE_AUTHOR = 'reference-author'
    REFERENCE_ARTICLE_TITLE = 'reference-article-title'
    REFERENCE_SOURCE = 'reference-source'
    REFERENCE_YEAR = 'reference-year'
    REFERENCE_VOLUME = 'reference-volume'
    REFERENCE_ISSUE = 'reference-issue'
    REFERENCE_FPAGE = 'reference-fpage'
    REFERENCE_LPAGE = 'reference-lpage'
    REFERENCE_DOI = 'reference-doi'
    REFERENCE_PMID = 'reference-pmid'
    REFERENCE_PMCID = 'reference-pmcid'
    REFERENCE_WEB = 'reference-web'
    REFERENCE_LABEL = 'reference-label'
    REFERENCE_PUBLISHER_NAME = 'reference-publisher-name'
    REFERENCE_PUBLISHER_LOC = 'reference-publisher-loc'
    AUTHOR_AFF_LABEL = 'author_aff-label'
    AUTHOR_AFF_INSTITUTION = 'author_aff-institution'
    AUTHOR_AFF_DEPARTMENT = 'author_aff-department'
    AUTHOR_AFF_CITY = 'author_aff-address-city'
    AUTHOR_AFF_POSTCODE = 'author_aff-address-postcode'
    AUTHOR_AFF_REGION = 'author_aff-address-state'
    AUTHOR_AFF_COUNTRY = 'author_aff-address-country'


# ── Segmentation label mapping (mirrors segmentation.conf [tags]) ─────────────
SEGMENTATION_LABEL_BY_FIELD: Dict[str, str] = {
    JatsFieldNames.TITLE:                  '<header>',
    JatsFieldNames.ABSTRACT:               '<header>',
    JatsFieldNames.KEYWORDS_TITLE:         '<header>',
    JatsFieldNames.KEYWORDS:               '<header>',
    JatsFieldNames.MANUSCRIPT_TYPE:        '<header>',
    JatsFieldNames.AUTHOR:                 '<header>',
    JatsFieldNames.AUTHOR_AFF:             '<header>',
    JatsFieldNames.AUTHOR_NOTES:           '<header>',
    JatsFieldNames.ARTICLE_NOTES:          '<header>',
    JatsFieldNames.FUNDING:               '<header>',
    JatsFieldNames.COPYRIGHT:             '<header>',
    JatsFieldNames.SUB_ARTICLE:           '<review>',
    JatsFieldNames.BODY_SECTION_TITLE:     '<body>',
    JatsFieldNames.BODY_SECTION_PARAGRAPH: '<body>',
    JatsFieldNames.BODY_FIGURE:            '<body>',
    JatsFieldNames.BODY_TABLE:             '<body>',
    # `<body>` wherever they print.  GROBID's own corpus labels a bare float
    # caption after the reference list `<body>` 902 times against `<annex>` 234,
    # and reserves `<annex>` for a named supplementary or appendix section, so a
    # corpus meant to mix with it has to agree.
    JatsFieldNames.FLOAT_FIGURE:           '<body>',
    JatsFieldNames.FLOAT_TABLE:            '<body>',
    JatsFieldNames.AVAILABILITY_SECTION_TITLE:     '<availability>',
    JatsFieldNames.AVAILABILITY_SECTION_PARAGRAPH: '<availability>',
    JatsFieldNames.FUNDING_SECTION_TITLE:          '<funding>',
    JatsFieldNames.FUNDING_SECTION_PARAGRAPH:      '<funding>',
    JatsFieldNames.BACK_FOOTNOTE:                  '<footnote>',
    JatsFieldNames.CONFLICT_SECTION_TITLE:         '<conflict>',
    JatsFieldNames.CONFLICT_SECTION_PARAGRAPH:     '<conflict>',
    JatsFieldNames.CONTRIBUTION_SECTION_TITLE:     '<contribution>',
    JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH: '<contribution>',
    # A declaration printed with the front matter is part of it, as GROBID's own
    # corpora have it both ways: 113 of its header documents hold a conflict note
    # and 366 of its segmentation documents a `<div type="conflict">` region.
    JatsFieldNames.CONFLICT_NOTE:                  '<header>',
    JatsFieldNames.CONTRIBUTION_NOTE:              '<header>',
    JatsFieldNames.ACK_SECTION_TITLE:      '<acknowledgement>',
    JatsFieldNames.ACK_SECTION_PARAGRAPH:  '<acknowledgement>',
    JatsFieldNames.APPENDIX_GROUP_TITLE:   '<annex>',
    JatsFieldNames.APPENDIX:               '<annex>',
    JatsFieldNames.BACK_SECTION_TITLE:     '<annex>',
    JatsFieldNames.BACK_SECTION_PARAGRAPH: '<annex>',
    JatsFieldNames.REFERENCE_LIST_TITLE:   '<references>',
    JatsFieldNames.REFERENCE:              '<references>',
    JatsFieldNames.PAGE_NO:                '<page>',
    # An email and an ORCID print inside the author block, so a line carrying one
    # is front matter: it claims two contact lines on `PPR459453` that read as
    # body without it, and an affiliation line on `PPR459468` that reads as a
    # page foot.
    JatsFieldNames.EMAIL:                  '<header>',
    JatsFieldNames.ORCID:                  '<header>',
}

# ── Header model label mapping ────────────────────────────────────────────────
HEADER_LABEL_BY_FIELD: Dict[str, str] = {
    JatsFieldNames.TITLE:            '<title>',
    JatsFieldNames.ABSTRACT:         '<abstract>',
    JatsFieldNames.KEYWORDS:         '<keyword>',
    JatsFieldNames.AUTHOR:           '<author>',
    JatsFieldNames.AUTHOR_AFF:       '<affiliation>',
    # `<note>` is not mapped to, and nothing should be: the TEI writes it as bare
    # text under `<front>`, exactly as it writes background, and the parser reads
    # text with no element of its own back as `O`.  A field mapped to it is
    # located, counted as marked, and taught as background.  GROBID writes a bare
    # `<note>` on 1 of its 835 documents, so there is nothing to agree with here
    # either -- a corresponding author's name, email and address take their own
    # labels there, and the words introducing them are `<other>`.
    JatsFieldNames.COPYRIGHT:        '<copyright>',
    JatsFieldNames.FUNDING:          '<funding>',
    JatsFieldNames.EMAIL:            '<email>',
    JatsFieldNames.ORCID:            '<pubnum>',
    # A statement's own heading is not part of it: GROBID opens none of its 115
    # conflict notes, 11 contribution notes or 878 abstracts with one, which is
    # why `keywords_title` is extracted for segmentation and left unmapped here.
    JatsFieldNames.CONFLICT_SECTION_PARAGRAPH:     '<conflict>',
    JatsFieldNames.CONTRIBUTION_SECTION_PARAGRAPH: '<contribution>',
    JatsFieldNames.CONFLICT_NOTE:                  '<conflict>',
    JatsFieldNames.CONTRIBUTION_NOTE:              '<contribution>',
    # What the article calls itself, which prints above the title.  GROBID writes
    # it as `<note type="doctype">` on 267 of its 835 header documents, "RESEARCH
    # ARTICLE" 88 times.
    JatsFieldNames.MANUSCRIPT_TYPE:  '<doctype>',
}

# ── Fulltext model label mapping ──────────────────────────────────────────────
FULLTEXT_LABEL_BY_FIELD: Dict[str, str] = {
    JatsFieldNames.BODY_SECTION_TITLE:     '<section>',
    JatsFieldNames.BODY_SECTION_PARAGRAPH: '<paragraph>',
    JatsFieldNames.BODY_FIGURE:            '<figure>',
    JatsFieldNames.BODY_TABLE:             '<table>',
    JatsFieldNames.ACK_SECTION_TITLE:      '<section>',
    JatsFieldNames.ACK_SECTION_PARAGRAPH:  '<paragraph>',
    JatsFieldNames.BACK_SECTION_TITLE:     '<section>',
    JatsFieldNames.BACK_SECTION_PARAGRAPH: '<paragraph>',
}

# ── Citation model label mapping (keyed by sub-field name) ────────────────────
# Tokens whose sub_field_name matches get this label; all others → <note>.
CITATION_LABEL_BY_SUB_FIELD: Dict[str, str] = {
    JatsSubFieldNames.REFERENCE_AUTHOR:          '<author>',
    JatsSubFieldNames.REFERENCE_ARTICLE_TITLE:   '<title>',
    JatsSubFieldNames.REFERENCE_SOURCE:          '<journal>',
    JatsSubFieldNames.REFERENCE_YEAR:            '<date>',
    JatsSubFieldNames.REFERENCE_VOLUME:          '<volume>',
    JatsSubFieldNames.REFERENCE_ISSUE:           '<issue>',
    JatsSubFieldNames.REFERENCE_FPAGE:           '<pages>',
    JatsSubFieldNames.REFERENCE_LPAGE:           '<pages>',
    JatsSubFieldNames.REFERENCE_DOI:             IDENTIFIER_LABEL,
    JatsSubFieldNames.REFERENCE_PMID:            IDENTIFIER_LABEL,
    JatsSubFieldNames.REFERENCE_PMCID:           IDENTIFIER_LABEL,
    JatsSubFieldNames.REFERENCE_WEB:             '<web>',
    JatsSubFieldNames.REFERENCE_LABEL:           '<note>',
    JatsSubFieldNames.REFERENCE_PUBLISHER_NAME:  '<publisher>',
    JatsSubFieldNames.REFERENCE_PUBLISHER_LOC:   '<location>',
}

# ── Affiliation-address model label mapping (keyed by sub-field name) ─────────
AFF_LABEL_BY_SUB_FIELD: Dict[str, str] = {
    JatsSubFieldNames.AUTHOR_AFF_LABEL:       '<marker>',
    JatsSubFieldNames.AUTHOR_AFF_INSTITUTION: '<institution>',
    JatsSubFieldNames.AUTHOR_AFF_DEPARTMENT:  '<department>',
    JatsSubFieldNames.AUTHOR_AFF_CITY:        '<settlement>',
    JatsSubFieldNames.AUTHOR_AFF_POSTCODE:    '<postCode>',
    JatsSubFieldNames.AUTHOR_AFF_REGION:      '<region>',
    JatsSubFieldNames.AUTHOR_AFF_COUNTRY:     '<country>',
}
