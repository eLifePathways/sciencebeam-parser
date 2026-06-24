from typing import Dict


class JatsFieldNames:
    TITLE = 'title'
    ABSTRACT = 'abstract'
    KEYWORDS_TITLE = 'keywords_title'
    KEYWORDS = 'keywords'
    MANUSCRIPT_TYPE = 'manuscript_type'
    AUTHOR = 'author'
    AUTHOR_AFF = 'author_aff'
    AUTHOR_NOTES = 'author_notes'
    FUNDING = 'funding'
    COPYRIGHT = 'copyright'
    SUB_ARTICLE = 'sub_article'
    BODY_SECTION_TITLE = 'body_section_title'
    BODY_SECTION_PARAGRAPH = 'body_section_paragraph'
    BODY_FIGURE = 'body_figure'
    BODY_TABLE = 'body_table'
    BACK_SECTION_TITLE = 'back_section_title'
    BACK_SECTION_PARAGRAPH = 'back_section_paragraph'
    ACK_SECTION_TITLE = 'acknowledgment_section_title'
    ACK_SECTION_PARAGRAPH = 'acknowledgment_section_paragraph'
    APPENDIX_GROUP_TITLE = 'appendix_group_title'
    APPENDIX = 'appendix'
    REFERENCE_LIST_TITLE = 'reference_list_title'
    REFERENCE = 'reference'
    PAGE_NO = 'page_no'


class JatsSubFieldNames:
    AUTHOR_AFF_ADDR = 'author_aff_addr'
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
    JatsFieldNames.FUNDING:               '<header>',
    JatsFieldNames.COPYRIGHT:             '<header>',
    JatsFieldNames.SUB_ARTICLE:           '<other>',
    JatsFieldNames.BODY_SECTION_TITLE:     '<body>',
    JatsFieldNames.BODY_SECTION_PARAGRAPH: '<body>',
    JatsFieldNames.BODY_FIGURE:            '<body>',
    JatsFieldNames.BODY_TABLE:             '<body>',
    JatsFieldNames.ACK_SECTION_TITLE:      '<acknowledgement>',
    JatsFieldNames.ACK_SECTION_PARAGRAPH:  '<acknowledgement>',
    JatsFieldNames.APPENDIX_GROUP_TITLE:   '<annex>',
    JatsFieldNames.APPENDIX:               '<annex>',
    JatsFieldNames.BACK_SECTION_TITLE:     '<annex>',
    JatsFieldNames.BACK_SECTION_PARAGRAPH: '<annex>',
    JatsFieldNames.REFERENCE_LIST_TITLE:   '<references>',
    JatsFieldNames.REFERENCE:              '<references>',
    JatsFieldNames.PAGE_NO:                '<page>',
}

# ── Header model label mapping ────────────────────────────────────────────────
HEADER_LABEL_BY_FIELD: Dict[str, str] = {
    JatsFieldNames.TITLE:            '<title>',
    JatsFieldNames.ABSTRACT:         '<abstract>',
    JatsFieldNames.KEYWORDS:         '<keyword>',
    JatsFieldNames.AUTHOR:           '<author>',
    JatsFieldNames.AUTHOR_AFF:       '<affiliation>',
    JatsFieldNames.AUTHOR_NOTES:     '<note>',
    JatsFieldNames.MANUSCRIPT_TYPE:  '<note>',
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
    JatsSubFieldNames.REFERENCE_DOI:             '<idno>',
    JatsSubFieldNames.REFERENCE_PMID:            '<pubnum>',
    JatsSubFieldNames.REFERENCE_PMCID:           '<pubnum>',
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
