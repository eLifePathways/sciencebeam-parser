from collections import defaultdict

from lxml import etree

from sciencebeam_parser.training.jats.field_extractor import JatsFieldExtractor
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames, JatsSubFieldNames


def _parse_jats(xml: str) -> etree._Element:
    return etree.fromstring(xml.encode())


def _field_values_for(xml: str):
    root = _parse_jats(xml)
    return list(JatsFieldExtractor().iter_field_values(root))


def _fields_by_name(values):
    d = defaultdict(list)
    for v in values:
        d[v.field_name].append(v)
    return d


class TestTitle:
    def test_extracts_title(self):
        fvs = _field_values_for(
            '<article>'
            '<front><article-meta><title-group>'
            '<article-title>My Title</article-title>'
            '</title-group></article-meta></front>'
            '</article>'
        )
        titles = [v for v in fvs if v.field_name == JatsFieldNames.TITLE]
        assert len(titles) == 1
        assert titles[0].text == 'My Title'
        assert titles[0].sub_field_name is None

    def test_no_title_gives_no_values(self):
        fvs = _field_values_for('<article><front><article-meta></article-meta></front></article>')
        titles = [v for v in fvs if v.field_name == JatsFieldNames.TITLE]
        assert titles == []


class TestAbstract:
    def test_extracts_abstract(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<abstract><p>This is the abstract.</p></abstract>'
            '</article-meta></front></article>'
        )
        abstracts = [v for v in fvs if v.field_name == JatsFieldNames.ABSTRACT]
        assert len(abstracts) == 1
        assert 'abstract' in abstracts[0].text.lower()


class TestAuthor:
    def test_extracts_author_name(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<contrib-group>'
            '<contrib contrib-type="author"><name>'
            '<surname>Smith</surname><given-names>John</given-names>'
            '</name></contrib>'
            '</contrib-group>'
            '</article-meta></front></article>'
        )
        authors = [v for v in fvs if v.field_name == JatsFieldNames.AUTHOR]
        assert len(authors) == 1
        assert authors[0].text == 'John Smith'

    def test_authors_merged_per_contrib_group(self):
        # All authors in one <contrib-group> are emitted as a single AUTHOR field
        # value so the aligner labels the full byline (including separators).
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<contrib-group>'
            '<contrib contrib-type="author">'
            '<name><surname>Smith</surname><given-names>John</given-names></name>'
            '<xref ref-type="aff">1</xref>'
            '</contrib>'
            '<contrib contrib-type="author">'
            '<name><surname>Jones</surname><given-names>Mary</given-names></name>'
            '<xref ref-type="aff">1</xref>'
            '<xref ref-type="corresp">*</xref>'
            '</contrib>'
            '</contrib-group>'
            '</article-meta></front></article>'
        )
        authors = [v for v in fvs if v.field_name == JatsFieldNames.AUTHOR]
        assert len(authors) == 1
        assert authors[0].text == 'John Smith 1 Mary Jones 1 *'

    def test_author_multiple_contrib_groups_emit_separately(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<contrib-group>'
            '<contrib contrib-type="author">'
            '<name><surname>Smith</surname><given-names>John</given-names></name>'
            '</contrib>'
            '</contrib-group>'
            '<contrib-group>'
            '<contrib contrib-type="author">'
            '<name><surname>Jones</surname><given-names>Mary</given-names></name>'
            '</contrib>'
            '</contrib-group>'
            '</article-meta></front></article>'
        )
        authors = [v for v in fvs if v.field_name == JatsFieldNames.AUTHOR]
        assert len(authors) == 2
        assert authors[0].text == 'John Smith'
        assert authors[1].text == 'Mary Jones'


class TestKeywords:
    def test_extracts_keyword_values(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<kwd-group>'
            '<title>Keywords</title>'
            '<kwd>machine learning</kwd>'
            '<kwd>deep learning</kwd>'
            '</kwd-group>'
            '</article-meta></front></article>'
        )
        keywords = [v for v in fvs if v.field_name == JatsFieldNames.KEYWORDS]
        assert len(keywords) == 1
        assert keywords[0].text == 'machine learning, deep learning'

    def test_keywords_title_extracted_for_segmentation(self):
        # KEYWORDS_TITLE is emitted so the segmentation model can label the heading
        # line as <header>.  It is intentionally absent from HEADER_LABEL_BY_FIELD
        # so the header model leaves the "Keywords" token unlabelled.
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<kwd-group>'
            '<title>Keywords</title>'
            '<kwd>machine learning</kwd>'
            '</kwd-group>'
            '</article-meta></front></article>'
        )
        kw_titles = [v for v in fvs if v.field_name == JatsFieldNames.KEYWORDS_TITLE]
        assert len(kw_titles) == 1
        assert kw_titles[0].text == 'Keywords'


class TestAffiliation:
    def test_extracts_affiliation_text(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<aff id="a1"><institution>MIT</institution>, Cambridge</aff>'
            '</article-meta></front></article>'
        )
        affs = [v for v in fvs if v.field_name == JatsFieldNames.AUTHOR_AFF]
        assert len(affs) >= 1
        assert 'MIT' in affs[0].text

    def test_extracts_aff_institution_subfield(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<aff id="a1"><institution>MIT</institution></aff>'
            '</article-meta></front></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.AUTHOR_AFF_INSTITUTION]
        assert len(sub) == 1
        assert sub[0].text == 'MIT'

    def test_extracts_aff_country_subfield(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<aff id="a1"><institution>MIT</institution><country>USA</country></aff>'
            '</article-meta></front></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.AUTHOR_AFF_COUNTRY]
        assert len(sub) == 1
        assert sub[0].text == 'USA'

    def test_addr_bulk_includes_institution_tail_when_no_addr_line(self):
        # Bioarxiv-style affs: <institution>UCL, GOSH</institution>, London WC1N 1EH,
        # <country>United Kingdom</country>  — city/postcode in institution tail, no <addr-line>.
        # The AUTHOR_AFF_ADDR bulk value must cover "London WC1N 1EH United Kingdom" so
        # those tokens get the <address> label rather than staying in <affiliation>.
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<aff id="a1">'
            '<label>1</label>'
            '<institution>UCL, GOSH</institution>'
            ', London WC1N 1EH, '
            '<country>United Kingdom</country>'
            '</aff>'
            '</article-meta></front></article>'
        )
        addr = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.AUTHOR_AFF_ADDR]
        assert len(addr) == 1
        assert 'London WC1N 1EH' in addr[0].text
        assert 'United Kingdom' in addr[0].text

    def test_addr_bulk_empty_when_aff_fully_unstructured(self):
        # Label-only affs (no <institution>, <addr-line>, <country>) cannot be split;
        # no AUTHOR_AFF_ADDR value should be emitted.
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<aff id="a1"><label>1</label>Institut Barcelona Spain</aff>'
            '</article-meta></front></article>'
        )
        addr = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.AUTHOR_AFF_ADDR]
        assert addr == []

    def test_institution_tail_excluded_when_no_country_or_addr_line(self):
        # Guard: institution tail is only address content when a <country> or <addr-line>
        # confirms the aff has structured address content.  Without that anchor the tail
        # may be continuation of the institution/department name: some publishers split a
        # single department name across two <institution> tags, leaving the second half
        # as the tail of the first — e.g.
        #   <institution>Dept of Microbiology</institution>, Immunology and Parasitology,
        #   <institution>University X</institution>, City, Country
        # "Immunology and Parasitology" is NOT an address.
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<aff id="a1">'
            '<label>1</label>'
            '<institution>Departamento de Microbiologia</institution>'
            ', Imunologia e Parasitologia, '
            '<institution>Universidade Federal de Santa Catarina</institution>'
            ', Florianopolis, Brasil'
            '</aff>'
            '</article-meta></front></article>'
        )
        addr = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.AUTHOR_AFF_ADDR]
        assert addr == []


class TestReference:
    def test_extracts_reference_text(self):
        fvs = _field_values_for(
            '<article><back><ref-list>'
            '<ref id="b1"><element-citation>'
            '<article-title>A Study</article-title>'
            '<year>2020</year>'
            '</element-citation></ref>'
            '</ref-list></back></article>'
        )
        refs = [v for v in fvs if v.field_name == JatsFieldNames.REFERENCE]
        assert len(refs) >= 1
        assert 'A Study' in refs[0].text

    def test_extracts_reference_article_title_subfield(self):
        fvs = _field_values_for(
            '<article><back><ref-list>'
            '<ref id="b1"><element-citation>'
            '<article-title>A Study</article-title>'
            '</element-citation></ref>'
            '</ref-list></back></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.REFERENCE_ARTICLE_TITLE]
        assert len(sub) == 1
        assert sub[0].text == 'A Study'

    def test_extracts_reference_year_subfield(self):
        fvs = _field_values_for(
            '<article><back><ref-list>'
            '<ref id="b1"><element-citation><year>2020</year></element-citation></ref>'
            '</ref-list></back></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.REFERENCE_YEAR]
        assert sub[0].text == '2020'


class TestBodySections:
    def test_extracts_body_section_title(self):
        fvs = _field_values_for(
            '<article><body><sec><title>Introduction</title>'
            '<p>Some text.</p></sec></body></article>'
        )
        titles = [v for v in fvs if v.field_name == JatsFieldNames.BODY_SECTION_TITLE]
        assert len(titles) == 1
        assert titles[0].text == 'Introduction'

    def test_extracts_body_paragraph(self):
        fvs = _field_values_for(
            '<article><body><sec><title>Intro</title>'
            '<p>Paragraph text.</p></sec></body></article>'
        )
        paras = [v for v in fvs if v.field_name == JatsFieldNames.BODY_SECTION_PARAGRAPH]
        assert len(paras) == 1
        assert 'Paragraph' in paras[0].text


class TestAcknowledgement:
    def test_extracts_ack_paragraph(self):
        fvs = _field_values_for(
            '<article><back><ack><p>We thank everyone.</p></ack></back></article>'
        )
        ack = [v for v in fvs if v.field_name == JatsFieldNames.ACK_SECTION_PARAGRAPH]
        assert len(ack) == 1
        assert 'thank' in ack[0].text
