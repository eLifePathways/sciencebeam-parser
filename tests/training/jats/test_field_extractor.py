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

    def test_extracts_reference_author_subfield(self):
        fvs = _field_values_for(
            '<article><back><ref-list>'
            '<ref id="b1"><element-citation>'
            '<person-group person-group-type="author">'
            '<name><surname>Smith</surname><given-names>A</given-names></name>'
            '<name><surname>Jones</surname><given-names>B C</given-names></name>'
            '</person-group>'
            '</element-citation></ref>'
            '</ref-list></back></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.REFERENCE_AUTHOR]
        assert len(sub) == 1
        assert 'Smith' in sub[0].text
        assert 'Jones' in sub[0].text

    def test_reference_author_includes_et_al(self):
        fvs = _field_values_for(
            '<article><back><ref-list>'
            '<ref id="b1"><element-citation>'
            '<person-group person-group-type="author">'
            '<name><surname>Smith</surname><given-names>A</given-names></name>'
            '<etal/>'
            '</person-group>'
            '</element-citation></ref>'
            '</ref-list></back></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.REFERENCE_AUTHOR]
        assert len(sub) == 1
        assert sub[0].text == 'Smith A et al.'

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

    def test_extracts_reference_doi_subfield(self):
        fvs = _field_values_for(
            '<article><back><ref-list>'
            '<ref id="b1"><element-citation>'
            '<pub-id pub-id-type="doi">10.1234/test</pub-id>'
            '</element-citation></ref>'
            '</ref-list></back></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.REFERENCE_DOI]
        assert len(sub) == 1
        assert sub[0].text == '10.1234/test'

    def test_extracts_reference_web_subfield_for_url_ext_link(self):
        fvs = _field_values_for(
            '<article xmlns:xlink="http://www.w3.org/1999/xlink"><back><ref-list>'
            '<ref id="b1"><element-citation>'
            '<ext-link ext-link-type="uri"'
            ' xlink:href="http://www.doi.org/10.5281/zenodo.6647010">'
            'http://www.doi.org/10.5281/zenodo.6647010'
            '</ext-link>'
            '</element-citation></ref>'
            '</ref-list></back></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.REFERENCE_WEB]
        assert len(sub) == 1
        assert sub[0].text == 'http://www.doi.org/10.5281/zenodo.6647010'

    def test_reference_web_ignores_reference_source_ext_link(self):
        fvs = _field_values_for(
            '<article xmlns:xlink="http://www.w3.org/1999/xlink"><back><ref-list>'
            '<ref id="b1"><element-citation>'
            '<ext-link ext-link-type="uri"'
            ' xlink:href="https://example.com/paper">Reference Source</ext-link>'
            '</element-citation></ref>'
            '</ref-list></back></article>'
        )
        sub = [v for v in fvs if v.sub_field_name == JatsSubFieldNames.REFERENCE_WEB]
        assert len(sub) == 0


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


class TestFunding:
    def test_extracts_funding_statement(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<funding-group>'
            '<funding-statement>Supported by grant 123.</funding-statement>'
            '</funding-group>'
            '</article-meta></front></article>'
        )
        funding = [v for v in fvs if v.field_name == JatsFieldNames.FUNDING]
        assert len(funding) == 1
        assert 'grant 123' in funding[0].text

    def test_extracts_multiple_funding_statements(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<funding-group>'
            '<funding-statement>Grant A funded this.</funding-statement>'
            '<funding-statement>The funder had no role.</funding-statement>'
            '</funding-group>'
            '</article-meta></front></article>'
        )
        funding = [v for v in fvs if v.field_name == JatsFieldNames.FUNDING]
        assert len(funding) == 2


class TestCopyright:
    def test_extracts_copyright_statement(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<permissions>'
            '<copyright-statement>Copyright 2022 Author et al.</copyright-statement>'
            '</permissions>'
            '</article-meta></front></article>'
        )
        cr = [v for v in fvs if v.field_name == JatsFieldNames.COPYRIGHT]
        assert len(cr) == 1
        assert '2022' in cr[0].text

    def test_extracts_license_paragraph(self):
        fvs = _field_values_for(
            '<article><front><article-meta>'
            '<permissions>'
            '<license>'
            '<license-p>Open access under CC-BY 4.0.</license-p>'
            '</license>'
            '</permissions>'
            '</article-meta></front></article>'
        )
        cr = [v for v in fvs if v.field_name == JatsFieldNames.COPYRIGHT]
        assert len(cr) == 1
        assert 'CC-BY' in cr[0].text


class TestSubArticle:
    def test_extracts_sub_article_paragraphs(self):
        fvs = _field_values_for(
            '<article>'
            '<sub-article article-type="peer-review">'
            '<body><p>This manuscript is well written.</p></body>'
            '</sub-article>'
            '</article>'
        )
        sub = [v for v in fvs if v.field_name == JatsFieldNames.SUB_ARTICLE]
        assert len(sub) == 1
        assert 'well written' in sub[0].text

    def test_extracts_sub_article_titles(self):
        fvs = _field_values_for(
            '<article>'
            '<sub-article article-type="peer-review">'
            '<front><title>Reviewer Report</title></front>'
            '<body><p>Some comments.</p></body>'
            '</sub-article>'
            '</article>'
        )
        sub = [v for v in fvs if v.field_name == JatsFieldNames.SUB_ARTICLE]
        texts = [v.text for v in sub]
        assert any('Reviewer Report' in t for t in texts)
        assert any('comments' in t for t in texts)

    def test_main_article_body_not_labeled_as_sub_article(self):
        fvs = _field_values_for(
            '<article>'
            '<body><sec><title>Introduction</title>'
            '<p>Main article text.</p></sec></body>'
            '<sub-article article-type="peer-review">'
            '<body><p>Review text.</p></body>'
            '</sub-article>'
            '</article>'
        )
        sub = [v for v in fvs if v.field_name == JatsFieldNames.SUB_ARTICLE]
        body = [v for v in fvs if v.field_name == JatsFieldNames.BODY_SECTION_PARAGRAPH]
        assert len(sub) == 1
        assert 'Review text' in sub[0].text
        assert len(body) == 1
        assert 'Main article' in body[0].text
