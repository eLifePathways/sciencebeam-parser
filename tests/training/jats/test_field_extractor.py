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
        assert 'Smith' in authors[0].text


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
