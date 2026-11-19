from pathlib import Path

from lxml import etree

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.transformers.tei_to_jats import TeiToJatsConverter
from sciencebeam_parser.utils.xml_serialization import serialize_xml_to_file


TEI_XML = (
    b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>'
    b'<titleStmt><title level="a" type="main">An invented title</title></titleStmt>'
    b'</fileDesc></teiHeader><text><body>'
    b'<div><head>Introduction</head><p>First <ref>1</ref> paragraph.</p></div>'
    b'<div><head>Methods</head><p>Second paragraph.</p></div>'
    b'</body><back><div type="acknowledgement"><p>Thanks to nobody.</p></div></back>'
    b'</text></TEI>'
)

STYLESHEET = (
    b'<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">'
    b'<xsl:param name="greeting" select="\'none\'"/>'
    b'<xsl:template match="/"><article><body><p><xsl:value-of select="$greeting"/></p>'
    b'</body></article></xsl:template>'
    b'</xsl:stylesheet>'
)


def _default_converter() -> TeiToJatsConverter:
    return TeiToJatsConverter.from_config(AppConfig.load_yaml(DEFAULT_CONFIG_FILE))


class TestTeiToJatsConverter:
    def test_should_write_jats_article(self, tmp_path: Path):
        filename = str(tmp_path / 'out.jats.xml')
        _default_converter().write_jats(etree.fromstring(TEI_XML), filename)
        jats = etree.parse(filename).getroot()
        assert jats.tag == 'article'
        assert jats.xpath('string(front/article-meta/title-group/article-title)') == (
            'An invented title'
        )

    def test_should_pass_the_configured_parameters(self, tmp_path: Path):
        converter = TeiToJatsConverter.from_stylesheet(STYLESHEET, AppConfig({
            'xslt': {'tei_to_jats': {'parameters': {'greeting': 'hello'}}},
            'output': {'pretty_print_xml': False}
        }))
        filename = tmp_path / 'out.jats.xml'
        converter.write_jats(etree.fromstring(TEI_XML), str(filename))
        assert filename.read_text(encoding='utf-8') == (
            '<article><body><p>hello</p></body></article>'
        )

    def test_should_indent_when_configured_to_pretty_print(self, tmp_path: Path):
        converter = TeiToJatsConverter.from_stylesheet(STYLESHEET, AppConfig({
            'output': {'pretty_print_xml': True}
        }))
        filename = tmp_path / 'out.jats.xml'
        converter.write_jats(etree.fromstring(TEI_XML), str(filename))
        assert filename.read_text(encoding='utf-8') == (
            '<article>\n  <body>\n    <p>none</p>\n  </body>\n</article>'
        )

    def test_should_write_the_same_jats_from_tei_read_back_from_a_pretty_printed_file(
        self, tmp_path: Path
    ):
        converter = _default_converter()
        tei_file = tmp_path / 'in.tei.xml'
        serialize_xml_to_file(etree.fromstring(TEI_XML), str(tei_file), pretty_print=True)
        from_memory = tmp_path / 'from-memory.jats.xml'
        from_file = tmp_path / 'from-file.jats.xml'
        converter.write_jats(etree.fromstring(TEI_XML), str(from_memory))
        converter.write_jats(etree.parse(str(tei_file)).getroot(), str(from_file))
        assert '\n' in tei_file.read_text(encoding='utf-8')
        assert from_file.read_bytes() == from_memory.read_bytes()
