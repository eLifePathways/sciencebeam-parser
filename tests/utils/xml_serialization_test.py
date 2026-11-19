from pathlib import Path

from lxml import etree

import pytest

from sciencebeam_parser.utils.xml_serialization import (
    serialize_xml_to_file,
    _JATS_BLOCK_ELEMENT_NAMES
)


TEI_NS_ATTRIB = b'<TEI xmlns="http://www.tei-c.org/ns/1.0">'


class TestSerializeXmlToFile:
    def test_should_indent_structural_tei_elements_when_pretty_print(
        self, tmp_path: Path
    ):
        root = etree.fromstring(
            TEI_NS_ATTRIB + b'<teiHeader><title>My Title</title></teiHeader></TEI>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(root, filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert '\n' in content
        assert 'My Title' in content

    def test_should_preserve_mixed_content_text_when_pretty_print(
        self, tmp_path: Path
    ):
        root = etree.fromstring(
            TEI_NS_ATTRIB + b'<p>some text <ref>1</ref> more text</p></TEI>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(root, filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert 'some text ' in content
        assert ' more text' in content

    def test_should_not_insert_whitespace_between_inline_sibling_elements(
        self, tmp_path: Path
    ):
        # title has no direct text; all content is in <hi> children whose
        # tails are None. Naively indenting would insert newlines between
        # the <hi> siblings, corrupting the concatenated title text.
        root = etree.fromstring(
            TEI_NS_ATTRIB + b'<teiHeader><titleStmt>'
            b'<title><hi>Part 1</hi><hi>Part 2</hi></title>'
            b'</titleStmt></teiHeader></TEI>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(root, filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert '\n' in content
        assert '<hi>Part 1</hi><hi>Part 2</hi>' in content

    def test_should_indent_structural_jats_elements_when_pretty_print(
        self, tmp_path: Path
    ):
        tree = etree.ElementTree(etree.fromstring(
            b'<article><body><sec><p>Some text</p><p>More text</p></sec></body></article>'
        ))
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(tree, filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert content == (
            '<article>\n'
            '  <body>\n'
            '    <sec>\n'
            '      <p>Some text</p>\n'
            '      <p>More text</p>\n'
            '    </sec>\n'
            '  </body>\n'
            '</article>'
        )

    @pytest.mark.parametrize('local_name', sorted(_JATS_BLOCK_ELEMENT_NAMES))
    def test_should_indent_every_listed_jats_element(
        self, tmp_path: Path, local_name: str
    ):
        root = etree.fromstring(
            b'<article><%(name)s><a/><b/></%(name)s></article>'
            % {b'name': local_name.encode()}
            if local_name != 'article'
            else b'<article><a/><b/></article>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(etree.ElementTree(root), filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert '<a/>\n' in content

    @pytest.mark.parametrize('local_name', [
        'ref', 'element-citation', 'name', 'person-group', 'contrib', 'aff',
        'title-group', 'tr', 'p'
    ])
    def test_should_not_enter_unlisted_jats_element(
        self, tmp_path: Path, local_name: str
    ):
        inner = b'<%(name)s><a/><b/></%(name)s>' % {b'name': local_name.encode()}
        root = etree.fromstring(b'<article><body>' + inner + b'</body></article>')
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(etree.ElementTree(root), filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert inner.decode() in content

    def test_should_not_change_string_value_of_unlisted_jats_elements(
        self, tmp_path: Path
    ):
        xml = (
            b'<article><front><article-meta><contrib-group>'
            b'<contrib><name><surname>Smith</surname>'
            b'<given-names>Jo</given-names></name></contrib>'
            b'</contrib-group></article-meta></front></article>'
        )
        before = etree.fromstring(xml)
        after = etree.fromstring(xml)
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(etree.ElementTree(after), filename, pretty_print=True)
        for before_element, after_element in zip(before.iter(), after.iter()):
            assert before_element.tag == after_element.tag
            if before_element.tag in _JATS_BLOCK_ELEMENT_NAMES:
                continue
            assert (
                before_element.xpath('string()') == after_element.xpath('string()')
            ), before_element.tag

    def test_should_not_indent_below_xml_space_preserve(
        self, tmp_path: Path
    ):
        preserved = (
            b'<sec xml:space="preserve"><p>1</p><p>2</p></sec>'
        )
        root = etree.fromstring(b'<article><body>' + preserved + b'</body></article>')
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(etree.ElementTree(root), filename, pretty_print=True)
        content = Path(filename).read_text(encoding='utf-8')
        assert preserved.decode() in content

    def test_should_not_indent_namespaced_root(
        self, tmp_path: Path
    ):
        xml = (
            b'<math xmlns="http://www.w3.org/1998/Math/MathML">'
            b'<mrow><mi>x</mi></mrow></math>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(
            etree.ElementTree(etree.fromstring(xml)), filename, pretty_print=True
        )
        assert Path(filename).read_bytes() == xml

    def test_should_not_indent_when_pretty_print_disabled(
        self, tmp_path: Path
    ):
        root = etree.fromstring(
            TEI_NS_ATTRIB + b'<teiHeader><title>My Title</title></teiHeader></TEI>'
        )
        filename = str(tmp_path / 'output.xml')
        serialize_xml_to_file(root, filename, pretty_print=False)
        content = Path(filename).read_text(encoding='utf-8')
        assert '\n' not in content
