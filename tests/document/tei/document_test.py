import logging

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutToken
)
from sciencebeam_parser.document.tei.attribution import DocumentAttribution
from sciencebeam_parser.document.tei.common import (
    get_tei_xpath_text_content_list,
    TEI_NS_MAP
)
from sciencebeam_parser.document.tei_document import (
    TeiDocument
)
from tests.document.tei.common_test import (
    ITALICS_FONT_1
)


LOGGER = logging.getLogger(__name__)


class TestTeiDocument:
    def test_should_be_able_to_set_title(self):
        document = TeiDocument()
        document.set_title('test')
        LOGGER.debug('xml: %r', etree.tostring(document.root))
        nodes = document.root.xpath(
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]',
            namespaces=TEI_NS_MAP
        )
        assert [e.text for e in nodes] == ['test']
        assert document.get_title() == 'test'

    def test_should_be_able_to_set_abstract(self):
        document = TeiDocument()
        document.set_abstract('test')
        LOGGER.debug('xml: %r', etree.tostring(document.root))
        nodes = document.root.xpath(
            '//tei:abstract/tei:p', namespaces=TEI_NS_MAP
        )
        assert [e.text for e in nodes] == ['test']
        assert document.get_abstract() == 'test'

    def test_should_be_able_to_set_title_with_italic_layout_tokens(self):
        title_block = LayoutBlock.for_tokens([
            LayoutToken('rend'),
            LayoutToken('italic1', font=ITALICS_FONT_1),
            LayoutToken('test')
        ])
        document = TeiDocument()
        document.set_title_layout_block(title_block)
        LOGGER.debug('xml: %r', etree.tostring(document.root))
        nodes = document.root.xpath(
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]',
            namespaces=TEI_NS_MAP
        )
        assert len(nodes) == 1
        title_node = nodes[0]
        assert get_tei_xpath_text_content_list(
            title_node,
            './tei:hi[@rend="italic"]'
        ) == ['italic1']
        assert document.get_title() == 'rend italic1 test'

    def test_should_write_every_title_it_is_given(self):
        document = TeiDocument()
        document.set_title_layout_block(LayoutBlock.for_text('The title'), language='en')
        document.set_title_layout_block(LayoutBlock.for_text('O titulo'), language='pt')
        nodes = document.root.xpath(
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]',
            namespaces=TEI_NS_MAP
        )
        assert [node.text for node in nodes] == ['The title', 'O titulo']
        # The first is what a caller asking for "the title" means.
        assert document.get_title() == 'The title'

    def test_should_set_the_language_of_the_title(self):
        document = TeiDocument()
        document.set_title_layout_block(LayoutBlock.for_text('The title'), language='en')
        nodes = document.root.xpath(
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]',
            namespaces=TEI_NS_MAP
        )
        assert nodes[0].attrib['{http://www.w3.org/XML/1998/namespace}lang'] == 'en'

    def test_should_leave_the_language_off_a_title_it_could_not_be_told_for(self):
        document = TeiDocument()
        document.set_title_layout_block(LayoutBlock.for_text('The title'))
        nodes = document.root.xpath(
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]',
            namespaces=TEI_NS_MAP
        )
        assert '{http://www.w3.org/XML/1998/namespace}lang' not in nodes[0].attrib

    def test_should_place_trailing_text_outside_title_element(self):
        title_block = LayoutBlock.for_text('The title')
        document = TeiDocument()
        document.set_title_layout_block(title_block, trailing_text='.')
        nodes = document.root.xpath(
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]',
            namespaces=TEI_NS_MAP
        )
        assert len(nodes) == 1
        title_node = nodes[0]
        assert document.get_title() == 'The title'
        assert title_node.tail == '.'


ATTRIBUTION_1 = DocumentAttribution(
    version='1.2.3',
    profile_digest='digest1',
    profile_name='profile1'
)


def _get_tei_header_child_names(document: TeiDocument) -> list:
    header = document.get_or_create_element_at(['teiHeader'])
    return [etree.QName(child).localname for child in header]


class TestTeiDocumentAbstractHeading:
    def _get_profile_desc_children(self, document: TeiDocument):
        return [
            (etree.QName(child).localname, child.get('type'), child.text)
            for child in document.root.xpath(
                '//tei:profileDesc/*', namespaces=TEI_NS_MAP
            )
        ]

    def test_should_add_the_heading_as_a_note_before_its_abstract(self):
        document = TeiDocument()
        document.add_abstract_layout_block(
            LayoutBlock.for_text('The abstract.'),
            heading_block=LayoutBlock.for_text('Plain language summary')
        )
        assert self._get_profile_desc_children(document) == [
            ('note', 'abstract-heading', 'Plain language summary'),
            ('abstract', None, None)
        ]

    def test_should_add_no_note_where_there_is_no_heading(self):
        document = TeiDocument()
        document.add_abstract_layout_block(LayoutBlock.for_text('The abstract.'))
        assert self._get_profile_desc_children(document) == [('abstract', None, None)]

    def test_should_keep_the_abstract_text_out_of_the_note(self):
        document = TeiDocument()
        document.add_abstract_layout_block(
            LayoutBlock.for_text('The abstract.'),
            heading_block=LayoutBlock.for_text('Plain language summary')
        )
        assert get_tei_xpath_text_content_list(
            document.root, '//tei:profileDesc/tei:abstract'
        ) == ['The abstract.']


class TestTeiDocumentAttribution:
    def test_should_add_application_where_grobid_puts_it(self):
        document = TeiDocument()
        document.set_attribution(ATTRIBUTION_1)
        nodes = document.root.xpath(
            'tei:teiHeader/tei:encodingDesc/tei:appInfo/tei:application',
            namespaces=TEI_NS_MAP
        )
        assert len(nodes) == 1
        assert nodes[0].attrib['version'] == ATTRIBUTION_1.version

    def test_should_add_encoding_desc_after_file_desc(self):
        document = TeiDocument()
        document.set_title('test')
        document.set_abstract('test')
        document.set_attribution(ATTRIBUTION_1)
        assert _get_tei_header_child_names(document) == [
            'fileDesc', 'encodingDesc', 'profileDesc'
        ]

    def test_should_add_encoding_desc_before_profile_desc_without_file_desc(self):
        document = TeiDocument()
        document.set_abstract('test')
        document.set_attribution(ATTRIBUTION_1)
        assert _get_tei_header_child_names(document) == [
            'encodingDesc', 'profileDesc'
        ]

    def test_should_add_encoding_desc_after_file_desc_without_profile_desc(self):
        document = TeiDocument()
        document.set_title('test')
        document.set_attribution(ATTRIBUTION_1)
        assert _get_tei_header_child_names(document) == ['fileDesc', 'encodingDesc']
