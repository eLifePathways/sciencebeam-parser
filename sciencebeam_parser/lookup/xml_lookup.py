import logging
from functools import cache

from lxml import etree

from sciencebeam_parser.utils.xml import get_text_content
from sciencebeam_parser.lookup import TextLookUp, SimpleTextLookUp


LOGGER = logging.getLogger(__name__)


TEI_NS = 'http://www.tei-c.org/ns/1.0'
TEI_NS_PREFIX = '{%s}' % TEI_NS

TEI_NS_MAP = {
    'tei': TEI_NS
}

TEI_CELL = TEI_NS_PREFIX + 'cell'


@cache
def load_xml_lookup_from_file(
    filename: str
) -> TextLookUp:
    root = etree.parse(filename)
    valid_texts = {
        get_text_content(node)
        for node in root.xpath('//tei:cell', namespaces=TEI_NS_MAP)
    }
    LOGGER.debug('valid_texts: %s', valid_texts)
    return SimpleTextLookUp(valid_texts)
