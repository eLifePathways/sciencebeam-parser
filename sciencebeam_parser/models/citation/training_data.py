import logging
import re

from lxml import etree

from sciencebeam_parser.document.tei.common import tei_xpath
from sciencebeam_parser.models.training_data import (
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser
)
from sciencebeam_parser.models.citation.extract import (
    get_detected_external_identifier_type_for_text
)
from sciencebeam_parser.utils.xml import get_text_content

# Matches "388 - 412", "281-282", "1199 -1207" etc.
_PAGE_RANGE_RE = re.compile(r'^(\S+)\s*[-–]\s*(\S+)$')


LOGGER = logging.getLogger(__name__)


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/CitationParser.java

ROOT_TRAINING_XML_ELEMENT_PATH = ['text', 'back', 'listBibl', 'bibl']

TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<title>': ROOT_TRAINING_XML_ELEMENT_PATH + ['title[@level="a"]'],
    '<author>': ROOT_TRAINING_XML_ELEMENT_PATH + ['author'],
    '<editor>': ROOT_TRAINING_XML_ELEMENT_PATH + ['editor'],
    '<institution>': ROOT_TRAINING_XML_ELEMENT_PATH + ['orgName'],
    '<collaboration>': ROOT_TRAINING_XML_ELEMENT_PATH + ['orgName[@type="collaboration"]'],
    '<journal>': ROOT_TRAINING_XML_ELEMENT_PATH + ['title[@level="j"]'],
    '<series>': ROOT_TRAINING_XML_ELEMENT_PATH + ['title[@level="s"]'],
    '<booktitle>': ROOT_TRAINING_XML_ELEMENT_PATH + ['title[@level="m"]'],
    '<date>': ROOT_TRAINING_XML_ELEMENT_PATH + ['date'],
    '<volume>': ROOT_TRAINING_XML_ELEMENT_PATH + ['biblScope[@unit="volume"]'],
    '<issue>': ROOT_TRAINING_XML_ELEMENT_PATH + ['biblScope[@unit="issue"]'],
    '<pages>': ROOT_TRAINING_XML_ELEMENT_PATH + ['biblScope[@unit="page"]'],
    '<publisher>': ROOT_TRAINING_XML_ELEMENT_PATH + ['publisher'],
    '<location>': ROOT_TRAINING_XML_ELEMENT_PATH + ['pubPlace'],
    '<tech>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@type="report"]'],
    '<web>': ROOT_TRAINING_XML_ELEMENT_PATH + ['ptr[@type="web"]'],
    '<idno>': ROOT_TRAINING_XML_ELEMENT_PATH + ['idno[@type="DOI"]'],
    '<pubnum>': ROOT_TRAINING_XML_ELEMENT_PATH + ['idno'],
    '<note>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note']
}


class CitationTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.references.tei.xml'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            use_tei_namespace=True,
            root_tag='TEI',
            default_tei_filename_suffix=(
                CitationTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=None,
            default_tei_sub_directory='citation/corpus'
        )

    def get_post_processed_xml_root(self, xml_root: etree.ElementBase):
        for idno_element in tei_xpath(xml_root, '//tei:idno'):
            external_identifier_type = get_detected_external_identifier_type_for_text(
                get_text_content(idno_element)
            )
            if not external_identifier_type:
                continue
            idno_element.attrib['type'] = external_identifier_type
        for page_el in tei_xpath(xml_root, '//tei:biblScope[@unit="page"]'):
            full_text = ' '.join(page_el.itertext()).replace('\n', ' ').strip()
            full_text = re.sub(r'\s+', ' ', full_text)
            m = _PAGE_RANGE_RE.match(full_text)
            if m and 'from' not in page_el.attrib:
                page_el.attrib['from'] = m.group(1)
                page_el.attrib['to'] = m.group(2)
        return xml_root


class CitationTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
            use_tei_namespace=True
        )
