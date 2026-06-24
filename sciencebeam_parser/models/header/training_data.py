import logging

from lxml.builder import ElementMaker

from sciencebeam_parser.models.training_data import (
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


TEI_E = ElementMaker()


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/HeaderParser.java
ROOT_TRAINING_XML_ELEMENT_PATH = ['text', 'front']

TRAINING_XML_ELEMENT_PATH_BY_LABEL_WITHOUT_ALIAS = {
    '<title>': ROOT_TRAINING_XML_ELEMENT_PATH + ['docTitle', 'titlePart'],
    '<author>': ROOT_TRAINING_XML_ELEMENT_PATH + ['byline', 'docAuthor'],
    '<address>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address'],
    '<date>': ROOT_TRAINING_XML_ELEMENT_PATH + ['date'],
    '<page>': ROOT_TRAINING_XML_ELEMENT_PATH + ['page'],
    '<publisher>': ROOT_TRAINING_XML_ELEMENT_PATH + ['publisher'],
    '<journal>': ROOT_TRAINING_XML_ELEMENT_PATH + ['journal'],
    '<affiliation>': ROOT_TRAINING_XML_ELEMENT_PATH + ['byline', 'affiliation'],
    '<note>': ROOT_TRAINING_XML_ELEMENT_PATH,
    '<abstract>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="abstract"]'],
    '<email>': ROOT_TRAINING_XML_ELEMENT_PATH + ['email'],
    '<pubnum>': ROOT_TRAINING_XML_ELEMENT_PATH + ['idno'],
    '<keyword>': ROOT_TRAINING_XML_ELEMENT_PATH + ['keyword'],
    '<phone>': ROOT_TRAINING_XML_ELEMENT_PATH + ['phone'],
    '<web>': ROOT_TRAINING_XML_ELEMENT_PATH + ['ptr[@type="web"]'],
    '<meeting>': ROOT_TRAINING_XML_ELEMENT_PATH + ['meeting'],
    '<submission>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@type="submission"]'],
    '<reference>': ROOT_TRAINING_XML_ELEMENT_PATH + ['reference'],
    '<copyright>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@type="copyright"]'],
    '<funding>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@type="funding"]'],
    '<doctype>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@type="doctype"]'],
    '<group>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@type="group"]']
}


TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    **TRAINING_XML_ELEMENT_PATH_BY_LABEL_WITHOUT_ALIAS,
    '<location>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address'],
    '<institution>': ROOT_TRAINING_XML_ELEMENT_PATH + ['byline', 'affiliation']
}

# Each new <affiliation> (B- prefix) resets to the front level so it gets its own
# <byline> wrapper rather than being appended to the author byline or a previous
# affiliation byline.
RESET_TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<affiliation>': ROOT_TRAINING_XML_ELEMENT_PATH,
}


class HeaderTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.header.tei.xml'
    DEFAULT_DATA_FILENAME_SUFFIX = '.header'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            element_maker=TEI_E,
            reset_training_xml_element_path_by_label=RESET_TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            default_tei_filename_suffix=(
                HeaderTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=(
                HeaderTeiTrainingDataGenerator.DEFAULT_DATA_FILENAME_SUFFIX
            ),
            default_tei_sub_directory='header/corpus/tei',
            default_data_sub_directory='header/corpus/raw'
        )


class HeaderTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL_WITHOUT_ALIAS
            ),
            use_tei_namespace=False
        )
