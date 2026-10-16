import logging

from sciencebeam_parser.document.tei.common import TEI_E
from sciencebeam_parser.models.training_data import (
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/AffiliationAddressParser.java

ROOT_TRAINING_XML_ELEMENT_PATH = [
    'teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic', 'author',
    'affiliation'
]

TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<marker>': ROOT_TRAINING_XML_ELEMENT_PATH + ['marker'],
    '<institution>': ROOT_TRAINING_XML_ELEMENT_PATH + ['orgName[@type="institution"]'],
    '<department>': ROOT_TRAINING_XML_ELEMENT_PATH + ['orgName[@type="department"]'],
    '<laboratory>': ROOT_TRAINING_XML_ELEMENT_PATH + ['orgName[@type="laboratory"]'],
    '<addrLine>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address', 'addrLine'],
    '<postCode>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address', 'postCode'],
    '<postBox>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address', 'postBox'],
    '<region>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address', 'region'],
    '<settlement>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address', 'settlement'],
    '<country>': ROOT_TRAINING_XML_ELEMENT_PATH + ['address', 'country']
}


class AffiliationAddressTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.affiliation.tei.xml'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            element_maker=TEI_E,
            default_tei_filename_suffix=(
                AffiliationAddressTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=None,
            default_tei_sub_directory='affiliation-address/corpus'
        )


class AffiliationAddressTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
        )
