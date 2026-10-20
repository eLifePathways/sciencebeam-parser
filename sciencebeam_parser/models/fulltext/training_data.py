import logging

from sciencebeam_parser.models.training_data import (
    NO_NS_TEI_E,
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/FullTextParser.java

ROOT_TRAINING_XML_ELEMENT_PATH = ['text']

TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<other>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@type="other"]'],
    '<section>': ROOT_TRAINING_XML_ELEMENT_PATH + ['head'],
    '<paragraph>': ROOT_TRAINING_XML_ELEMENT_PATH + ['p'],
    '<citation_marker>': ROOT_TRAINING_XML_ELEMENT_PATH + ['p', 'ref[@type="biblio"]'],
    '<figure_marker>': ROOT_TRAINING_XML_ELEMENT_PATH + ['p', 'ref[@type="figure"]'],
    '<table_marker>': ROOT_TRAINING_XML_ELEMENT_PATH + ['p', 'ref[@type="table"]'],
    '<equation_marker>': ROOT_TRAINING_XML_ELEMENT_PATH + ['p', 'ref[@type="formula"]'],
    '<section_marker>': ROOT_TRAINING_XML_ELEMENT_PATH + ['p', 'ref[@type="section"]'],
    '<figure>': ROOT_TRAINING_XML_ELEMENT_PATH + ['figure'],
    '<table>': ROOT_TRAINING_XML_ELEMENT_PATH + ['figure[@type="table"]'],
    '<equation>': ROOT_TRAINING_XML_ELEMENT_PATH + ['formula'],
    '<equation_label>': ROOT_TRAINING_XML_ELEMENT_PATH + ['formula', 'label'],
    '<item>': ROOT_TRAINING_XML_ELEMENT_PATH + ['item']
}


class FullTextTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.fulltext.tei.xml'
    DEFAULT_DATA_FILENAME_SUFFIX = '.fulltext'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            element_maker=NO_NS_TEI_E,
            default_tei_filename_suffix=(
                FullTextTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=(
                FullTextTeiTrainingDataGenerator.DEFAULT_DATA_FILENAME_SUFFIX
            ),
            default_tei_sub_directory='fulltext/corpus/tei',
            default_data_sub_directory='fulltext/corpus/raw'
        )


class FullTextTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
        )
