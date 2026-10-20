import logging

from sciencebeam_parser.models.training_data import (
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/TableParser.java

ROOT_TRAINING_XML_ELEMENT_PATH = ['text', 'figure[@type="table"]']

# Note:
#   The table training data generation is different to figures in
#   how the following labels are mapped: `content`, `other`, `note`
TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<figure_head>': ROOT_TRAINING_XML_ELEMENT_PATH + ['head'],
    '<label>': ROOT_TRAINING_XML_ELEMENT_PATH + ['head', 'label'],
    '<figDesc>': ROOT_TRAINING_XML_ELEMENT_PATH + ['figDesc'],
    '<content>': ROOT_TRAINING_XML_ELEMENT_PATH + ['table'],
    '<other>': ROOT_TRAINING_XML_ELEMENT_PATH + ['other'],
    '<note>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note']
}


class TableTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.table.tei.xml'
    DEFAULT_DATA_FILENAME_SUFFIX = '.table'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            use_tei_namespace=False,
            root_tag='tei',
            default_tei_filename_suffix=(
                TableTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=(
                TableTeiTrainingDataGenerator.DEFAULT_DATA_FILENAME_SUFFIX
            ),
            default_tei_sub_directory='table/corpus/tei',
            default_data_sub_directory='table/corpus/raw'
        )


class TableTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
        )
