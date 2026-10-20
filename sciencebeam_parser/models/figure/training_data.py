import logging

from sciencebeam_parser.models.training_data import (
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/FigureParser.java

ROOT_TRAINING_XML_ELEMENT_PATH = ['text', 'figure']

TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<figure_head>': ROOT_TRAINING_XML_ELEMENT_PATH + ['head'],
    '<label>': ROOT_TRAINING_XML_ELEMENT_PATH + ['head', 'label'],
    '<figDesc>': ROOT_TRAINING_XML_ELEMENT_PATH + ['figDesc']
}


class FigureTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.figure.tei.xml'
    DEFAULT_DATA_FILENAME_SUFFIX = '.figure'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            use_tei_namespace=False,
            root_tag='tei',
            default_tei_filename_suffix=(
                FigureTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=(
                FigureTeiTrainingDataGenerator.DEFAULT_DATA_FILENAME_SUFFIX
            ),
            default_tei_sub_directory='figure/corpus/tei',
            default_data_sub_directory='figure/corpus/raw'
        )


class FigureTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
        )
