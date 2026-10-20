import logging

from sciencebeam_parser.models.training_data import (
    NO_NS_TEI_E,
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser
)


LOGGER = logging.getLogger(__name__)


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/ReferenceSegmenterParser.java

ROOT_TRAINING_XML_ELEMENT_PATH = ['text', 'listBibl']

TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<reference>': ROOT_TRAINING_XML_ELEMENT_PATH + ['bibl'],
    '<label>': ROOT_TRAINING_XML_ELEMENT_PATH + ['bibl', 'label']
}

RESET_TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<label>': ROOT_TRAINING_XML_ELEMENT_PATH
}


class ReferenceSegmenterTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.references.referenceSegmenter.tei.xml'
    DEFAULT_DATA_FILENAME_SUFFIX = '.references.referenceSegmenter'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            reset_training_xml_element_path_by_label=RESET_TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            element_maker=NO_NS_TEI_E,
            default_tei_filename_suffix=(
                ReferenceSegmenterTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=(
                ReferenceSegmenterTeiTrainingDataGenerator.DEFAULT_DATA_FILENAME_SUFFIX
            ),
            default_tei_sub_directory='reference-segmenter/corpus/tei',
            default_data_sub_directory='reference-segmenter/corpus/raw'
        )


class ReferenceSegmenterTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
        )
