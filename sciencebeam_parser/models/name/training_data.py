import logging
from typing import Iterable, Set, Union
from sciencebeam_parser.document.semantic_document import SemanticAuthor

from sciencebeam_parser.models.data import LayoutModelData
from sciencebeam_parser.models.model import (
    LabeledLayoutToken,
    iter_entity_layout_blocks_for_labeled_layout_tokens
)
from sciencebeam_parser.models.name.extract import NameSemanticExtractor
from sciencebeam_parser.models.training_data import (
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser,
    ExtractInstruction,
    ResetExtractInstruction,
    get_model_data_label
)


LOGGER = logging.getLogger(__name__)


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/AuthorParser.java

ROOT_TRAINING_XML_ELEMENT_PATH = [
    'teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic', 'author', 'persName'
]

TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<marker>': ROOT_TRAINING_XML_ELEMENT_PATH + ['marker'],
    '<title>': ROOT_TRAINING_XML_ELEMENT_PATH + ['roleName'],
    '<forename>': ROOT_TRAINING_XML_ELEMENT_PATH + ['forename'],
    '<middlename>': ROOT_TRAINING_XML_ELEMENT_PATH + ['middlename'],
    '<surname>': ROOT_TRAINING_XML_ELEMENT_PATH + ['surname'],
    '<suffix>': ROOT_TRAINING_XML_ELEMENT_PATH + ['suffix']
}


def iter_model_data_with_reset_instruction_iterable(
    model_data_or_instruction_iterable: Iterable[Union[LayoutModelData, ExtractInstruction]]
) -> Iterable[Union[LayoutModelData, ExtractInstruction]]:
    # using extractor to re-use logic to split author names
    # here we will split on the first token of the extracted semantic content
    extractor = NameSemanticExtractor()
    model_data_or_instruction_list = list(
        model_data_or_instruction_iterable
    )
    entity_tokens = iter_entity_layout_blocks_for_labeled_layout_tokens([
        LabeledLayoutToken(
            label=get_model_data_label(model_data) or '',
            layout_token=model_data.layout_token
        )
        for model_data in model_data_or_instruction_list
        if (
            isinstance(model_data, LayoutModelData)
            and model_data.layout_token is not None
        )
    ])
    LOGGER.debug('entity_tokens: %r', entity_tokens)
    reset_token_ids: Set[int] = set()
    for index, semantic_content in enumerate(extractor.iter_semantic_content_for_entity_blocks(
        entity_tokens=entity_tokens,
        name_type=SemanticAuthor
    )):
        if index == 0:
            continue
        for semantic_token in semantic_content.iter_tokens():
            reset_token_ids.add(id(semantic_token))
            break
    for model_data_or_instruction in model_data_or_instruction_list:
        if isinstance(model_data_or_instruction, LayoutModelData):
            model_data = model_data_or_instruction
            if id(model_data.layout_token) in reset_token_ids:
                yield ResetExtractInstruction(
                    ROOT_TRAINING_XML_ELEMENT_PATH[:-1]
                )
        yield model_data_or_instruction


class NameTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.authors.tei.xml'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            use_tei_namespace=True,
            root_tag='TEI',
            default_tei_filename_suffix=(
                NameTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=None
        )

    def iter_model_data_or_instruction_for_model_data_iterable(
        self,
        model_data_iterable: Iterable[LayoutModelData]
    ) -> Iterable[Union[LayoutModelData, ExtractInstruction]]:
        parent_model_data_or_instruction_iterable = (
            super().iter_model_data_or_instruction_for_model_data_iterable(
                model_data_iterable
            )
        )
        return iter_model_data_with_reset_instruction_iterable(
            parent_model_data_or_instruction_iterable
        )


class NameTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH[:-1],
            training_xml_element_path_by_label=(
                TRAINING_XML_ELEMENT_PATH_BY_LABEL
            ),
        )
