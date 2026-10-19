import logging
from typing import Iterable, List, Optional

from lxml.builder import ElementMaker

from sciencebeam_parser.utils.xml_writer import XmlTreeWriter
from sciencebeam_parser.document.layout_document import (
    LayoutLine,
    LayoutToken,
    join_layout_tokens
)
from sciencebeam_parser.models.data import LayoutModelData
from sciencebeam_parser.models.model import (
    get_split_prefix_label
)
from sciencebeam_parser.models.training_data import (
    AbstractTeiTrainingDataGenerator,
    AbstractTrainingTeiParser,
    get_model_data_label
)


LOGGER = logging.getLogger(__name__)


TEI_E = ElementMaker()


# based on:
# https://github.com/kermitt2/grobid/blob/0.7.0/grobid-core/src/main/java/org/grobid/core/engines/Segmentation.java
ROOT_TRAINING_XML_ELEMENT_PATH = ['text']
TRAINING_XML_ELEMENT_PATH_BY_LABEL = {
    '<other>': ROOT_TRAINING_XML_ELEMENT_PATH,
    'O': ROOT_TRAINING_XML_ELEMENT_PATH,
    '<header>': ROOT_TRAINING_XML_ELEMENT_PATH + ['front'],
    '<headnote>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@place="headnote"]'],
    '<footnote>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@place="footnote"]'],
    '<marginnote>': ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@place="marginnote"]'],
    '<page>': ROOT_TRAINING_XML_ELEMENT_PATH + ['page'],
    '<references>': ROOT_TRAINING_XML_ELEMENT_PATH + ['listBibl'],
    '<body>': ROOT_TRAINING_XML_ELEMENT_PATH + ['body'],
    '<cover>': ROOT_TRAINING_XML_ELEMENT_PATH + ['titlePage'],
    '<toc>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="toc"]'],
    '<annex>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="annex"]'],
    '<availability>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="availability"]'],
    '<review>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="review"]'],
    '<contribution>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="contribution"]'],
    '<conflict>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="conflict"]'],
    '<funding>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="funding"]'],
    '<acknowledgement>': ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="acknowledgement"]'],
}


# GROBID's own reader accepts several spellings for the same region, and its
# corpus uses them -- `div[@type="acknowledgment"]` on 200 lines at 0.9.0. They
# are read, never written: our generator emits one spelling per region.
# From TEISegmentationSaxParser.startElement.
ADDITIONAL_TRAINING_XML_ELEMENT_PATHS_BY_LABEL = {
    '<acknowledgement>': [
        ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="acknowledgements"]'],
        ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="acknowledgment"]'],
        ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="acknowledgments"]'],
    ],
    '<availability>': [
        ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="data_availability"]'],
        ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="data-availability"]'],
    ],
    '<conflict>': [ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="conflicts"]']],
    '<contribution>': [ROOT_TRAINING_XML_ELEMENT_PATH + ['div[@type="contributions"]']],
    '<footnote>': [ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@place="foot"]']],
    '<headnote>': [ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@place="head"]']],
    '<marginnote>': [ROOT_TRAINING_XML_ELEMENT_PATH + ['note[@place="margin"]']],
    '<page>': [ROOT_TRAINING_XML_ELEMENT_PATH + ['pages']],
    '<toc>': [ROOT_TRAINING_XML_ELEMENT_PATH + ['toc']],
}


def iter_tokens_from_model_data(model_data: LayoutModelData) -> Iterable[LayoutToken]:
    if model_data.layout_token is not None:
        yield model_data.layout_token
        return
    assert model_data.layout_line is not None
    yield from model_data.layout_line.tokens


def iter_layout_lines_from_layout_tokens(
    layout_tokens: Iterable[LayoutToken]
) -> Iterable[LayoutLine]:
    line_layout_tokens: List[LayoutToken] = []
    for layout_token in layout_tokens:
        if not line_layout_tokens:
            line_layout_tokens.append(layout_token)
            continue
        if (
            layout_token.line_meta.line_id
            == line_layout_tokens[0].line_meta.line_id
        ):
            LOGGER.debug('line id matching: %r - %r', layout_token, line_layout_tokens[0])
            line_layout_tokens.append(layout_token)
            continue
        yield LayoutLine(tokens=line_layout_tokens)
        line_layout_tokens = [layout_token]
    if line_layout_tokens:
        yield LayoutLine(tokens=line_layout_tokens)


class SegmentationTeiTrainingDataGenerator(AbstractTeiTrainingDataGenerator):
    DEFAULT_TEI_FILENAME_SUFFIX = '.segmentation.tei.xml'
    DEFAULT_DATA_FILENAME_SUFFIX = '.segmentation'

    def __init__(self):
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            element_maker=TEI_E,
            default_tei_filename_suffix=(
                SegmentationTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
            ),
            default_data_filename_suffix=(
                SegmentationTeiTrainingDataGenerator.DEFAULT_DATA_FILENAME_SUFFIX
            ),
            default_tei_sub_directory='segmentation/corpus/tei',
            default_data_sub_directory='segmentation/corpus/raw'
        )

    def write_xml_line_for_layout_tokens(
        self,
        xml_writer: XmlTreeWriter,
        layout_tokens: Iterable[LayoutToken],
        source: Optional[LayoutModelData] = None
    ):
        xml_writer.append_text(join_layout_tokens(layout_tokens), source=source)
        xml_writer.append(TEI_E('lb'))

    def write_xml_for_model_data_iterable(
        self,
        xml_writer: XmlTreeWriter,
        model_data_iterable: Iterable[LayoutModelData]
    ):
        default_path = xml_writer.current_path
        pending_whitespace = ''
        for model_data in model_data_iterable:
            prefixed_label = get_model_data_label(model_data)
            _prefix, label = get_split_prefix_label(prefixed_label or '')
            xml_element_path = self.get_training_xml_path_for_label(
                label,
                current_path=xml_writer.current_path
            )
            LOGGER.debug('label: %r (%r)', label, xml_element_path)
            if xml_writer.current_path != xml_element_path:
                xml_writer.require_path(default_path)
            xml_writer.append_text(pending_whitespace)
            pending_whitespace = ''
            xml_writer.require_path(xml_element_path)
            for layout_line in iter_layout_lines_from_layout_tokens(
                iter_tokens_from_model_data(model_data)
            ):
                self.write_xml_line_for_layout_tokens(
                    xml_writer,
                    layout_line.tokens,
                    source=model_data
                )
                pending_whitespace = '\n'
        xml_writer.require_path(default_path)
        xml_writer.append_text(pending_whitespace)


class SegmentationTrainingTeiParser(AbstractTrainingTeiParser):
    def __init__(self) -> None:
        super().__init__(
            root_training_xml_element_path=ROOT_TRAINING_XML_ELEMENT_PATH,
            training_xml_element_path_by_label=TRAINING_XML_ELEMENT_PATH_BY_LABEL,
            line_as_token=True,
            # GROBID's writer empties its accumulator wherever an element opens or
            # closes: that ends the line, and the next chunk starts a new span. So a
            # `<page>` nested in a `<body>` is its own line and the body resumes as a
            # new span. Its corpus nests where ours writes siblings.
            flushes_at_element_boundary=True,
            additional_element_paths_by_label=(
                ADDITIONAL_TRAINING_XML_ELEMENT_PATHS_BY_LABEL
            )
        )
