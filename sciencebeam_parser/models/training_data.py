from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import re
from typing import (
    Counter as CounterType,
    Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, TypeVar, Union
)

from lxml import etree
from lxml.builder import ElementMaker

from sciencebeam_parser.utils.xml_writer import (
    TracedItem,
    TracingXmlTreeWriter,
    XmlTreeWriter
)
from sciencebeam_parser.utils.labels import OTHER_LABELS, get_split_prefix_label
from sciencebeam_parser.utils.tokenizer import get_tokenized_tokens
from sciencebeam_parser.document.tei.common import TEI_E, tei_xpath
from sciencebeam_parser.document.layout_document import (
    LayoutLine,
    LayoutLineMeta,
    LayoutToken
)
from sciencebeam_parser.models.data import (
    NEW_DOCUMENT_MARKER,
    LabeledLayoutModelData,
    LabeledLayoutToken,
    LayoutModelData,
    NewDocumentMarker
)


LOGGER = logging.getLogger(__name__)


NO_NS_TEI_E = ElementMaker()


class ExtractInstruction:
    pass


class NewLineExtractInstruction(ExtractInstruction):
    pass


@dataclass
class ResetExtractInstruction(ExtractInstruction):
    reset_element_path: List[str]


def get_model_data_label(model_data: LayoutModelData) -> Optional[str]:
    if isinstance(model_data, LabeledLayoutModelData):
        return model_data.label
    return None


def is_same_layout_line(
    layout_line_1: Optional[LayoutLine],
    layout_line_2: Optional[LayoutLine]
) -> bool:
    assert layout_line_1 is not None
    assert layout_line_2 is not None
    return id(layout_line_1) == id(layout_line_2)


def is_same_model_data_layout_line(
    model_data_1: LayoutModelData,
    model_data_2: LayoutModelData
) -> bool:
    return is_same_layout_line(model_data_1.layout_line, model_data_2.layout_line)


def iter_group_model_data_by_line(
    model_data_iterable: Iterable[LayoutModelData]
) -> Iterable[Sequence[LayoutModelData]]:
    line_model_data_list: List[LayoutModelData] = []
    for model_data in model_data_iterable:
        if not line_model_data_list:
            line_model_data_list.append(model_data)
            continue
        previous_model_data = line_model_data_list[-1]
        if is_same_model_data_layout_line(
            model_data,
            previous_model_data
        ):
            LOGGER.debug('same line: %r - %r', model_data, previous_model_data)
            line_model_data_list.append(model_data)
            continue
        yield line_model_data_list
        line_model_data_list = [model_data]
    if line_model_data_list:
        yield line_model_data_list


def iter_model_data_with_new_line_instruction(
    model_data_iterable: Iterable[LayoutModelData]
) -> Iterable[Union[LayoutModelData, ExtractInstruction]]:
    line_model_data_list: List[LayoutModelData] = []
    for model_data in model_data_iterable:
        if not line_model_data_list:
            line_model_data_list.append(model_data)
            continue
        previous_model_data = line_model_data_list[-1]
        if is_same_model_data_layout_line(
            model_data,
            previous_model_data
        ):
            LOGGER.debug('same line: %r - %r', model_data, previous_model_data)
            line_model_data_list.append(model_data)
            continue
        yield from line_model_data_list
        yield NewLineExtractInstruction()
        line_model_data_list = [model_data]
    if line_model_data_list:
        yield from line_model_data_list
        yield NewLineExtractInstruction()


def get_default_note_type_for_label(label: str) -> str:
    return label.strip('<>')


def is_parent_path_of(
    parent_path: Sequence[str],
    child_path: Sequence[str]
) -> bool:
    if len(parent_path) >= len(child_path):
        return False
    return tuple(child_path[:len(parent_path)]) == tuple(parent_path)


def is_same_or_parent_path_of(
    parent_path: Sequence[str],
    child_path: Sequence[str]
) -> bool:
    return (
        tuple(parent_path) == tuple(child_path)
        or is_parent_path_of(parent_path, child_path)
    )


class TeiTrainingDataGenerator(ABC):
    @abstractmethod
    def get_training_tei_xml_for_multiple_model_data_iterables(
        self,
        model_data_iterables: Iterable[Iterable[LayoutModelData]]
    ) -> etree.ElementBase:
        pass

    @abstractmethod
    def get_training_tei_xml_for_model_data_iterable(
        self,
        model_data_iterable: Iterable[LayoutModelData]
    ) -> etree.ElementBase:
        pass

    @abstractmethod
    def get_default_tei_filename_suffix(self) -> Optional[str]:
        pass

    def get_default_data_filename_suffix(self) -> Optional[str]:
        return None

    def get_default_tei_sub_directory(self) -> Optional[str]:
        pass

    def get_default_data_sub_directory(self) -> Optional[str]:
        pass


class AbstractTeiTrainingDataGenerator(TeiTrainingDataGenerator):
    def __init__(
        self,
        root_training_xml_element_path: Sequence[str],
        training_xml_element_path_by_label: Mapping[str, Sequence[str]],
        *,
        root_tag: str = 'tei',
        use_tei_namespace: bool = True,
        element_maker: Optional[ElementMaker] = None,
        reset_training_xml_element_path_by_label: Optional[Mapping[str, Sequence[str]]] = None,
        default_tei_filename_suffix: Optional[str] = None,
        default_data_filename_suffix: Optional[str] = None,
        default_tei_sub_directory: Optional[str] = None,
        default_data_sub_directory: Optional[str] = None
    ):
        self.root_training_xml_element_path = root_training_xml_element_path
        self.root_parent_training_xml_element_path = root_training_xml_element_path[:-1]
        self.training_xml_element_path_by_label = training_xml_element_path_by_label
        self.reset_training_xml_element_path_by_label = (
            reset_training_xml_element_path_by_label or {}
        )
        self._training_xml_element_paths = {
            tuple(element_path)
            for label, element_path in training_xml_element_path_by_label.items()
            if (
                label not in OTHER_LABELS
                and tuple(element_path) != tuple(root_training_xml_element_path)
            )
        }
        self.other_element_path = training_xml_element_path_by_label.get('<other>')
        if element_maker is None:
            element_maker = TEI_E if use_tei_namespace else NO_NS_TEI_E
        self.element_maker = element_maker
        self.root_tag = root_tag
        self.default_tei_filename_suffix = default_tei_filename_suffix
        self.default_data_filename_suffix = default_data_filename_suffix
        self.default_tei_sub_directory = default_tei_sub_directory
        self.default_data_sub_directory = default_data_sub_directory

    def get_default_tei_filename_suffix(self) -> Optional[str]:
        return self.default_tei_filename_suffix

    def get_default_data_filename_suffix(self) -> Optional[str]:
        return self.default_data_filename_suffix

    def get_default_tei_sub_directory(self) -> Optional[str]:
        return self.default_tei_sub_directory

    def get_default_data_sub_directory(self) -> Optional[str]:
        return self.default_data_sub_directory

    def get_training_xml_path_for_label(
        self,
        label: Optional[str],
        current_path: Sequence[str]
    ) -> Sequence[str]:
        if not label or label in OTHER_LABELS:
            if label and self.other_element_path is not None:
                return self.other_element_path
            if tuple(current_path) in self._training_xml_element_paths:
                LOGGER.debug(
                    'found current path in element paths, returning parent: %r', current_path
                )
                return current_path[:-1]
            LOGGER.debug(
                'not found current path in element paths, returning current: %r', current_path
            )
            return current_path
        training_xml_path = self.training_xml_element_path_by_label.get(label or '')
        if not training_xml_path:
            note_type = get_default_note_type_for_label(label)
            LOGGER.info('label not mapped, creating note: %r', label)
            training_xml_path = (
                list(self.root_training_xml_element_path) + [f'note[@type="{note_type}"]']
            )
        return training_xml_path

    def get_parent_training_xml_path_within_root(
        self,
        element_path: Sequence[str]
    ) -> Sequence[str]:
        parent_path = element_path[:-1]
        if len(parent_path) < len(self.root_training_xml_element_path):
            return self.root_training_xml_element_path
        return parent_path

    def get_reset_training_xml_path_for_label(
        self,
        label: Optional[str],
        prefix: Optional[str]
    ) -> Optional[Sequence[str]]:
        if prefix != 'B' or not label:
            return None
        return self.reset_training_xml_element_path_by_label.get(label)

    def write_xml_for_model_data_with_instructions_iterable(
        self,
        xml_writer: XmlTreeWriter,
        model_data_or_instruction_iterable: Iterable[Union[LayoutModelData, ExtractInstruction]]
    ):
        default_path = xml_writer.current_path
        LOGGER.debug('default_path: %r', default_path)
        pending_whitespace = ''
        prev_label: str = ''
        pending_reset_path: Optional[List[str]] = None
        for model_data_or_instruction in model_data_or_instruction_iterable:
            if isinstance(model_data_or_instruction, LayoutModelData):
                model_data = model_data_or_instruction
                layout_token = model_data.layout_token
                assert layout_token is not None
                prefixed_label = get_model_data_label(model_data)
                prefix, label = get_split_prefix_label(prefixed_label or '')
                xml_element_path = self.get_training_xml_path_for_label(
                    label,
                    current_path=xml_writer.current_path
                )
                reset_path = self.get_reset_training_xml_path_for_label(
                    label=label,
                    prefix=prefix
                )
                if pending_reset_path is not None:
                    reset_path = pending_reset_path
                    pending_reset_path = None
                LOGGER.debug(
                    'label: %r (%r: %r; reset_path=%r)',
                    label, prefix, xml_element_path, reset_path
                )
                if reset_path is not None:
                    xml_writer.require_path(reset_path)
                elif (
                    prev_label not in OTHER_LABELS
                    and pending_whitespace
                    and not is_same_or_parent_path_of(xml_writer.current_path, xml_element_path)
                ):
                    LOGGER.debug(
                        'closing element before adding whitespace, %r -> %r',
                        xml_writer.current_path, xml_element_path
                    )
                    xml_writer.require_path(xml_writer.current_path[:-1])
                elif prefix == 'B' and label not in OTHER_LABELS:
                    xml_writer.require_path(
                        self.get_parent_training_xml_path_within_root(xml_element_path)
                    )
                xml_writer.require_path_or_below(xml_element_path)
                xml_writer.append_text(pending_whitespace)
                pending_whitespace = ''
                xml_writer.require_path(xml_element_path)
                xml_writer.append_text(layout_token.text, source=model_data)
                pending_whitespace = layout_token.whitespace
                prev_label = label
            elif isinstance(model_data_or_instruction, ResetExtractInstruction):
                pending_reset_path = model_data_or_instruction.reset_element_path
            elif isinstance(model_data_or_instruction, NewLineExtractInstruction):
                xml_writer.append(self.element_maker('lb'))
                pending_whitespace = '\n'
        xml_writer.require_path(default_path)
        xml_writer.append_text(pending_whitespace)

    def iter_model_data_or_instruction_for_model_data_iterable(
        self,
        model_data_iterable: Iterable[LayoutModelData]
    ) -> Iterable[Union[LayoutModelData, ExtractInstruction]]:
        return iter_model_data_with_new_line_instruction(
            model_data_iterable
        )

    def write_xml_for_model_data_iterable(
        self,
        xml_writer: XmlTreeWriter,
        model_data_iterable: Iterable[LayoutModelData]
    ):
        self.write_xml_for_model_data_with_instructions_iterable(
            xml_writer,
            self.iter_model_data_or_instruction_for_model_data_iterable(
                model_data_iterable
            )
        )

    def _get_xml_writer(self) -> TracingXmlTreeWriter:
        return TracingXmlTreeWriter(
            self.element_maker(self.root_tag),
            element_maker=self.element_maker
        )

    def get_post_processed_xml_root(self, xml_root: etree.ElementBase):
        return xml_root

    def get_training_tei_xml_and_trace(
        self,
        model_data_iterables: Iterable[Iterable[LayoutModelData]]
    ) -> Tuple[etree.ElementBase, Sequence[TracedItem]]:
        """The training TEI, with a record of what was written where.

        The trace says which element instance each run of text went into and
        which model data it came from, neither of which the finished tree can
        answer.
        """
        xml_writer = self._get_xml_writer()
        xml_writer.require_path(self.root_parent_training_xml_element_path)
        for model_data_iterable in model_data_iterables:
            xml_writer.require_path(self.root_parent_training_xml_element_path)
            xml_writer.require_path(self.root_training_xml_element_path)
            self.write_xml_for_model_data_iterable(
                xml_writer,
                model_data_iterable=model_data_iterable
            )
        return self.get_post_processed_xml_root(xml_writer.root), xml_writer.trace

    def get_training_tei_xml_for_multiple_model_data_iterables(
        self,
        model_data_iterables: Iterable[Iterable[LayoutModelData]]
    ) -> etree.ElementBase:
        return self.get_training_tei_xml_and_trace(model_data_iterables)[0]

    def get_training_tei_xml_for_model_data_iterable(
        self,
        model_data_iterable: Iterable[LayoutModelData]
    ) -> etree.ElementBase:
        return self.get_training_tei_xml_for_multiple_model_data_iterables(
            [model_data_iterable]
        )


TEI_LB = 'lb'

_TEI_LINE_WHITESPACE_RE = re.compile(r'[ \t\r\n]+')


def normalise_tei_line_text(text: str) -> str:
    """One line of a training TEI as a single line of text.

    Collapses the whitespace the XML carries for its own layout, and leaves
    anything else -- a non-breaking space is content.
    """
    return _TEI_LINE_WHITESPACE_RE.sub(' ', text).strip()


def get_local_tag_name(tag) -> str:
    """The tag without its namespace, or '' for a comment or processing instruction.

    Training TEI is read the way GROBID reads it, which is namespace-unaware: its
    own segmentation corpus ships both dialects.
    """
    if not isinstance(tag, str):
        return ''
    return tag.rsplit('}', maxsplit=1)[-1]


_NON_PATH_ATTRIBUTES = frozenset({'from', 'to'})


def _get_tag_expression_for_element(element: etree.ElementBase) -> str:
    tag = get_local_tag_name(element.tag)
    attrib = {
        get_local_tag_name(key): value
        for key, value in element.attrib.items()
        if key not in _NON_PATH_ATTRIBUTES
    }
    if not attrib:
        return tag
    if len(attrib) > 1:
        raise ValueError('only supporting up to one attribute')
    key, value = list(attrib.items())[0]
    return '{tag}[@{key}="{value}"]'.format(tag=tag, key=key, value=value)


class TeiTrainingElementPath(NamedTuple):
    element_list: Sequence[etree.ElementBase] = tuple([])

    def get_path(self) -> Sequence[str]:
        return [
            _get_tag_expression_for_element(element)
            for element in self.element_list
        ]

    def append(self, element: etree.ElementBase) -> 'TeiTrainingElementPath':
        return TeiTrainingElementPath(
            list(self.element_list) + [element]
        )


EMPTY_TEI_TRAINING_ELEMENT_PATH = TeiTrainingElementPath()


class TeiTrainingText(NamedTuple):
    text: str
    path: TeiTrainingElementPath
    is_start: bool


class TeiTrainingLine(NamedTuple):
    text_list: Sequence[TeiTrainingText]


def is_line_break_element(element: etree.ElementBase) -> bool:
    return get_local_tag_name(element.tag) == TEI_LB


def is_comment_or_processing_instruction(element: etree.ElementBase) -> bool:
    return not isinstance(element.tag, str)


def _has_text_content(element: etree.ElementBase) -> bool:
    return any(not text.isspace() for text in element.itertext() if text)


def _iter_flat_tei_training_text_from_element(
    parent_element: etree.ElementBase,
    current_path: TeiTrainingElementPath = EMPTY_TEI_TRAINING_ELEMENT_PATH,
    restarts_span_after_nested_element: bool = False
) -> Iterable[Union[TeiTrainingText, ExtractInstruction]]:
    LOGGER.debug('current_path: %s', current_path)
    is_start = True
    if parent_element.text:
        yield TeiTrainingText(
            text=parent_element.text,
            path=current_path,
            is_start=is_start
        )
        if not parent_element.text.isspace():
            is_start = False

    for child_element in parent_element:
        if is_line_break_element(child_element):
            yield NewLineExtractInstruction()
        elif not is_comment_or_processing_instruction(child_element):
            child_path = current_path.append(child_element)
            yield from _iter_flat_tei_training_text_from_element(
                child_element,
                child_path,
                restarts_span_after_nested_element
            )
            if (
                restarts_span_after_nested_element
                and not is_start
                and _has_text_content(child_element)
            ):
                # what follows a nested element that carried text of its own is a
                # new span rather than the same one resumed, which is how GROBID
                # reads its `<page>` inside a `<body>`
                is_start = True

        if child_element.tail:
            yield TeiTrainingText(
                text=child_element.tail,
                path=current_path,
                is_start=is_start
            )
            if not child_element.tail.isspace():
                is_start = False


def _iter_tei_training_lines_from_element(
    parent_element: etree.ElementBase,
    current_path: TeiTrainingElementPath = EMPTY_TEI_TRAINING_ELEMENT_PATH,
    restarts_span_after_nested_element: bool = False
) -> Iterable[TeiTrainingLine]:
    line_text_list = []
    for item in _iter_flat_tei_training_text_from_element(
        parent_element,
        current_path,
        restarts_span_after_nested_element
    ):
        if isinstance(item, TeiTrainingText):
            line_text_list.append(item)
        elif isinstance(item, NewLineExtractInstruction):
            yield TeiTrainingLine(line_text_list)
            line_text_list = []
        else:
            raise RuntimeError('unrecognised item: %r' % item)
    if line_text_list:
        yield TeiTrainingLine(line_text_list)


T = TypeVar('T')


def iter_group_doc_items_with_new_doc_marker(
    flat_item_iterable: Iterable[Union[T, NewDocumentMarker]]
) -> Iterable[List[T]]:
    doc_items: List[T] = []
    for item in flat_item_iterable:
        if isinstance(item, NewDocumentMarker):
            yield doc_items
            doc_items = []
            continue
        doc_items.append(item)


def iter_tag_result_for_flat_tag_result(
    flat_tag_result_iterable: Iterable[Union[Tuple[str, str], NewDocumentMarker]]
) -> Iterable[List[Tuple[str, str]]]:
    doc_tag_result: List[Tuple[str, str]] = []
    for token_tag_result in flat_tag_result_iterable:
        if isinstance(token_tag_result, NewDocumentMarker):
            yield doc_tag_result
            doc_tag_result = []
            continue
        doc_tag_result.append(token_tag_result)


def get_tag_result_for_flat_tag_result(
    flat_tag_result_iterable: Iterable[Union[Tuple[str, str], NewDocumentMarker]]
) -> List[List[Tuple[str, str]]]:
    return list(iter_tag_result_for_flat_tag_result(flat_tag_result_iterable))


class TrainingTeiParser(ABC):
    @abstractmethod
    def parse_training_tei_to_tag_result(
        self,
        tei_root: etree.ElementBase
    ) -> List[List[Tuple[str, str]]]:
        pass

    @abstractmethod
    def parse_training_tei_to_labeled_layout_tokens_list(
        self,
        tei_root: etree.ElementBase,
        unrecognised_path_counts: Optional[CounterType[str]] = None
    ) -> Sequence[Sequence[LabeledLayoutToken]]:
        pass


_ELEMENT_PATH_STEP_RE = re.compile(r'^([^\[]+)(\[.*\])?$')


def get_namespace_agnostic_xpath_step(element_path_step: str) -> str:
    """One step of a training element path, matched whatever namespace it is in.

    `figure[@type="table"]` becomes `*[local-name()="figure"][@type="table"]`.
    """
    match = _ELEMENT_PATH_STEP_RE.match(element_path_step)
    assert match is not None
    tag, predicate = match.groups()
    return '*[local-name()="%s"]%s' % (tag, predicate or '')


class AbstractTrainingTeiParser(TrainingTeiParser):
    def __init__(
        self,
        root_training_xml_element_path: Sequence[str],
        training_xml_element_path_by_label: Mapping[str, Sequence[str]],
        line_as_token: bool = False,
        restarts_span_after_nested_element: bool = False,
    ) -> None:
        self.label_by_relative_element_path_map = {
            tuple(element_path[len(root_training_xml_element_path):]): label
            for label, element_path in training_xml_element_path_by_label.items()
        }
        for element_path in list(self.label_by_relative_element_path_map.keys()):
            if len(element_path) < 2:
                continue
            parent_element_path = element_path[:-1]
            if parent_element_path not in self.label_by_relative_element_path_map:
                self.label_by_relative_element_path_map[parent_element_path] = 'O'
        self.root_training_xml_xpath = './' + '/'.join(
            get_namespace_agnostic_xpath_step(element_path_step)
            for element_path_step in root_training_xml_element_path
        )
        self.line_as_token = line_as_token
        self.restarts_span_after_nested_element = restarts_span_after_nested_element

    def get_label_for_element_path(
        self,
        tei_training_element_path: TeiTrainingElementPath,
        text: str,  # pylint: disable=unused-argument
        unrecognised_path_counts: Optional[CounterType[str]] = None
    ) -> str:
        """The label of the innermost element on the path we recognise.

        GROBID's own reader tracks a current tag rather than an element path, so an
        element it has no rule for leaves the enclosing label in place. Reading its
        corpus requires the same, and what our stricter lookup was protecting is kept
        as a count: it is zero for training TEI we wrote ourselves.
        """
        element_path = list(tei_training_element_path.get_path())
        for end_index in range(len(element_path), -1, -1):
            for start_index in range(end_index + 1):
                label = self.label_by_relative_element_path_map.get(
                    tuple(element_path[start_index:end_index])
                )
                if not label:
                    continue
                if end_index - start_index < len(element_path):
                    if unrecognised_path_counts is not None:
                        unrecognised_path_counts['/'.join(element_path)] += 1
                return label
        if unrecognised_path_counts is not None:
            unrecognised_path_counts['/'.join(element_path)] += 1
        return 'O'

    def iter_parse_training_tei_to_flat_labeled_layout_tokens(
        self,
        tei_root: etree.ElementBase,
        unrecognised_path_counts: Optional[CounterType[str]] = None
    ) -> Iterable[Union[LabeledLayoutToken, NewDocumentMarker]]:
        for text_node in tei_xpath(tei_root, self.root_training_xml_xpath):
            tei_training_lines = list(
                _iter_tei_training_lines_from_element(
                    text_node, EMPTY_TEI_TRAINING_ELEMENT_PATH,
                    self.restarts_span_after_nested_element
                )
            )
            LOGGER.debug('tei_training_lines: %r', tei_training_lines)
            for line_index, line in enumerate(tei_training_lines):
                line_meta = LayoutLineMeta(
                    line_id=1 + line_index,
                    text=normalise_tei_line_text(
                        ''.join(text.text for text in line.text_list)
                    )
                )
                for text in line.text_list:
                    if text.text.isspace():
                        continue
                    token_count = 0
                    if text.path.element_list:
                        label = self.get_label_for_element_path(
                            text.path,
                            text=text.text,
                            unrecognised_path_counts=unrecognised_path_counts
                        )
                        prefix = 'B-' if text.is_start else 'I-'
                    else:
                        label = 'O'
                        prefix = ''
                    if label in OTHER_LABELS:
                        prefix = ''
                    for token_text in get_tokenized_tokens(text.text):
                        yield LabeledLayoutToken(
                            label=prefix + label,
                            layout_token=LayoutToken(
                                text=token_text,
                                line_meta=line_meta
                            )
                        )
                        token_count += 1
                        if prefix:
                            prefix = 'I-'
                        if self.line_as_token:
                            break
                    if token_count and self.line_as_token:
                        # we are only outputting the first token of each line
                        break
            yield NEW_DOCUMENT_MARKER

    def iter_parse_training_tei_to_flat_tag_result(
        self,
        tei_root: etree.ElementBase
    ) -> Iterable[Union[Tuple[str, str], NewDocumentMarker]]:
        for item in self.iter_parse_training_tei_to_flat_labeled_layout_tokens(
            tei_root
        ):
            if isinstance(item, NewDocumentMarker):
                yield item
                continue
            assert isinstance(item, LabeledLayoutToken)
            yield item.layout_token.text, item.label

    def parse_training_tei_to_tag_result(
        self,
        tei_root: etree.ElementBase
    ) -> List[List[Tuple[str, str]]]:
        return list(iter_group_doc_items_with_new_doc_marker(
            self.iter_parse_training_tei_to_flat_tag_result(
                tei_root
            )
        ))

    def parse_training_tei_to_labeled_layout_tokens_list(
        self,
        tei_root: etree.ElementBase,
        unrecognised_path_counts: Optional[CounterType[str]] = None
    ) -> Sequence[Sequence[LabeledLayoutToken]]:
        return list(iter_group_doc_items_with_new_doc_marker(
            self.iter_parse_training_tei_to_flat_labeled_layout_tokens(
                tei_root,
                unrecognised_path_counts=unrecognised_path_counts
            )
        ))
