import logging
from typing import Dict, Iterable, List, Optional, Sequence, Union

from lxml import etree
from lxml.builder import ElementMaker

from sciencebeam_parser.utils.xml import get_text_content
from sciencebeam_parser.utils.xml_writer import parse_tag_expression
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutPageCoordinates,
    LayoutToken
)
from sciencebeam_parser.document.semantic_document import (
    SemanticContentWrapper,
    SemanticMixedContentWrapper
)


LOGGER = logging.getLogger(__name__)


XML_NS = 'http://www.w3.org/XML/1998/namespace'
XML_NS_PREFIX = '{%s}' % XML_NS
XML_ID = XML_NS_PREFIX + 'id'

TEI_NS = 'http://www.tei-c.org/ns/1.0'
TEI_NS_PREFIX = '{%s}' % TEI_NS

TEI_NS_MAP = {
    'tei': TEI_NS
}

TEI_E = ElementMaker(namespace=TEI_NS, nsmap={
    None: TEI_NS
})


def get_or_create_element_at(parent: etree.ElementBase, path: List[str]) -> etree.ElementBase:
    if not path:
        return parent
    child = parent.find(TEI_NS_PREFIX + path[0])
    if child is None:
        LOGGER.debug('creating element: %s', path[0])
        tag_expression = parse_tag_expression(path[0])
        child = tag_expression.create_node(
            element_maker=TEI_E
        )
        parent.append(child)
    return get_or_create_element_at(child, path[1:])


def get_or_create_element_after(
    parent: etree.ElementBase,
    tag: str,
    preceding_tags: Sequence[str]
) -> etree.ElementBase:
    """Create a child after the last of the children it must follow.

    `get_or_create_element_at` appends, which holds while every child is created
    in schema order. `encodingDesc` is not: it is written once the rest of the
    header is built, by which point `profileDesc` may already be there.
    """
    child = parent.find(TEI_NS_PREFIX + tag)
    if child is not None:
        return child
    child = TEI_E(tag)
    preceding = {TEI_NS_PREFIX + preceding_tag for preceding_tag in preceding_tags}
    index = 0
    for position, existing_child in enumerate(parent):
        if existing_child.tag in preceding:
            index = position + 1
    parent.insert(index, child)
    return child


def tei_xpath(parent: etree.ElementBase, xpath: str) -> List[etree.ElementBase]:
    return parent.xpath(xpath, namespaces=TEI_NS_MAP)


def get_tei_xpath_text_content_list(parent: etree.ElementBase, xpath: str) -> List[str]:
    return [get_text_content(node) for node in tei_xpath(parent, xpath)]


def get_required_styles(layout_token: LayoutToken) -> List[str]:
    required_styles = []
    if layout_token.font.is_bold:
        required_styles.append('bold')
    if layout_token.font.is_italics:
        required_styles.append('italic')
    if layout_token.font.is_subscript:
        required_styles.append('subscript')
    if layout_token.font.is_superscript:
        required_styles.append('superscript')
    return required_styles


def get_element_for_styles(styles: List[str], text: str) -> etree.ElementBase:
    if not styles:
        return text
    child: Optional[etree.ElementBase] = None
    for style in reversed(styles):
        LOGGER.debug('style: %r, child: %r, text: %r', style, child, text)
        if child is not None:
            child = TEI_E('hi', {'rend': style}, child)
        else:
            child = TEI_E('hi', {'rend': style}, text)
    return child


def format_coordinates(coordinates: LayoutPageCoordinates) -> str:
    return '%d,%.2f,%.2f,%.2f,%.2f' % (
        coordinates.page_number,
        coordinates.x,
        coordinates.y,
        coordinates.width,
        coordinates.height
    )


def format_coordinates_list(coordinates_list: List[LayoutPageCoordinates]) -> str:
    return ';'.join((
        format_coordinates(coordinates)
        for coordinates in coordinates_list
    ))


def get_default_attributes_for_layout_block(
    layout_block: LayoutBlock,
    enable_coordinates: bool = True
) -> Dict[str, str]:
    if enable_coordinates:
        formatted_coords = format_coordinates_list(
            layout_block.get_merged_coordinates_list()
        )
        if formatted_coords:
            return {'coords': formatted_coords}
    return {}


def iter_layout_block_tei_children(
    layout_block: LayoutBlock,
    enable_coordinates: bool = True
) -> Iterable[Union[str, etree.ElementBase]]:
    pending_styles: List[str] = []
    pending_text = ''
    pending_whitespace = ''
    if enable_coordinates:
        yield get_default_attributes_for_layout_block(
            layout_block=layout_block,
            enable_coordinates=enable_coordinates
        )
    for line in layout_block.lines:
        for token in line.tokens:
            required_styles = get_required_styles(token)
            LOGGER.debug('token: %r, required_styles=%r', token, required_styles)
            if required_styles != pending_styles:
                if pending_text:
                    yield get_element_for_styles(
                        pending_styles,
                        pending_text
                    )
                    pending_text = ''
                if pending_whitespace:
                    yield pending_whitespace
                    pending_whitespace = ''
                pending_styles = required_styles
            if pending_whitespace:
                pending_text += pending_whitespace
                pending_whitespace = ''
            pending_text += token.text
            pending_whitespace = token.whitespace
    if pending_text:
        yield get_element_for_styles(
            pending_styles,
            pending_text
        )


def extend_element(
    element: etree.ElementBase,
    children_or_attributes: Iterable[etree.ElementBase]
):
    for item in children_or_attributes:
        if isinstance(item, dict):
            element.attrib.update(item)
            continue
        if isinstance(item, str):
            try:
                previous_element = element[-1]
            except IndexError:
                previous_element = None
            if previous_element is not None:
                previous_element.tail = (
                    (previous_element.tail or '')
                    + item
                )
            else:
                element.text = (
                    (element.text or '')
                    + item
                )
            continue
        element.append(item)


def create_tei_note_element(
    note_type: str,
    layout_block: LayoutBlock
) -> etree.EntityBase:
    return TEI_E(
        'note',
        {'type': note_type},
        *iter_layout_block_tei_children(layout_block)
    )


def get_default_attributes_for_semantic_content(
    semantic_content: SemanticContentWrapper,
    **kwargs
) -> Dict[str, str]:
    attrib = get_default_attributes_for_layout_block(
        semantic_content.merged_block,
        **kwargs
    )
    if isinstance(semantic_content, SemanticMixedContentWrapper):
        if semantic_content.content_id:
            attrib = {
                **attrib,
                XML_ID: semantic_content.content_id
            }
    return attrib


def _create_tei_note_element(
    note_type: str,
    layout_block: LayoutBlock
) -> etree.EntityBase:
    return TEI_E(
        'note',
        {'type': note_type},
        *iter_layout_block_tei_children(layout_block)
    )


class TeiElementWrapper:
    def __init__(self, element: etree.ElementBase):
        self.element = element

    def xpath_nodes(self, xpath: str) -> List[etree.ElementBase]:
        return tei_xpath(self.element, xpath)

    def xpath(self, xpath: str) -> List['TeiElementWrapper']:
        return [TeiElementWrapper(node) for node in self.xpath_nodes(xpath)]

    def get_xpath_text_content_list(self, xpath: str) -> List[str]:
        return get_tei_xpath_text_content_list(self.element, xpath)

    def get_notes_text_list(self, note_type: str) -> List[str]:
        return get_tei_xpath_text_content_list(
            self.element,
            '//tei:note[@type="%s"]' % note_type,
        )

    def add_note(self, note_type: str, layout_block: LayoutBlock):
        self.element.append(create_tei_note_element(note_type, layout_block))


class TeiElementBuilder:
    def __init__(
        self,
        element: etree.ElementBase,
    ):
        self.element = element
        self.builder_by_path_fragment: Dict[str, 'TeiElementBuilder'] = {}

    def get_or_create(
        self,
        path: Optional[List[str]]
    ) -> 'TeiElementBuilder':
        if not path:
            return self
        key = path[0]
        builder = self.builder_by_path_fragment.get(key)
        if not builder:
            builder = TeiElementBuilder(TEI_E(key))
            self.element.append(builder.element)
            self.builder_by_path_fragment[key] = builder
        return builder.get_or_create(path[1:])

    def add_dict(self, attrib: dict):
        _attrib = self.element.attrib
        for k, v in attrib.items():
            _attrib[k] = v

    def append(
        self,
        child: Union[dict, etree.ElementBase]
    ):
        if isinstance(child, dict):
            self.add_dict(child)
            return
        self.element.append(child)

    def extend(self, children: List[Union[dict, etree.ElementBase]]):
        for child in children:
            self.append(child)
