from typing import FrozenSet

from lxml import etree

from sciencebeam_parser.document.tei.common import TEI_NS


def get_xml_tree(xml_root: etree.ElementBase) -> etree._ElementTree:
    if isinstance(xml_root, etree._ElementTree):  # pylint: disable=protected-access
        # Note: _XSLTResultTree is extending _ElementTree
        return xml_root
    return etree.ElementTree(xml_root)


_TEI_BLOCK_ELEMENT_NAMES = frozenset({
    'TEI', 'teiHeader', 'fileDesc', 'titleStmt', 'publicationStmt',
    'profileDesc', 'sourceDesc', 'abstract',
    'text', 'body', 'back', 'front',
    'div', 'listBibl', 'biblStruct', 'analytic', 'monogr',
    'figure', 'table', 'row',
})

_JATS_BLOCK_ELEMENT_NAMES = frozenset({
    'article', 'front', 'body', 'back',
    'journal-meta', 'article-meta', 'contrib-group', 'custom-meta-group',
    'abstract', 'sec', 'fig', 'caption', 'table-wrap', 'table', 'tbody',
    'ref-list', 'app-group', 'app', 'ack',
})

_BLOCK_ELEMENT_NAMES_BY_NAMESPACE = {
    TEI_NS: _TEI_BLOCK_ELEMENT_NAMES,
    None: _JATS_BLOCK_ELEMENT_NAMES,
}

_XML_SPACE_ATTRIB_NAME = '{http://www.w3.org/XML/1998/namespace}space'


def _indent_element(
    element: etree.ElementBase,
    block_element_names: FrozenSet[str],
    space: str,
    level: int
) -> None:
    # A preserve subtree is returned from rather than descended into, which is
    # why the inherited value needs no tracking.
    if element.get(_XML_SPACE_ATTRIB_NAME) == 'preserve':
        return
    if etree.QName(element).localname not in block_element_names:
        return
    children = [child for child in element if isinstance(child.tag, str)]
    if not children:
        return
    indent = '\n' + level * space
    child_indent = indent + space
    if not (element.text and element.text.strip()):
        element.text = child_indent
    for i, child in enumerate(children):
        _indent_element(child, block_element_names, space, level + 1)
        if not (child.tail and child.tail.strip()):
            child.tail = child_indent if i < len(children) - 1 else indent


def _indent_xml(root: etree.ElementBase, space: str = '  ') -> None:
    block_element_names = _BLOCK_ELEMENT_NAMES_BY_NAMESPACE.get(
        etree.QName(root).namespace
    )
    if block_element_names is None:
        return
    _indent_element(root, block_element_names, space=space, level=0)


def serialize_xml_to_file(
    xml_root: etree.ElementBase,
    filename: str,
    pretty_print: bool = False
):
    xml_tree = get_xml_tree(xml_root)
    if pretty_print:
        _indent_xml(xml_tree.getroot())
    xml_tree.write(
        filename,
        encoding='utf-8',
        pretty_print=False
    )
