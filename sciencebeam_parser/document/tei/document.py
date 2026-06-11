from typing import List, Optional

from lxml import etree

from sciencebeam_parser.document.layout_document import LayoutBlock
from sciencebeam_parser.document.tei.common import (
    TEI_E,
    TeiElementWrapper,
    extend_element,
    get_or_create_element_at,
    get_tei_xpath_text_content_list,
    iter_layout_block_tei_children,
    tei_xpath
)


class TeiAuthor(TeiElementWrapper):
    pass


class TeiAffiliation(TeiElementWrapper):
    pass


class TeiSectionParagraph(TeiElementWrapper):
    def __init__(self, element: etree.ElementBase):
        super().__init__(element)
        self._pending_whitespace: Optional[str] = None

    def add_content(self, layout_block: LayoutBlock):
        if self._pending_whitespace:
            extend_element(self.element, [self._pending_whitespace])
        self._pending_whitespace = layout_block.whitespace
        extend_element(
            self.element,
            iter_layout_block_tei_children(layout_block)
        )


class TeiSection(TeiElementWrapper):
    def get_title_text(self) -> str:
        return '\n'.join(get_tei_xpath_text_content_list(
            self.element,
            '//tei:head',
        ))

    def get_paragraph_text_list(self) -> List[str]:
        return get_tei_xpath_text_content_list(
            self.element,
            '//tei:p',
        )

    def add_paragraph(self, paragraph: TeiSectionParagraph):
        self.element.append(paragraph.element)

    def create_paragraph(self) -> TeiSectionParagraph:
        return TeiSectionParagraph(TEI_E('p'))


class TeiDocument(TeiElementWrapper):
    def __init__(self, root: Optional[etree.ElementBase] = None):
        if root is None:
            self.root = TEI_E('TEI')
        else:
            self.root = root
        self._reference_element: Optional[etree.ElementBase] = None
        super().__init__(self.root)

    def get_or_create_element_at(self, path: List[str]) -> etree.ElementBase:
        return get_or_create_element_at(self.root, path)

    def set_child_element_at(self, path: List[str], child: etree.ElementBase):
        parent = self.get_or_create_element_at(path)
        parent.append(child)

    def get_title(self) -> str:
        return '\n'.join(get_tei_xpath_text_content_list(
            self.root,
            '//tei:fileDesc/tei:titleStmt/tei:title[@level="a"][@type="main"]',
        ))

    def set_title(self, title: str):
        self.set_child_element_at(
            ['teiHeader', 'fileDesc', 'titleStmt'],
            TEI_E('title', title, level="a", type="main")
        )

    def set_title_layout_block(self, title_block: LayoutBlock, trailing_text: str = ''):
        title_elem = TEI_E(
            'title',
            {'level': 'a', 'type': 'main'},
            *iter_layout_block_tei_children(title_block)
        )
        if trailing_text:
            title_elem.tail = trailing_text
        self.set_child_element_at(
            ['teiHeader', 'fileDesc', 'titleStmt'],
            title_elem
        )

    def get_abstract(self) -> str:
        return '\n'.join(get_tei_xpath_text_content_list(
            self.root,
            '//tei:abstract/tei:p',
        ))

    def set_abstract(self, abstract: str):
        self.set_child_element_at(
            ['teiHeader', 'profileDesc', 'abstract'],
            TEI_E('p', abstract)
        )

    def set_abstract_layout_block(self, abstract_block: LayoutBlock):
        self.set_child_element_at(
            ['teiHeader', 'profileDesc', 'abstract'],
            TEI_E('p', *iter_layout_block_tei_children(abstract_block))
        )

    def get_body_element(self) -> etree.ElementBase:
        return self.get_or_create_element_at(['text', 'body'])

    def get_body(self) -> TeiElementWrapper:
        return TeiElementWrapper(self.get_body_element())

    def get_back_element(self) -> etree.ElementBase:
        return self.get_or_create_element_at(['text', 'back'])

    def get_back_annex_element(self) -> etree.ElementBase:
        return self.get_or_create_element_at(['text', 'back', 'div[@type="annex"]'])

    def get_back_annex(self) -> TeiElementWrapper:
        return TeiElementWrapper(self.get_back_annex_element())

    def get_references_element(self) -> etree.ElementBase:
        if self._reference_element is not None:
            return self._reference_element
        self._reference_element = self.get_or_create_element_at(
            ['text', 'back', 'div[@type="references"]']
        )
        return self._reference_element

    def get_references(self) -> TeiElementWrapper:
        return TeiElementWrapper(self.get_references_element())

    def get_body_sections(self) -> List[TeiSection]:
        return [
            TeiSection(element)
            for element in tei_xpath(self.get_body_element(), './tei:div')
        ]

    def add_body_section(self, section: TeiSection):
        self.get_body_element().append(section.element)

    def add_back_annex_section(self, section: TeiSection):
        self.get_back_annex_element().append(section.element)

    def add_acknowledgement_section(self, section: TeiSection):
        section.element.attrib['type'] = 'acknowledgement'
        self.get_back_element().append(section.element)

    def create_section(self) -> TeiSection:
        return TeiSection(TEI_E('div'))
