import logging
from typing import Dict, List

from lxml import etree

from sciencebeam_parser.document.layout_document import (
    LayoutGraphic,
    LayoutLineMeta,
    LayoutPageCoordinates,
    LayoutFont,
    LayoutPageMeta,
    LayoutToken,
    LayoutLine,
    LayoutBlock,
    LayoutPage,
    LayoutDocument,
    EMPTY_FONT
)


LOGGER = logging.getLogger(__name__)

ALTO_NS = 'http://www.loc.gov/standards/alto/ns-v3#'
ALTO_NS_MAP = {
    'alto': ALTO_NS
}


def alto_xpath(parent: etree.ElementBase, xpath: str) -> List[etree.ElementBase]:
    return parent.xpath(xpath, namespaces=ALTO_NS_MAP)


class AltoParser:
    def __init__(self):
        self.font_by_id_map: Dict[str, LayoutFont] = {}

    def parse_page_coordinates(
        self,
        node: etree.ElementBase,
        page_number: int
    ) -> LayoutPageCoordinates:
        width = float(node.attrib.get('WIDTH', 0))
        height = float(node.attrib.get('HEIGHT', 0))
        if width < 0 or height < 0:
            LOGGER.warning(
                'Clamping negative dimensions to zero: width=%r, height=%r, attrib=%r',
                width, height, dict(node.attrib)
            )
            width = max(0.0, width)
            height = max(0.0, height)
        return LayoutPageCoordinates(
            x=float(node.attrib.get('HPOS', 0)),
            y=float(node.attrib.get('VPOS', 0)),
            width=width,
            height=height,
            page_number=page_number
        )

    def parse_token(
        self,
        token_node: etree.ElementBase,
        page_number: int,
        layout_line_meta: LayoutLineMeta
    ) -> LayoutToken:
        return LayoutToken(
            text=token_node.attrib.get('CONTENT') or '',
            font=self.font_by_id_map.get(
                token_node.attrib.get('STYLEREFS'),
                EMPTY_FONT
            ),
            coordinates=self.parse_page_coordinates(token_node, page_number=page_number),
            line_meta=layout_line_meta
        )

    def parse_line(
        self,
        line_node: etree.ElementBase,
        page_number: int,
        page_meta: LayoutPageMeta
    ) -> LayoutLine:
        return LayoutLine(tokens=[
            self.parse_token(
                token_node,
                page_number=page_number,
                layout_line_meta=LayoutLineMeta(
                    line_id=id(line_node),
                    page_meta=page_meta
                )
            )
            for token_node in alto_xpath(line_node, './/alto:String')
        ])

    def parse_block(
        self,
        block_node: etree.ElementBase,
        page_number: int,
        page_meta: LayoutPageMeta
    ) -> LayoutBlock:
        return LayoutBlock(lines=[
            self.parse_line(line_node, page_number=page_number, page_meta=page_meta)
            for line_node in alto_xpath(block_node, './/alto:TextLine[alto:String]')
        ])

    def parse_graphic(
        self,
        graphic_node: etree.ElementBase,
        page_number: int,
        page_meta: LayoutPageMeta
    ) -> LayoutGraphic:
        attrib = graphic_node.attrib
        return LayoutGraphic(
            local_file_path=attrib.get('FILEID'),
            coordinates=self.parse_page_coordinates(graphic_node, page_number=page_number),
            graphic_type=attrib.get('TYPE'),
            page_meta=page_meta
        )

    def parse_page(
        self,
        page_node: etree.ElementBase,
        page_index: int
    ) -> LayoutPage:
        page_number_str = page_node.attrib.get('PHYSICAL_IMG_NR')
        page_number = int(page_number_str) if page_number_str else 1 + page_index
        width_str = page_node.attrib.get('WIDTH')
        height_str = page_node.attrib.get('HEIGHT')
        coordinates = (
            LayoutPageCoordinates(
                x=0,
                y=0,
                width=float(width_str),
                height=float(height_str),
                page_number=page_number
            )
            if width_str and height_str
            else None
        )
        page_meta = LayoutPageMeta(
            page_number=page_number,
            coordinates=coordinates
        )
        return LayoutPage(
            meta=page_meta,
            blocks=[
                self.parse_block(block_node, page_number=page_number, page_meta=page_meta)
                for block_node in alto_xpath(page_node, './/alto:TextBlock')
            ],
            graphics=[
                self.parse_graphic(graphic_node, page_number=page_number, page_meta=page_meta)
                for graphic_node in alto_xpath(page_node, './/alto:Illustration')
            ]
        )

    def parse_font(self, font_node: etree.ElementBase) -> LayoutFont:
        font_styles = (font_node.attrib.get('FONTSTYLE') or '').split(' ')
        return LayoutFont(
            font_id=font_node.attrib.get('ID'),
            font_family=font_node.attrib.get('FONTFAMILY'),
            font_size=float(font_node.attrib.get('FONTSIZE')),
            is_bold='bold' in font_styles,
            is_italics='italics' in font_styles,
            is_subscript='subscript' in font_styles,
            is_superscript='superscript' in font_styles
        )

    def parse_font_by_id_map(self, root: etree.ElementBase) -> Dict[str, LayoutFont]:
        fonts = [
            self.parse_font(font_node)
            for font_node in alto_xpath(root, './alto:Styles/alto:TextStyle')
        ]
        return {
            font.font_id: font
            for font in fonts
        }

    def parse_root(self, root: etree.ElementBase) -> LayoutDocument:
        self.font_by_id_map = self.parse_font_by_id_map(root)
        return LayoutDocument(pages=[
            self.parse_page(page_node, page_index=page_index)
            for page_index, page_node in enumerate(alto_xpath(root, './/alto:Page'))
        ])


def parse_alto_root(root: etree.ElementBase) -> LayoutDocument:
    return AltoParser().parse_root(root)
