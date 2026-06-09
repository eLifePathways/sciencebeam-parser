from lxml.builder import ElementMaker

from sciencebeam_parser.document.layout_document import LayoutPageCoordinates
from sciencebeam_parser.external.pdfalto.parser import AltoParser, parse_alto_root, ALTO_NS


ALTO_E = ElementMaker(namespace=ALTO_NS, nsmap={
    None: ALTO_NS
})


FONT_ID_1 = 'font1'
FONTFAMILY_1 = 'fontfamily1'
FONTSIZE_1 = 11.1

FONT_ID_2 = 'font2'
FONTFAMILY_2 = 'fontfamily2'
FONTSIZE_2 = 22.2

BOLD = 'bold'
ITALICS = 'italics'
SUBSCRIPT = 'subscript'
SUPERSCRIPT = 'superscript'

TOKEN_1 = 'token1'
TOKEN_2 = 'token2'

COORDINATES_1 = LayoutPageCoordinates(
    x=100.1, y=101.1, width=102.2, height=103.3, page_number=1
)

COORDINATES_2 = LayoutPageCoordinates(
    x=200.1, y=201.1, width=202.2, height=203.3, page_number=1
)


class TestAltoParser:
    def test_should_parse_font_without_fontstyle(self):
        font = AltoParser().parse_font(ALTO_E.TextStyle(
            ID=FONT_ID_1,
            FONTFAMILY=FONTFAMILY_1,
            FONTSIZE=str(FONTSIZE_1)
        ))
        assert font.is_bold is False
        assert font.is_italics is False
        assert font.is_subscript is False
        assert font.is_superscript is False

    def test_should_parse_font_with_bold_fontstyle(self):
        font = AltoParser().parse_font(ALTO_E.TextStyle(
            ID=FONT_ID_1,
            FONTFAMILY=FONTFAMILY_1,
            FONTSIZE=str(FONTSIZE_1),
            FONTSTYLE=BOLD
        ))
        assert font.is_bold is True
        assert font.is_italics is False
        assert font.is_subscript is False
        assert font.is_superscript is False

    def test_should_parse_font_with_italics_fontstyle(self):
        font = AltoParser().parse_font(ALTO_E.TextStyle(
            ID=FONT_ID_1,
            FONTFAMILY=FONTFAMILY_1,
            FONTSIZE=str(FONTSIZE_1),
            FONTSTYLE=ITALICS
        ))
        assert font.is_bold is False
        assert font.is_italics is True
        assert font.is_subscript is False
        assert font.is_superscript is False

    def test_should_parse_font_with_subscript_fontstyle(self):
        font = AltoParser().parse_font(ALTO_E.TextStyle(
            ID=FONT_ID_1,
            FONTFAMILY=FONTFAMILY_1,
            FONTSIZE=str(FONTSIZE_1),
            FONTSTYLE=SUBSCRIPT
        ))
        assert font.is_bold is False
        assert font.is_italics is False
        assert font.is_subscript is True
        assert font.is_superscript is False

    def test_should_parse_font_with_superscript_fontstyle(self):
        font = AltoParser().parse_font(ALTO_E.TextStyle(
            ID=FONT_ID_1,
            FONTFAMILY=FONTFAMILY_1,
            FONTSIZE=str(FONTSIZE_1),
            FONTSTYLE=SUPERSCRIPT
        ))
        assert font.is_bold is False
        assert font.is_italics is False
        assert font.is_subscript is False
        assert font.is_superscript is True

    def test_should_parse_illustration_as_layout_graphic(self):
        page = AltoParser().parse_page(
            ALTO_E.Page(ALTO_E.PrintSpace(
                ALTO_E.Illustration(
                    ID='graphic1',
                    HPOS=str(COORDINATES_1.x),
                    VPOS=str(COORDINATES_1.y),
                    WIDTH=str(COORDINATES_1.width),
                    HEIGHT=str(COORDINATES_1.height),
                    FILEID="/path/to/graphic.svg",
                    TYPE="svg"
                )
            )),
            page_index=COORDINATES_1.page_number - 1
        )
        assert len(page.graphics) == 1
        graphic = page.graphics[0]
        assert graphic.local_file_path == '/path/to/graphic.svg'
        assert graphic.coordinates == COORDINATES_1
        assert graphic.graphic_type == 'svg'
        assert graphic.page_meta.page_number == COORDINATES_1.page_number

    def test_should_parse_page_meta_data(self):
        page = AltoParser().parse_page(
            ALTO_E.Page(
                {'PHYSICAL_IMG_NR': '10', 'WIDTH': '101', 'HEIGHT': '102'},
                ALTO_E.PrintSpace(),
            ),
            page_index=0
        )
        assert page.meta.page_number == 10
        assert page.meta.coordinates == LayoutPageCoordinates(
            x=0, y=0, width=101, height=102, page_number=10
        )

    def test_should_use_default_page_number(self):
        page = AltoParser().parse_page(
            ALTO_E.Page(
                ALTO_E.PrintSpace(),
            ),
            page_index=10
        )
        assert page.meta.page_number == 11

    def test_should_clamp_negative_token_dimensions_to_zero(self):
        page = AltoParser().parse_page(
            ALTO_E.Page(
                ALTO_E.PrintSpace(
                    ALTO_E.TextBlock(
                        ALTO_E.TextLine(
                            ALTO_E.String(
                                CONTENT=TOKEN_1,
                                STYLEREFS=FONT_ID_1,
                                HPOS='253.44',
                                VPOS='-94923152',
                                WIDTH='61.65',
                                HEIGHT='-48132729'
                            )
                        )
                    )
                )
            ),
            page_index=0
        )
        tokens = page.blocks[0].lines[0].tokens
        assert len(tokens) == 1
        coords = tokens[0].coordinates
        assert coords is not None
        assert coords.width == 61.65
        assert coords.height == 0.0


class TestParseAltoRoot:
    def test_should_parse_simple_document(self):
        layout_document = parse_alto_root(ALTO_E.alto(
            ALTO_E.Styles(
                ALTO_E.TextStyle(
                    ID=FONT_ID_1,
                    FONTFAMILY=FONTFAMILY_1,
                    FONTSIZE=str(FONTSIZE_1)
                ),
                ALTO_E.TextStyle(
                    ID=FONT_ID_2,
                    FONTFAMILY=FONTFAMILY_2,
                    FONTSIZE=str(FONTSIZE_2),
                    FONTSTYLE=f'{BOLD} {ITALICS}'
                )
            ),
            ALTO_E.Layout(
                ALTO_E.Page(
                    ALTO_E.PrintSpace(
                        ALTO_E.TextBlock(
                            ALTO_E.TextLine(
                                ALTO_E.String(
                                    CONTENT=TOKEN_1,
                                    STYLEREFS=FONT_ID_1,
                                    HPOS=str(COORDINATES_1.x),
                                    VPOS=str(COORDINATES_1.y),
                                    WIDTH=str(COORDINATES_1.width),
                                    HEIGHT=str(COORDINATES_1.height)
                                ),
                                ALTO_E.String(
                                    CONTENT=TOKEN_2,
                                    STYLEREFS=FONT_ID_2,
                                    HPOS=str(COORDINATES_2.x),
                                    VPOS=str(COORDINATES_2.y),
                                    WIDTH=str(COORDINATES_2.width),
                                    HEIGHT=str(COORDINATES_2.height)
                                )
                            )
                        )
                    )
                )
            )
        ))
        assert len(layout_document.pages) == 1
        assert len(layout_document.pages[0].blocks) == 1
        assert len(layout_document.pages[0].blocks[0].lines) == 1
        tokens = layout_document.pages[0].blocks[0].lines[0].tokens
        assert len(tokens) == 2
        token = tokens[0]
        assert token.text == TOKEN_1
        assert token.font.font_family == FONTFAMILY_1
        assert token.font.font_size == FONTSIZE_1
        assert token.font.is_bold is False
        assert token.font.is_italics is False
        assert token.coordinates == COORDINATES_1
        token = tokens[1]
        assert token.text == TOKEN_2
        assert token.font.font_family == FONTFAMILY_2
        assert token.font.font_size == FONTSIZE_2
        assert token.font.is_bold is True
        assert token.font.is_italics is True
        assert token.coordinates == COORDINATES_2
        assert tokens[0].line_meta is not None
        assert tokens[0].line_meta == tokens[1].line_meta
        assert tokens[0].line_meta.page_meta == layout_document.pages[0].meta
