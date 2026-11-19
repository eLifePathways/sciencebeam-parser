from lxml import etree

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.xslt import TEI_TO_JATS_XSLT_FILE
from sciencebeam_parser.transformers.xslt import XsltTransformerWrapper
from sciencebeam_parser.utils.xml_serialization import serialize_xml_to_file


class TeiToJatsConverter:
    """The JATS the service returns for a TEI document."""

    def __init__(self, xslt_transformer: XsltTransformerWrapper, pretty_print: bool):
        self.xslt_transformer = xslt_transformer
        self.pretty_print = pretty_print

    @staticmethod
    def from_stylesheet(stylesheet: bytes, config: AppConfig) -> 'TeiToJatsConverter':
        return TeiToJatsConverter(
            XsltTransformerWrapper.from_template_string(
                stylesheet,
                xslt_template_parameters=(
                    config.get('xslt', {}).get('tei_to_jats', {}).get('parameters', {})
                )
            ),
            pretty_print=config.get('output', {}).get('pretty_print_xml', True)
        )

    @staticmethod
    def from_config(config: AppConfig) -> 'TeiToJatsConverter':
        return TeiToJatsConverter.from_stylesheet(
            etree.tostring(etree.parse(TEI_TO_JATS_XSLT_FILE)), config
        )

    def get_jats_root(self, tei_root: etree.ElementBase) -> etree.ElementBase:
        return self.xslt_transformer(tei_root)

    def write_jats(self, tei_root: etree.ElementBase, filename: str) -> None:
        serialize_xml_to_file(
            self.get_jats_root(tei_root), filename, pretty_print=self.pretty_print
        )
