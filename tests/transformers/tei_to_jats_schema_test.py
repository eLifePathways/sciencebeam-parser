from pathlib import Path

import pytest
from lxml import etree

from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.transformers.tei_to_jats import TeiToJatsConverter
from sciencebeam_parser.transformers.xslt import XsltTransformerWrapper
from sciencebeam_parser.resources.xslt import TEI_TO_JATS_XSLT_FILE


JATS_ARCHIVING_1_4_XSD = (
    Path(__file__).parent / 'resources' / 'jats-archiving-1.4-xsd' / 'JATS-archivearticle1-4.xsd'
)


# Every template the stylesheet has, with invented content.
TEI_FILE = Path(__file__).parent / 'resources' / 'every-template.tei.xml'

PARAMETER_SETS = [
    {},
    {'output_parameters': 'true', 'output_bold': 'true', 'output_italic': 'true'},
    {'acknowledgement_target': 'body', 'annex_target': 'body'},
    {'annex_target': 'app'},
]


@pytest.fixture(name='jats_schema', scope='module')
def _jats_schema() -> etree.XMLSchema:
    return etree.XMLSchema(etree.parse(
        str(JATS_ARCHIVING_1_4_XSD), etree.XMLParser(no_network=True)
    ))


def _assert_valid(jats_schema: etree.XMLSchema, jats: etree.ElementBase) -> None:
    assert jats_schema.validate(jats), [
        f'{error.line}: {error.message}' for error in jats_schema.error_log
    ]


@pytest.mark.parametrize('parameters', PARAMETER_SETS)
def test_should_write_jats_the_1_4_archiving_schema_accepts(
    jats_schema: etree.XMLSchema, parameters: dict
):
    transformer = XsltTransformerWrapper.from_template_file(
        TEI_TO_JATS_XSLT_FILE, xslt_template_parameters=parameters
    )
    _assert_valid(jats_schema, transformer(etree.parse(str(TEI_FILE)).getroot()))


def test_should_write_a_file_the_1_4_archiving_schema_accepts(
    jats_schema: etree.XMLSchema, tmp_path: Path
):
    converter = TeiToJatsConverter.from_config(AppConfig.load_yaml(DEFAULT_CONFIG_FILE))
    jats_file = tmp_path / 'jats.xml'
    converter.write_jats(etree.parse(str(TEI_FILE)).getroot(), str(jats_file))
    jats = etree.parse(str(jats_file))
    assert jats.docinfo.public_id.startswith('-//NLM//DTD JATS (Z39.96) Journal Archiving')
    _assert_valid(jats_schema, jats)
