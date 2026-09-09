# pylint: disable=too-many-lines
import logging
import os
import re
import gzip
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from lxml import etree


from sciencebeam_parser.utils.xml import get_text_content_list
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutPage,
    join_layout_tokens,
)
from sciencebeam_parser.models.data import LayoutModelData
from sciencebeam_parser.document.tei.common import get_tei_xpath_text_content_list
from sciencebeam_parser.models.data import DEFAULT_DOCUMENT_FEATURES_CONTEXT
from sciencebeam_parser.models.training_data import TeiTrainingDataGenerator
from sciencebeam_parser.models.fulltext.training_data import FullTextTeiTrainingDataGenerator
from sciencebeam_parser.models.reference_segmenter.training_data import (
    ReferenceSegmenterTeiTrainingDataGenerator
)
from sciencebeam_parser.models.segmentation.training_data import (
    SegmentationTeiTrainingDataGenerator
)
from sciencebeam_parser.models.header.training_data import HeaderTeiTrainingDataGenerator
from sciencebeam_parser.models.affiliation_address.training_data import (
    AffiliationAddressTeiTrainingDataGenerator
)
from sciencebeam_parser.models.name.training_data import (
    NameTeiTrainingDataGenerator
)
from sciencebeam_parser.models.figure.training_data import (
    FigureTeiTrainingDataGenerator
)
from sciencebeam_parser.models.table.training_data import (
    TableTeiTrainingDataGenerator
)
from sciencebeam_parser.models.citation.training_data import (
    CitationTeiTrainingDataGenerator
)
from sciencebeam_parser.models.citation.labels import IDENTIFIER_LABEL
from sciencebeam_parser.training.jats.annotated_document import JatsAnnotatedLayoutDocument
from sciencebeam_parser.training.jats.field_vocab import JatsFieldNames, JatsSubFieldNames
import sciencebeam_parser.training.cli.generate_data as generate_data_module
from sciencebeam_parser.training.cli.generate_data import (
    CitationModelTrainingDataGenerator,
    ModelResultCache,
    NameCitationModelTrainingDataGenerator,
    ReferenceSegmenterModelTrainingDataGenerator,
    TrainingDataDocumentContext,
    _split_references_by_jats_instance,
    generate_training_data_for_layout_document,
    main,
)
from sciencebeam_parser.training.quality.record import DocumentStatus, JatsStatus

from tests.processors.fulltext.model_mocks import MockFullTextModels
from tests.test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)

MINIMAL_EXAMPLE_PDF = 'test-data/minimal-example.pdf'
MINIMAL_EXAMPLE_PDF_PATTERN = 'test-data/minimal-example*.pdf'

NON_EXISTING_PDF_PATTERN = 'test-data/non-existing*.pdf'


SOURCE_FILENAME_1 = 'test1.pdf'


class SampleLayoutDocument:
    def __init__(self) -> None:
        self.title_block = LayoutBlock.for_text('This is the title')

        self.author_surname_block = LayoutBlock.for_text('Author Surname 1')
        self.author_block = LayoutBlock.merge_blocks([self.author_surname_block])

        self.institution_block = LayoutBlock.for_text('Institution 1')
        self.affiliation_block = LayoutBlock.merge_blocks([self.institution_block])

        self.header_block = LayoutBlock.merge_blocks([
            self.title_block,
            self.author_block,
            self.affiliation_block
        ])

        self.figure_head_block = LayoutBlock.for_text('Figure 1')
        self.figure_block = LayoutBlock.merge_blocks([self.figure_head_block])

        self.table_head_block = LayoutBlock.for_text('Table 1')
        self.table_block = LayoutBlock.merge_blocks([self.table_head_block])

        self.body_section_title_block = LayoutBlock.for_text('Section 1')
        self.body_section_paragraph_block = LayoutBlock.for_text('Paragraph 1')
        self.body_block = LayoutBlock.merge_blocks([
            self.body_section_title_block,
            self.body_section_paragraph_block,
            self.figure_block,
            self.table_block
        ])

        self.ref_author_surname_block = LayoutBlock.for_text('Ref Author Surname 1')
        self.ref_author_block = LayoutBlock.merge_blocks([self.ref_author_surname_block])

        self.ref_label_block = LayoutBlock.for_text('1')
        self.ref_title_block = LayoutBlock.for_text('Reference 1')
        self.ref_text_block = LayoutBlock.merge_blocks([
            self.ref_title_block,
            self.ref_author_block
        ])
        self.ref_ref_block = LayoutBlock.merge_blocks([
            self.ref_label_block,
            self.ref_text_block
        ])

        self.layout_document = LayoutDocument(pages=[LayoutPage(blocks=[
            self.header_block,
            self.body_block,
            self.ref_ref_block
        ])])


def configure_fulltext_models_mock_with_sample_document(
    fulltext_models_mock: MockFullTextModels,
    sample_layout_document: SampleLayoutDocument
):
    doc = sample_layout_document
    segmentation_model_mock = fulltext_models_mock.segmentation_model_mock
    header_model_mock = fulltext_models_mock.header_model_mock
    name_header_model_mock = fulltext_models_mock.name_header_model_mock
    name_citation_model_mock = fulltext_models_mock.name_citation_model_mock
    affiliation_address_model_mock = fulltext_models_mock.affiliation_address_model_mock
    fulltext_model_mock = fulltext_models_mock.fulltext_model_mock
    reference_segmenter_model_mock = fulltext_models_mock.reference_segmenter_model_mock
    citation_model_mock = fulltext_models_mock.citation_model_mock
    figure_model_mock = fulltext_models_mock.figure_model_mock
    table_model_mock = fulltext_models_mock.table_model_mock

    segmentation_model_mock.update_label_by_layout_block(
        doc.header_block, '<header>'
    )
    segmentation_model_mock.update_label_by_layout_block(
        doc.body_block, '<body>'
    )
    segmentation_model_mock.update_label_by_layout_block(
        doc.ref_ref_block, '<references>'
    )

    header_model_mock.update_label_by_layout_block(
        doc.title_block, '<title>'
    )
    header_model_mock.update_label_by_layout_block(
        doc.author_block, '<author>'
    )
    header_model_mock.update_label_by_layout_block(
        doc.affiliation_block, '<affiliation>'
    )

    affiliation_address_model_mock.update_label_by_layout_block(
        doc.institution_block, '<institution>'
    )

    name_header_model_mock.update_label_by_layout_block(
        doc.author_surname_block, '<surname>'
    )

    fulltext_model_mock.update_label_by_layout_block(
        doc.body_section_title_block, '<section>'
    )
    fulltext_model_mock.update_label_by_layout_block(
        doc.body_section_paragraph_block, '<paragraph>'
    )
    fulltext_model_mock.update_label_by_layout_block(
        doc.figure_block, '<figure>'
    )
    fulltext_model_mock.update_label_by_layout_block(
        doc.table_block, '<table>'
    )

    figure_model_mock.update_label_by_layout_block(
        doc.figure_head_block, '<figure_head>'
    )

    table_model_mock.update_label_by_layout_block(
        doc.table_head_block, '<figure_head>'
    )

    reference_segmenter_model_mock.update_label_by_layout_block(
        doc.ref_label_block, '<label>'
    )
    reference_segmenter_model_mock.update_label_by_layout_block(
        doc.ref_text_block, '<reference>'
    )

    citation_model_mock.update_label_by_layout_block(
        doc.ref_title_block, '<title>'
    )
    citation_model_mock.update_label_by_layout_block(
        doc.ref_author_block, '<author>'
    )

    name_citation_model_mock.update_label_by_layout_block(
        doc.ref_author_surname_block, '<surname>'
    )


@pytest.fixture(autouse=True)
def _patch_sciencebeam_parser_class_mock(
    sciencebeam_parser_class_mock: MagicMock
) -> Iterator[MagicMock]:
    with patch.object(
        generate_data_module, 'ScienceBeamParser', sciencebeam_parser_class_mock
    ) as mock:
        yield mock


@pytest.fixture(name='sample_layout_document')
def _sample_layout_document() -> SampleLayoutDocument:
    return SampleLayoutDocument()


@pytest.fixture(name='sciencebeam_parser_session_mock', autouse=True)
def _sciencebeam_parser_session_mock(
    sciencebeam_parser_mock: MagicMock
) -> MockFullTextModels:
    mock = MagicMock(name='ScienceBeamParserSession')
    sciencebeam_parser_mock.get_new_session.return_value.__enter__.return_value = mock
    return mock


@pytest.fixture(name='sciencebeam_parser_source_mock', autouse=True)
def _sciencebeam_parser_source_mock(
    sciencebeam_parser_session_mock: MagicMock,
    sample_layout_document: SampleLayoutDocument
) -> MockFullTextModels:
    mock = MagicMock(name='ScienceBeamParserSource')
    mock.get_layout_document.return_value = sample_layout_document.layout_document
    sciencebeam_parser_session_mock.get_source.return_value = mock
    return mock


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def normalize_whitespace_list(text_iterable: Iterable[str]) -> Sequence[str]:
    return [
        normalize_whitespace(text)
        for text in text_iterable
    ]


def _get_expected_file_path_with_suffix(
    output_path: Path,
    source_filename: str,
    suffix: Optional[str],
    pre_file_path_suffix: str = ''
) -> Path:
    assert suffix
    source_name = os.path.splitext(os.path.basename(source_filename))[0]
    return output_path.joinpath(source_name + pre_file_path_suffix + suffix)


def _check_tei_training_data_generator_output_and_return_xml_root(
    tei_training_data_generator: TeiTrainingDataGenerator,
    output_path: Path,
    expect_raw_data: bool,
    source_filename: str = SOURCE_FILENAME_1,
    pre_file_path_suffix: str = ''
) -> etree.ElementBase:
    expected_tei_path = _get_expected_file_path_with_suffix(
        output_path,
        source_filename,
        tei_training_data_generator.get_default_tei_filename_suffix(),
        pre_file_path_suffix=pre_file_path_suffix
    )
    assert expected_tei_path.exists()
    if expect_raw_data:
        expected_data_path = _get_expected_file_path_with_suffix(
            output_path,
            source_filename,
            tei_training_data_generator.get_default_data_filename_suffix(),
            pre_file_path_suffix=pre_file_path_suffix
        )
        assert expected_data_path.exists()
    xml_root = etree.parse(str(expected_tei_path)).getroot()
    LOGGER.debug('xml: %r', etree.tostring(xml_root))
    return xml_root


def _check_tei_training_data_generator_output(
    tei_training_data_generator: TeiTrainingDataGenerator,
    output_path: Path,
    expect_raw_data: bool,
    tei_xml_xpath: str,
    tei_expected_values: Sequence[str],
    **kwargs
):
    xml_root = _check_tei_training_data_generator_output_and_return_xml_root(
        tei_training_data_generator=tei_training_data_generator,
        output_path=output_path,
        expect_raw_data=expect_raw_data,
        **kwargs
    )
    assert normalize_whitespace_list(
        get_tei_xpath_text_content_list(xml_root, tei_xml_xpath)
    ) == [
        normalize_whitespace(tei_expected_value)
        for tei_expected_value in tei_expected_values
    ]


@log_on_exception
class TestGenerateTrainingDataForLayoutDocument:
    def test_should_generate_data_using_mock_models(  # noqa pylint: disable=too-many-locals, too-many-statements
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )

        output_path = tmp_path / 'output'
        output_path.mkdir()
        generate_training_data_for_layout_document(
            layout_document=sample_layout_document.layout_document,
            output_path=str(output_path),
            source_filename=SOURCE_FILENAME_1,
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            fulltext_models=fulltext_models_mock,
            use_model=True,
            use_directory_structure=False
        )

        _check_tei_training_data_generator_output(
            SegmentationTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            tei_xml_xpath='text/front',
            tei_expected_values=[sample_layout_document.header_block.text]
        )

        _check_tei_training_data_generator_output(
            HeaderTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            tei_xml_xpath='text/front/docTitle/titlePart',
            tei_expected_values=[sample_layout_document.title_block.text]
        )

        _check_tei_training_data_generator_output(
            NameTeiTrainingDataGenerator(),
            pre_file_path_suffix='.header',
            output_path=output_path,
            expect_raw_data=False,
            tei_xml_xpath='//tei:author//tei:surname',
            tei_expected_values=[sample_layout_document.author_surname_block.text]
        )

        _check_tei_training_data_generator_output(
            NameTeiTrainingDataGenerator(),
            pre_file_path_suffix='.citations',
            output_path=output_path,
            expect_raw_data=False,
            tei_xml_xpath='//tei:author//tei:surname',
            tei_expected_values=[sample_layout_document.ref_author_surname_block.text]
        )

        _check_tei_training_data_generator_output(
            AffiliationAddressTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=False,
            tei_xml_xpath='//tei:affiliation/tei:orgName[@type="institution"]',
            tei_expected_values=[sample_layout_document.institution_block.text]
        )

        _check_tei_training_data_generator_output(
            FullTextTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            tei_xml_xpath='//head',
            tei_expected_values=[sample_layout_document.body_section_title_block.text]
        )

        _check_tei_training_data_generator_output(
            FigureTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            tei_xml_xpath='//head',
            tei_expected_values=[sample_layout_document.figure_head_block.text]
        )

        _check_tei_training_data_generator_output(
            TableTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            tei_xml_xpath='//head',
            tei_expected_values=[sample_layout_document.table_head_block.text]
        )

        _check_tei_training_data_generator_output(
            ReferenceSegmenterTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            tei_xml_xpath='//bibl',
            tei_expected_values=[sample_layout_document.ref_ref_block.text]
        )

        _check_tei_training_data_generator_output(
            CitationTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=False,
            tei_xml_xpath='//tei:bibl/tei:title[@level="a"]',
            tei_expected_values=[sample_layout_document.ref_title_block.text]
        )

    def test_should_return_a_quality_record_with_the_count_per_model(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )

        output_path = tmp_path / 'output'
        output_path.mkdir()
        record = generate_training_data_for_layout_document(
            layout_document=sample_layout_document.layout_document,
            output_path=str(output_path),
            source_filename=SOURCE_FILENAME_1,
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            fulltext_models=fulltext_models_mock,
            use_model=True,
            use_directory_structure=False
        )

        json_dict_by_model = record.to_json_dict_by_model(
            ['reference-segmenter', 'citation', 'segmentation']
        )
        reference_segmenter_json_dict = json_dict_by_model['reference-segmenter']
        assert reference_segmenter_json_dict['document_id'] == 'test1'
        assert reference_segmenter_json_dict['status'] == DocumentStatus.OK
        assert reference_segmenter_json_dict['written'] is True
        assert reference_segmenter_json_dict['entity_element_count'] == 1
        assert json_dict_by_model['citation']['entity_element_count'] == 1
        assert 'entity_element_count' not in json_dict_by_model['segmentation']

    def test_should_report_a_missing_jats_without_a_reference_count(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )

        output_path = tmp_path / 'output'
        output_path.mkdir()
        record = generate_training_data_for_layout_document(
            layout_document=sample_layout_document.layout_document,
            output_path=str(output_path),
            source_filename=SOURCE_FILENAME_1,
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            fulltext_models=fulltext_models_mock,
            use_model=True,
            use_directory_structure=False
        )

        assert record.to_json_dict_by_model(
            ['reference-segmenter']
        )['reference-segmenter']['jats'] == {'status': JatsStatus.MISSING}

    def test_should_report_a_jats_that_could_not_be_parsed(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        empty_jats_path = tmp_path / 'empty.jats.xml'
        empty_jats_path.write_bytes(b'')

        output_path = tmp_path / 'output'
        output_path.mkdir()
        record = generate_training_data_for_layout_document(
            layout_document=sample_layout_document.layout_document,
            output_path=str(output_path),
            source_filename=SOURCE_FILENAME_1,
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            fulltext_models=fulltext_models_mock,
            use_model=True,
            use_directory_structure=False,
            jats_xml_filename=str(empty_jats_path)
        )

        assert record.to_json_dict_by_model(
            ['reference-segmenter']
        )['reference-segmenter']['jats'] == {'status': JatsStatus.UNPARSABLE}

    def test_should_report_a_jats_declaring_no_references_as_a_zero_count(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        jats_path = tmp_path / 'no-references.jats.xml'
        jats_path.write_text(
            '<article><back><sec><p>Appendix text.</p></sec></back></article>',
            encoding='utf-8'
        )

        output_path = tmp_path / 'output'
        output_path.mkdir()
        record = generate_training_data_for_layout_document(
            layout_document=sample_layout_document.layout_document,
            output_path=str(output_path),
            source_filename=SOURCE_FILENAME_1,
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            fulltext_models=fulltext_models_mock,
            use_model=True,
            use_directory_structure=False,
            jats_xml_filename=str(jats_path)
        )

        assert record.to_json_dict_by_model(
            ['reference-segmenter']
        )['reference-segmenter']['jats'] == {
            'status': JatsStatus.OK,
            'reference_count': 0,
            'aligned_reference_count': 0
        }

    def test_should_count_the_jats_references_and_the_citation_labels(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        jats_path = tmp_path / 'references.jats.xml'
        jats_path.write_text(
            '<article><back><ref-list>'
            '<ref><label>1</label><element-citation>'
            '<article-title>Reference 1</article-title>'
            '<person-group person-group-type="author">'
            '<name><surname>Ref Author Surname 1</surname></name>'
            '</person-group>'
            '<pub-id pub-id-type="doi">10.1234/not-printed</pub-id>'
            '</element-citation></ref>'
            '</ref-list></back></article>',
            encoding='utf-8'
        )

        output_path = tmp_path / 'output'
        output_path.mkdir()
        record = generate_training_data_for_layout_document(
            layout_document=sample_layout_document.layout_document,
            output_path=str(output_path),
            source_filename=SOURCE_FILENAME_1,
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            fulltext_models=fulltext_models_mock,
            use_model=True,
            use_directory_structure=False,
            jats_xml_filename=str(jats_path)
        )

        json_dict = record.to_json_dict_by_model(['citation'])['citation']
        assert json_dict['jats']['reference_count'] == 1
        assert json_dict['jats']['aligned_reference_count'] == 1
        label_counts = json_dict['label_counts']
        assert label_counts['<title>'] == {'jats': 1, 'marked': 1}
        # The DOI is in the JATS and not on the page: counted, and not marked.
        assert label_counts[IDENTIFIER_LABEL] == {'jats': 1, 'marked': 0}

    def test_not_should_generate_figure_data_if_not_present(  # noqa pylint: disable=too-many-locals, too-many-statements
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        fulltext_models_mock.fulltext_model_mock.update_label_by_layout_block(
            sample_layout_document.figure_block, '<paragraph>'
        )

        output_path = tmp_path / 'output'
        output_path.mkdir()
        generate_training_data_for_layout_document(
            layout_document=sample_layout_document.layout_document,
            output_path=str(output_path),
            source_filename=SOURCE_FILENAME_1,
            document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
            fulltext_models=fulltext_models_mock,
            use_model=True,
            use_directory_structure=False
        )

        example_name = os.path.splitext(os.path.basename(SOURCE_FILENAME_1))[0]

        expected_figure_tei_path = output_path.joinpath(
            example_name + FigureTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
        )
        expected_figure_data_path = output_path.joinpath(
            example_name + FigureTeiTrainingDataGenerator.DEFAULT_DATA_FILENAME_SUFFIX
        )
        assert not expected_figure_tei_path.exists()
        assert not expected_figure_data_path.exists()


@log_on_exception
class TestMain:
    def test_should_fail_if_no_files_were_found(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        output_path = tmp_path / 'generated-data'
        with pytest.raises(FileNotFoundError):
            main([
                f'--source-path={NON_EXISTING_PDF_PATTERN}',
                f'--output-path={output_path}'
            ])
        assert not output_path.exists()

    def test_should_be_able_to_generate_segmentation_training_data(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        output_path = tmp_path / 'generated-data'
        main([
            f'--source-path={MINIMAL_EXAMPLE_PDF_PATTERN}',
            f'--output-path={output_path}'
        ])
        assert output_path.exists()

        xml_root = _check_tei_training_data_generator_output_and_return_xml_root(
            SegmentationTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            source_filename=MINIMAL_EXAMPLE_PDF
        )
        assert get_text_content_list(xml_root.xpath('text'))
        assert not get_text_content_list(xml_root.xpath('text/front'))

        xml_root = _check_tei_training_data_generator_output_and_return_xml_root(
            HeaderTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            source_filename=MINIMAL_EXAMPLE_PDF
        )
        assert get_text_content_list(xml_root.xpath('text/front'))

    def test_should_add_gz_suffix_if_enabled(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        output_path = tmp_path / 'generated-data'
        main([
            f'--source-path={MINIMAL_EXAMPLE_PDF_PATTERN}',
            f'--output-path={output_path}',
            '--gzip'
        ])
        assert output_path.exists()

        tei_training_data_generator = SegmentationTeiTrainingDataGenerator()
        tei_filename_suffix = tei_training_data_generator.get_default_tei_filename_suffix()
        assert tei_filename_suffix
        expected_tei_path = _get_expected_file_path_with_suffix(
            output_path,
            MINIMAL_EXAMPLE_PDF,
            tei_filename_suffix + '.gz',
        )
        assert expected_tei_path.exists()
        with gzip.open(expected_tei_path, 'r') as fp:
            etree.parse(fp)

        data_filename_suffix = tei_training_data_generator.get_default_data_filename_suffix()
        assert data_filename_suffix
        expected_data_path = _get_expected_file_path_with_suffix(
            output_path,
            MINIMAL_EXAMPLE_PDF,
            data_filename_suffix + '.gz',
        )
        assert expected_data_path.exists()
        with gzip.open(expected_data_path, 'r') as fp:
            fp.read()

    def test_should_be_able_to_generate_segmentation_training_data_using_model(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        output_path = tmp_path / 'generated-data'
        main([
            f'--source-path={MINIMAL_EXAMPLE_PDF_PATTERN}',
            f'--output-path={output_path}',
            '--use-model'
        ])
        assert output_path.exists()

        xml_root = _check_tei_training_data_generator_output_and_return_xml_root(
            SegmentationTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            source_filename=MINIMAL_EXAMPLE_PDF
        )
        assert get_text_content_list(xml_root.xpath('text/front'))

        xml_root = _check_tei_training_data_generator_output_and_return_xml_root(
            HeaderTeiTrainingDataGenerator(),
            output_path=output_path,
            expect_raw_data=True,
            source_filename=MINIMAL_EXAMPLE_PDF
        )
        assert get_text_content_list(xml_root.xpath('text/front'))

    def test_should_allow_to_use_directory_structure(
        self,
        tmp_path: Path,
        sample_layout_document: SampleLayoutDocument,
        fulltext_models_mock: MockFullTextModels
    ):
        configure_fulltext_models_mock_with_sample_document(
            fulltext_models_mock,
            sample_layout_document
        )
        output_path = tmp_path / 'generated-data'
        main([
            '--use-directory-structure',
            f'--source-path={MINIMAL_EXAMPLE_PDF_PATTERN}',
            f'--output-path={output_path}'
        ])
        assert output_path.exists()

        expected_output_path_and_suffix_list = [(
            output_path / 'segmentation' / 'corpus' / 'tei',
            SegmentationTeiTrainingDataGenerator().get_default_tei_filename_suffix()
        ), (
            output_path / 'segmentation' / 'corpus' / 'raw',
            SegmentationTeiTrainingDataGenerator().get_default_data_filename_suffix()
        ), (
            output_path / 'header' / 'corpus' / 'tei',
            HeaderTeiTrainingDataGenerator().get_default_tei_filename_suffix()
        ), (
            output_path / 'header' / 'corpus' / 'raw',
            HeaderTeiTrainingDataGenerator().get_default_data_filename_suffix()
        ), (
            output_path / 'fulltext' / 'corpus' / 'tei',
            FullTextTeiTrainingDataGenerator().get_default_tei_filename_suffix()
        ), (
            output_path / 'fulltext' / 'corpus' / 'raw',
            FullTextTeiTrainingDataGenerator().get_default_data_filename_suffix()
        ), (
            output_path / 'figure' / 'corpus' / 'tei',
            FigureTeiTrainingDataGenerator().get_default_tei_filename_suffix()
        ), (
            output_path / 'figure' / 'corpus' / 'raw',
            FigureTeiTrainingDataGenerator().get_default_data_filename_suffix()
        ), (
            output_path / 'table' / 'corpus' / 'tei',
            TableTeiTrainingDataGenerator().get_default_tei_filename_suffix()
        ), (
            output_path / 'table' / 'corpus' / 'raw',
            TableTeiTrainingDataGenerator().get_default_data_filename_suffix()
        ), (
            output_path / 'reference-segmenter' / 'corpus' / 'tei',
            ReferenceSegmenterTeiTrainingDataGenerator().get_default_tei_filename_suffix()
        ), (
            output_path / 'reference-segmenter' / 'corpus' / 'raw',
            ReferenceSegmenterTeiTrainingDataGenerator().get_default_data_filename_suffix()
        ), (
            output_path / 'affiliation-address' / 'corpus',
            AffiliationAddressTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
        ), (
            output_path / 'citation' / 'corpus',
            CitationTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
        ), (
            output_path / 'name' / 'header' / 'corpus',
            '.header' + NameTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
        ), (
            output_path / 'name' / 'citation' / 'corpus',
            '.citations' + NameTeiTrainingDataGenerator.DEFAULT_TEI_FILENAME_SUFFIX
        )]

        for model_output_path, suffix in expected_output_path_and_suffix_list:
            file_path = _get_expected_file_path_with_suffix(
                model_output_path,
                MINIMAL_EXAMPLE_PDF,
                suffix
            )
            assert file_path.exists()


# ── Helpers for JATS-based tests ─────────────────────────────────────────────


def _doc_text(doc: LayoutDocument) -> str:
    return join_layout_tokens(list(doc.iter_all_tokens()))


def _annotate_block_as_reference(
    block: LayoutBlock,
    annotated: JatsAnnotatedLayoutDocument,
    instance_id: int,
    sub_field: Optional[str] = None,
) -> None:
    for token in block.iter_all_tokens():
        annotated.set_token_label(
            token,
            JatsFieldNames.REFERENCE,
            sub_field_name=sub_field,
            instance_id=instance_id,
        )


def _make_jats_context(
    annotated: JatsAnnotatedLayoutDocument,
    ref_blocks: List[LayoutBlock],
) -> TrainingDataDocumentContext:
    seg_labels: Dict[int, str] = {
        id(line): '<references>'
        for block in ref_blocks
        for line in block.lines
    }
    return TrainingDataDocumentContext(
        output_path='/tmp/test',
        source_filename='test.pdf',
        document_features_context=DEFAULT_DOCUMENT_FEATURES_CONTEXT,
        fulltext_models=MagicMock(),
        use_model=False,
        use_directory_structure=False,
        model_result_cache=ModelResultCache(),
        gzip_enabled=False,
        jats_annotated_document=annotated,
        jats_segmentation_labels=seg_labels,
    )


# ── _split_references_by_jats_instance ────────────────────────────────────────

@log_on_exception
class TestSplitReferencesByJatsInstance:
    def test_returns_one_document_per_reference_instance(self):
        ref1_block = LayoutBlock.for_text('Smith 2020 Title Journal')
        ref2_block = LayoutBlock.for_text('Jones 2019 Article Nature')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[ref1_block, ref2_block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        _annotate_block_as_reference(ref1_block, annotated, instance_id=1)
        _annotate_block_as_reference(ref2_block, annotated, instance_id=2)

        result = _split_references_by_jats_instance(refs_doc, annotated)

        assert len(result) == 2
        assert _doc_text(result[0]) == ref1_block.text
        assert _doc_text(result[1]) == ref2_block.text

    def test_skips_blocks_with_no_reference_tokens(self):
        title_block = LayoutBlock.for_text('References')
        ref1_block = LayoutBlock.for_text('Smith 2020')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[title_block, ref1_block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        _annotate_block_as_reference(ref1_block, annotated, instance_id=1)

        result = _split_references_by_jats_instance(refs_doc, annotated)

        assert len(result) == 1
        assert _doc_text(result[0]) == ref1_block.text

    def test_returns_empty_list_when_no_reference_annotations(self):
        block = LayoutBlock.for_text('Some text')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)

        result = _split_references_by_jats_instance(refs_doc, annotated)

        assert not result

    def test_assigns_mixed_block_to_dominant_instance(self):
        # A block where most tokens belong to instance 1, one token to instance 2
        block = LayoutBlock.for_text('Smith Jones Brown Extra')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        tokens = list(block.iter_all_tokens())
        for t in tokens[:3]:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=1)
        annotated.set_token_label(tokens[3], JatsFieldNames.REFERENCE, instance_id=2)

        result = _split_references_by_jats_instance(refs_doc, annotated)

        assert len(result) == 1  # dominant instance 1 wins, block not split

    def test_gap_fill_includes_unlabeled_continuation_block(self):
        # Simulates a DOI continuation line: ref1 block has labeled tokens, the
        # continuation block (e.g. the second line of a wrapped DOI) has none.
        # It should be attached to ref1, not silently dropped.
        ref1_block = LayoutBlock.for_text('Smith 2020 doi')
        cont_block = LayoutBlock.for_text('10.1234/continuation')
        ref2_block = LayoutBlock.for_text('Jones 2019 Article Nature')
        refs_doc = LayoutDocument(
            pages=[LayoutPage(blocks=[ref1_block, cont_block, ref2_block])]
        )
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        _annotate_block_as_reference(ref1_block, annotated, instance_id=1)
        _annotate_block_as_reference(ref2_block, annotated, instance_id=2)
        # cont_block is intentionally left unannotated

        result = _split_references_by_jats_instance(refs_doc, annotated)

        assert len(result) == 2
        doc1_text = _doc_text(result[0])
        assert 'Smith' in doc1_text
        assert '10.1234/continuation' in doc1_text  # gap-filled into ref1
        assert 'Jones' not in doc1_text
        doc2_text = _doc_text(result[1])
        assert 'Jones' in doc2_text
        assert '10.1234/continuation' not in doc2_text

    def test_leading_unlabeled_block_is_still_skipped(self):
        # A block appearing BEFORE any labeled content has no preceding instance
        # to gap-fill from, so it should be dropped (e.g. the "References" title).
        title_block = LayoutBlock.for_text('References')
        ref1_block = LayoutBlock.for_text('Smith 2020')
        refs_doc = LayoutDocument(
            pages=[LayoutPage(blocks=[title_block, ref1_block])]
        )
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        _annotate_block_as_reference(ref1_block, annotated, instance_id=1)

        result = _split_references_by_jats_instance(refs_doc, annotated)

        assert len(result) == 1
        assert 'References' not in _doc_text(result[0])
        assert 'Smith' in _doc_text(result[0])

    def test_splits_block_at_instance_boundary(self):
        # Two references share one LayoutBlock (Burguete/Carvalho pattern):
        # line 1 belongs to instance 1, line 2 belongs to instance 2.
        # The block must be split so each reference gets its own sub-document.
        line1 = LayoutLine.for_text('Burguete 2020 Title Journal')
        line2 = LayoutLine.for_text('Carvalho 2019 Another Paper')
        shared_block = LayoutBlock(lines=[line1, line2])
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[shared_block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        for token in line1.tokens:
            annotated.set_token_label(token, JatsFieldNames.REFERENCE, instance_id=1)
        for token in line2.tokens:
            annotated.set_token_label(token, JatsFieldNames.REFERENCE, instance_id=2)

        result = _split_references_by_jats_instance(refs_doc, annotated)

        assert len(result) == 2
        assert 'Burguete' in _doc_text(result[0])
        assert 'Carvalho' not in _doc_text(result[0])
        assert 'Carvalho' in _doc_text(result[1])
        assert 'Burguete' not in _doc_text(result[1])

    def test_excludes_headnote_lines_from_gap_fill(self):
        # A page header that slipped into the references sub-document must not
        # be included in any reference's sub-document.
        ref1_block = LayoutBlock.for_text('Smith 2020 doi')
        header_line = LayoutLine.for_text('Journal Name | Volume 1 | 2020')
        header_block = LayoutBlock(lines=[header_line])
        ref2_block = LayoutBlock.for_text('Jones 2019 Article')
        refs_doc = LayoutDocument(
            pages=[LayoutPage(blocks=[ref1_block, header_block, ref2_block])]
        )
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        _annotate_block_as_reference(ref1_block, annotated, instance_id=1)
        _annotate_block_as_reference(ref2_block, annotated, instance_id=2)
        jats_seg_labels = {id(header_line): '<headnote>'}

        result = _split_references_by_jats_instance(refs_doc, annotated, jats_seg_labels)

        assert len(result) == 2
        assert 'Journal Name' not in _doc_text(result[0])
        assert 'Journal Name' not in _doc_text(result[1])


# ── ReferenceSegmenterModelTrainingDataGenerator JATS label fn ───────────────


def _make_md(line: LayoutLine, token_idx: int = 0) -> LayoutModelData:
    return LayoutModelData(
        data_line='token feats',
        layout_line=line,
        layout_token=line.tokens[token_idx],
    )


def _get_citation_label_list_for_sub_fields(
    text: str,
    sub_field_by_token_index: Dict[int, str]
) -> List[Optional[str]]:
    line = LayoutLine.for_text(text)
    citation_doc = LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=[line])])])
    annotated = JatsAnnotatedLayoutDocument(layout_document=citation_doc)
    for token_index, sub_field in sub_field_by_token_index.items():
        annotated.set_token_label(
            line.tokens[token_index], JatsFieldNames.REFERENCE,
            sub_field_name=sub_field, instance_id=1,
        )
    label_fn = CitationModelTrainingDataGenerator().get_jats_label_fn()
    assert label_fn is not None
    return [
        label_fn(annotated, {}, _make_md(line, token_index))
        for token_index in range(len(line.tokens))
    ]


@log_on_exception
class TestCitationJatsLabelFn:
    def test_should_label_identifier_sub_fields_with_the_identifier_label(self):
        assert _get_citation_label_list_for_sub_fields(
            'doi 10 unrelated',
            {1: JatsSubFieldNames.REFERENCE_DOI}
        ) == [None, IDENTIFIER_LABEL, None]

    def test_should_start_a_new_identifier_when_the_kind_changes(self):
        # 'doi' and 'pmid' tokens stand for the identifier values; without a B- prefix on the
        # second one the generator would write both into a single <idno>
        assert _get_citation_label_list_for_sub_fields(
            'doi pmid',
            {
                0: JatsSubFieldNames.REFERENCE_DOI,
                1: JatsSubFieldNames.REFERENCE_PMID
            }
        ) == [IDENTIFIER_LABEL, 'B-' + IDENTIFIER_LABEL]

    def test_should_not_start_a_new_identifier_within_one_kind(self):
        assert _get_citation_label_list_for_sub_fields(
            'doi doi doi',
            {
                0: JatsSubFieldNames.REFERENCE_DOI,
                1: JatsSubFieldNames.REFERENCE_DOI,
                2: JatsSubFieldNames.REFERENCE_DOI
            }
        ) == [IDENTIFIER_LABEL, IDENTIFIER_LABEL, IDENTIFIER_LABEL]

    def test_should_not_start_a_new_identifier_after_unlabelled_text(self):
        # the unlabelled token already closes the <idno> element
        assert _get_citation_label_list_for_sub_fields(
            'doi and pmid',
            {
                0: JatsSubFieldNames.REFERENCE_DOI,
                2: JatsSubFieldNames.REFERENCE_PMID
            }
        ) == [IDENTIFIER_LABEL, None, IDENTIFIER_LABEL]

    def test_should_not_start_a_new_identifier_across_references(self):
        # each reference is its own training document, so the first identifier of the next
        # one needs no prefix even though its kind differs from the previous reference's
        line = LayoutLine.for_text('pmid doi')
        citation_doc = LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=[line])])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=citation_doc)
        annotated.set_token_label(
            line.tokens[0], JatsFieldNames.REFERENCE,
            sub_field_name=JatsSubFieldNames.REFERENCE_PMID, instance_id=1,
        )
        annotated.set_token_label(
            line.tokens[1], JatsFieldNames.REFERENCE,
            sub_field_name=JatsSubFieldNames.REFERENCE_DOI, instance_id=2,
        )
        label_fn = CitationModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None
        assert [
            label_fn(annotated, {}, _make_md(line, 0)),
            label_fn(annotated, {}, _make_md(line, 1))
        ] == [IDENTIFIER_LABEL, IDENTIFIER_LABEL]

    def test_should_not_start_a_new_identifier_after_another_label(self):
        assert _get_citation_label_list_for_sub_fields(
            'doi 2020 pmid',
            {
                0: JatsSubFieldNames.REFERENCE_DOI,
                1: JatsSubFieldNames.REFERENCE_YEAR,
                2: JatsSubFieldNames.REFERENCE_PMID
            }
        ) == [IDENTIFIER_LABEL, '<date>', IDENTIFIER_LABEL]


@log_on_exception
class TestReferenceSegmenterJatsLabelFn:
    def test_labels_whole_line_as_reference_when_any_token_labeled(self):
        line = LayoutLine.for_text('Smith 2020 : Journal')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=[line])])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        # Label only the first token
        annotated.set_token_label(line.tokens[0], JatsFieldNames.REFERENCE, instance_id=1)

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        # First token (labeled) → '<reference>'
        assert label_fn(annotated, {}, _make_md(line, 0)) == '<reference>'
        # Unlabeled token on the same line → also '<reference>'
        assert label_fn(annotated, {}, _make_md(line, 2)) == '<reference>'

    def test_returns_none_for_line_with_no_reference_tokens(self):
        line = LayoutLine.for_text('Publisher Full Text')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=[line])])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        assert label_fn(annotated, {}, _make_md(line, 0)) is None

    def test_labels_line_as_label_when_reference_label_token_present(self):
        line = LayoutLine.for_text('[1] Smith 2020')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=[line])])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        annotated.set_token_label(
            line.tokens[0], JatsFieldNames.REFERENCE,
            sub_field_name=JatsSubFieldNames.REFERENCE_LABEL, instance_id=1,
        )
        # remaining tokens are plain REFERENCE
        for t in line.tokens[1:]:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=1)

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        # Only the REFERENCE_LABEL token itself gets '<label>'; the rest get '<reference>'
        assert label_fn(annotated, {}, _make_md(line, 0)) == '<label>'
        for idx in range(1, len(line.tokens)):
            assert label_fn(annotated, {}, _make_md(line, idx)) == '<reference>'

    def test_unlabeled_token_on_reference_line_gets_reference_not_label(self):
        # "10. Author Name..." where "10." is REFERENCE_LABEL and the rest are plain REFERENCE.
        # An additional unlabeled token on the same line should expand to '<reference>',
        # NOT '<label>'.
        line = LayoutLine.for_text('10. Author Name unlabeled')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[LayoutBlock(lines=[line])])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        annotated.set_token_label(
            line.tokens[0], JatsFieldNames.REFERENCE,
            sub_field_name=JatsSubFieldNames.REFERENCE_LABEL, instance_id=1,
        )
        annotated.set_token_label(line.tokens[1], JatsFieldNames.REFERENCE, instance_id=1)
        annotated.set_token_label(line.tokens[2], JatsFieldNames.REFERENCE, instance_id=1)
        # tokens[3] ("unlabeled") has no annotation

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        assert label_fn(annotated, {}, _make_md(line, 0)) == '<label>'      # "10."
        assert label_fn(annotated, {}, _make_md(line, 1)) == '<reference>'  # "Author"
        assert label_fn(annotated, {}, _make_md(line, 2)) == '<reference>'  # "Name"
        assert label_fn(annotated, {}, _make_md(line, 3)) == '<reference>'  # "unlabeled" expanded

    def test_plain_reference_transition_emits_b_prefix_for_bibl_boundary(self):
        # When a plain <reference> token starts a new instance (no <label>),
        # the label fn returns 'B-<reference>' so the TEI generator creates a new
        # <bibl> without losing the token.
        line1 = LayoutLine.for_text('Smith 2020')
        line2 = LayoutLine.for_text('Doe 2019')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock(lines=[line1, line2])
        ])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        for t in line1.tokens:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=1)
        for t in line2.tokens:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=2)

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        # All of line 1 → '<reference>' (first instance, no prior instance)
        for idx in range(len(line1.tokens)):
            assert label_fn(annotated, {}, _make_md(line1, idx)) == '<reference>'
        # First token of line 2 → 'B-<reference>' (B-prefix creates new bibl)
        assert label_fn(annotated, {}, _make_md(line2, 0)) == 'B-<reference>'
        # Remaining tokens of line 2 → '<reference>' (same instance, no transition)
        for idx in range(1, len(line2.tokens)):
            assert label_fn(annotated, {}, _make_md(line2, idx)) == '<reference>'

    def test_unlabeled_token_on_new_instance_line_emits_b_reference(self):
        # "9. Moraes R." — "9." is unlabeled but "Moraes R." is annotated as instance 2.
        # The label fn should emit 'B-<reference>' for "9." so it claims the whole line
        # for the new bibl, rather than appending "9." to the previous bibl.
        line1 = LayoutLine.for_text('https://some.url/paper')
        line2 = LayoutLine.for_text('9. Moraes R. Title')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock(lines=[line1, line2])
        ])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        for t in line1.tokens:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=1)
        # "9." (line2.tokens[0]) is intentionally NOT annotated (unlabeled in JATS)
        for t in line2.tokens[1:]:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=2)

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        for idx in range(len(line1.tokens)):
            label_fn(annotated, {}, _make_md(line1, idx))  # advance state
        # "9." is unlabeled but its line's first annotated token is instance 2 → B-prefix
        assert label_fn(annotated, {}, _make_md(line2, 0)) == 'B-<reference>'
        # Remaining tokens (annotated as instance 2) → same instance, no transition
        for idx in range(1, len(line2.tokens)):
            assert label_fn(annotated, {}, _make_md(line2, idx)) == '<reference>'

    def test_label_token_at_instance_transition_emits_b_label(self):
        # When a REFERENCE_LABEL token starts a new instance the label fn returns
        # 'B-<label>' so the reset mechanism (which fires only on B-prefix) creates a
        # new <bibl> with the label text correctly placed inside it.
        line1 = LayoutLine.for_text('Smith 2020')
        line2 = LayoutLine.for_text('[2] Doe 2019')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock(lines=[line1, line2])
        ])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        for t in line1.tokens:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=1)
        annotated.set_token_label(
            line2.tokens[0], JatsFieldNames.REFERENCE,
            sub_field_name=JatsSubFieldNames.REFERENCE_LABEL, instance_id=2,
        )
        for t in line2.tokens[1:]:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=2)

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        for idx in range(len(line1.tokens)):
            label_fn(annotated, {}, _make_md(line1, idx))  # advance state
        # REFERENCE_LABEL at instance transition → 'B-<label>' (triggers reset + new bibl)
        assert label_fn(annotated, {}, _make_md(line2, 0)) == 'B-<label>'
        # Remaining tokens of line 2 → '<reference>'
        for idx in range(1, len(line2.tokens)):
            assert label_fn(annotated, {}, _make_md(line2, idx)) == '<reference>'

    def test_unannotated_line_between_references_is_gap_filled(self):
        # "DOI:" may appear on a line by itself with no annotated reference tokens
        # (JATS only stores the DOI value, not the "DOI:" prefix).  Returning None
        # for that line would back the TEI writer up to listBibl level, creating a
        # spurious <bibl> for the URL that follows.  The fix gap-fills to '<reference>'
        # so "DOI:" stays inside the current bibl.
        line1 = LayoutLine.for_text('Smith J 2020 A title DOI')
        line_doi = LayoutLine.for_text('DOI:')          # no annotated tokens
        line_url = LayoutLine.for_text('10.1234/abcd')
        refs_doc = LayoutDocument(pages=[LayoutPage(blocks=[
            LayoutBlock(lines=[line1, line_doi, line_url])
        ])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=refs_doc)
        for t in line1.tokens:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=1)
        for t in line_url.tokens:
            annotated.set_token_label(t, JatsFieldNames.REFERENCE, instance_id=1)
        # line_doi tokens are intentionally unannotated

        label_fn = ReferenceSegmenterModelTrainingDataGenerator().get_jats_label_fn()
        assert label_fn is not None

        # Advance through line1 tokens
        for idx in range(len(line1.tokens)):
            label_fn(annotated, {}, _make_md(line1, idx))
        # "DOI:" line has no annotations → gap-fill returns '<reference>', not None
        doi_label = label_fn(annotated, {}, _make_md(line_doi, 0))
        assert doi_label == '<reference>', (
            f'Unannotated line after a reference should gap-fill to <reference>, got {doi_label!r}'
        )


# ── CitationModelTrainingDataGenerator JATS path ──────────────────────────────

@log_on_exception
class TestCitationModelJatsPath:
    def test_uses_jats_instance_split_and_skips_reference_segmenter(self):
        ref1_block = LayoutBlock.for_text('Smith 2020 A Title')
        ref2_block = LayoutBlock.for_text('Jones 2019 B Journal')
        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[ref1_block, ref2_block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=layout_document)
        _annotate_block_as_reference(ref1_block, annotated, instance_id=1)
        _annotate_block_as_reference(ref2_block, annotated, instance_id=2)
        context = _make_jats_context(annotated, [ref1_block, ref2_block])

        result = list(
            CitationModelTrainingDataGenerator().iter_model_layout_documents(
                layout_document, context
            )
        )

        assert len(result) == 2
        assert _doc_text(result[0]) == ref1_block.text
        assert _doc_text(result[1]) == ref2_block.text
        context.fulltext_models.reference_segmenter_model.assert_not_called()

    def test_returns_empty_when_no_references_annotated(self):
        block = LayoutBlock.for_text('Some text')
        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=layout_document)
        context = _make_jats_context(annotated, [block])

        result = list(
            CitationModelTrainingDataGenerator().iter_model_layout_documents(
                layout_document, context
            )
        )

        assert not result


# ── NameCitationModelTrainingDataGenerator JATS path ─────────────────────────

@log_on_exception
class TestNameCitationModelJatsPath:
    def test_returns_author_tokens_without_running_models(self):
        author_block = LayoutBlock.for_text('Smith J Jones M')
        other_block = LayoutBlock.for_text('Title of Paper Journal 2020')
        layout_document = LayoutDocument(
            pages=[LayoutPage(blocks=[author_block, other_block])]
        )
        annotated = JatsAnnotatedLayoutDocument(layout_document=layout_document)
        _annotate_block_as_reference(
            author_block, annotated, instance_id=1,
            sub_field=JatsSubFieldNames.REFERENCE_AUTHOR
        )
        _annotate_block_as_reference(other_block, annotated, instance_id=1)
        context = _make_jats_context(annotated, [author_block, other_block])

        result = list(
            NameCitationModelTrainingDataGenerator().iter_model_layout_documents(
                layout_document, context
            )
        )

        assert len(result) == 1
        result_text = _doc_text(result[0])
        assert 'Smith' in result_text
        assert 'Jones' in result_text
        # Non-author tokens should not be present
        assert 'Title' not in result_text
        context.fulltext_models.reference_segmenter_model.assert_not_called()
        context.fulltext_models.citation_model.assert_not_called()

    def test_returns_empty_when_no_author_annotations(self):
        block = LayoutBlock.for_text('Smith 2020')
        layout_document = LayoutDocument(pages=[LayoutPage(blocks=[block])])
        annotated = JatsAnnotatedLayoutDocument(layout_document=layout_document)
        _annotate_block_as_reference(block, annotated, instance_id=1)
        context = _make_jats_context(annotated, [block])

        result = list(
            NameCitationModelTrainingDataGenerator().iter_model_layout_documents(
                layout_document, context
            )
        )

        assert not result

    def test_collects_authors_from_multiple_references(self):
        author1_block = LayoutBlock.for_text('Smith J')
        author2_block = LayoutBlock.for_text('Jones M')
        layout_document = LayoutDocument(
            pages=[LayoutPage(blocks=[author1_block, author2_block])]
        )
        annotated = JatsAnnotatedLayoutDocument(layout_document=layout_document)
        _annotate_block_as_reference(
            author1_block, annotated, instance_id=1,
            sub_field=JatsSubFieldNames.REFERENCE_AUTHOR
        )
        _annotate_block_as_reference(
            author2_block, annotated, instance_id=2,
            sub_field=JatsSubFieldNames.REFERENCE_AUTHOR
        )
        context = _make_jats_context(annotated, [author1_block, author2_block])

        result = list(
            NameCitationModelTrainingDataGenerator().iter_model_layout_documents(
                layout_document, context
            )
        )

        assert len(result) == 1
        result_text = _doc_text(result[0])
        assert 'Smith' in result_text
        assert 'Jones' in result_text
