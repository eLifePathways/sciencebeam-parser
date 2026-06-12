# pylint: disable=not-callable
import logging
import gzip
from pathlib import Path
from typing import Iterator, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from lxml import etree
from lxml.builder import E

from sciencebeam_trainer_delft.sequence_labelling.reader import (
    load_data_crf_lines,
    load_data_and_labels_crf_file,
    load_data_and_labels_crf_lines
)
from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument,
    LayoutLine,
    LayoutToken
)

from sciencebeam_parser.document.tei.common import TEI_E
from sciencebeam_parser.models.data import DocumentFeaturesContext, ModelDataGenerator

import sciencebeam_parser.training.cli.generate_delft_data as generate_delft_data_module
from sciencebeam_parser.training.cli.generate_delft_data import (
    main,
    translate_tag_result_tags_IOB_to_grobid,
    translate_tags_IOB_to_grobid
)
from sciencebeam_parser.utils.xml_writer import XmlTreeWriter
from tests.processors.fulltext.model_mocks import MockFullTextModels

from tests.test_utils import log_on_exception


LOGGER = logging.getLogger(__name__)

MINIMAL_EXAMPLE_PDF = 'test-data/minimal-example.pdf'
MINIMAL_EXAMPLE_PDF_PATTERN = 'test-data/minimal-example*.pdf'


SOURCE_FILENAME_1 = 'test1.pdf'


TOKEN_1 = 'token1'
TOKEN_2 = 'token2'
TOKEN_3 = 'token3'


@pytest.fixture(autouse=True)
def _patch_sciencebeam_parser_class_mock(
    sciencebeam_parser_class_mock: MagicMock
) -> Iterator[MagicMock]:
    with patch.object(
        generate_delft_data_module, 'ScienceBeamParser', sciencebeam_parser_class_mock
    ) as mock:
        yield mock


@pytest.fixture(name='document_features_context')
def _document_features_context(
    sciencebeam_parser_mock: MagicMock
) -> DocumentFeaturesContext:
    return DocumentFeaturesContext(
        sciencebeam_parser_mock.app_features_context
    )


def _test_generate_delft_with_multiple_tokens_tei_and_raw(
    tmp_path: Path,
    model_name: str,
    file_suffix: str,
    tei_root: etree.ElementBase,
    tokens: Sequence[str],
    expected_labels: Sequence[str]
):
    assert len(tokens) == len(expected_labels)
    tei_source_path = tmp_path / 'tei'
    raw_source_path = tmp_path / 'raw'
    output_path = tmp_path / 'output.data'
    tei_source_path.mkdir(parents=True, exist_ok=True)
    (tei_source_path / f'sample{file_suffix}.tei.xml').write_bytes(
        etree.tostring(tei_root)
    )
    raw_source_path.mkdir(parents=True, exist_ok=True)
    expected_features = [[
        [f'{i}.1', f'{i}.2', f'{i}.3']
        for i in range(len(tokens))
    ]]
    (raw_source_path / f'sample{file_suffix}').write_text('\n'.join([
        f'{token} {" ".join(expected_token_features)}'
        for token, expected_token_features in zip(tokens, expected_features[0])
    ]))
    main([
        f'--model-name={model_name}',
        f'--tei-source-path={tei_source_path}/*.tei.xml',
        f'--raw-source-path={raw_source_path}',
        f'--delft-output-path={output_path}'
    ])
    assert output_path.exists()
    texts, _labels, _features = load_data_and_labels_crf_file(
        str(output_path)
    )
    LOGGER.debug('texts: %r', texts)
    assert len(texts) == 1
    assert list(texts[0]) == tokens
    assert list(_labels[0]) == expected_labels
    assert _features.tolist() == expected_features


def _test_generate_delft_with_two_tokens_tei_and_raw(
    tmp_path: Path,
    model_name: str,
    file_suffix: str,
    tei_root: etree.ElementBase,
    expected_labels: Sequence[str]
):
    _test_generate_delft_with_multiple_tokens_tei_and_raw(
        tmp_path=tmp_path,
        model_name=model_name,
        file_suffix=file_suffix,
        tei_root=tei_root,
        expected_labels=expected_labels,
        tokens=[TOKEN_1, TOKEN_2]
    )


def _test_generate_delft_with_multiple_tokens_tei_only(  # pylint: disable=too-many-locals
    tmp_path: Path,
    model_name: str,
    file_suffix: str,
    tei_root: etree.ElementBase,
    tokens: Sequence[str],
    expected_labels: Sequence[str],
    data_generator: ModelDataGenerator,
    layout_document: Optional[LayoutDocument] = None
):
    tei_source_path = tmp_path / 'tei'
    output_path = tmp_path / 'output.data'
    tei_source_path.mkdir(parents=True, exist_ok=True)
    (tei_source_path / f'sample{file_suffix}.tei.xml').write_bytes(
        etree.tostring(tei_root)
    )
    main([
        f'--model-name={model_name}',
        f'--tei-source-path={tei_source_path}/*.tei.xml',
        f'--delft-output-path={output_path}'
    ])
    assert output_path.exists()
    if layout_document is None:
        layout_document = LayoutDocument.for_blocks([
            LayoutBlock.for_text(' '.join(tokens))
        ])
    expected_data_lines = list(data_generator.iter_data_lines_for_layout_document(
        layout_document
    ))
    _expected_texts, expected_features = load_data_crf_lines(expected_data_lines)
    LOGGER.debug('expected_features: %r', expected_features)
    texts, labels, features = load_data_and_labels_crf_file(
        str(output_path)
    )
    LOGGER.debug('texts: %r', texts)
    LOGGER.debug('labels: %r', labels)
    LOGGER.debug('features: %r', features)
    LOGGER.debug('training tei: %r', etree.tostring(tei_root))
    assert len(texts) == 1
    assert list(texts[0]) == tokens
    assert list(labels[0]) == expected_labels
    assert features.tolist() == expected_features.tolist()


def _test_generate_delft_with_two_tokens_tei_only(
    tmp_path: Path,
    model_name: str,
    file_suffix: str,
    tei_root: etree.ElementBase,
    expected_labels: Sequence[str],
    data_generator: ModelDataGenerator
):
    _test_generate_delft_with_multiple_tokens_tei_only(
        tmp_path=tmp_path,
        model_name=model_name,
        file_suffix=file_suffix,
        tei_root=tei_root,
        tokens=[TOKEN_1, TOKEN_2],
        expected_labels=expected_labels,
        data_generator=data_generator
    )


class TestTranslateTagsIOBToGrobid:
    def test_outside_tag_becomes_other(self):
        assert translate_tags_IOB_to_grobid('O') == '<other>'

    def test_begin_tag_becomes_i_prefixed(self):
        assert translate_tags_IOB_to_grobid('B-<title>') == 'I-<title>'

    def test_inside_tag_drops_prefix(self):
        assert translate_tags_IOB_to_grobid('I-<title>') == '<title>'

    def test_outside_after_annotated_becomes_i_other(self):
        # GROBID uses I-<other> for the first "other" token immediately after any
        # annotated span. prev_grobid_tag carries the previous translated label.
        assert translate_tags_IOB_to_grobid('O', prev_grobid_tag='I-<title>') == 'I-<other>'
        assert translate_tags_IOB_to_grobid('O', prev_grobid_tag='<title>') == 'I-<other>'

    def test_outside_after_other_stays_other(self):
        assert translate_tags_IOB_to_grobid('O', prev_grobid_tag='<other>') == '<other>'
        assert translate_tags_IOB_to_grobid('O', prev_grobid_tag='I-<other>') == '<other>'

    def test_outside_at_document_start_stays_other(self):
        assert translate_tags_IOB_to_grobid('O', prev_grobid_tag=None) == '<other>'

    def test_translate_doc_applies_i_other_convention(self):
        # Full document translation: O after B-<title> → I-<other>; subsequent O → <other>
        tag_result = [[
            ('token1', 'B-<title>'),
            ('token2', 'O'),
            ('token3', 'O'),
        ]]
        translated = translate_tag_result_tags_IOB_to_grobid(tag_result)
        assert translated == [[
            ('token1', 'I-<title>'),
            ('token2', 'I-<other>'),
            ('token3', '<other>'),
        ]]


@log_on_exception
class TestMain:
    def test_should_be_able_to_generate_segmentation_training_data(
        self,
        tmp_path: Path
    ):
        _test_generate_delft_with_two_tokens_tei_and_raw(
            tmp_path=tmp_path,
            model_name='segmentation',
            file_suffix='.segmentation',
            tei_root=E('tei', E('text', *[
                E('front', TOKEN_1, E('lb')),
                '\n',
                E('body', TOKEN_2, E('lb')),
                '\n'
            ])),
            expected_labels=['B-<header>', 'B-<body>']
        )

    def test_should_be_able_to_generate_header_training_data(
        self,
        tmp_path: Path
    ):
        _test_generate_delft_with_two_tokens_tei_and_raw(
            tmp_path=tmp_path,
            model_name='header',
            file_suffix='.header',
            tei_root=E('tei', E('text', E('front', *[
                E('docTitle', E('titlePart', TOKEN_1, E('lb'))),
                '\n',
                E('byline', E('docAuthor', TOKEN_2, E('lb'))),
                '\n'
            ]))),
            expected_labels=['B-<title>', 'B-<author>']
        )

    def test_should_be_able_to_generate_fulltext_training_data(
        self,
        tmp_path: Path
    ):
        _test_generate_delft_with_two_tokens_tei_and_raw(
            tmp_path=tmp_path,
            model_name='fulltext',
            file_suffix='.fulltext',
            tei_root=E('tei', E('text', *[
                E('p', TOKEN_1, ' ', TOKEN_2, E('lb')),
                '\n'
            ])),
            expected_labels=['B-<paragraph>', 'I-<paragraph>']
        )

    def test_should_continue_fulltext_paragraph(
        self,
        tmp_path: Path
    ):
        _test_generate_delft_with_multiple_tokens_tei_and_raw(
            tmp_path=tmp_path,
            model_name='fulltext',
            file_suffix='.fulltext',
            tei_root=E('tei', E('text', *[
                E(
                    'p',
                    TOKEN_1,
                    ' ',
                    E('ref', {'type': 'biblio'}, TOKEN_2),
                    ' ',
                    TOKEN_3,
                    E('lb')
                ),
                '\n'
            ])),
            tokens=[TOKEN_1, TOKEN_2, TOKEN_3],
            expected_labels=['B-<paragraph>', 'B-<citation_marker>', 'I-<paragraph>']
        )

    def test_should_be_able_to_generate_figure_training_data(
        self,
        tmp_path: Path
    ):
        _test_generate_delft_with_two_tokens_tei_and_raw(
            tmp_path=tmp_path,
            model_name='figure',
            file_suffix='.figure',
            tei_root=E('tei', E('text', *[
                E('figure', E('figDesc', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb'))),
                '\n'
            ])),
            expected_labels=['B-<figDesc>', 'I-<figDesc>']
        )

    def test_should_be_able_to_generate_table_training_data(
        self,
        tmp_path: Path
    ):
        _test_generate_delft_with_two_tokens_tei_and_raw(
            tmp_path=tmp_path,
            model_name='table',
            file_suffix='.table',
            tei_root=E('tei', E('text', *[
                E(
                    'figure',
                    {'type': 'table'},
                    E('figDesc', TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb'))
                ),
                '\n'
            ])),
            expected_labels=['B-<figDesc>', 'I-<figDesc>']
        )

    def test_should_be_able_to_generate_reference_segmenter_training_data(
        self,
        tmp_path: Path
    ):
        _test_generate_delft_with_two_tokens_tei_and_raw(
            tmp_path=tmp_path,
            model_name='reference_segmenter',
            file_suffix='.references.referenceSegmenter',
            tei_root=E('tei', E('text', E('listBibl', *[
                E(
                    'bibl',
                    TOKEN_1, E('lb'), '\n', TOKEN_2, E('lb')
                ),
                '\n'
            ]))),
            expected_labels=['B-<reference>', 'I-<reference>']
        )

    def test_should_be_able_to_generate_affiliation_address_training_data(
        self,
        tmp_path: Path,
        fulltext_models_mock: MockFullTextModels,
        document_features_context: DocumentFeaturesContext
    ):
        data_generator = fulltext_models_mock.affiliation_address_model.get_data_generator(
            document_features_context=document_features_context
        )
        xml_writer = XmlTreeWriter(TEI_E('TEI'), element_maker=TEI_E)
        xml_writer.require_path([
            'teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic', 'author',
            'affiliation', 'orgName[@type="institution"]'
        ])
        xml_writer.append_text(' '.join([TOKEN_1, TOKEN_2]))
        _test_generate_delft_with_two_tokens_tei_only(
            tmp_path=tmp_path,
            model_name='affiliation_address',
            file_suffix='.affiliation',
            tei_root=xml_writer.root,
            expected_labels=['B-<institution>', 'I-<institution>'],
            data_generator=data_generator
        )

    def test_should_be_able_to_generate_name_header_training_data(
        self,
        tmp_path: Path,
        fulltext_models_mock: MockFullTextModels,
        document_features_context: DocumentFeaturesContext
    ):
        data_generator = fulltext_models_mock.name_header_model.get_data_generator(
            document_features_context=document_features_context
        )
        xml_writer = XmlTreeWriter(TEI_E('TEI'), element_maker=TEI_E)
        xml_writer.require_path([
            'teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic', 'author',
            'persName', 'forename'
        ])
        xml_writer.append_text(' '.join([TOKEN_1, TOKEN_2]))
        _test_generate_delft_with_two_tokens_tei_only(
            tmp_path=tmp_path,
            model_name='name_header',
            file_suffix='.header.authors',
            tei_root=xml_writer.root,
            expected_labels=['B-<forename>', 'I-<forename>'],
            data_generator=data_generator
        )

    def test_should_be_able_to_generate_name_citation_training_data(
        self,
        tmp_path: Path,
        fulltext_models_mock: MockFullTextModels,
        document_features_context: DocumentFeaturesContext
    ):
        data_generator = fulltext_models_mock.name_citation_model.get_data_generator(
            document_features_context=document_features_context
        )
        xml_writer = XmlTreeWriter(TEI_E('TEI'), element_maker=TEI_E)
        xml_writer.require_path([
            'teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic', 'author',
            'persName', 'forename'
        ])
        xml_writer.append_text(' '.join([TOKEN_1, TOKEN_2]))
        _test_generate_delft_with_two_tokens_tei_only(
            tmp_path=tmp_path,
            model_name='name_citation',
            file_suffix='.citations.authors',
            tei_root=xml_writer.root,
            expected_labels=['B-<forename>', 'I-<forename>'],
            data_generator=data_generator
        )

    def test_should_generate_lineend_on_lb_for_citation_model(
        self,
        tmp_path: Path,
        fulltext_models_mock: MockFullTextModels,
        document_features_context: DocumentFeaturesContext
    ):
        data_generator = fulltext_models_mock.name_citation_model.get_data_generator(
            document_features_context=document_features_context
        )
        xml_writer = XmlTreeWriter(TEI_E('TEI'), element_maker=TEI_E)
        xml_writer.require_path([
            'teiHeader', 'fileDesc', 'sourceDesc', 'biblStruct', 'analytic', 'author',
            'persName', 'forename'
        ])
        xml_writer.append_all(
            TOKEN_1,
            TEI_E('lb'),
            '\n',
            TOKEN_2
        )
        layout_document = LayoutDocument.for_blocks([
            LayoutBlock(lines=[
                LayoutLine(tokens=[LayoutToken(TOKEN_1)]),
                LayoutLine(tokens=[LayoutToken(TOKEN_2)])
            ])
        ])
        _test_generate_delft_with_multiple_tokens_tei_only(
            tmp_path=tmp_path,
            model_name='name_citation',
            file_suffix='.citations.authors',
            tei_root=xml_writer.root,
            tokens=[TOKEN_1, TOKEN_2],
            layout_document=layout_document,
            expected_labels=['B-<forename>', 'I-<forename>'],
            data_generator=data_generator
        )

    def test_should_be_able_to_generate_citation_training_data(
        self,
        tmp_path: Path,
        fulltext_models_mock: MockFullTextModels,
        document_features_context: DocumentFeaturesContext
    ):
        data_generator = fulltext_models_mock.citation_model.get_data_generator(
            document_features_context=document_features_context
        )
        xml_writer = XmlTreeWriter(TEI_E('tei'), element_maker=TEI_E)
        xml_writer.require_path(['text', 'back', 'listBibl', 'bibl', 'author'])
        xml_writer.append_text(' '.join([TOKEN_1, TOKEN_2]))
        _test_generate_delft_with_two_tokens_tei_only(
            tmp_path=tmp_path,
            model_name='citation',
            file_suffix='.references',
            tei_root=xml_writer.root,
            expected_labels=['B-<author>', 'I-<author>'],
            data_generator=data_generator
        )

    def test_should_be_able_to_load_and_generate_gzipped_training_data(
        self,
        tmp_path: Path
    ):
        model_name = 'segmentation'
        tokens = [TOKEN_1, TOKEN_2]
        file_suffix = '.segmentation'
        tei_root = E('tei', E('text', *[
            E('front', TOKEN_1, E('lb')),
            '\n',
            E('body', TOKEN_2, E('lb')),
            '\n'
        ]))
        tei_source_path = tmp_path / 'tei'
        raw_source_path = tmp_path / 'raw'
        output_path = tmp_path / 'output.data.gz'
        tei_source_path.mkdir(parents=True, exist_ok=True)
        (tei_source_path / f'sample{file_suffix}.tei.xml.gz').write_bytes(
            gzip.compress(etree.tostring(tei_root))
        )
        raw_source_path.mkdir(parents=True, exist_ok=True)
        expected_features = [[
            [f'{i}.1', f'{i}.2', f'{i}.3']
            for i in range(len(tokens))
        ]]
        (raw_source_path / f'sample{file_suffix}.gz').write_text('\n'.join([
            f'{token} {" ".join(expected_token_features)}'
            for token, expected_token_features in zip(tokens, expected_features[0])
        ]))
        main([
            f'--model-name={model_name}',
            f'--tei-source-path={tei_source_path}/*.tei.xml.gz',
            f'--raw-source-path={raw_source_path}',
            f'--delft-output-path={output_path}'
        ])
        assert output_path.exists()
        texts, _labels, _features = load_data_and_labels_crf_lines(
            gzip.decompress(output_path.read_bytes()).decode('utf-8').splitlines()
        )
        LOGGER.debug('texts: %r', texts)
        assert len(texts) == 1
        assert list(texts[0]) == tokens
