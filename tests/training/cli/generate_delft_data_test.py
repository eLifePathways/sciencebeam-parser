# pylint: disable=not-callable
import json
import logging
import gzip
from pathlib import Path
from typing import Iterator, List, Optional, Sequence
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
    get_document_id_for_tei_file,
    main,
    translate_tag_result_tags_IOB_to_grobid,
    translate_tags_IOB_to_grobid
)
from sciencebeam_parser.training.grobid_column_layout import (
    get_grobid_column_layout_for_model_name
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


def _get_raw_feature_rows(model_name: str, token_count: int) -> List[List[str]]:
    """Distinguishable placeholder values, one row per token, at the width the
    data generator would have produced."""
    layout = get_grobid_column_layout_for_model_name(model_name)
    column_count = len(layout.get_data_generator_column_names()) - 1
    return [
        [f'{token_index}.{column_index}' for column_index in range(column_count)]
        for token_index in range(token_count)
    ]


def _get_expected_training_feature_rows(
    model_name: str,
    raw_feature_rows: Sequence[Sequence[str]],
    include_extra_columns: bool = False
) -> List[List[str]]:
    layout = get_grobid_column_layout_for_model_name(model_name)
    column_count = len(
        layout.get_training_data_column_names(include_extra_columns)
    ) - 1
    assert layout.get_training_data_feature_indices(include_extra_columns) == list(
        range(column_count)
    ), 'expected the emitted columns to be a prefix of the generated ones'
    return [list(row[:column_count]) for row in raw_feature_rows]


def _test_generate_delft_with_multiple_tokens_tei_and_raw(  # pylint: disable=too-many-locals
    tmp_path: Path,
    model_name: str,
    file_suffix: str,
    tei_root: etree.ElementBase,
    tokens: Sequence[str],
    expected_labels: Sequence[str],
    include_extra_columns: bool = False
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
    raw_feature_rows = _get_raw_feature_rows(model_name, len(tokens))
    (raw_source_path / f'sample{file_suffix}').write_text('\n'.join([
        f'{token} {" ".join(raw_token_features)}'
        for token, raw_token_features in zip(tokens, raw_feature_rows)
    ]))
    main([
        f'--model-name={model_name}',
        f'--tei-source-path={tei_source_path}/*.tei.xml',
        f'--raw-source-path={raw_source_path}',
        f'--delft-output-path={output_path}'
    ] + (['--include-extra-columns'] if include_extra_columns else []))
    assert output_path.exists()
    expected_features = [_get_expected_training_feature_rows(
        model_name, raw_feature_rows, include_extra_columns
    )]
    assert [
        len(line.split())
        for line in output_path.read_text().splitlines()
        if line.strip()
    ] == [1 + len(expected_features[0][0]) + 1] * len(tokens)
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
    expected_labels: Sequence[str],
    include_extra_columns: bool = False
):
    _test_generate_delft_with_multiple_tokens_tei_and_raw(
        tmp_path=tmp_path,
        model_name=model_name,
        file_suffix=file_suffix,
        tei_root=tei_root,
        expected_labels=expected_labels,
        tokens=[TOKEN_1, TOKEN_2],
        include_extra_columns=include_extra_columns
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
    _expected_texts, generated_features = load_data_crf_lines(expected_data_lines)
    expected_features = [_get_expected_training_feature_rows(
        model_name, generated_features.tolist()[0]
    )]
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
    assert features.tolist() == expected_features


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

    def test_should_include_the_segmentation_extra_column_when_asked_for(
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
            expected_labels=['B-<header>', 'B-<body>'],
            include_extra_columns=True
        )

    def test_should_reject_a_raw_file_of_another_width(
        self,
        tmp_path: Path
    ):
        tei_source_path = tmp_path / 'tei'
        raw_source_path = tmp_path / 'raw'
        tei_source_path.mkdir(parents=True, exist_ok=True)
        (tei_source_path / 'sample.segmentation.tei.xml').write_bytes(etree.tostring(
            E('tei', E('text', *[
                E('front', TOKEN_1, E('lb')),
                '\n',
                E('body', TOKEN_2, E('lb')),
                '\n'
            ]))
        ))
        raw_source_path.mkdir(parents=True, exist_ok=True)
        (raw_source_path / 'sample.segmentation').write_text('\n'.join([
            f'{TOKEN_1} feature1 feature2',
            f'{TOKEN_2} feature1 feature2'
        ]))
        with pytest.raises(ValueError):
            main([
                '--model-name=segmentation',
                f'--tei-source-path={tei_source_path}/*.tei.xml',
                f'--raw-source-path={raw_source_path}',
                f'--delft-output-path={tmp_path}/output.data'
            ])

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
        raw_feature_rows = _get_raw_feature_rows(model_name, len(tokens))
        (raw_source_path / f'sample{file_suffix}.gz').write_text('\n'.join([
            f'{token} {" ".join(raw_token_features)}'
            for token, raw_token_features in zip(tokens, raw_feature_rows)
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


@log_on_exception
class TestQualityRecord:
    def _write_reference_segmenter_tei(
        self, tei_source_path: Path, document_id: str, bibl_count: int
    ) -> None:
        tei_source_path.mkdir(parents=True, exist_ok=True)
        (
            tei_source_path / f'{document_id}.references.referenceSegmenter.tei.xml'
        ).write_bytes(etree.tostring(E('tei', E('text', E('listBibl', *[
            child
            for index in range(bibl_count)
            for child in (E('bibl', f'reference{index}', E('lb')), '\n')
        ])))))

    def test_should_record_the_entity_count_the_parse_returns(self, tmp_path: Path):
        tei_source_path = tmp_path / 'tei'
        self._write_reference_segmenter_tei(tei_source_path, 'document1', bibl_count=3)
        output_path = tmp_path / 'output.data'
        main([
            '--model-name=reference_segmenter',
            f'--tei-source-path={tei_source_path}/*.tei.xml',
            f'--delft-output-path={output_path}'
        ])
        quality_record_path = Path(str(output_path) + '.quality.jsonl')
        rows = [
            json.loads(line)
            for line in quality_record_path.read_text(encoding='utf-8').splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]['document_id'] == 'document1'
        assert rows[0]['model'] == 'reference-segmenter'
        assert rows[0]['entity_start_count'] == 3

    def test_should_join_the_record_generation_wrote(self, tmp_path: Path):
        tei_source_path = tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'corpus' / 'tei'
        self._write_reference_segmenter_tei(tei_source_path, 'document1', bibl_count=3)
        generated_record_path = (
            tmp_path / 'train' / 'ore' / 'reference-segmenter' / 'quality.jsonl'
        )
        generated_record_path.write_text(json.dumps({
            'document_id': 'document1',
            'model': 'reference-segmenter',
            'status': 'ok',
            'jats': {'status': 'ok', 'reference_count': 4},
            'written': True,
            'entity_element_count': 3,
        }) + '\n', encoding='utf-8')
        output_path = tmp_path / 'output.data'
        main([
            '--model-name=reference_segmenter',
            f'--tei-source-path={tei_source_path}/*.tei.xml',
            f'--quality-record-path={generated_record_path}',
            f'--delft-output-path={output_path}'
        ])
        row = json.loads(
            Path(str(output_path) + '.quality.jsonl').read_text(encoding='utf-8')
        )
        assert row['corpus'] == 'ore'
        assert row['entity_start_count'] == 3
        assert row['generated']['entity_element_count'] == 3
        assert row['generated']['jats']['reference_count'] == 4

    def test_should_write_the_record_where_asked(self, tmp_path: Path):
        tei_source_path = tmp_path / 'tei'
        self._write_reference_segmenter_tei(tei_source_path, 'document1', bibl_count=1)
        quality_output_path = tmp_path / 'elsewhere' / 'quality.jsonl'
        main([
            '--model-name=reference_segmenter',
            f'--tei-source-path={tei_source_path}/*.tei.xml',
            f'--delft-output-path={tmp_path}/output.data',
            f'--quality-output-path={quality_output_path}'
        ])
        assert quality_output_path.exists()

    def test_should_record_the_labels_a_citation_sequence_marks(self, tmp_path: Path):
        tei_source_path = tmp_path / 'tei'
        tei_source_path.mkdir(parents=True)
        (tei_source_path / 'document1.references.tei.xml').write_bytes(etree.tostring(
            TEI_E('TEI', TEI_E('text', TEI_E('back', TEI_E('listBibl', *[
                TEI_E('bibl', TEI_E('title', TOKEN_1, {'level': 'a'}), ' ', TOKEN_2),
                '\n',
            ]))))
        ))
        output_path = tmp_path / 'output.data'
        main([
            '--model-name=citation',
            f'--tei-source-path={tei_source_path}/*.tei.xml',
            f'--delft-output-path={output_path}'
        ])
        row = json.loads(
            Path(str(output_path) + '.quality.jsonl').read_text(encoding='utf-8')
        )
        assert row['sequence_count'] == 1
        assert row['label_start_counts'] == {'<title>': 1}
        # every bibl is its own sequence, so there is no entity count to take
        assert 'entity_start_count' not in row


class TestGetDocumentIdForTeiFile:
    def test_should_strip_the_model_suffix(self):
        assert get_document_id_for_tei_file(
            '/tei/PPR459453.references.referenceSegmenter.tei.xml',
            '.references.referenceSegmenter.tei.xml'
        ) == 'PPR459453'

    def test_should_strip_a_gzip_suffix_first(self):
        assert get_document_id_for_tei_file(
            '/tei/PPR459453.references.tei.xml.gz', '.references.tei.xml'
        ) == 'PPR459453'

    def test_should_fall_back_for_a_model_with_no_declared_suffix(self):
        assert get_document_id_for_tei_file('/tei/PPR459453.tei.xml', None) == 'PPR459453'

    def test_should_fall_back_when_the_file_does_not_carry_the_suffix(self):
        assert get_document_id_for_tei_file(
            '/tei/PPR459453.something-else.tei.xml', '.references.tei.xml'
        ) == 'PPR459453'
