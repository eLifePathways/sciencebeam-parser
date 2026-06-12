import argparse
import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple

from lxml import etree

from sciencebeam_trainer_delft.utils.io import (
    auto_download_input_file
)
from sciencebeam_trainer_delft.sequence_labelling.reader import (
    load_data_crf_lines
)
from sciencebeam_trainer_delft.sequence_labelling.tag_formatter import (
    TagOutputFormats,
    iter_format_tag_result
)

from sciencebeam_parser.utils.io import (
    auto_uploading_output_file,
    glob
)

from sciencebeam_parser.document.layout_document import (
    LayoutBlock,
    LayoutDocument
)

from sciencebeam_parser.models.data import (
    DocumentFeaturesContext,
    LabeledLayoutToken,
    ModelDataGenerator
)
from sciencebeam_parser.models.training_data import TrainingTeiParser

from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.app.parser import ScienceBeamParser


LOGGER = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'ScienceBeam Parser: Generate DELFT Training Data'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--tei-source-path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--raw-source-path',
        type=str,
        required=False
    )
    parser.add_argument(
        '--delft-output-path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    return parser.parse_args(argv)


_GROBID_OTHER_LABELS = frozenset({'<other>', 'I-<other>'})


def translate_tags_IOB_to_grobid(
    tag: str,
    prev_grobid_tag: Optional[str] = None
) -> str:
    """
    Convert labels from IOB2 to the ones used by GROBID (expected by the wapiti model).

    When prev_grobid_tag is supplied the function also replicates GROBID's convention of
    emitting I-<other> for the first "other" token immediately after an annotated span.
    """
    if tag == 'O':
        # GROBID uses I-<other> for the first "other" token after any annotated span
        if prev_grobid_tag is not None and prev_grobid_tag not in _GROBID_OTHER_LABELS:
            return 'I-<other>'
        return '<other>'
    if tag.startswith('B-'):
        # begin
        return 'I-' + tag[2:]
    if tag.startswith('I-'):
        # inside
        return '' + tag[2:]
    return tag


def _translate_doc_tags_IOB_to_grobid(
    doc_tag_result: Sequence[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    translated: List[Tuple[str, str]] = []
    prev_grobid_tag: Optional[str] = None
    for token_text, tag in doc_tag_result:
        grobid_tag = translate_tags_IOB_to_grobid(tag, prev_grobid_tag)
        translated.append((token_text, grobid_tag))
        prev_grobid_tag = grobid_tag
    return translated


def translate_tag_result_tags_IOB_to_grobid(
    tag_result: Sequence[Sequence[Tuple[str, str]]]
) -> List[List[Tuple[str, str]]]:
    return [
        _translate_doc_tags_IOB_to_grobid(doc_tag_result)
        for doc_tag_result in tag_result
    ]


def get_tag_result_for_labeled_layout_tokens_list(
    labeled_layout_tokens_list: Sequence[Sequence[LabeledLayoutToken]]
) -> List[List[Tuple[str, str]]]:
    return [
        [
            (
                labeled_layout_token.layout_token.text,
                labeled_layout_token.label
            )
            for labeled_layout_token in labeled_layout_tokens
        ]
        for labeled_layout_tokens in labeled_layout_tokens_list
    ]


def get_raw_file_for_tei_file(
    tei_file: str,
    raw_source_path: str
) -> str:
    compression_suffix = ''
    if tei_file.endswith('.gz'):
        compression_suffix = '.gz'
        tei_file = tei_file[:-len(compression_suffix)]
    tei_suffix = '.tei.xml'
    assert tei_file.endswith(tei_suffix)
    return os.path.join(
        raw_source_path,
        os.path.basename(tei_file[:-len(tei_suffix)] + compression_suffix)
    )


def get_raw_file_list_for_tei_file_list(
    tei_file_list: Iterable[str],
    raw_source_path: str
) -> Sequence[str]:
    return [
        get_raw_file_for_tei_file(tei_file, raw_source_path=raw_source_path)
        for tei_file in tei_file_list
    ]


def get_training_tei_parser_for_model_name(
    model_name: str,
    sciencebeam_parser: ScienceBeamParser
) -> TrainingTeiParser:
    model = sciencebeam_parser.fulltext_models.get_sequence_model_by_name(model_name)
    try:
        training_tei_parser = model.get_training_tei_parser()
        assert training_tei_parser is not None
        return training_tei_parser
    except NotImplementedError as exc:
        training_tei_parser = None
        raise RuntimeError('unsupported model: %r' % model_name) from exc


def get_data_generator_for_model_name(
    model_name: str,
    sciencebeam_parser: ScienceBeamParser
) -> ModelDataGenerator:
    model = sciencebeam_parser.fulltext_models.get_sequence_model_by_name(model_name)
    return model.get_data_generator(
        document_features_context=DocumentFeaturesContext(
            app_features_context=sciencebeam_parser.app_features_context
        )
    )


def iter_generate_delft_training_data_lines_for_document(  # pylint: disable=too-many-locals
    tei_file: str,
    raw_file: Optional[str],
    training_tei_parser: TrainingTeiParser,
    data_generator: ModelDataGenerator
) -> Iterable[str]:
    with auto_download_input_file(
        tei_file,
        auto_decompress=True
    ) as local_tei_file:
        tei_root = etree.parse(local_tei_file).getroot()
    labeled_layout_tokens_list = (
        training_tei_parser.parse_training_tei_to_labeled_layout_tokens_list(
            tei_root
        )
    )
    LOGGER.debug('labeled_layout_tokens_list: %r', labeled_layout_tokens_list)
    translated_tag_result = translate_tag_result_tags_IOB_to_grobid(
        get_tag_result_for_labeled_layout_tokens_list(
            labeled_layout_tokens_list
        )
    )
    LOGGER.debug('translated_tag_result: %r', translated_tag_result)
    if raw_file:
        with auto_download_input_file(
            raw_file,
            auto_decompress=True
        ) as local_raw_file:
            with open(local_raw_file, 'r', encoding='utf-8') as raw_fp:
                texts, features = load_data_crf_lines(
                    raw_fp
                )
        assert len(texts) == len(translated_tag_result)
        for doc_tokens, doc_tag_result in zip(texts, translated_tag_result):
            assert len(doc_tokens) == len(doc_tag_result)
    else:
        layout_documents = [
            LayoutDocument.for_blocks([
                LayoutBlock.for_tokens([
                    labeled_layout_token.layout_token
                    for labeled_layout_token in labeled_layout_tokens
                ])
            ])
            for labeled_layout_tokens in labeled_layout_tokens_list
        ]
        LOGGER.debug('layout_documents: %r', layout_documents)
        data_line_iterable = list(data_generator.iter_data_lines_for_layout_documents(
            layout_documents
        ))
        _texts, features = load_data_crf_lines(data_line_iterable)
    LOGGER.debug('features: %r', features)
    yield from iter_format_tag_result(
        tag_result=translated_tag_result,
        output_format=TagOutputFormats.DATA,
        texts=None,
        features=features
    )


def generate_delft_training_data(
    model_name: str,
    tei_source_path: str,
    raw_source_path: str,
    delft_output_path: str,
    sciencebeam_parser: ScienceBeamParser
):
    training_tei_parser = get_training_tei_parser_for_model_name(
        model_name,
        sciencebeam_parser=sciencebeam_parser
    )
    data_generator = get_data_generator_for_model_name(
        model_name,
        sciencebeam_parser=sciencebeam_parser
    )
    LOGGER.debug('tei_source_path: %r', tei_source_path)
    tei_file_list = glob(tei_source_path)
    if not tei_file_list:
        raise RuntimeError('no files found for file pattern %r' % tei_source_path)
    LOGGER.info('tei_file_list: %r', tei_file_list)
    if raw_source_path:
        raw_file_list: Sequence[Optional[str]] = get_raw_file_list_for_tei_file_list(
            tei_file_list,
            raw_source_path=raw_source_path
        )
    else:
        raw_file_list = [None] * len(tei_file_list)
    LOGGER.info('raw_file_list: %r', raw_file_list)
    LOGGER.info('writing to : %r', delft_output_path)
    with auto_uploading_output_file(
        delft_output_path,
        mode='w',
        encoding='utf-8',
    ) as data_fp:
        for document_index, (tei_file, raw_file) in enumerate(zip(tei_file_list, raw_file_list)):
            if document_index > 0:
                data_fp.write('\n\n')
            data_fp.writelines(iter_generate_delft_training_data_lines_for_document(
                tei_file=tei_file,
                raw_file=raw_file,
                training_tei_parser=training_tei_parser,
                data_generator=data_generator
            ))


def run(args: argparse.Namespace):
    LOGGER.info('args: %r', args)
    config = AppConfig.load_yaml(
        DEFAULT_CONFIG_FILE
    )
    sciencebeam_parser = ScienceBeamParser.from_config(config)
    generate_delft_training_data(
        model_name=args.model_name,
        tei_source_path=args.tei_source_path,
        raw_source_path=args.raw_source_path,
        delft_output_path=args.delft_output_path,
        sciencebeam_parser=sciencebeam_parser
    )


def main(argv: Optional[List[str]] = None):
    LOGGER.debug('argv: %r', argv)
    args = parse_args(argv)
    if args.debug:
        for name in [__name__, 'sciencebeam_parser', 'sciencebeam_trainer_delft']:
            logging.getLogger(name).setLevel('DEBUG')
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
