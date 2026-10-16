import argparse
import logging
import os
from typing import Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

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
from sciencebeam_parser.training.extra_feature_columns import (
    is_missing_extra_columns,
    iter_rows_with_extra_columns
)
from sciencebeam_parser.training.tei_raw_pairing import (
    PairingResult,
    get_paired_rows,
    iter_match_keys_for_labelled_line
)
from sciencebeam_parser.training.grobid_column_layout import (
    GrobidColumnLayout,
    get_grobid_column_layout_for_model_name,
    get_validated_training_data_feature_indices,
    select_feature_columns
)

from sciencebeam_parser.training.quality.assembly import (
    AssembledDocumentRecord,
    GeneratedDocumentRecord,
    get_assembly_summary_by_corpus,
    get_document_ids_without_generated_output,
    read_generated_document_records,
    write_assembly_records
)
from sciencebeam_parser.training.quality.gate import (
    TRAINING_QUALITY_CONFIG_FILE,
    TrainingQualityConfig,
    CorpusRefusedError,
    check_corpus_loss_or_fail,
    get_gate_summary_by_corpus,
    get_quality_verdict,
    load_training_quality_config
)
from sciencebeam_parser.training.quality.requirement import (
    get_requirement,
    load_requirement_by_corpus_and_model
)
from sciencebeam_parser.training.quality.review import ReviewStanding
from sciencebeam_parser.training.quality.counting import (
    count_entity_starts,
    count_label_starts_per_sequence,
    get_canonical_model_name,
    is_model_counted_by_label
)
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.app.parser import ScienceBeamParser


LOGGER = logging.getLogger(__name__)


_TEI_XML_PARSER = etree.XMLParser(recover=True)


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
        '--quality-record-path',
        type=str,
        required=False,
        help=(
            'File pattern of the per-document records written at generation, e.g.'
            ' "<data>/train/*/reference-segmenter/quality/*.quality.json". Their counts'
            ' are joined with the entity count only this step can take. Without it the'
            ' entity count is still recorded, with nothing to compare it against.'
        )
    )
    parser.add_argument(
        '--quality-output-path',
        type=str,
        required=False,
        help=(
            'Where to write the assembly quality record'
            ' (default: the delft output path with ".quality.jsonl" appended).'
        )
    )
    parser.add_argument(
        '--quality-filter',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Exclude documents failing the quality thresholds, and documents a'
            ' corpus requires a review for. Assembly refuses when a corpus loses'
            ' more than the configured share of its documents. On by default,'
            ' because assembling data the gate would have refused is the unusual'
            ' case; --no-quality-filter records the counts without acting on them.'
        )
    )
    parser.add_argument(
        '--quality-config-path',
        type=str,
        default=TRAINING_QUALITY_CONFIG_FILE,
        help='Thresholds the quality filter applies (default: the shipped config).'
    )
    parser.add_argument(
        '--require-config-path',
        type=str,
        required=False,
        help=(
            'The corpus\'s own training-source.yml, whose `require` block states what'
            ' each corpus and model demands before its data is used: accepted, not'
            ' rejected, or measured only. Without it nothing is demanded beyond the'
            ' thresholds, which is what `measured` means.'
        )
    )
    parser.add_argument(
        '--max-unreviewed-ratio',
        type=float,
        required=False,
        help=(
            'Override the configured share of a corpus that may go unreviewed before'
            ' assembly refuses. Counted apart from the share excluded by a measurement.'
        )
    )
    parser.add_argument(
        '--max-excluded-ratio',
        type=float,
        required=False,
        help=(
            'Override the configured share of a corpus beyond which assembly refuses.'
            ' Raising it is a decision to record, not a way around the refusal.'
        )
    )
    parser.add_argument(
        '--include-extra-columns',
        action='store_true',
        help=(
            'Emit the columns this project adds on top of GROBID\'s layout'
            ' (segmentation\'s whole_line_text, read by delft models as a text feature).'
            ' Without it the output matches GROBID\'s column layout and can be mixed'
            ' with GROBID\'s own corpus.'
        )
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
    prev_grobid_tag: Optional[str] = None,
    marks_other_span_start: bool = True
) -> str:
    """
    Convert labels from IOB2 to the ones used by GROBID (expected by the wapiti model).

    When prev_grobid_tag is supplied the function also replicates GROBID's convention of
    emitting I-<other> for the first "other" token immediately after an annotated span.
    Whether a corpus marks that start at all is a decision GROBID takes per model, and
    `marks_other_span_start` in the recorded column layout says which way.
    """
    if tag == 'O':
        # GROBID uses I-<other> for the first "other" token after any annotated span
        if (
            marks_other_span_start
            and prev_grobid_tag is not None
            and prev_grobid_tag not in _GROBID_OTHER_LABELS
        ):
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
    doc_tag_result: Sequence[Tuple[str, str]],
    marks_other_span_start: bool = True
) -> List[Tuple[str, str]]:
    translated: List[Tuple[str, str]] = []
    prev_grobid_tag: Optional[str] = None
    for token_text, tag in doc_tag_result:
        grobid_tag = translate_tags_IOB_to_grobid(
            tag, prev_grobid_tag, marks_other_span_start=marks_other_span_start
        )
        translated.append((token_text, grobid_tag))
        prev_grobid_tag = grobid_tag
    return translated


def translate_tag_result_tags_IOB_to_grobid(
    tag_result: Sequence[Sequence[Tuple[str, str]]],
    marks_other_span_start: bool = True
) -> List[List[Tuple[str, str]]]:
    return [
        _translate_doc_tags_IOB_to_grobid(
            doc_tag_result, marks_other_span_start=marks_other_span_start
        )
        for doc_tag_result in tag_result
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


class DelftDocumentResult(NamedTuple):
    data_lines: Sequence[str]
    labeled_layout_tokens_list: Sequence[Sequence[LabeledLayoutToken]]
    pairing_results: Sequence[PairingResult] = ()


def get_pairing_result_for_sequence(
    raw_row_tokens: Sequence[str],
    labeled_layout_tokens: Sequence[LabeledLayoutToken]
) -> PairingResult:
    return get_paired_rows(
        raw_row_tokens,
        [
            list(iter_match_keys_for_labelled_line(
                token_text=labeled_layout_token.layout_token.text,
                line_text=labeled_layout_token.layout_token.line_meta.text
            ))
            for labeled_layout_token in labeled_layout_tokens
        ]
    )


def get_delft_training_data_for_document(  # pylint: disable=too-many-locals
    tei_file: str,
    raw_file: Optional[str],
    training_tei_parser: TrainingTeiParser,
    data_generator: ModelDataGenerator,
    column_layout: GrobidColumnLayout,
    include_extra_columns: bool = False
) -> DelftDocumentResult:
    with auto_download_input_file(
        tei_file,
        auto_decompress=True
    ) as local_tei_file:
        # GROBID's own corpus carries xml:id values that are not NCNames, which a
        # validating parse refuses outright
        tei_root = etree.parse(local_tei_file, _TEI_XML_PARSER).getroot()
    labeled_layout_tokens_list = (
        training_tei_parser.parse_training_tei_to_labeled_layout_tokens_list(
            tei_root
        )
    )
    LOGGER.debug('labeled_layout_tokens_list: %r', labeled_layout_tokens_list)
    if raw_file:
        with auto_download_input_file(
            raw_file,
            auto_decompress=True
        ) as local_raw_file:
            with open(local_raw_file, 'r', encoding='utf-8') as raw_fp:
                texts, features = load_data_crf_lines(
                    raw_fp
                )
        if len(texts) != len(labeled_layout_tokens_list):
            raise ValueError(
                '%r has %d sequence(s) and %r has %d' % (
                    raw_file, len(texts), tei_file, len(labeled_layout_tokens_list)
                )
            )
        pairing_results = [
            get_pairing_result_for_sequence(doc_tokens, labeled_layout_tokens)
            for doc_tokens, labeled_layout_tokens
            in zip(texts, labeled_layout_tokens_list)
        ]
        # a row the TEI has no line for has no label and no line text, so it is
        # not emitted rather than given the line before it, which is what GROBID
        # does with it
        labeled_layout_tokens_list = [
            [labeled_layout_tokens[pair.labelled_index] for pair in result.paired_rows]
            for labeled_layout_tokens, result
            in zip(labeled_layout_tokens_list, pairing_results)
        ]
        selectable_features: Sequence[Sequence[Sequence[str]]] = [
            [document_features[pair.raw_index] for pair in result.paired_rows]
            for document_features, result in zip(features.tolist(), pairing_results)
        ]
        # the token column belongs to the row, not to the line it was matched
        # against: GROBID's raw file leads with the line's first whitespace field
        # where our tokenizer would have stopped at `LIMIX` in `LIMIX: genetic`
        row_tokens = [
            [doc_tokens[pair.raw_index] for pair in result.paired_rows]
            for doc_tokens, result in zip(texts, pairing_results)
        ]
        if selectable_features and selectable_features[0] and is_missing_extra_columns(
            column_layout, feature_column_count=len(selectable_features[0][0])
        ):
            LOGGER.info(
                '%r stops where GROBID\'s own features do; reading %r off the TEI',
                raw_file, list(column_layout.extra_columns)
            )
            extended_features: List[List[List[str]]] = []
            extended_tokens: List[List[str]] = []
            extended_labeled: List[List[LabeledLayoutToken]] = []
            for document_features, doc_row_tokens, labeled_layout_tokens in zip(
                selectable_features, row_tokens, labeled_layout_tokens_list
            ):
                kept = [
                    (feature_row, row_token, labeled_layout_token)
                    for (feature_row, labeled_layout_token), row_token in zip(
                        iter_rows_with_extra_columns(
                            column_layout, document_features, labeled_layout_tokens
                        ),
                        doc_row_tokens
                    )
                ]
                extended_features.append([item[0] for item in kept])
                extended_tokens.append([item[1] for item in kept])
                extended_labeled.append([item[2] for item in kept])
            selectable_features = extended_features
            row_tokens = extended_tokens
            labeled_layout_tokens_list = extended_labeled
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
        selectable_features = features.tolist()
        row_tokens = [
            [
                labeled_layout_token.layout_token.text
                for labeled_layout_token in labeled_layout_tokens
            ]
            for labeled_layout_tokens in labeled_layout_tokens_list
        ]
        pairing_results = [
            get_paired_rows(
                [labeled_layout_token.layout_token.text
                 for labeled_layout_token in labeled_layout_tokens],
                [[labeled_layout_token.layout_token.text]
                 for labeled_layout_token in labeled_layout_tokens]
            )
            for labeled_layout_tokens in labeled_layout_tokens_list
        ]
    # after the pairing, so that where a span starts is decided by the rows that
    # are written rather than by rows that were dropped
    translated_tag_result = translate_tag_result_tags_IOB_to_grobid(
        [
            [
                (row_token, labeled_layout_token.label)
                for row_token, labeled_layout_token
                in zip(doc_row_tokens, labeled_layout_tokens)
            ]
            for doc_row_tokens, labeled_layout_tokens
            in zip(row_tokens, labeled_layout_tokens_list)
        ],
        marks_other_span_start=column_layout.marks_other_span_start
    )
    LOGGER.debug('translated_tag_result: %r', translated_tag_result)
    LOGGER.debug('features: %r', selectable_features)
    # a document every row of which was dropped has no features to read a column
    # count from, and nothing to write
    first_feature_row = next(
        (
            feature_row
            for document_features in selectable_features
            for feature_row in document_features
        ),
        None
    )
    if first_feature_row is None:
        return DelftDocumentResult([], labeled_layout_tokens_list, pairing_results)
    feature_indices = get_validated_training_data_feature_indices(
        column_layout,
        feature_column_count=len(first_feature_row),
        data_generator_name=type(data_generator).__name__,
        data_generator_column_names=data_generator.feature_names,
        include_extra_columns=include_extra_columns
    )
    return DelftDocumentResult(
        list(iter_format_tag_result(
            tag_result=translated_tag_result,
            output_format=TagOutputFormats.DATA,
            texts=None,
            features=select_feature_columns(selectable_features, feature_indices)
        )),
        labeled_layout_tokens_list,
        pairing_results
    )


def get_document_id_for_tei_file(
    tei_file: str,
    tei_filename_suffix: Optional[str]
) -> str:
    """The document id generation recorded, which is the source name.

    A model with no declared suffix, or a file that does not carry it, falls back
    to everything before the first dot.
    """
    basename = os.path.basename(tei_file)
    if basename.endswith('.gz'):
        basename = basename[:-len('.gz')]
    if tei_filename_suffix and basename.endswith(tei_filename_suffix):
        return basename[:-len(tei_filename_suffix)]
    return basename.split('.', maxsplit=1)[0]


def get_tei_filename_suffix_for_model_name(
    model_name: str,
    sciencebeam_parser: ScienceBeamParser
) -> Optional[str]:
    model = sciencebeam_parser.fulltext_models.get_sequence_model_by_name(model_name)
    return model.get_tei_training_data_generator().get_default_tei_filename_suffix()


def get_assembled_document_record(
    document_id: str,
    model_name: str,
    result: DelftDocumentResult,
    generated_record_by_document_id: Mapping[str, GeneratedDocumentRecord],
    quality_config: Optional[TrainingQualityConfig] = None,
    requirement_by_pair: Optional[Mapping[Tuple[str, str], str]] = None,
) -> AssembledDocumentRecord:
    generated = generated_record_by_document_id.get(document_id)
    entity_start_count = count_entity_starts(
        model_name, result.labeled_layout_tokens_list
    )
    canonical_model_name = get_canonical_model_name(model_name)
    review = generated.review if generated else ReviewStanding.UNREVIEWED
    verdict = None
    if quality_config is not None:
        verdict = get_quality_verdict(
            document_id=document_id,
            thresholds=quality_config.get_thresholds_for_model(
                canonical_model_name
            ),
            requirement=get_requirement(
                requirement_by_pair,
                generated.corpus if generated else None,
                canonical_model_name
            ),
            review=review,
            jats_status=generated.jats_status if generated else None,
            jats_reference_count=generated.jats_reference_count if generated else None,
            written=generated.written if generated else None,
            entity_element_count=generated.entity_element_count if generated else None,
            entity_start_count=entity_start_count,
            sequence_count=len(result.labeled_layout_tokens_list),
        )
    return AssembledDocumentRecord(
        document_id=document_id,
        model_name=canonical_model_name,
        corpus=generated.corpus if generated else None,
        review=review,
        sequence_count=len(result.labeled_layout_tokens_list),
        entity_start_count=entity_start_count,
        label_start_counts=(
            count_label_starts_per_sequence(result.labeled_layout_tokens_list)
            if is_model_counted_by_label(model_name)
            else None
        ),
        generated=generated,
        verdict=verdict,
    )


def discard_refused_output(delft_output_path: str) -> None:
    """Leave nothing usable behind when assembly refuses.

    The data is written as documents are assembled, so by the time a corpus is
    known to be mostly excluded the file exists and would look like a corpus to
    whatever reads it next. A remote path cannot be removed here, and says so.
    """
    if not os.path.exists(delft_output_path):
        LOGGER.warning(
            'assembly refused; the output at %r could not be removed and'
            ' holds only the documents that passed',
            delft_output_path
        )
        return
    os.remove(delft_output_path)
    LOGGER.warning('assembly refused; removed the partial output at %r', delft_output_path)


def log_gate_summary(
    model_name: str,
    assembled_records: Sequence[AssembledDocumentRecord],
    quality_config: TrainingQualityConfig,
    max_excluded_ratio: Optional[float] = None,
    max_unreviewed_ratio: Optional[float] = None,
    delft_output_path: Optional[str] = None,
) -> None:
    summary_by_corpus = get_gate_summary_by_corpus([
        (record.verdict, record.corpus)
        for record in assembled_records
        if record.verdict is not None
    ])
    for record in assembled_records:
        if record.verdict is not None and record.verdict.is_excluded:
            LOGGER.warning('excluding %s', record.verdict)
    for corpus, summary in sorted(
        summary_by_corpus.items(), key=lambda item: item[0] or ''
    ):
        LOGGER.info(
            '%s / %s: %s', corpus or 'corpus not known',
            get_canonical_model_name(model_name), summary
        )
    try:
        check_corpus_loss_or_fail(
            summary_by_corpus,
            max_excluded_ratio
            if max_excluded_ratio is not None
            else quality_config.max_excluded_ratio,
            max_unreviewed_ratio
            if max_unreviewed_ratio is not None
            else quality_config.max_unreviewed_ratio
        )
    except CorpusRefusedError:
        if delft_output_path:
            discard_refused_output(delft_output_path)
        raise


def log_assembly_summary(
    model_name: str,
    assembled_records: Sequence[AssembledDocumentRecord],
    generated_record_by_document_id: Mapping[str, GeneratedDocumentRecord],
) -> None:
    canonical_model_name = get_canonical_model_name(model_name)
    for corpus, summary in sorted(
        get_assembly_summary_by_corpus(assembled_records).items(),
        key=lambda item: item[0] or ''
    ):
        LOGGER.info(
            '%s / %s: %s', corpus or 'corpus not known', canonical_model_name, summary
        )
    document_ids_without_output = get_document_ids_without_generated_output(
        generated_record_by_document_id
    )
    if document_ids_without_output:
        LOGGER.warning(
            '%d documents generation wrote no %s file for: %r',
            len(document_ids_without_output), canonical_model_name,
            document_ids_without_output
        )


def generate_delft_training_data(  # pylint: disable=too-many-locals
    model_name: str,
    tei_source_path: str,
    raw_source_path: str,
    delft_output_path: str,
    sciencebeam_parser: ScienceBeamParser,
    include_extra_columns: bool = False,
    quality_record_path: Optional[str] = None,
    quality_output_path: Optional[str] = None,
    quality_filter: bool = False,
    quality_config_path: str = TRAINING_QUALITY_CONFIG_FILE,
    require_config_path: Optional[str] = None,
    max_excluded_ratio: Optional[float] = None,
    max_unreviewed_ratio: Optional[float] = None
):
    training_tei_parser = get_training_tei_parser_for_model_name(
        model_name,
        sciencebeam_parser=sciencebeam_parser
    )
    data_generator = get_data_generator_for_model_name(
        model_name,
        sciencebeam_parser=sciencebeam_parser
    )
    column_layout = get_grobid_column_layout_for_model_name(model_name)
    LOGGER.info(
        'column layout for %r: %d columns, label_slot=%r, extra_columns=%r (included: %r)',
        model_name, len(column_layout.columns), column_layout.label_slot,
        list(column_layout.extra_columns), include_extra_columns
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
    generated_record_by_document_id = (
        read_generated_document_records(quality_record_path)
        if quality_record_path
        else {}
    )
    tei_filename_suffix = get_tei_filename_suffix_for_model_name(
        model_name, sciencebeam_parser=sciencebeam_parser
    )
    quality_config = (
        load_training_quality_config(quality_config_path) if quality_filter else None
    )
    requirement_by_pair = (
        load_requirement_by_corpus_and_model(require_config_path)
        if require_config_path
        else None
    )
    if quality_filter and not quality_record_path:
        LOGGER.warning(
            'no --quality-record-path given, so the thresholds against the JATS and'
            ' the generated elements cannot be applied and are reported as unchecked'
        )
    if quality_filter and not require_config_path:
        LOGGER.warning(
            'no --require-config-path given, so nothing is demanded of a document'
            ' beyond the thresholds and every verdict is recorded rather than read'
        )
    assembled_records: List[AssembledDocumentRecord] = []
    LOGGER.info('writing to : %r', delft_output_path)
    with auto_uploading_output_file(
        delft_output_path,
        mode='w',
        encoding='utf-8',
    ) as data_fp:
        written_document_count = 0
        for tei_file, raw_file in zip(tei_file_list, raw_file_list):
            result = get_delft_training_data_for_document(
                tei_file=tei_file,
                raw_file=raw_file,
                training_tei_parser=training_tei_parser,
                data_generator=data_generator,
                column_layout=column_layout,
                include_extra_columns=include_extra_columns
            )
            record = get_assembled_document_record(
                document_id=get_document_id_for_tei_file(tei_file, tei_filename_suffix),
                model_name=model_name,
                result=result,
                generated_record_by_document_id=generated_record_by_document_id,
                quality_config=quality_config,
                requirement_by_pair=requirement_by_pair,
            )
            assembled_records.append(record)
            if record.verdict is not None and record.verdict.is_excluded:
                continue
            # The blank line separates documents, so it follows what was written
            # rather than the position in the source list.
            if written_document_count:
                data_fp.write('\n\n')
            data_fp.writelines(result.data_lines)
            written_document_count += 1
    assembly_record_path = quality_output_path or delft_output_path + '.quality.jsonl'
    write_assembly_records(assembly_record_path, assembled_records)
    log_assembly_summary(
        model_name, assembled_records, generated_record_by_document_id
    )
    if quality_config is not None:
        log_gate_summary(
            model_name, assembled_records, quality_config,
            max_excluded_ratio=max_excluded_ratio,
            max_unreviewed_ratio=max_unreviewed_ratio,
            delft_output_path=delft_output_path
        )
    # Last, so that it is the line left on the screen -- and only where the gate
    # let the assembly stand, since a refusal removes what was written.
    LOGGER.info(
        'wrote %d documents to %r, with what was measured about each in %r',
        written_document_count, delft_output_path, assembly_record_path
    )


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
        sciencebeam_parser=sciencebeam_parser,
        include_extra_columns=args.include_extra_columns,
        quality_record_path=args.quality_record_path,
        quality_output_path=args.quality_output_path,
        quality_filter=args.quality_filter,
        quality_config_path=args.quality_config_path,
        require_config_path=args.require_config_path,
        max_excluded_ratio=args.max_excluded_ratio,
        max_unreviewed_ratio=args.max_unreviewed_ratio
    )


def main(argv: Optional[List[str]] = None):
    LOGGER.debug('argv: %r', argv)
    args = parse_args(argv)
    # The import chain installs a root handler and raises the root level, so this
    # CLI's own output -- the quality summary included -- is otherwise dropped.
    for name in [__name__, 'sciencebeam_parser']:
        logging.getLogger(name).setLevel('DEBUG' if args.debug else 'INFO')
    if args.debug:
        logging.getLogger('sciencebeam_trainer_delft').setLevel('DEBUG')
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    main()
