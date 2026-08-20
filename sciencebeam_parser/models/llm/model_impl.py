import logging
from typing import List, Optional, Tuple

from sciencebeam_parser.models.llm.client import (
    LlmClient,
    LlmCompletionClient,
    LlmRequestError,
    get_response_content
)
from sciencebeam_parser.models.llm.config import LlmConfigError, LlmEngineConfig
from sciencebeam_parser.models.llm.decode import (
    EVIDENCE_RESPONSE_SCHEMA,
    LINES_RESPONSE_SCHEMA,
    LlmInputTooLargeError,
    LlmResponseError,
    decode_evidence_response,
    decode_line_starts_response,
    get_line_numbers,
    render_numbered_lines
)
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.prompt import get_prompt
from sciencebeam_parser.models.llm.tasks import get_citation_labels
from sciencebeam_parser.models.llm.telemetry import llm_span, set_response_attributes
from sciencebeam_parser.models.llm.values import (
    decode_batched_values_response,
    get_batched_values_response_schema,
    render_numbered_references
)
from sciencebeam_parser.models.model_impl import ModelImpl


LOGGER = logging.getLogger(__name__)

LINE_STATUS_FEATURE_NAME = 'line_status'

LINES_SHAPE = 'lines'
EVIDENCE_SHAPE = 'evidence'
VALUES_SHAPE = 'values'

LINE_BASED_SHAPES = (LINES_SHAPE, EVIDENCE_SHAPE)

SUPPORTED_RESPONSE_SHAPES = (LINES_SHAPE, EVIDENCE_SHAPE, VALUES_SHAPE)


def _get_content_or_none(response_json) -> Optional[str]:
    """The raw content even when it will fail to decode, so the trace shows it."""
    choices = response_json.get('choices') or []
    if not choices:
        return None
    return choices[0].get('message', {}).get('content')


class LlmModelImpl(ModelImpl):
    def __init__(
        self,
        config: LlmEngineConfig,
        client: Optional[LlmCompletionClient] = None
    ):
        if config.response_shape not in SUPPORTED_RESPONSE_SHAPES:
            raise LlmConfigError(
                f'unsupported response_shape {config.response_shape!r};'
                f' supported: {list(SUPPORTED_RESPONSE_SHAPES)}'
            )
        self.config = config
        self.client = client if client is not None else LlmClient(config)
        self.labels = (
            get_citation_labels() if config.response_shape == VALUES_SHAPE else []
        )
        self.line_status_index = (
            get_feature_column_index(config.task, LINE_STATUS_FEATURE_NAME)
            if config.response_shape in LINE_BASED_SHAPES else -1
        )

    def __repr__(self) -> str:
        return '%s(task=%r, model=%r, shape=%r, prompt=%r)' % (
            type(self).__name__, self.config.task, self.config.model,
            self.config.response_shape, self.config.prompt_version
        )

    def preload(self):
        self.client.validate_configuration()

    def predict_labels(
        self,
        texts: List[List[str]],
        features: List[List[List[str]]],
        output_format: Optional[str] = None
    ) -> List[List[Tuple[str, str]]]:
        if output_format:
            raise NotImplementedError(
                f'{type(self).__name__} does not support output_format={output_format!r}'
            )
        if self.config.response_shape == VALUES_SHAPE:
            return self._predict_labels_in_batches(texts)
        return [
            self._predict_labels_for_sequence(sequence_texts, sequence_features)
            for sequence_texts, sequence_features in zip(texts, features)
        ]

    def _predict_labels_in_batches(
        self, texts: List[List[str]]
    ) -> List[List[Tuple[str, str]]]:
        batch_size = max(1, self.config.max_references_per_request)
        results: List[List[Tuple[str, str]]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            results.extend(self._predict_labels_from_values(batch))
        return results

    def _get_content(self, response_json, token_count: int) -> str:
        try:
            return get_response_content(response_json)
        except LlmRequestError as exc:
            raise LlmRequestError(
                f'{exc} [task={self.config.task!r} model={self.config.model!r}'
                f' shape={self.config.response_shape!r}'
                f' prompt={self.config.prompt_version!r} tokens={token_count}]'
            ) from exc

    def _check_evidence(self, mismatches: int) -> None:
        if not mismatches:
            return
        message = (
            f'{mismatches} reference(s) quoted words that are not on the line they'
            ' named, or the line below it'
        )
        if self.config.evidence_mismatch_raises:
            raise LlmResponseError(message)
        LOGGER.warning('llm %s: %s', self.config.task, message)

    def _check_input_size(self, line_count: int, token_count: int) -> None:
        """A references region far larger than a reference list is a segmentation
        failure upstream, not something to extract from. Warn rather than raise by
        default: raising would fail exactly the documents where the CRF engine
        produces poor output, which flatters a comparison rather than informing it.
        """
        if self.config.max_input_lines and line_count > self.config.max_input_lines:
            raise LlmInputTooLargeError(
                f'{line_count} lines ({token_count} tokens) exceeds'
                f' max_input_lines={self.config.max_input_lines};'
                ' a references region this large is usually a mislabelled'
                ' segmentation region rather than a reference list'
            )
        if self.config.warn_input_lines and line_count > self.config.warn_input_lines:
            LOGGER.warning(
                'llm %s input is %d lines (%d tokens), which is larger than a'
                ' reference list usually is; check whether the segmentation model'
                ' labelled the right region',
                self.config.task, line_count, token_count
            )

    def _predict_labels_for_sequence(
        self,
        tokens: List[str],
        feature_rows: List[List[str]]
    ) -> List[Tuple[str, str]]:
        line_status_values = [row[self.line_status_index] for row in feature_rows]
        line_numbers = get_line_numbers(line_status_values)
        self._check_input_size(max(line_numbers) + 1, len(tokens))
        prompt = get_prompt(
            self.config.task,
            self.config.prompt_version,
            render_numbered_lines(tokens, line_numbers)
        )
        is_evidence = self.config.response_shape == EVIDENCE_SHAPE
        schema = EVIDENCE_RESPONSE_SCHEMA if is_evidence else LINES_RESPONSE_SCHEMA
        with llm_span(self.config, prompt, self.config.record_trace_content) as span:
            response_json = self.client.get_completion(prompt, schema)
            span.set_attribute('sciencebeam.input_lines', max(line_numbers) + 1)
            span.set_attribute('sciencebeam.input_tokens', len(tokens))
            set_response_attributes(
                span, response_json, _get_content_or_none(response_json),
                self.config.record_trace_content
            )
            content = self._get_content(response_json, len(tokens))
            if is_evidence:
                labeled, mismatches = decode_evidence_response(
                    content, tokens, line_status_values
                )
                span.set_attribute('sciencebeam.evidence_mismatches', mismatches)
                self._check_evidence(mismatches)
            else:
                labeled = decode_line_starts_response(
                    content, tokens, line_status_values
                )
        LOGGER.info(
            'llm labelled %d tokens over %d lines (model=%r provider=%r)',
            len(tokens), max(line_numbers) + 1, self.config.model,
            response_json.get('provider')
        )
        return labeled

    def _predict_labels_from_values(
        self, token_lists: List[List[str]]
    ) -> List[List[Tuple[str, str]]]:
        token_count = sum(len(tokens) for tokens in token_lists)
        prompt = get_prompt(
            self.config.task,
            self.config.prompt_version,
            render_numbered_references(token_lists)
        )
        with llm_span(self.config, prompt, self.config.record_trace_content) as span:
            response_json = self.client.get_completion(
                prompt, get_batched_values_response_schema(self.labels)
            )
            span.set_attribute('sciencebeam.input_tokens', token_count)
            span.set_attribute('sciencebeam.batch_size', len(token_lists))
            set_response_attributes(
                span, response_json, _get_content_or_none(response_json),
                self.config.record_trace_content
            )
            content = self._get_content(response_json, token_count)
            labeled = decode_batched_values_response(content, token_lists, self.labels)
        LOGGER.info(
            'llm labelled %d references, %d tokens (model=%r provider=%r)',
            len(token_lists), token_count, self.config.model,
            response_json.get('provider')
        )
        return labeled
