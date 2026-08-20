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
    LINES_RESPONSE_SCHEMA,
    decode_line_starts_response,
    get_line_numbers,
    render_numbered_lines
)
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.prompt import get_prompt
from sciencebeam_parser.models.llm.tasks import get_citation_labels
from sciencebeam_parser.models.llm.values import (
    decode_values_response,
    get_values_response_schema
)
from sciencebeam_parser.models.model_impl import ModelImpl


LOGGER = logging.getLogger(__name__)

LINE_STATUS_FEATURE_NAME = 'line_status'

LINES_SHAPE = 'lines'
VALUES_SHAPE = 'values'

SUPPORTED_RESPONSE_SHAPES = (LINES_SHAPE, VALUES_SHAPE)


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
            if config.response_shape == LINES_SHAPE else -1
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
        return [
            self._predict_labels_for_sequence(sequence_texts, sequence_features)
            for sequence_texts, sequence_features in zip(texts, features)
        ]

    def _get_content(self, response_json, token_count: int) -> str:
        try:
            return get_response_content(response_json)
        except LlmRequestError as exc:
            raise LlmRequestError(
                f'{exc} [task={self.config.task!r} model={self.config.model!r}'
                f' shape={self.config.response_shape!r}'
                f' prompt={self.config.prompt_version!r} tokens={token_count}]'
            ) from exc

    def _predict_labels_for_sequence(
        self,
        tokens: List[str],
        feature_rows: List[List[str]]
    ) -> List[Tuple[str, str]]:
        if self.config.response_shape == VALUES_SHAPE:
            return self._predict_labels_from_values(tokens)
        line_status_values = [row[self.line_status_index] for row in feature_rows]
        line_numbers = get_line_numbers(line_status_values)
        prompt = get_prompt(
            self.config.task,
            self.config.prompt_version,
            render_numbered_lines(tokens, line_numbers)
        )
        response_json = self.client.get_completion(prompt, LINES_RESPONSE_SCHEMA)
        content = self._get_content(response_json, len(tokens))
        labeled = decode_line_starts_response(content, tokens, line_status_values)
        LOGGER.info(
            'llm labelled %d tokens over %d lines (model=%r provider=%r)',
            len(tokens), max(line_numbers) + 1, self.config.model,
            response_json.get('provider')
        )
        return labeled

    def _predict_labels_from_values(self, tokens: List[str]) -> List[Tuple[str, str]]:
        prompt = get_prompt(
            self.config.task, self.config.prompt_version, ' '.join(tokens)
        )
        response_json = self.client.get_completion(
            prompt, get_values_response_schema(self.labels)
        )
        content = self._get_content(response_json, len(tokens))
        labeled = decode_values_response(content, tokens, self.labels)
        LOGGER.info(
            'llm labelled %d tokens from values (model=%r provider=%r)',
            len(tokens), self.config.model, response_json.get('provider')
        )
        return labeled
