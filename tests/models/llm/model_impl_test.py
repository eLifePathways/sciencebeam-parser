import json
import threading
from contextvars import ContextVar
from typing import Any, List, Mapping, Optional

import pytest

from sciencebeam_parser.models.llm.client import LlmTruncatedResponseError
from sciencebeam_parser.models.llm.config import LlmConfigError, LlmEngineConfig
from sciencebeam_parser.models.llm.decode import (
    LlmInputTooLargeError,
    LlmResponseError
)
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.model_impl import LlmModelImpl
from sciencebeam_parser.models.llm.values import render_numbered_references
from sciencebeam_parser.utils.telemetry import span


CONFIG = {
    'task': 'reference_segmenter',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'lines-v1',
    'provider': 'siliconflow',
}

TOKENS = ['1', '.', 'Fleming', 'PS', 'High', 'quality']
LINE_STATUS = ['LINESTART', 'LINEEND', 'LINESTART', 'LINEIN', 'LINEIN', 'LINEEND']

LINE_STATUS_INDEX = get_feature_column_index('reference_segmenter', 'line_status')


def feature_rows():
    return [
        ['x'] * LINE_STATUS_INDEX + [status] + ['y']
        for status in LINE_STATUS
    ]


class FakeClient:
    """Stands in for the network. No test in this module reaches an endpoint.

    `content` may be a list, one entry per call, so a test can make the first
    response leave a reference out and the next one answer it.
    """
    def __init__(self, content=None, error: Optional[Exception] = None):
        self.contents = content if isinstance(content, list) else [content]
        self.error = error
        self.prompts: List[str] = []
        self.lock = threading.Lock()

    def validate_configuration(self) -> None:
        if self.error:
            raise self.error

    def get_completion(self, prompt: str, response_schema: Mapping[str, Any]):
        assert response_schema['type'] == 'object'
        with self.lock:
            self.prompts.append(prompt)
            index = min(len(self.prompts) - 1, len(self.contents) - 1)
        if self.error:
            raise self.error
        return {
            'choices': [{'message': {'content': self.contents[index]}}],
            'provider': 'SiliconFlow'
        }


def get_model_impl(content: Optional[str] = None, error: Optional[Exception] = None):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config(CONFIG),
        client=FakeClient(content=content, error=error)
    )


class TestLlmModelImpl:
    def test_should_resolve_the_line_status_column_by_name(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        assert model_impl.line_status_index == LINE_STATUS_INDEX

    def test_should_return_labels_for_every_input_token(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        result = model_impl.predict_labels([TOKENS], [feature_rows()])
        assert len(result) == 1
        assert [token for token, _ in result[0]] == TOKENS

    def test_should_include_numbered_lines_and_the_prompt_version_text(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        model_impl.predict_labels([TOKENS], [feature_rows()])
        prompt = model_impl.client.prompts[0]
        assert '0\t1 .' in prompt
        assert '1\tFleming PS High quality' in prompt
        assert 'line number' in prompt

    def test_should_predict_for_each_sequence(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        result = model_impl.predict_labels([TOKENS, TOKENS], [feature_rows(), feature_rows()])
        assert len(result) == 2

    def test_should_raise_rather_than_fall_back_on_an_invalid_response(self):
        model_impl = get_model_impl(json.dumps({'starts': [99]}))
        with pytest.raises(LlmResponseError):
            model_impl.predict_labels([TOKENS], [feature_rows()])

    def test_should_raise_for_an_unsupported_output_format(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        with pytest.raises(NotImplementedError):
            model_impl.predict_labels([TOKENS], [feature_rows()], output_format='json')

    def test_should_reject_an_unsupported_response_shape(self):
        with pytest.raises(LlmConfigError, match='response_shape'):
            LlmModelImpl(LlmEngineConfig.from_model_config({
                **CONFIG, 'response_shape': 'spans'
            }), client=FakeClient())

    def test_preload_should_surface_a_configuration_failure(self):
        model_impl = get_model_impl(error=RuntimeError('endpoint unreachable'))
        with pytest.raises(RuntimeError, match='unreachable'):
            model_impl.preload()


CITATION_CONFIG = {
    'task': 'citation',
    'response_shape': 'values',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'values-v3',
}

CITATION_TOKENS = ['Fleming', 'PS', ',', 'Koletsi', 'D', ':', 'High', 'quality']
# the prompt's worked example uses REFERENCE markers of its own
EXAMPLES_IN_CITATION_PROMPT = 4
SECOND_REFERENCE = ['Rada', 'G', ':', 'What', 'is', 'best']


def batched(*per_reference) -> str:
    return json.dumps({'references': [
        {'index': index, 'fields': [
            {'label': label, 'text': text} for label, text in fields
        ]}
        for index, fields in enumerate(per_reference)
    ]})


def get_citation_model_impl(content: str, **overrides):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config({**CITATION_CONFIG, **overrides}),
        client=FakeClient(content=content)
    )


def no_features(token_lists):
    return [[[]] * len(tokens) for tokens in token_lists]


class TestLlmModelImplValuesShape:
    def test_should_not_need_a_line_status_column(self):
        assert get_citation_model_impl(batched([])).line_status_index == -1

    def test_should_label_from_located_values(self):
        model_impl = get_citation_model_impl(
            batched([('author', 'Fleming PS , Koletsi D')])
        )
        result = model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert [label for _, label in result[0]][:5] == [
            'B-<author>', 'I-<author>', 'I-<author>', 'I-<author>', 'I-<author>'
        ]

    def test_should_return_the_input_tokens_unchanged(self):
        model_impl = get_citation_model_impl(batched([('title', 'High quality')]))
        result = model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert [token for token, _ in result[0]] == CITATION_TOKENS

    def test_should_drop_a_value_that_is_not_in_the_source(self, caplog):
        model_impl = get_citation_model_impl(batched([('title', 'a paraphrased title')]))
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels(
                [CITATION_TOKENS], no_features([CITATION_TOKENS])
            )
        assert [label for _, label in result[0]] == ['O'] * len(CITATION_TOKENS)
        assert 'could not be located' in caplog.text

    def test_should_raise_for_a_dropped_field_when_configured_to(self):
        model_impl = get_citation_model_impl(
            batched([('title', 'a paraphrased title')]), dropped_field_raises=True
        )
        with pytest.raises(LlmResponseError, match='could not be located'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))

    def test_should_send_the_conventions_and_the_numbered_references(self):
        model_impl = get_citation_model_impl(batched([]))
        model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        prompt = model_impl.client.prompts[0]
        assert 'REFERENCE 0\nFleming PS , Koletsi D : High quality' in prompt
        assert 'page range is TWO' in prompt


class TestCitationBatching:
    def test_should_send_several_references_in_one_call(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(
            batched([('author', 'Fleming PS')], [('author', 'Rada G')])
        )
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 1
        assert len(result) == 2
        assert result[1][0] == ('Rada', 'B-<author>')

    def test_should_number_each_reference_in_the_prompt(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(batched([], []))
        model_impl.predict_labels(token_lists, no_features(token_lists))
        prompt = model_impl.client.prompts[0]
        assert 'REFERENCE 0\nFleming PS' in prompt
        assert 'REFERENCE 1\nRada G' in prompt

    def test_should_split_into_calls_at_the_configured_bound(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE, CITATION_TOKENS]
        # no fields, so the same stubbed response is valid for every reference
        model_impl = get_citation_model_impl(batched([]), max_references_per_request=1)
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 3
        assert len(result) == 3

    def test_should_locate_values_within_their_own_reference_only(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(
            batched([('author', 'Rada G')], [('author', 'Rada G')])
        )
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        # reference 0 does not contain "Rada G", so that claim is dropped rather
        # than matched against the other reference
        assert [label for _, label in result[0]] == ['O'] * len(CITATION_TOKENS)
        assert result[1][0] == ('Rada', 'B-<author>')


class TestEmptyInput:
    """A references region can come back with no tokens. Asking the model about
    nothing wastes a call, and the line-based path used to reach max() on an
    empty sequence — an unhandled ValueError, so the service answered 500.
    """
    def test_should_return_no_labels_for_an_empty_sequence(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        assert model_impl.predict_labels([[]], [[]]) == [[]]

    def test_should_not_call_the_model_for_an_empty_sequence(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        model_impl.predict_labels([[]], [[]])
        assert model_impl.client.prompts == []

    def test_should_still_predict_the_non_empty_sequences(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        result = model_impl.predict_labels([[], TOKENS], [[], feature_rows()])
        assert result[0] == []
        assert [token for token, _ in result[1]] == TOKENS

    def test_should_return_no_labels_for_an_empty_reference(self):
        model_impl = get_citation_model_impl(batched([]))
        assert model_impl.predict_labels([[]], no_features([[]])) == [[]]

    def test_should_not_call_the_model_for_an_empty_reference(self):
        model_impl = get_citation_model_impl(batched([]))
        model_impl.predict_labels([[]], no_features([[]]))
        assert model_impl.client.prompts == []

    def test_should_keep_an_empty_reference_out_of_the_batch(self):
        token_lists = [[], CITATION_TOKENS, []]
        model_impl = get_citation_model_impl(batched([('author', 'Fleming PS')]))
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 1
        assert render_numbered_references([CITATION_TOKENS]) in model_impl.client.prompts[0]
        assert result[0] == [] and result[2] == []
        assert result[1][0] == ('Fleming', 'B-<author>')


class TruncatingClient:
    """Truncates any batch larger than `answers_up_to`, as a provider does when a
    response runs past max_tokens: valid-looking JSON, finish_reason='length'.
    """
    def __init__(self, answers_up_to: int):
        self.answers_up_to = answers_up_to
        self.prompts: List[str] = []
        self.batch_sizes: List[int] = []

    def validate_configuration(self) -> None:
        pass

    def get_completion(self, prompt: str, response_schema: Mapping[str, Any]):
        assert response_schema['type'] == 'object'
        self.prompts.append(prompt)
        size = prompt.count('REFERENCE ') - EXAMPLES_IN_CITATION_PROMPT
        self.batch_sizes.append(size)
        if size > self.answers_up_to:
            return {'choices': [{
                'message': {'content': '{"references": [{"index": 0, "fie'},
                'finish_reason': 'length',
            }], 'usage': {'completion_tokens': 16000}}
        return {
            'choices': [{'message': {'content': batched(*([[]] * size))},
                         'finish_reason': 'stop'}],
            'provider': 'Venice',
        }


def get_truncating_model_impl(answers_up_to: int, **overrides):
    return LlmModelImpl(
        LlmEngineConfig.from_model_config({**CITATION_CONFIG, **overrides}),
        client=TruncatingClient(answers_up_to)
    )


class TestRetryOnMalformedResponse:
    """Generation is not bit-reproducible even at temperature 0, so a response
    that will not parse is worth asking for again — but only that: a strictness
    setting that fires is a decision, not a bad sample.
    """
    def test_should_ask_again_when_the_response_is_not_json(self, caplog):
        model_impl = get_model_impl(['{"starts": [0', json.dumps({'starts': [0]})])
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels([TOKENS], [feature_rows()])
        assert len(model_impl.client.prompts) == 2
        assert [token for token, _ in result[0]] == TOKENS
        assert 'could not be parsed, asking again' in caplog.text

    def test_should_send_the_same_prompt_again(self):
        model_impl = get_model_impl(['{"starts": [0', json.dumps({'starts': [0]})])
        model_impl.predict_labels([TOKENS], [feature_rows()])
        first, second = model_impl.client.prompts
        assert first == second

    def test_should_ask_again_when_a_key_is_missing(self):
        model_impl = get_model_impl(['{}', json.dumps({'starts': [0]})])
        model_impl.predict_labels([TOKENS], [feature_rows()])
        assert len(model_impl.client.prompts) == 2

    def test_should_give_up_after_the_configured_retries(self):
        model_impl = get_model_impl('{"starts": [0')
        with pytest.raises(LlmResponseError, match='not json'):
            model_impl.predict_labels([TOKENS], [feature_rows()])
        assert len(model_impl.client.prompts) == 2

    def test_should_not_retry_when_disabled(self):
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config(
                {**CONFIG, 'max_malformed_response_retries': 0}
            ),
            client=FakeClient(content='{"starts": [0')
        )
        with pytest.raises(LlmResponseError):
            model_impl.predict_labels([TOKENS], [feature_rows()])
        assert len(model_impl.client.prompts) == 1

    def test_should_not_ask_again_when_the_response_parsed(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        model_impl.predict_labels([TOKENS], [feature_rows()])
        assert len(model_impl.client.prompts) == 1

    def test_should_ask_again_for_a_citation_batch(self):
        token_lists = [CITATION_TOKENS]
        model_impl = get_citation_model_impl(
            ['{"references": [', batched([('author', 'Fleming PS')])]
        )
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 2
        assert result[0][0] == ('Fleming', 'B-<author>')

    def test_should_not_retry_a_strictness_failure(self):
        # dropped_field_raises firing is a decision, so the same answer would
        # only come back again
        model_impl = get_citation_model_impl(
            batched([('title', 'a paraphrased title')]), dropped_field_raises=True
        )
        with pytest.raises(LlmResponseError, match='could not be located'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert len(model_impl.client.prompts) == 1


class TestSplitOnTruncation:
    """An answer that ran out of room is unusable, and the batch is the only thing
    the engine can change about it.
    """
    def test_should_halve_a_batch_whose_answer_was_truncated(self, caplog):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE, CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_truncating_model_impl(answers_up_to=2)
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert model_impl.client.batch_sizes == [4, 2, 2]
        assert [[t for t, _ in r] for r in result] == token_lists
        assert 'hit the output limit' in caplog.text

    def test_should_halve_repeatedly_until_the_answer_fits(self):
        token_lists = [CITATION_TOKENS] * 4
        model_impl = get_truncating_model_impl(answers_up_to=1)
        model_impl.predict_labels(token_lists, no_features(token_lists))
        assert model_impl.client.batch_sizes == [4, 2, 1, 1, 2, 1, 1]

    def test_should_raise_when_a_single_reference_still_truncates(self):
        model_impl = get_truncating_model_impl(answers_up_to=0)
        with pytest.raises(LlmTruncatedResponseError, match='output token limit'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))

    def test_should_not_split_a_batch_that_answered(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_truncating_model_impl(answers_up_to=10)
        model_impl.predict_labels(token_lists, no_features(token_lists))
        assert model_impl.client.batch_sizes == [2]

    def test_should_keep_the_truncation_type_rather_than_the_base_error(self):
        model_impl = get_truncating_model_impl(answers_up_to=0)
        with pytest.raises(LlmTruncatedResponseError):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))


class TestRetryForMissingReferences:
    """Skipped references are always the tail of the batch, so the repair call
    sends only those — which puts them at the start of a short batch.
    """
    def test_should_ask_again_for_a_reference_the_batch_left_out(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        first = batched([('author', 'Fleming PS')])            # answers 0 only
        second = batched([('author', 'Rada G')])               # answers the retry's 0
        model_impl = get_citation_model_impl([first, second])
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 2
        assert result[1][0] == ('Rada', 'B-<author>')

    def test_should_send_only_the_missing_reference_in_the_retry(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl([
            batched([('author', 'Fleming PS')]), batched([('author', 'Rada G')])
        ])
        model_impl.predict_labels(token_lists, no_features(token_lists))
        # the prompt template's worked example contains REFERENCE markers of its
        # own, so compare the rendered block rather than search the whole prompt
        first, retry = model_impl.client.prompts
        assert render_numbered_references([CITATION_TOKENS, SECOND_REFERENCE]) in first
        assert render_numbered_references([SECOND_REFERENCE]) in retry
        assert render_numbered_references([CITATION_TOKENS, SECOND_REFERENCE]) not in retry

    def test_should_keep_the_first_answer_for_the_references_already_given(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl([
            batched([('author', 'Fleming PS')]), batched([('author', 'Rada G')])
        ])
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert result[0][0] == ('Fleming', 'B-<author>')

    def test_should_not_ask_again_when_the_batch_was_complete(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(
            batched([('author', 'Fleming PS')], [('author', 'Rada G')])
        )
        model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 1

    def test_should_give_up_rather_than_loop_when_the_retry_answers_nothing(self, caplog):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        empty = json.dumps({'references': []})
        model_impl = get_citation_model_impl(
            [batched([('author', 'Fleming PS')]), empty, empty],
            max_missing_reference_retries=5
        )
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 2
        assert [label for _, label in result[1]] == ['O'] * len(SECOND_REFERENCE)
        assert 'still unanswered' in caplog.text

    def test_should_not_retry_when_disabled(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(
            batched([('author', 'Fleming PS')]), max_missing_reference_retries=0
        )
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 1
        assert [label for _, label in result[1]] == ['O'] * len(SECOND_REFERENCE)

    def test_should_return_every_input_token_after_a_retry(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl([
            batched([('author', 'Fleming PS')]), batched([('author', 'Rada G')])
        ])
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert [[token for token, _ in r] for r in result] == token_lists


class TestUnansweredReferences:
    """A batch answer the engine cannot honour costs those references, not the
    document. Every one of these cases used to raise, which lost all the
    references in the batch and every field the other models had produced.
    """
    def test_should_leave_an_unanswered_reference_unlabelled(self, caplog):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(batched([('author', 'Fleming PS')]))
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert [label for _, label in result[1]] == ['O'] * len(SECOND_REFERENCE)
        assert 'no answer for reference(s) [1]' in caplog.text

    def test_should_keep_the_answered_references_in_the_same_batch(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(batched([('author', 'Fleming PS')]))
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert result[0][0] == ('Fleming', 'B-<author>')

    def test_should_return_every_reference_sent_even_when_unanswered(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(batched([('author', 'Fleming PS')]))
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert [[token for token, _ in labelled] for labelled in result] == token_lists

    def test_should_keep_the_first_answer_when_an_index_appears_twice(self, caplog):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        content = json.dumps({'references': [
            {'index': 0, 'fields': [{'label': 'author', 'text': 'Fleming PS'}]},
            {'index': 0, 'fields': [{'label': 'title', 'text': 'High quality'}]},
            {'index': 1, 'fields': []},
        ]})
        model_impl = get_citation_model_impl(content)
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels(token_lists, no_features(token_lists))
        labels = [label for _, label in result[0]]
        assert 'B-<author>' in labels
        assert 'B-<title>' not in labels
        assert 'twice' in caplog.text

    def test_should_skip_an_index_outside_the_batch(self, caplog):
        content = json.dumps({'references': [
            {'index': 5, 'fields': []},
            {'index': 0, 'fields': [{'label': 'author', 'text': 'Fleming PS'}]},
        ]})
        model_impl = get_citation_model_impl(content)
        with caplog.at_level('WARNING'):
            result = model_impl.predict_labels(
                [CITATION_TOKENS], no_features([CITATION_TOKENS])
            )
        assert result[0][0] == ('Fleming', 'B-<author>')
        assert 'outside the 1 sent' in caplog.text

    def test_should_raise_when_configured_to_be_strict(self):
        token_lists = [CITATION_TOKENS, SECOND_REFERENCE]
        model_impl = get_citation_model_impl(
            batched([('author', 'Fleming PS')]), unanswered_reference_raises=True,
            max_missing_reference_retries=0
        )
        with pytest.raises(LlmResponseError, match='still unanswered'):
            model_impl.predict_labels(token_lists, no_features(token_lists))

    def test_should_still_raise_for_a_response_that_cannot_be_parsed(self):
        model_impl = get_citation_model_impl('{"references": [')
        with pytest.raises(LlmResponseError, match='not json'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))

    def test_should_still_raise_for_a_malformed_reference_entry(self):
        content = json.dumps({'references': [{'fields': []}]})
        model_impl = get_citation_model_impl(content)
        with pytest.raises(LlmResponseError, match='malformed'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))


BIG_LINE_STATUS = ['LINESTART', 'LINEEND'] * 400


def get_big_input():
    tokens = [str(index) for index in range(len(BIG_LINE_STATUS))]
    features = [
        ['x'] * LINE_STATUS_INDEX + [status] + ['y'] for status in BIG_LINE_STATUS
    ]
    return tokens, features


class TestInputSize:
    def test_should_warn_above_the_warn_threshold(self, caplog):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        tokens, features = get_big_input()
        with caplog.at_level('WARNING'):
            model_impl.predict_labels([tokens], [features])
        assert 'segmentation model' in caplog.text

    def test_should_not_warn_for_an_ordinary_reference_list(self, caplog):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        with caplog.at_level('WARNING'):
            model_impl.predict_labels([TOKENS], [feature_rows()])
        assert 'segmentation model' not in caplog.text

    def test_should_raise_when_a_hard_limit_is_configured_and_exceeded(self):
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({**CONFIG, 'max_input_lines': 10}),
            client=FakeClient(content=json.dumps({'starts': [0]}))
        )
        tokens, features = get_big_input()
        with pytest.raises(LlmInputTooLargeError, match='max_input_lines=10'):
            model_impl.predict_labels([tokens], [features])

    def test_should_not_raise_when_no_hard_limit_is_configured(self):
        model_impl = get_model_impl(json.dumps({'starts': [0]}))
        tokens, features = get_big_input()
        assert model_impl.predict_labels([tokens], [features])


class TestCitationReferenceMarker:
    def test_should_not_use_a_bracketed_number_as_the_marker(self):
        model_impl = get_citation_model_impl(batched([]))
        model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        prompt = model_impl.client.prompts[0]
        assert '[0]' not in prompt

    def test_should_tell_the_model_the_marker_is_not_content(self):
        model_impl = get_citation_model_impl(batched([]))
        model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert 'not part of the reference' in model_impl.client.prompts[0]

    def test_should_name_the_reference_and_field_when_a_value_is_dropped(self, caplog):
        model_impl = get_citation_model_impl(batched([('note', '1')]))
        with caplog.at_level('WARNING'):
            model_impl.predict_labels([CITATION_TOKENS], no_features([CITATION_TOKENS]))
        assert 'reference index 0' in caplog.text
        assert 'label=note' in caplog.text
        assert 'Fleming' in caplog.text


class TestCitationConcurrency:
    def test_should_preserve_order_across_parallel_batches(self):
        token_lists = [
            ['Alpha', 'A'], ['Bravo', 'B'], ['Charlie', 'C'], ['Delta', 'D'],
        ]
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({
                **CITATION_CONFIG,
                'max_references_per_request': 1,
                'max_concurrent_requests': 4,
            }),
            client=FakeClient(content=batched([]))
        )
        result = model_impl.predict_labels(token_lists, no_features(token_lists))
        assert [tokens[0] for tokens in token_lists] == [
            labelled[0][0] for labelled in result
        ]

    def test_should_carry_the_calling_context_into_each_worker(self):
        """What makes a batch's span nest under the document's.

        A worker thread starts with no context, so the current span would
        otherwise be absent there and every batch would begin its own trace.
        Asserted on a plain context variable rather than on a span, so it holds
        whether or not opentelemetry is installed.
        """
        current_document: ContextVar[str] = ContextVar('current_document')
        seen: List[str] = []
        lock = threading.Lock()

        class ContextObservingClient(FakeClient):
            def get_completion(self, prompt: str, response_schema):
                with lock:
                    seen.append(current_document.get('missing'))
                return super().get_completion(prompt, response_schema)

        token_lists = [['Alpha'], ['Bravo'], ['Charlie']]
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({
                **CITATION_CONFIG,
                'max_references_per_request': 1,
                'max_concurrent_requests': 3,
            }),
            client=ContextObservingClient(content=batched([]))
        )
        current_document.set('example.pdf')
        model_impl.predict_labels(token_lists, no_features(token_lists))
        assert seen == ['example.pdf'] * len(token_lists)

    def test_should_nest_parallel_batch_spans_under_the_document(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The same propagation, asserted against opentelemetry itself.

        The tracer is replaced rather than a global provider installed, since
        the global one can only be set once per process and would leak into
        every other test.
        """
        # Imported here rather than at the top, because the sdk is an optional
        # extra: this module has to import without it, and the lint image does
        # not install it.
        # pylint: disable=import-outside-toplevel,import-error
        pytest.importorskip('opentelemetry.sdk')
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        for module in (
            'sciencebeam_parser.utils.telemetry',
            'sciencebeam_parser.models.llm.telemetry'
        ):
            monkeypatch.setattr(
                f'{module}.get_tracer', lambda name=None: provider.get_tracer(__name__)
            )

        token_lists = [['Alpha'], ['Bravo'], ['Charlie'], ['Delta']]
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({
                **CITATION_CONFIG,
                'max_references_per_request': 1,
                'max_concurrent_requests': 4,
            }),
            client=FakeClient(content=batched([]))
        )
        with span('process_document', {'sciencebeam.document.name': 'example.pdf'}):
            model_impl.predict_labels(token_lists, no_features(token_lists))

        spans = exporter.get_finished_spans()
        document = [one for one in spans if one.name == 'process_document']
        calls = [one for one in spans if one.name != 'process_document']
        assert len(document) == 1
        assert len(calls) == len(token_lists)
        assert all(
            one.parent is not None
            and one.parent.span_id == document[0].context.span_id
            for one in calls
        )
        assert len({one.context.trace_id for one in spans}) == 1

    def test_should_make_one_call_per_batch_when_parallel(self):
        token_lists = [['Alpha'], ['Bravo'], ['Charlie']]
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({
                **CITATION_CONFIG,
                'max_references_per_request': 1,
                'max_concurrent_requests': 3,
            }),
            client=FakeClient(content=batched([]))
        )
        model_impl.predict_labels(token_lists, no_features(token_lists))
        assert len(model_impl.client.prompts) == 3

    def test_should_propagate_a_failure_from_a_worker(self):
        token_lists = [['Alpha'], ['Bravo']]
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({
                **CITATION_CONFIG,
                'max_references_per_request': 1,
                'max_concurrent_requests': 2,
            }),
            client=FakeClient(content='{"references": [')
        )
        with pytest.raises(LlmResponseError, match='not json'):
            model_impl.predict_labels(token_lists, no_features(token_lists))

    def test_should_stay_sequential_when_concurrency_is_one(self):
        token_lists = [['Alpha'], ['Bravo']]
        model_impl = LlmModelImpl(
            LlmEngineConfig.from_model_config({
                **CITATION_CONFIG,
                'max_references_per_request': 1,
                'max_concurrent_requests': 1,
            }),
            client=FakeClient(content=batched([]))
        )
        assert len(model_impl.predict_labels(token_lists, no_features(token_lists))) == 2
