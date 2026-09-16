import contextvars
import json
import os
import shutil
import threading
from dataclasses import fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pytest

from sciencebeam_parser.models.llm.cache import (
    META_FILENAME,
    REQUEST_FILENAME,
    CachingLlmClient,
    LlmResponseCache,
    get_request_key,
    get_response_cache
)
from sciencebeam_parser.models.llm.client import FIRST_ATTEMPT, get_request_body
from sciencebeam_parser.models.llm.config import LlmEngineConfig
from sciencebeam_parser.models.llm.features import get_feature_column_index
from sciencebeam_parser.models.llm.model_impl import LlmModelImpl
from sciencebeam_parser.models.llm.usage import (
    REPLAYED_RESPONSE_KEY,
    get_request_llm_usage_header_value,
    start_request_llm_usage
)


CONFIG = {
    'task': 'reference_segmenter',
    'model': 'qwen/qwen3.5-9b',
    'prompt_version': 'lines-v1',
    'response_shape': 'lines',
}

SCHEMA: Dict[str, Any] = {'type': 'object', 'properties': {'starts': {'type': 'array'}}}

PROMPT = 'label these lines'

TOKENS = ['1', '.', 'Fleming', 'PS']
LINE_STATUS = ['LINESTART', 'LINEEND', 'LINESTART', 'LINEEND']

LINE_STATUS_INDEX = get_feature_column_index('reference_segmenter', 'line_status')


def feature_rows() -> List[List[str]]:
    return [
        ['x'] * LINE_STATUS_INDEX + [status] + ['y']
        for status in LINE_STATUS
    ]


def get_config(**extra) -> LlmEngineConfig:
    return LlmEngineConfig.from_model_config({**CONFIG, **extra})


def get_response(content: str) -> Dict[str, Any]:
    return {
        'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
        'provider': 'venice',
        'usage': {'prompt_tokens': 10, 'completion_tokens': 20},
    }


def get_stored_responses(cache_dir) -> List[Any]:
    """The stored answers, not the two once-per-key files beside them."""
    return sorted(
        path for path in cache_dir.rglob('*.json')
        if path.name not in (REQUEST_FILENAME, META_FILENAME)
    )


def get_replayed_response(content: str) -> Dict[str, Any]:
    """What a replay returns: the stored body, marked so usage counts it as
    replayed rather than as spend."""
    return {**get_response(content), REPLAYED_RESPONSE_KEY: True}


class FakeClient:
    """Stands in for the network. No test in this module reaches an endpoint.

    `contents` is one entry per call, so a test can make the first response
    unparseable and the next one usable.
    """
    def __init__(self, contents: Optional[List[str]] = None):
        self.contents = list(contents) if contents is not None else ['{"starts": [0, 1]}']
        self.prompts: List[str] = []
        self.attempts: List[Tuple[int, ...]] = []
        self.lock = threading.Lock()

    def validate_configuration(self) -> None:
        pass

    def get_completion(
        self,
        prompt: str,
        response_schema: Mapping[str, Any],
        attempt: Tuple[int, ...] = FIRST_ATTEMPT
    ) -> Dict[str, Any]:
        assert response_schema
        with self.lock:
            index = min(len(self.prompts), len(self.contents) - 1)
            self.prompts.append(prompt)
            self.attempts.append(attempt)
        return get_response(self.contents[index])

    @property
    def call_count(self) -> int:
        return len(self.prompts)


class NeverCalledClient:
    def validate_configuration(self) -> None:
        pass

    def get_completion(
        self,
        prompt: str,
        response_schema: Mapping[str, Any],
        attempt: Tuple[int, ...] = FIRST_ATTEMPT
    ):
        raise AssertionError(
            f'the cache should have answered {prompt[:40]!r} at {attempt}'
        )


def get_caching_client(config: LlmEngineConfig, delegate, cache_dir) -> CachingLlmClient:
    return CachingLlmClient(config, delegate, LlmResponseCache(str(cache_dir)))


class TestGetRequestKey:
    def test_should_be_stable_for_the_same_request(self):
        config = get_config()
        body = get_request_body(config, PROMPT, SCHEMA)
        assert get_request_key(config.endpoint, body) == get_request_key(
            config.endpoint, get_request_body(config, PROMPT, SCHEMA)
        )

    def test_should_not_depend_on_the_order_keys_were_added(self):
        config = get_config()
        body = get_request_body(config, PROMPT, SCHEMA)
        reordered = dict(reversed(list(body.items())))
        assert get_request_key(config.endpoint, reordered) == get_request_key(
            config.endpoint, body
        )

    def test_should_change_with_the_prompt(self):
        config = get_config()
        assert get_request_key(
            config.endpoint, get_request_body(config, PROMPT, SCHEMA)
        ) != get_request_key(
            config.endpoint, get_request_body(config, PROMPT + ' again', SCHEMA)
        )

    def test_should_change_with_the_response_schema(self):
        config = get_config()
        assert get_request_key(
            config.endpoint, get_request_body(config, PROMPT, SCHEMA)
        ) != get_request_key(
            config.endpoint, get_request_body(config, PROMPT, {'type': 'array'})
        )

    def test_should_change_with_the_endpoint(self):
        config = get_config()
        body = get_request_body(config, PROMPT, SCHEMA)
        assert get_request_key(config.endpoint, body) != get_request_key(
            'https://example.org/v1', body
        )

    @pytest.mark.parametrize('changed', [
        {'model': 'other/model'},
        {'temperature': 0.7},
        {'max_output_tokens': 200},
        {'provider': 'siliconflow'},
        {'reasoning_enabled': False},
        {'extra_body': {'top_p': 0.5}},
    ])
    def test_should_change_with_any_parameter_that_reaches_the_body(self, changed: dict):
        # The omission this key exists to prevent: a parameter added to the request
        # and not to the key would replay an answer generated under another setting.
        config = get_config()
        other = get_config(**changed)
        assert get_request_key(
            other.endpoint, get_request_body(other, PROMPT, SCHEMA)
        ) != get_request_key(config.endpoint, get_request_body(config, PROMPT, SCHEMA))

    def test_should_have_no_way_to_reach_the_cache_directory(self):
        # It is configured for the app rather than per model, so it is not on the
        # engine config the request body is built from — structural rather than a
        # rule get_request_body has to keep.
        assert 'response_cache_dir' not in {
            field_.name for field_ in fields(LlmEngineConfig)
        }

    def test_should_not_change_with_a_setting_the_request_does_not_carry(self):
        config = get_config()
        other = get_config(timeout_seconds=1, max_attempts=2)
        assert get_request_key(
            other.endpoint, get_request_body(other, PROMPT, SCHEMA)
        ) == get_request_key(config.endpoint, get_request_body(config, PROMPT, SCHEMA))


class TestCachingLlmClient:
    def test_should_ask_live_on_a_miss_and_store_the_answer(self, tmp_path):
        delegate = FakeClient()
        client = get_caching_client(get_config(), delegate, tmp_path)
        assert client.get_completion(PROMPT, SCHEMA) == get_response('{"starts": [0, 1]}')
        assert delegate.call_count == 1
        assert client.cache.miss_count == 1

    def test_should_replay_a_stored_answer_without_asking(self, tmp_path):
        config = get_config()
        get_caching_client(config, FakeClient(), tmp_path).get_completion(PROMPT, SCHEMA)
        replay = get_caching_client(config, NeverCalledClient(), tmp_path)
        assert replay.get_completion(PROMPT, SCHEMA) == get_replayed_response(
            '{"starts": [0, 1]}'
        )
        assert replay.cache.hit_count == 1

    def test_should_replay_the_same_request_at_the_same_attempt_again(self, tmp_path):
        # Two calls to the api for one document, against a server that stays up:
        # the second asks the same question at the same point in the retry logic,
        # so it replays rather than paying again.
        config = get_config()
        delegate = FakeClient()
        client = get_caching_client(config, delegate, tmp_path)
        assert client.get_completion(PROMPT, SCHEMA) == get_response('{"starts": [0, 1]}')
        assert client.get_completion(PROMPT, SCHEMA) == get_replayed_response(
            '{"starts": [0, 1]}'
        )
        assert delegate.call_count == 1

    def test_should_ask_live_for_a_repeat_the_engine_numbers_differently(self, tmp_path):
        # A parse retry, or a re-ask for the references a batch left out, sends
        # the same bytes on purpose and needs a different answer. Serving the
        # stored one would defeat the retry it is part of.
        config = get_config()
        delegate = FakeClient(['{"starts": [0]}', '{"starts": [0, 1]}'])
        client = get_caching_client(config, delegate, tmp_path)
        assert client.get_completion(PROMPT, SCHEMA, (0,)) == get_response('{"starts": [0]}')
        assert client.get_completion(PROMPT, SCHEMA, (1,)) == get_response(
            '{"starts": [0, 1]}'
        )
        assert delegate.call_count == 2

    def test_should_replay_a_repeated_request_attempt_for_attempt(self, tmp_path):
        config = get_config()
        delegate = FakeClient(['{"starts": [0]}', '{"starts": [0, 1]}'])
        client = get_caching_client(config, delegate, tmp_path)
        client.get_completion(PROMPT, SCHEMA, (0, 0))
        client.get_completion(PROMPT, SCHEMA, (0, 1))
        replay = get_caching_client(config, NeverCalledClient(), tmp_path)
        assert replay.get_completion(PROMPT, SCHEMA, (0, 0)) == get_replayed_response(
            '{"starts": [0]}'
        )
        assert replay.get_completion(PROMPT, SCHEMA, (0, 1)) == get_replayed_response(
            '{"starts": [0, 1]}'
        )

    def test_should_go_live_for_an_attempt_that_was_never_stored(self, tmp_path):
        config = get_config()
        get_caching_client(config, FakeClient(['{"starts": [0]}']), tmp_path).get_completion(
            PROMPT, SCHEMA, (0,)
        )
        delegate = FakeClient(['{"starts": [0, 1]}'])
        client = get_caching_client(config, delegate, tmp_path)
        assert client.get_completion(PROMPT, SCHEMA, (1,)) == get_response(
            '{"starts": [0, 1]}'
        )
        assert delegate.call_count == 1

    def test_should_name_entries_by_where_the_engine_was(self, tmp_path):
        config = get_config()
        client = get_caching_client(config, FakeClient(), tmp_path)
        client.get_completion(PROMPT, SCHEMA, (0,))
        client.get_completion(PROMPT, SCHEMA, (1,))
        client.get_completion(PROMPT, SCHEMA, (0, 1))
        names = [path.name for path in get_stored_responses(tmp_path)]
        assert names == ['000-001.json', '000.json', '001.json']

    def test_should_store_nothing_when_the_request_fails_outright(self, tmp_path):
        class FailingClient(NeverCalledClient):
            def get_completion(self, prompt, response_schema, attempt=FIRST_ATTEMPT):
                raise RuntimeError('http 400')

        config = get_config()
        client = get_caching_client(config, FailingClient(), tmp_path)
        with pytest.raises(RuntimeError):
            client.get_completion(PROMPT, SCHEMA)
        assert not list(tmp_path.rglob('*.json'))

    def test_should_store_the_request_beside_the_response(self, tmp_path):
        config = get_config()
        get_caching_client(config, FakeClient(), tmp_path).get_completion(PROMPT, SCHEMA)
        request_paths = list(tmp_path.rglob(REQUEST_FILENAME))
        assert len(request_paths) == 1
        stored = json.loads(request_paths[0].read_text(encoding='utf-8'))
        assert stored == get_request_body(config, PROMPT, SCHEMA)
        assert stored['messages'][0]['content'] == PROMPT

    def test_should_write_the_request_once_for_a_repeated_key(self, tmp_path):
        config = get_config()
        client = get_caching_client(config, FakeClient(), tmp_path)
        client.get_completion(PROMPT, SCHEMA)
        client.get_completion(PROMPT, SCHEMA)
        assert len(list(tmp_path.rglob(REQUEST_FILENAME))) == 1

    def test_should_ask_live_when_an_entry_is_unreadable(self, tmp_path):
        config = get_config()
        get_caching_client(config, FakeClient(), tmp_path).get_completion(PROMPT, SCHEMA)
        entry = get_stored_responses(tmp_path)[0]
        entry.write_text('{ truncated', encoding='utf-8')
        delegate = FakeClient(['{"starts": [1]}'])
        client = get_caching_client(config, delegate, tmp_path)
        assert client.get_completion(PROMPT, SCHEMA) == get_response('{"starts": [1]}')
        assert delegate.call_count == 1

    def test_should_not_write_the_replay_marker_to_disk(self, tmp_path):
        # It is added on the way out, so a stored entry stays the body as returned.
        config = get_config()
        client = get_caching_client(config, FakeClient(), tmp_path)
        client.get_completion(PROMPT, SCHEMA)
        client.get_completion(PROMPT, SCHEMA)
        entry = get_stored_responses(tmp_path)[0]
        assert REPLAYED_RESPONSE_KEY not in json.loads(entry.read_text(encoding='utf-8'))

    def test_should_record_which_model_asked(self, tmp_path):
        # The body carries the rendered prompt but not the version it came from,
        # and `model` is one string for two tasks, so neither says which sequence
        # model an entry belongs to.
        config = get_config()
        get_caching_client(config, FakeClient(), tmp_path).get_completion(PROMPT, SCHEMA)
        meta_paths = list(tmp_path.rglob(META_FILENAME))
        assert len(meta_paths) == 1
        assert json.loads(meta_paths[0].read_text(encoding='utf-8')) == {
            'task': 'reference_segmenter',
            'prompt_version': 'lines-v1',
            'response_shape': 'lines',
            'model': 'qwen/qwen3.5-9b',
            'provider': None,
            'endpoint': config.endpoint,
        }

    def test_should_keep_document_text_out_of_the_meta_file(self, tmp_path):
        # The one file in the cache that can be read without reading a manuscript.
        get_caching_client(get_config(), FakeClient(), tmp_path).get_completion(
            'a prompt holding the manuscript', SCHEMA
        )
        meta_text = next(tmp_path.rglob(META_FILENAME)).read_text(encoding='utf-8')
        assert 'manuscript' not in meta_text

    def test_should_group_entries_by_task(self, tmp_path):
        get_caching_client(get_config(), FakeClient(), tmp_path).get_completion(
            PROMPT, SCHEMA
        )
        get_caching_client(
            get_config(task='citation', response_shape='values'),
            FakeClient(), tmp_path
        ).get_completion('other prompt', SCHEMA)
        assert sorted(path.name for path in tmp_path.iterdir()) == [
            'citation', 'reference_segmenter'
        ]

    def test_should_let_one_task_be_cleared_on_its_own(self, tmp_path):
        # The per-model control that is deliberately not a config setting.
        config = get_config()
        get_caching_client(config, FakeClient(), tmp_path).get_completion(PROMPT, SCHEMA)
        shutil.rmtree(tmp_path / 'reference_segmenter')
        delegate = FakeClient()
        get_caching_client(config, delegate, tmp_path).get_completion(PROMPT, SCHEMA)
        assert delegate.call_count == 1

    def test_should_not_let_a_task_name_escape_the_cache_directory(self, tmp_path):
        cache_dir = tmp_path / 'cache'
        get_caching_client(
            get_config(task='../../escaped'), FakeClient(), cache_dir
        ).get_completion(PROMPT, SCHEMA)
        assert get_stored_responses(cache_dir)
        assert not list(tmp_path.glob('escaped'))

    def test_should_pass_validation_through(self, tmp_path):
        class RaisingClient(NeverCalledClient):
            def validate_configuration(self) -> None:
                raise RuntimeError('not reachable')

        client = get_caching_client(get_config(), RaisingClient(), tmp_path)
        with pytest.raises(RuntimeError, match='not reachable'):
            client.validate_configuration()


class TestConcurrentMisses:
    def test_should_leave_exactly_one_valid_entry(self, tmp_path):
        # All four miss and all four call, since waiting on a miss would cost
        # more than the duplicate does. They write the same entry, and the rename
        # is what keeps a reader from meeting half a file.
        started = threading.Barrier(4)
        config = get_config()
        cache = LlmResponseCache(str(tmp_path))

        class SlowClient(FakeClient):
            def get_completion(self, prompt, response_schema, attempt=FIRST_ATTEMPT):
                started.wait(timeout=5)
                return super().get_completion(prompt, response_schema, attempt)

        client = CachingLlmClient(config, SlowClient(), cache)
        results: List[Any] = []
        threads = [
            threading.Thread(
                target=lambda: results.append(client.get_completion(PROMPT, SCHEMA))
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert len(results) == 4
        entries = get_stored_responses(tmp_path)
        assert len(entries) == 1
        assert json.loads(entries[0].read_text(encoding='utf-8'))
        assert not list(tmp_path.rglob('.tmp-*'))


class TestGetResponseCache:
    def test_should_reuse_one_cache_per_directory(self, tmp_path):
        assert get_response_cache(str(tmp_path)) is get_response_cache(str(tmp_path))
        assert get_response_cache(str(tmp_path)) is not get_response_cache(
            str(tmp_path / 'other')
        )


class TestLlmModelImplWithACache:
    def test_should_not_wrap_the_client_when_no_directory_is_configured(self):
        delegate = FakeClient()
        model_impl = LlmModelImpl(get_config(), client=delegate)
        assert model_impl.client is delegate

    def test_should_produce_the_same_labels_from_a_warm_cache(self, tmp_path):
        config = get_config()
        live = LlmModelImpl(config, client=FakeClient(), response_cache_dir=str(tmp_path))
        expected = live.predict_labels([TOKENS], [feature_rows()])
        replay = LlmModelImpl(config, client=NeverCalledClient(), response_cache_dir=str(tmp_path))
        assert replay.predict_labels([TOKENS], [feature_rows()]) == expected

    def test_should_replay_a_malformed_then_parseable_sequence_as_a_success(self, tmp_path):
        config = get_config(max_malformed_response_retries=1)
        delegate = FakeClient(['not json at all', '{"starts": [0, 1]}'])
        live = LlmModelImpl(config, client=delegate, response_cache_dir=str(tmp_path))
        expected = live.predict_labels([TOKENS], [feature_rows()])
        assert delegate.call_count == 2

        replay = LlmModelImpl(config, client=NeverCalledClient(), response_cache_dir=str(tmp_path))
        assert replay.predict_labels([TOKENS], [feature_rows()]) == expected

    def test_should_keep_the_unparseable_body_a_decoder_fix_needs(self, tmp_path):
        config = get_config(max_malformed_response_retries=1)
        LlmModelImpl(
            config, client=FakeClient(['not json at all', '{"starts": [0, 1]}']),
            response_cache_dir=str(tmp_path)
        ).predict_labels([TOKENS], [feature_rows()])
        stored = [
            json.loads(path.read_text(encoding='utf-8'))
            for path in get_stored_responses(tmp_path)
        ]
        contents = [entry['choices'][0]['message']['content'] for entry in stored]
        assert 'not json at all' in contents

    def test_should_report_a_replayed_run_as_having_spent_nothing(self, tmp_path):
        # The whole point of the accounting: the stored body carries the original
        # call's tokens and credits, and this run paid neither.
        config = get_config()
        LlmModelImpl(config, client=FakeClient(), response_cache_dir=str(tmp_path)).predict_labels(
            [TOKENS], [feature_rows()]
        )

        def replay_run():
            start_request_llm_usage()
            LlmModelImpl(
                config, client=NeverCalledClient(), response_cache_dir=str(tmp_path)
            ).predict_labels([TOKENS], [feature_rows()])
            return get_request_llm_usage_header_value()

        usage = json.loads(contextvars.copy_context().run(replay_run) or '{}')
        assert usage['calls'] == 0
        assert usage['output_tokens'] == 0
        assert 'cost_credits' not in usage
        assert usage['replayed']['calls'] == 1
        assert usage['replayed']['output_tokens'] == 20

    def test_should_keep_the_resolved_provider_and_usage(self, tmp_path):
        config = get_config()
        LlmModelImpl(config, client=FakeClient(), response_cache_dir=str(tmp_path)).predict_labels(
            [TOKENS], [feature_rows()]
        )
        entry = get_stored_responses(tmp_path)[0]
        stored = json.loads(entry.read_text(encoding='utf-8'))
        assert stored['provider'] == 'venice'
        assert stored['usage']['completion_tokens'] == 20

    def test_should_not_reach_the_cache_directory_of_another_run(self, tmp_path):
        LlmModelImpl(
            get_config(), client=FakeClient(),
            response_cache_dir=str(tmp_path / 'one')
        ).predict_labels([TOKENS], [feature_rows()])
        delegate = FakeClient()
        LlmModelImpl(
            get_config(), client=delegate, response_cache_dir=str(tmp_path / 'two')
        ).predict_labels([TOKENS], [feature_rows()])
        assert delegate.call_count == 1
        assert os.path.isdir(tmp_path / 'two')
