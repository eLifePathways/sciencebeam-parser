import hashlib
import json
import logging
import os
import tempfile
import threading
from functools import lru_cache
from typing import Any, Mapping, Optional, Tuple

from sciencebeam_parser.models.llm.client import (
    FIRST_ATTEMPT,
    LlmCompletionClient,
    get_request_body
)
from sciencebeam_parser.models.llm.config import LlmEngineConfig
from sciencebeam_parser.models.llm.usage import REPLAYED_RESPONSE_KEY
from sciencebeam_parser.utils.telemetry import set_current_span_attribute


LOGGER = logging.getLogger(__name__)

CACHE_HIT_ATTRIBUTE = 'sciencebeam.llm.cache_hit'

REQUEST_FILENAME = 'request.json'


def get_request_key(endpoint: str, request_body: Mapping[str, Any]) -> str:
    """The request as sent, hashed.

    Derived from the body rather than from a list of parameters believed to
    matter, so a parameter added to the body later is covered without anyone
    remembering to add it here.
    """
    serialised = json.dumps(
        {'endpoint': endpoint, 'request': request_body},
        sort_keys=True, ensure_ascii=False, separators=(',', ':')
    )
    return hashlib.sha256(serialised.encode('utf-8')).hexdigest()


def _write_atomically(path: str, value: Any) -> None:
    """A reader meets a whole entry or no entry, however many workers are writing."""
    handle, temp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.tmp-')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as temp_file:
            json.dump(value, temp_file, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_entry_name(attempt: Tuple[int, ...]) -> str:
    """The file an attempt is stored as: `000` the first ask, `001` the same
    question after an unparseable answer, `000-001` a re-ask for the references
    a batch left out."""
    return '-'.join(f'{index:03d}' for index in attempt)


class LlmResponseCache:
    """Completions on disk, one per attempt, so a run replays as it happened.

    Holds document text: a prompt is the manuscript region, and a `values`
    response is field values copied out of it.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        self._lock = threading.Lock()
        self._has_logged_a_hit = False
        self.hit_count = 0
        self.miss_count = 0
        LOGGER.info(
            'llm response cache enabled at %s; it holds document text, and'
            ' clearing it is deleting the directory',
            self.cache_dir
        )

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.cache_dir!r})'

    def get_summary(self) -> str:
        return f'{self.hit_count} replayed, {self.miss_count} live'

    def get_response(
        self, key: str, attempt: Tuple[int, ...]
    ) -> Optional[Mapping[str, Any]]:
        path = self._get_response_path(key, attempt)
        try:
            with open(path, 'r', encoding='utf-8') as response_file:
                response = json.load(response_file)
        except FileNotFoundError:
            self._count(hit=False)
            return None
        except (OSError, ValueError) as exc:
            LOGGER.warning('llm cache entry %s is unreadable, asking live: %s', path, exc)
            self._count(hit=False)
            return None
        self._count(hit=True)
        return response

    def put(
        self,
        key: str,
        attempt: Tuple[int, ...],
        request_body: Mapping[str, Any],
        response: Mapping[str, Any]
    ) -> None:
        key_dir = self._get_key_dir(key)
        os.makedirs(key_dir, exist_ok=True)
        request_path = os.path.join(key_dir, REQUEST_FILENAME)
        if not os.path.exists(request_path):
            # Once per key, so a directory named by a one-way hash can still be
            # read back to the question it answers.
            _write_atomically(request_path, request_body)
        _write_atomically(self._get_response_path(key, attempt), response)

    def _get_key_dir(self, key: str) -> str:
        return os.path.join(self.cache_dir, key[:2], key)

    def _get_response_path(self, key: str, attempt: Tuple[int, ...]) -> str:
        return os.path.join(self._get_key_dir(key), get_entry_name(attempt) + '.json')

    def _count(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.hit_count += 1
                should_log = not self._has_logged_a_hit
                self._has_logged_a_hit = True
            else:
                self.miss_count += 1
                should_log = False
        if should_log:
            LOGGER.info(
                'llm response cache is warm; this run replays stored answers and'
                ' spends less than a cold one would'
            )


@lru_cache(maxsize=None)
def get_response_cache(cache_dir: str) -> LlmResponseCache:
    """One cache per directory for the life of the process.

    Two models configured against the same directory then share the counts,
    which makes them the run's rather than one model's. It holds no other state:
    which entry a call reads comes from the caller, so nothing here has to be
    scoped to a run or reset between documents.
    """
    return LlmResponseCache(cache_dir)


class CachingLlmClient:
    """Replays completions from disk, and lets everything else through.

    Wraps the client rather than the transport, so a 429 or a connection error is
    still retried live and never stored, and a clean response is stored even when
    it goes on to fail decoding — that body is what a decoder fix is developed
    against.
    """

    def __init__(
        self,
        config: LlmEngineConfig,
        delegate: LlmCompletionClient,
        cache: LlmResponseCache
    ):
        self.config = config
        self.delegate = delegate
        self.cache = cache

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.delegate!r}, {self.cache!r})'

    def validate_configuration(self) -> None:
        self.delegate.validate_configuration()

    def get_completion(
        self,
        prompt: str,
        response_schema: Mapping[str, Any],
        attempt: Tuple[int, ...] = FIRST_ATTEMPT
    ) -> Mapping[str, Any]:
        request_body = get_request_body(self.config, prompt, response_schema)
        key = get_request_key(self.config.endpoint, request_body)
        cached = self.cache.get_response(key, attempt)
        set_current_span_attribute(CACHE_HIT_ATTRIBUTE, cached is not None)
        if cached is not None:
            LOGGER.debug(
                'llm cache replaying %s %s', key[:12], get_entry_name(attempt)
            )
            # Marked on the way out, so usage counts it as replayed rather than
            # adding the stored call's tokens and credits to what this run spent.
            return {**cached, REPLAYED_RESPONSE_KEY: True}
        # An exception stores nothing, so a failure is asked again live.
        response = self.delegate.get_completion(prompt, response_schema, attempt)
        self.cache.put(key, attempt, request_body, response)
        return response
