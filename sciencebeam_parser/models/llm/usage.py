import contextvars
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


USAGE_HEADER_NAME = 'X-ScienceBeam-LLM-Usage'

# Set on a body the response cache replayed, so this module can tell a call that
# spent from one that did not. Added when the entry is read, so nothing stored on
# disk carries it, and no decoder looks at it.
REPLAYED_RESPONSE_KEY = 'sciencebeam_replayed_from_cache'


@dataclass
class LlmUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # A subset of output_tokens, as the provider reports it, never an addition to
    # it: prompt_tokens + completion_tokens is the reported total_tokens.
    reasoning_tokens: int = 0
    # A subset of input_tokens, served from the provider's own prefix cache. Cost
    # is already net of the discount, so two runs with the same input tokens can
    # differ in credits.
    cached_input_tokens: int = 0
    peak_output_tokens: int = 0
    cost: Optional[float] = None
    models: list = field(default_factory=list)
    providers: list = field(default_factory=list)

    def add_usage(self, usage: Mapping[str, Any]) -> None:
        output_tokens = _get_int(usage.get('completion_tokens'))
        self.calls += 1
        self.input_tokens += _get_int(usage.get('prompt_tokens'))
        self.output_tokens += output_tokens
        self.reasoning_tokens += _get_int(
            (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        )
        self.cached_input_tokens += _get_int(
            (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        )
        self.peak_output_tokens = max(self.peak_output_tokens, output_tokens)
        cost = usage.get('cost')
        if cost is not None:
            self.cost = (self.cost or 0.0) + float(cost)

    def add_model(self, model: Optional[str], provider: Optional[str]) -> None:
        for value, values in ((model, self.models), (provider, self.providers)):
            if value and value not in values:
                values.append(value)

    def to_dict(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            'calls': self.calls,
            'input_tokens': self.input_tokens,
            'cached_input_tokens': self.cached_input_tokens,
            'output_tokens': self.output_tokens,
            'reasoning_tokens': self.reasoning_tokens,
            'peak_output_tokens': self.peak_output_tokens,
        }
        # Absent rather than zero: a self-hosted endpoint reports no cost, and a
        # zero would read as free.
        if self.cost is not None:
            entry['cost_credits'] = self.cost
        if self.models:
            entry['models'] = list(self.models)
        if self.providers:
            entry['providers'] = list(self.providers)
        return entry


def _get_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _replayed_entry(usage: Optional[LlmUsage]) -> Dict[str, Any]:
    """Absent unless something was replayed, so a cold run's record is unchanged.

    What those responses cost when they were generated, which is not a forecast of
    a cold run: prices and providers move, and generation is not reproducible, so
    a fresh run would not make exactly the same calls.
    """
    if usage is None or not usage.calls:
        return {}
    return {'replayed': usage.to_dict()}


class LlmUsageAccumulator:
    """What one request spent, in total and per task.

    Replayed calls are counted apart rather than added in: they cost nothing now,
    and folding their stored `usage` into the totals would report a cold run's
    spend as this run's.

    Mutable and lock-guarded, because the batches of one document are answered
    from a thread pool and several requests are in flight at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = LlmUsage()
        self._replayed = LlmUsage()
        self._by_task: Dict[str, LlmUsage] = {}
        self._replayed_by_task: Dict[str, LlmUsage] = {}

    @property
    def calls(self) -> int:
        with self._lock:
            return self._total.calls + self._replayed.calls

    def add_response(
        self, task: str, response_json: Mapping[str, Any], replayed: bool = False
    ) -> None:
        usage = response_json.get('usage') or {}
        model = response_json.get('model')
        provider = response_json.get('provider')
        total, by_task = (
            (self._replayed, self._replayed_by_task) if replayed
            else (self._total, self._by_task)
        )
        with self._lock:
            for entry in (total, by_task.setdefault(task, LlmUsage())):
                entry.add_usage(usage)
                entry.add_model(model, provider)

    def to_dict(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._total.calls and not self._replayed.calls:
                return None
            tasks = sorted(set(self._by_task) | set(self._replayed_by_task))
            return {
                **self._total.to_dict(),
                **_replayed_entry(self._replayed),
                'by_task': {
                    task: {
                        **self._by_task.get(task, LlmUsage()).to_dict(),
                        **_replayed_entry(self._replayed_by_task.get(task)),
                    }
                    for task in tasks
                },
            }


REQUEST_LLM_USAGE: contextvars.ContextVar[Optional[LlmUsageAccumulator]] = (
    contextvars.ContextVar('request_llm_usage', default=None)
)


def start_request_llm_usage() -> LlmUsageAccumulator:
    """Bind a fresh accumulator to the calling task's context.

    Deliberately not reset afterwards: an exception handler runs above whatever
    binds this and has to still be able to read it, and every request task and
    every threadpool call gets its own copy of the context anyway.
    """
    accumulator = LlmUsageAccumulator()
    REQUEST_LLM_USAGE.set(accumulator)
    return accumulator


def record_llm_usage(task: str, response_json: Mapping[str, Any]) -> None:
    accumulator = REQUEST_LLM_USAGE.get()
    if accumulator is not None:
        accumulator.add_response(
            task, response_json,
            replayed=bool(response_json.get(REPLAYED_RESPONSE_KEY))
        )


def get_request_llm_usage_header_value() -> Optional[str]:
    accumulator = REQUEST_LLM_USAGE.get()
    if accumulator is None:
        return None
    usage = accumulator.to_dict()
    return None if usage is None else json.dumps(usage)
