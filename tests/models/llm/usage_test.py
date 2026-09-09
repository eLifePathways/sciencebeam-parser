import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from sciencebeam_parser.models.llm.usage import (
    LlmUsageAccumulator,
    get_request_llm_usage_header_value,
    record_llm_usage,
    start_request_llm_usage
)


TASK_1 = 'citation'
TASK_2 = 'reference_segmenter'


def _response(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning_tokens: int = 0,
    cost=0.0004,
    model: str = 'qwen/qwen3.5-9b',
    provider: str = 'SiliconFlow'
) -> dict:
    usage: dict = {
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'completion_tokens_details': {'reasoning_tokens': reasoning_tokens},
    }
    if cost is not None:
        usage['cost'] = cost
    return {'usage': usage, 'model': model, 'provider': provider}


class TestLlmUsageAccumulator:
    def test_should_report_nothing_without_a_call(self):
        assert LlmUsageAccumulator().to_dict() is None

    def test_should_sum_tokens_and_cost_across_calls(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, _response())
        accumulator.add_response(TASK_1, _response())
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['calls'] == 2
        assert usage['input_tokens'] == 200
        assert usage['output_tokens'] == 100
        assert usage['cost_credits'] == 0.0008

    def test_should_keep_the_peak_of_a_single_call_rather_than_the_sum(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, _response(completion_tokens=50))
        accumulator.add_response(TASK_1, _response(completion_tokens=16000))
        accumulator.add_response(TASK_1, _response(completion_tokens=50))
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['output_tokens'] == 16100
        assert usage['peak_output_tokens'] == 16000

    def test_should_treat_reasoning_tokens_as_part_of_the_output_tokens(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(
            TASK_1, _response(completion_tokens=500, reasoning_tokens=400)
        )
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['output_tokens'] == 500
        assert usage['reasoning_tokens'] == 400

    def test_should_leave_cost_absent_rather_than_zero_when_unreported(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, _response(cost=None))
        usage = accumulator.to_dict()
        assert usage is not None
        assert 'cost_credits' not in usage
        assert usage['output_tokens'] == 50

    def test_should_count_a_response_without_usage_as_a_call(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, {'model': 'qwen/qwen3.5-9b'})
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['calls'] == 1
        assert usage['input_tokens'] == 0
        assert 'cost_credits' not in usage

    def test_should_break_usage_down_by_task(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, _response(completion_tokens=50))
        accumulator.add_response(TASK_2, _response(completion_tokens=900, model='other'))
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['output_tokens'] == 950
        assert usage['by_task'][TASK_1]['output_tokens'] == 50
        assert usage['by_task'][TASK_2]['output_tokens'] == 900
        assert usage['by_task'][TASK_2]['models'] == ['other']

    def test_should_record_the_resolved_provider(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, _response(provider='Venice'))
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['providers'] == ['Venice']

    def test_should_be_safe_to_add_to_from_several_threads(self):
        accumulator = LlmUsageAccumulator()
        barrier = threading.Barrier(4, timeout=5)

        def add(_):
            barrier.wait()
            accumulator.add_response(TASK_1, _response())

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(add, range(4)))

        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['calls'] == 4
        assert usage['input_tokens'] == 400


class TestRequestLlmUsage:
    def test_should_record_nothing_without_an_accumulator(self):
        def call_without_binding():
            record_llm_usage(TASK_1, _response())
            return get_request_llm_usage_header_value()

        assert contextvars.copy_context().run(call_without_binding) is None

    def test_should_have_no_header_value_when_no_call_was_made(self):
        def bind_only():
            start_request_llm_usage()
            return get_request_llm_usage_header_value()

        assert contextvars.copy_context().run(bind_only) is None

    def test_should_report_what_the_request_spent_as_json(self):
        def spend():
            start_request_llm_usage()
            record_llm_usage(TASK_1, _response())
            return get_request_llm_usage_header_value()

        header_value = contextvars.copy_context().run(spend)
        assert header_value
        assert json.loads(header_value)['input_tokens'] == 100

    def test_should_keep_concurrent_requests_apart(self):
        barrier = threading.Barrier(2, timeout=5)

        def request(call_count: int) -> str:
            start_request_llm_usage()
            barrier.wait()
            for _ in range(call_count):
                record_llm_usage(TASK_1, _response())
            header_value = get_request_llm_usage_header_value()
            assert header_value
            return header_value

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda count: contextvars.copy_context().run(request, count),
                [3, 1]
            ))

        assert [json.loads(result)['calls'] for result in results] == [3, 1]

    def test_should_reach_the_creator_when_mutated_from_a_copied_context(self):
        def request() -> int:
            accumulator = start_request_llm_usage()
            contextvars.copy_context().run(record_llm_usage, TASK_1, _response())
            return accumulator.calls

        assert contextvars.copy_context().run(request) == 1


class TestProviderReportedShape:
    """Anchored on real OpenRouter responses: `prompt_tokens + completion_tokens`
    is the reported `total_tokens`, so reasoning and cached tokens are breakdowns
    of those two rather than additions to them."""
    GPT_OSS_RESPONSE = {
        'model': 'openai/gpt-oss-120b',
        'provider': 'AkashML',
        'usage': {
            'prompt_tokens': 81,
            'prompt_tokens_details': {'cached_tokens': 64, 'cache_write_tokens': 0},
            'completion_tokens': 42,
            'completion_tokens_details': {'reasoning_tokens': 20},
            'total_tokens': 123,
            'cost': 9.57e-06,
            'cost_details': {'upstream_inference_cost': 9.57e-06},
        },
    }

    def test_should_not_add_reasoning_tokens_to_the_output_tokens(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, self.GPT_OSS_RESPONSE)
        usage = accumulator.to_dict()
        assert usage is not None
        reported = self.GPT_OSS_RESPONSE['usage']
        assert usage['input_tokens'] + usage['output_tokens'] == reported['total_tokens']
        assert usage['reasoning_tokens'] == 20

    def test_should_record_provider_cached_input_tokens_as_part_of_the_input(self):
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, self.GPT_OSS_RESPONSE)
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['input_tokens'] == 81
        assert usage['cached_input_tokens'] == 64

    def test_should_record_what_a_truncated_response_spent(self):
        truncated = {
            'model': 'openai/gpt-oss-120b',
            'provider': 'AkashML',
            'choices': [{'message': {'content': ''}, 'finish_reason': 'length'}],
            'usage': {
                'prompt_tokens': 81,
                'completion_tokens': 16,
                'completion_tokens_details': {'reasoning_tokens': 10},
                'total_tokens': 97,
                'cost': 5.15e-06,
            },
        }
        accumulator = LlmUsageAccumulator()
        accumulator.add_response(TASK_1, truncated)
        usage = accumulator.to_dict()
        assert usage is not None
        assert usage['output_tokens'] == 16
        assert usage['cost_credits'] == 5.15e-06
