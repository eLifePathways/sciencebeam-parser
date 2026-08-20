# LLM engine (experimental)

A third sequence-model engine alongside `wapiti` and `delft`, serving the `reference_segmenter` and
`citation` models. It is **opt-in**: the shipped default profile stays `grobid_crf`, and a default
install acquires no network dependency and no credential requirement.

## Using it

```sh
export OPENROUTER_API_KEY=...            # or SCIENCEBEAM_LLM_API_KEY
export SCIENCEBEAM_PARSER__PROFILE=llm_reference_segmenter
```

Override the profile by environment rather than editing `profile:` in `config.yml`. The shipped
default is asserted by test, so changing it there fails the build. Env keys use the
`SCIENCEBEAM_PARSER__` prefix with `__` between levels, so a single setting can be overridden the
same way — for example
`SCIENCEBEAM_PARSER__SEQUENCE_MODEL_PROFILES__LLM_REFERENCE_SEGMENTER__CITATION__MODEL`.

Three profiles, all extending `grobid_crf_0_9_0` so every other model stays on wapiti:

| profile | replaces |
| --- | --- |
| `llm_reference_segmenter` | `reference_segmenter` |
| `llm_citation` | `citation` |
| `llm_references` | both |

One per model matters for attribution: when a run fails, the per-model profiles say which model did
it without having to read a stack trace.

## Configuration

```yaml
reference_segmenter:
  engine: 'llm'
  task: 'reference_segmenter'      # selects the prompt and the feature layout
  response_shape: 'lines'          # line numbers where each reference begins
  model: 'qwen/qwen3.5-9b'
  provider: 'siliconflow'          # pinned; routing fails closed without a match
  prompt_version: 'lines-v1'       # sciencebeam_parser/models/llm/prompts/<task>/<version>.md
  reasoning: 'off'                 # models that think by default must be told not to
citation:
  engine: 'llm'
  task: 'citation'
  response_shape: 'values'         # field values, located back in the token sequence
  model: 'qwen/qwen3.5-9b'
  provider: 'siliconflow'
  prompt_version: 'values-v1'
  reasoning: 'off'
```

`response_shape` is configuration rather than a fixed choice, because the best shape differs by
model and by task and moves with each new checkpoint. Comparing shapes is therefore defining a
second profile and running the benchmark, not building a second evaluation route.

Also accepted: `endpoint` (any OpenAI-compatible base URL, so a self-hosted vLLM works),
`temperature`, `timeout_seconds`, `max_output_tokens`, `max_attempts`, `extra_body`.

## What it guarantees

No text reaches a document that was not in the source. Under `lines` the model returns line numbers
and never text at all. Under `values` it returns text, and every value is located back in the token
sequence — one that cannot be found, or that a previous field already claimed, raises. Either way
`Model._iter_flat_label_model_data_lists_to` independently rejects any result whose tokens are not
the input tokens.

The `citation` label vocabulary is read from the model's own label map rather than restated in the
prompt source, so it cannot drift from the labels the extractor understands.

Every request enforces zero data retention — `zdr`, `data_collection: deny`,
`allow_fallbacks: false`, `require_parameters: true`, and `only: [provider]` when pinned. A `:free`
model id is refused at load, because that tier requires allowing training on prompts.

A response that cannot be decoded raises. There is no fallback to a CRF engine and no partial
labelling: a score is only meaningful if every label came from the model under test.

A response cut off at the output limit raises `LlmTruncatedResponseError` naming
`finish_reason`, the completion token count and the task, rather than surfacing as a JSON parse
error. Raise `max_output_tokens`, or send less per request.
