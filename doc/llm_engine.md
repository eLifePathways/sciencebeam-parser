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
  provider: 'venice'               # pinned; routing fails closed without a match
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
`temperature`, `timeout_seconds`, `max_output_tokens`, `max_attempts`, `extra_body`,
`max_references_per_request`, `record_trace_content`, `warn_input_lines`, `max_input_lines`,
`unanswered_reference_raises`, `max_missing_reference_retries`,
`max_malformed_response_retries`.

### What the segmenter is told to skip

The region it receives is whatever segmentation labelled `<references>`, which is sometimes a table
or body text. The prompt says to report only bibliographic references, and an empty answer is
accepted — "no references in this region" is a valid conclusion.

### Response shapes for the reference segmenter

`lines` returns a line number per reference. `evidence` returns the line number **and** the first
words on it, so a wrong quote costs a check rather than the reference. `lines` is the default;
`evidence` costs about four times the output tokens and earns it on a weaker model than the one
shipped here.

A quote is accepted against the line it names or the one below it, since models name the line
holding the reference number while quoting the words underneath. Anything else counts on
`sciencebeam.evidence_mismatches` and logs a warning; `evidence_mismatch_raises` makes the check
load-bearing.

### Batching (citation)

The citation model receives every reference of a document at once and batches them, each numbered in
the prompt, with one entry per reference expected in the response. Values are located within their
own reference's tokens only, so a value belonging to a neighbour is dropped rather than labelled.

```yaml
max_references_per_request: 10   # references per call
max_concurrent_requests: 4       # calls in flight at once
```

Batching is worth roughly 2x over one call per reference, and concurrency about 2x again up to 4
calls; beyond that provider throughput is the bound. A larger batch does not generate fewer tokens —
it generates them in one long stream that cannot be overlapped, and long generations draw provider
timeouts, so more smaller batches is the direction that helps.

A reference the model skips, answers twice, or numbers outside the batch is left unlabelled and
counted on `sciencebeam.unanswered_references`; the rest of the batch stands. Skipped ones are asked
for again in a batch containing only them (`max_missing_reference_retries`, default 1), stopping as
soon as a round recovers nothing. `unanswered_reference_raises` makes what remains fatal.

An answer that hits `max_output_tokens` is unusable, since a field cut off cannot be told from one
never sent, so the batch is halved and asked again down to a single reference. The default of 16000
allows about 4x the input, which is what this shape returns.

Retries use a jittered backoff and honour `Retry-After`, spending longer on a 429 than on a server
error. A 429 from OpenRouter is its pooled allocation with the provider rather than a limit on the
key, so a provider key on the account — or `endpoint` pointed straight at the provider — buys a
separate limit. Lower `max_concurrent_requests` if they persist.

Spans from worker threads are siblings rather than nested, and the service may already handle
documents concurrently, which multiplies with this.

## What it guarantees

No text reaches a document that was not in the source. Under `lines` the model returns line numbers
and never text at all. Under `values` it returns text, and every value is located back in the token
sequence. Either way `Model._iter_flat_label_model_data_lists_to` independently rejects any result
whose tokens are not the input tokens.

**A response the engine cannot parse is asked for again and then raises; a claim the engine cannot
honour is dropped and counted.** Bad JSON or a malformed entry are the first: re-requested with the
same prompt `max_malformed_response_retries` times (default 1), since generation is not
bit-reproducible even at temperature 0. A value that is not in the reference, or one a previous
field already claimed, are the second — dropped with a warning naming the reference, label and text,
counted on `sciencebeam.dropped_fields`. Dropping keeps the guarantee rather than weakening it: a
discarded value never becomes a label, so the guarantee constrains what a response can *add* and not
what it may lose.

The `*_raises` settings make each loss strict instead. Those are not retried — a strictness setting
firing is a decision, and the same answer would come back.

There is no fallback to a CRF engine and no partial labelling: a score is only meaningful if every
label came from the model under test.

The `citation` label vocabulary is read from the model's own label map rather than restated in the
prompt source, so it cannot drift from the labels the extractor understands.

Every request enforces zero data retention — `zdr`, `data_collection: deny`,
`allow_fallbacks: false`, `require_parameters: true`, and `only: [provider]` when pinned. A `:free`
model id is refused at load, because that tier requires allowing training on prompts.

## Tracing (optional)

Spans follow OpenTelemetry's GenAI semantic conventions — `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens` and so on — so they are meaningful in any OTLP
backend. [OpenInference](https://github.com/Arize-ai/openinference) names are emitted alongside
them, so a backend that reads those rather than the GenAI conventions still renders the span as an
LLM call; [Phoenix](https://phoenix.arize.com/) is the one used in development.

```sh
make dev-install                                       # includes the telemetry extra
make docker-start-telemetry                            # Phoenix on http://localhost:6006
export SCIENCEBEAM_PARSER__PROFILE=llm_references      # or llm_citation, llm_reference_segmenter
make dev-start-with-telemetry                          # the host parser, endpoint already set
```

Phoenix runs as a compose service behind the `telemetry` profile, so a plain `docker-start` does
not bring an observability server up with it. The image is pinned in
`docker-compose.override.yml`; bump it deliberately rather than tracking `latest`. Traces persist in a named volume across restarts;
`make docker-stop-telemetry` removes the container and keeps them, and
`make docker-logs-telemetry` follows its logs.

`dev-start-with-telemetry` is the host parser with the collector endpoint already set; it does not
choose a profile, which is set the usual way and is not specific to this engine. A parser running
inside compose would point at `http://phoenix:6006` instead, since `localhost` there is the
container.

The endpoint variable is what turns emission on, so a parser already running has to be restarted to
start tracing.

Without the extra installed, or without a collector endpoint set, nothing is emitted and the engine
behaves identically. An existing tracer provider is left alone rather than replaced.

Configuration is plain OTLP: `OTEL_EXPORTER_OTLP_ENDPOINT` or
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, read by the exporter itself. Nothing in the engine names a
backend — Phoenix is only what listens in development, and any OTLP collector works in its place.
`PHOENIX_COLLECTOR_ENDPOINT` is *not* read: it is Phoenix's own variable, and honouring it would put
one backend's configuration into the engine.

The response body is attached to the span even when it fails to decode, which is the point: a
truncated or malformed response is visible rather than inferred from an exception.

A call made while serving a request nests under a `process_document` span naming the document, so a
trace says which document it is about. The log line for a completed call names its trace id, which
is what joins the log to the prompt and response on the span.

Tracing is not specific to this engine and lives in `utils/telemetry.py`.

**A span carrying prompt text is a copy of manuscript text.** Sending it to a collector on localhost
is not a new disclosure when the same text is already going to the model, but sending it anywhere
else is. Set `record_trace_content: false` in the model config to keep the metrics and drop the
text.

## Input size

What the engine receives is whatever the *segmentation* model labelled `<references>`, which is not
necessarily a reference list, so `sciencebeam.input_lines` and `sciencebeam.input_tokens` are on
every span to make an oversized region visible.

Above `warn_input_lines` (default 300) the engine logs a warning naming the count. `max_input_lines`
(default 0, off) raises instead, for a run where failing fast is wanted.

## Choosing a provider

The same model id served by two providers is not the same service. For
`qwen/qwen3.5-9b`, measured on 33 references in 4 batches:

| provider | quantisation | seconds | token-acc | reported uptime (30m / 1d) |
| --- | --- | --- | --- | --- |
| Venice | fp8 | **12** | 0.860 | 100% / 100% |
| SiliconFlow | fp8 | 39 | 0.867 | 96% / 99% |

Venice is shipped: 3.3x faster at the same quantisation and price, for a token-accuracy difference
inside the run-to-run spread.

`GET /api/v1/models/{author}/{slug}/endpoints` lists the providers for a model with their
quantisation, pricing and reported uptime, which move over time. Latency and throughput usually come
back `None` there, and not every provider is reachable under the zero-retention pin, so a candidate
has to be measured rather than looked up.

## CI

The benchmark workflow passes `OPENROUTER_API_KEY` from the `benchmark` environment into the parser
container as a bare `-e OPENROUTER_API_KEY`, so the value never appears in a command line and an
unset secret leaves it unset rather than empty. Every non-LLM profile ignores it, so it is passed
unconditionally.

Select the engine the same way as any other configuration: a `profile:llm_references` label on the
PR, or the `profile` input on `workflow_dispatch`.

`SCIENCEBEAM_PARSER__PRELOAD_ON_STARTUP=true` is already set there, which means an LLM profile
validates its endpoint and model id while the container starts. A missing key or an unreachable
endpoint therefore fails at "Wait for parser" with the reason in the container logs, rather than
part-way through a run.

**PLOS cannot be combined with an LLM profile.** `benchmarks/run.py` refuses it, because provider
zero-retention does not cover OpenRouter itself and those manuscripts are not redistributable. The
workflow adds `--include-corpus plos-manuscripts` automatically on `main`, so an LLM profile there
fails rather than sending private manuscripts to a third party. Point the engine at a self-hosted
endpoint if that corpus needs covering.

Nothing is traced in CI: no collector endpoint is set, so the engine emits no spans.

Unit tests reach no network. The engine's tests are fixture-driven and the client is a `Protocol`, so
the ordinary CI job needs no secret at all.
