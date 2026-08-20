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
`temperature`, `timeout_seconds`, `max_output_tokens`, `max_attempts`, `extra_body`,
`max_references_per_request`, `record_trace_content`, `warn_input_lines`, `max_input_lines`.

### What the segmenter is told to skip

The region it receives is whatever segmentation labelled `<references>`, which is sometimes a table
or body text (see
`.project-notes/references-region-includes-non-reference-content.md`). The prompt therefore says to
report only actual bibliographic references and to return an empty list if none of the lines are
references — deciding what is a reference is the segmenter's job, so this is not the same as papering
over the upstream defect in the citation prompt.

An empty answer is accepted rather than rejected, since "no references in this region" is a valid
thing for the model to conclude.

### Response shapes for the reference segmenter

`lines` returns a line number per reference. `evidence` returns the line number **and** the first
words on it — the number is still the answer, the words are only a check on it, so a wrong quote
costs a check rather than the reference.

The pilot behind this found `evidence` worth about 0.19 `partial_list` on a 12B and 0.003 on the 9B
shipped here, at roughly four times the output tokens, which is why `lines` is the default. On one
real reference list both found 33 references against a gold of 33, in 5.3s and 6.9s. `evidence` earns
its cost on a weaker model, not on a capable one.

A quote is accepted against the line it names or the one below it, since models name the line
holding the reference number while quoting the words underneath. Anything else increments
`sciencebeam.evidence_mismatches` on the span and logs a warning; set `evidence_mismatch_raises` to
make the check load-bearing instead. It is off by default because a mismatch is a well-formed answer
whose evidence disagrees, not a protocol violation — and in the pilot a 12B mismatched on 158 of 210
references while the 9B mismatched on 1 of 118.

### Batching (citation)

`processor.py` hands the engine every reference of a document at once, so the citation model batches
them: `max_references_per_request` (default 10) references per call, each numbered in the prompt,
with one entry per reference required in the response. Requiring an answer for every reference sent
makes a merged or omitted reference fail validation rather than score, and values are located within
their own reference's tokens only — which also catches the model attributing one reference's author
to another, a mistake a flat field list would have matched against the whole document and labelled
silently. Lowering the bound is the lever if it happens often.

Both settings are in `config.yml` rather than only in code, since they are the knobs worth turning:

```yaml
max_references_per_request: 10   # references per call
max_concurrent_requests: 4       # calls in flight at once
```

Measured on 10 references: one batched call took 22s against 48s for ten single calls, at token
accuracy 0.906 against 0.904 — twice as fast at no cost in accuracy. On 33 references in 4 batches,
concurrency took 111s down to 58s with accuracy unchanged.

**Raising the batch size is the wrong lever.** Decode cost is linear in output tokens however they
are grouped, so a bigger batch does not generate less — it generates the same amount in one long
stream that cannot be overlapped, and long generations are what draw provider timeouts (one cost
about five minutes in retry). More, smaller batches running concurrently is the direction that helps.

Lower `max_concurrent_requests` if the provider answers with 429s; the client retries them with
backoff, but a rate-limited provider can make concurrency a net loss. Note also that spans from
worker threads are siblings rather than nested, and that the service may already handle documents
concurrently, which multiplies with this.

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

**A response the engine cannot parse raises; a claim the engine cannot honour is dropped and
counted.** Bad JSON, a missing reference entry, an index out of range are the first. An invented
value, or two fields over one span, are the second — dropped with a warning naming the reference,
label and text, counted on `sciencebeam.dropped_fields`.

Dropping satisfies the guarantee rather than weakening it: a discarded value never becomes a label,
so no text reaches the document that was not in the source. Raising would be stronger than the
guarantee needs, and it destroys every reference in a document over one invented field — which is
common, because the region handed over is often not a reference list. Set `dropped_field_raises` for
a run that wants strictness.

A response cut off at the output limit raises `LlmTruncatedResponseError` naming
`finish_reason`, the completion token count and the task, rather than surfacing as a JSON parse
error. Raise `max_output_tokens`, or send less per request.

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

**A span carrying prompt text is a copy of manuscript text.** Sending it to a collector on localhost
is not a new disclosure when the same text is already going to the model, but sending it anywhere
else is. Set `record_trace_content: false` in the model config to keep the metrics and drop the
text.

## Input size

`sciencebeam.input_lines` and `sciencebeam.input_tokens` are on every span, because what the engine
receives is whatever the *segmentation* model labelled `<references>` — not necessarily a reference
list. A region of a thousand lines is a mislabelled region, and it degrades the `wapiti` path
identically, so it is worth being able to see and filter on.

Above `warn_input_lines` (default 300) the engine logs a warning naming the count. `max_input_lines`
(default 0, off) raises instead, for a run where failing fast is wanted — off by default because
raising would fail exactly the documents where the CRF path produces poor output, which flatters a
comparison rather than informing it.
