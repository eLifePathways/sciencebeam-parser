# Profiles

A profile names a set of sequence models and the processor settings that go with
them. `config.yml` declares them under `profiles:`, and `profile:` selects the
one a deployment serves — `grobid_crf` as shipped, overridable with
`SCIENCEBEAM_PARSER__PROFILE`.

## Selecting a profile per request

Every route that serves a document takes a `profile` query parameter:

```bash
curl --fail --show-error \
  --header 'Accept: application/tei+xml' \
  --form "input=@$PWD/test-data/minimal-example.pdf;filename=test.pdf" \
  'http://localhost:8080/api/processFulltextDocument?profile=llm_references'
```

A request that names nothing is served the deployment's own profile, so adding
the parameter changes nothing for an existing caller.

The parameter is accepted on `/convert`, `/pdfalto`, the `/models/*` endpoints
and the GROBID-compatible endpoints. On the last of those it is a deliberate
departure from the API being mimicked: no GROBID client sends it, and a route
that ignored it would make a comparison read as a null result rather than as a
mistake.

## What a deployment will serve

Only the deployment's own profile is selectable unless the config says
otherwise:

```yaml
selectable_profiles:
  - wapiti_scielo_preprints_ore
  - llm_references
```

or by environment, which takes a YAML list:

```bash
export SCIENCEBEAM_PARSER__SELECTABLE_PROFILES='[wapiti_scielo_preprints_ore, llm_references]'
```

Naming a profile that is not selectable, or that does not exist, is a `400`
listing what is available.

Profiles overlap — the `wapiti_*` profiles differ from their base by one model
of ten — and a model is shared by every profile whose configuration for it is
identical, so serving a second profile usually costs one or two models rather
than ten. `max_loaded_models` bounds how many distinct models one process will
hold across every profile it has served, 40 by default. A request for a profile
that would take the process over the bound is refused with a `503` rather than
loading it; what is already loaded keeps being served.

The first request for a profile pays for loading its models, which for a delft
profile includes downloading and converting the artefacts. `preload_on_startup`
warms the deployment's own profile only.

## What a profile may set

A profile may set `sequence_models`, `models` and `processors`. Everything else
in the config — the download directory, the lookups, the wapiti binary, the LLM
response cache — is built once and shared by every profile in the process, so a
profile setting one of those keys is a startup error naming the profile and the
key.

## Which profile served a response

Two response headers say what answered a request:

| header | value |
| --- | --- |
| `X-ScienceBeam-Profile` | the profile name, after aliases |
| `X-ScienceBeam-Profile-Digest` | a digest of the models that profile resolved to |

The digest is there because the name alone is not enough to identify what ran:
environment variables are applied after the profile is resolved and so win over
it, which means two deployments can answer the same profile name with different
models.

## See also

* [LLM engine](llm_engine.md) — the `llm_*` profiles
* [Benchmarks](benchmarks.md)
