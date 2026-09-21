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

`/pdfalto` takes the parameter and reports the profile that served it, but
nothing a profile currently sets reaches that route: converting a PDF to ALTO
uses the pdfalto wrapper and the page range, neither of which a profile may
change. It is uniform rather than special-cased so that the day `pdfalto`
settings become part of a profile, the route already asks for one — and so that
`?profile=` never means one thing on one route and another elsewhere. Naming a
profile there does resolve it, which counts towards `max_loaded_models`.

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

`all` means every profile the config declares, which is the convenient setting
for a benchmark sweep or an experiment:

```bash
export SCIENCEBEAM_PARSER__SELECTABLE_PROFILES=all
```

`max_loaded_models` still bounds what `all` may load, so it opts in to choosing
between the profiles rather than to unbounded memory. No profile may be named
`all`; the config fails to start if one is.

Naming a profile that is not selectable, or that does not exist, is a `400`
listing what is available. The selectable names are also listed in the OpenAPI
schema, so `/api/docs` offers them as a choice on every route that takes the
parameter.

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
models. A deployment whose config names no profile sends the digest alone.

## Which profile a converted document says produced it

Every converted document carries the same two values, plus the parser version,
so a saved file stays attributable without the response it came from.

The TEI records them where GROBID records its own, in
`teiHeader/encodingDesc/appInfo/application`:

```xml
<encodingDesc>
  <appInfo>
    <application ident="sciencebeam-parser" version="1.2.3">
      <label type="profile">wapiti_grobid_only</label>
      <label type="profile-digest">a1b2c3d4e5f6</label>
    </application>
  </appInfo>
</encodingDesc>
```

The JATS carries the same values as `custom-meta` entries in
`article-meta/custom-meta-group`, under the names
`sciencebeam-parser-version`, `sciencebeam-parser-profile` and
`sciencebeam-parser-profile-digest`. They are read from the TEI during the
transform, so the two outputs cannot disagree. The asset zip's `tei.xml` is the
same serialization and carries them too.

There is no timestamp: two conversions of the same document, under the same
profile and the same build, produce byte-identical output. A configuration that
names no profile records the digest and the version without a profile name.

The `/api/models/*` routes and `/api/pdfalto` return training data, tagged
sequences and ALTO rather than a converted document, and carry no attribution.

## See also

* [LLM engine](llm_engine.md) — the `llm_*` profiles
* [Benchmarks](benchmarks.md)
