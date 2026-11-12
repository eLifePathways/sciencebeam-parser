# Profiles

A profile names a set of sequence models and the processor settings that go with
them. `config.yml` declares them under `profiles:`, and `profile:` selects the
one a deployment serves — `recommended` as shipped, overridable with
`SCIENCEBEAM_PARSER__PROFILE`.

## Which profile to point at

Five profiles are names rather than combinations: each tracks a profile declared
beside it, and what it tracks changes between releases.

| name | what it serves |
| --- | --- |
| `recommended` | the combination we currently recommend |
| `recommended_cv` | the same, locating figures and tables on the rendered page; slower |
| `lightweight` | retrained CRF models throughout, for a deployment that will not run deep-learning models |
| `grobid_crf` | the CRF models GROBID ships, as a baseline |
| `grobid_dl` | the deep-learning models GROBID publishes to the Hugging Face Hub |

Point a deployment at one of these to follow what we would choose. Name the
profile it currently tracks instead — `GET /api/profiles` lists both — to pin one
combination for as long as it is declared.

A profile that tracks another declares `extends` and may add only `label` and
`description`, so it resolves to exactly what it names and cannot drift from it:

```yaml
profiles:
  recommended:
    extends: delft_all_scielo_preprints_ore
    label: 'Recommended'
    description: 'The combination we currently recommend.'
```

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
  - wapiti_references_scielo_preprints_ore
  - llm_references
```

or by environment, which takes a YAML list:

```bash
export SCIENCEBEAM_PARSER__SELECTABLE_PROFILES='[wapiti_references_scielo_preprints_ore, llm_references]'
```

As shipped it declares the five names above; each brings the profile it tracks,
since that is the name the response headers report. The per-model profiles —
`delft_header_scielo_preprints_ore` and the rest, which exist so a change can be
attributed to one model — are declared and not selectable, so a deployment opts
into them explicitly.

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

## What each profile is for

`GET /api/profiles` says what a caller is choosing between:

```bash
curl --fail --show-error 'http://localhost:8080/api/profiles'
```

```json
{
  "profiles": [
    {
      "name": "delft_all_scielo_preprints_ore",
      "label": "DeLFT retrained models, SciELO preprints and ORE",
      "description": "GROBID 0.9.0 with segmentation, the header and both reference models retrained as DeLFT models on SciELO preprints and ORE.",
      "alias_names": [],
      "is_default": false,
      "tracks": null
    },
    {
      "name": "recommended",
      "label": "Recommended",
      "description": "The combination we currently recommend. Its models change between releases, and the profile digest on each response identifies them.",
      "alias_names": [],
      "is_default": true,
      "tracks": "delft_all_scielo_preprints_ore"
    }
  ]
}
```

It lists what this deployment declares selectable, so a deployment that narrows
`selectable_profiles` advertises only what it serves. A profile that tracks
another is listed beside what it tracks rather than in place of it, because the
two are different offers: one follows what we would choose, the other stays put.

`tracks` is what tells them apart: the profile this one currently serves the
configuration of, or `null` where it declares its own. The two answer a request
identically, so a client offering a menu has one conversion under two names and
should pick one — the page this service serves offers the tracking name and
drops what it tracks, since a response reports the profile that answered either
way. A client pinning a combination rather than offering a menu wants the other
one.

Under `selectable_profiles: all` the per-model profiles outnumber the rest, so
the page leads with the names that track and heads the two parts "Updated
between releases" and "Fixed combinations"; a deployment serving one kind gets
no heading.

`profile_aliases` is for a name a profile no longer has — an old name kept
working after a rename. An alias is reported as `alias_names` of the profile it
resolves to rather than as a choice of its own, and a name may not be both a
profile and an alias. The shipped config declares none.

The text comes from the profile itself:

```yaml
profiles:
  llm_references:
    label: 'LLM references'
    description: 'GROBID 0.9.0 with both reference models served by an LLM. Needs an API key.'
    sequence_models: llm_references
```

Both keys are optional — a profile without them is still selectable and still
listed, reporting `null` for each — and neither reaches the configuration that
serves a document, so adding or editing one leaves the profile digest where it
is.

`/api/docs` documents the list as the response example for that route, built
from the registry when the app starts. That is where a reader of the generated
documentation sees it: the `profile` parameter's schema names the profiles it
accepts, but the documentation renderer has no surface for a word about each
value, so the prose arrives as a response rather than in the parameter.

## What a profile may set

A profile may set `sequence_models`, `models` and `processors`, and may describe
itself with `label` and `description`. Everything else in the config — the
download directory, the lookups, the wapiti binary, the LLM response cache — is
built once and shared by every profile in the process, so a profile setting one
of those keys is a startup error naming the profile and the key.

A profile may instead set `extends`, naming the profile it tracks, beside
`label` and `description` and nothing else. Setting `extends` alongside
`sequence_models`, `models` or `processors` is a startup error, as is pointing
at a profile that itself sets `extends`: what a name serves should be readable
without following a chain.

## Which profile served a response

Two response headers say what answered a request:

| header | value |
| --- | --- |
| `X-ScienceBeam-Profile` | the profile name, after aliases and after a profile that tracks another |
| `X-ScienceBeam-Profile-Digest` | a digest of the models that profile resolved to |
| `X-ScienceBeam-Profile-Requested` | the name the request named, where that is not the one above |

A request naming `recommended` is answered with the profile `recommended`
tracks, so a document saved today names a combination that still means that
combination once `recommended` has moved. That is also why a deployment serving
a tracking profile serves what it tracks: the name in the header is one a caller
may send, so naming the tracking profile is enough to get both.

Being handed a name it did not use would leave a caller to work out the
connection, so a request that named something else is told both:

```http
X-ScienceBeam-Profile-Requested: recommended
X-ScienceBeam-Profile: delft_all_scielo_preprints_ore
X-ScienceBeam-Profile-Digest: f5d45b3293c9
```

The requested header is absent where the name a request sent is the one that
served it, so its presence is what says a name resolved to another. It is a
property of the exchange rather than of the document: a converted document
records what produced it, below, and not what it was asked for.

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
