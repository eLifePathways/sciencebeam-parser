# Cross-origin requests

A browser only lets a page read a response from another origin where the service says it may. Nothing is permitted as shipped, so a page served from anywhere other than the service itself cannot call the API until a deployment names its origin.

This is a rule browsers enforce on pages, not an access control: the service still answers `curl`, a script, and anybody else. Naming an origin stops other people's web pages from using the API; it does not limit what the service spends.

## Permitting an origin

```yaml
cors:
  allowed_origins:
    - https://demo.example.org
    - http://localhost:4321
```

Or per deployment, without building an image:

```bash
export SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS='[https://demo.example.org, http://localhost:4321]'
```

An origin is a scheme, a host and a port, as the browser reports it, and is matched exactly — `https://example.org` does not cover `http://example.org`, `https://www.example.org` or a different port. A single origin may be given on its own, without a list.

`*` permits every origin:

```bash
export SCIENCEBEAM_PARSER__CORS__ALLOWED_ORIGINS='*'
```

## What a permitted page gets

Only `/api` answers cross-origin; the page at `/` does not.

Beside the body, a permitted page may read the headers naming what served the request — `X-ScienceBeam-Profile`, `X-ScienceBeam-Profile-Digest` and `X-ScienceBeam-LLM-Usage`. Failures carry the same access headers as a success, so a 406 for an unsupported media type, or a 500, reaches the page as the error it is.

Credentials are never permitted: a browser may not send cookies or an `Authorization` header to the API, whatever origin it is on.

The browser's permission check (an `OPTIONS` preflight) is answered without a document being parsed. Note that a preflight carries no credentials, so an authenticating proxy in front of the service has to let `OPTIONS` through for any of this to be reached.

## What a running deployment permits

An unpermitted origin's preflight is refused, and a permitted one's is not:

```bash
curl -i -X OPTIONS https://parser.example.org/api/convert \
  -H 'Origin: https://demo.example.org' \
  -H 'Access-Control-Request-Method: POST'
```

`200` with an `Access-Control-Allow-Origin` header is a permitted origin; `400 Disallowed CORS origin` is not. The process also logs what it permits at startup.
