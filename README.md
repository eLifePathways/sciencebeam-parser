# ScienceBeam Parser

## Overview

ScienceBeam Parser allows you to parse scientific documents.
Initially is starting as a partial Python variation of GROBID and allows you to re-use some of the models.
However, it may deviate more in the future.

## Pre-requisites

Docker containers are provided that can be used on multiple operating systems.
It can be used as an example setup for Linux / Ubuntu based systems.

Otherwise the following paragraphs list some of the pre-requisits when not using Docker:

This currently only supports Linux due to the binaries used (`pdfalto`, `wapiti`).
It may also be used on other platforms without Docker, provided matching binaries are configured.

For sqlite or lmdb you may need:

```bash
apt-get install libsqlite3-dev liblmdb-dev
```

For Computer Vision PyTorch is required.

For OCR, tesseract needs to be installed. On Ubuntu the following command can be used:

```bash
apt-get install tesseract-ocr-eng libtesseract-dev libleptonica-dev
```

The Word* to PDF conversion requires [LibreOffice](https://www.libreoffice.org/).

## Development

### Create Virtual Environment and install Dependencies

```bash
make dev-venv
```

### Configuration

There is no implicit "grobid-home" directory. The only configuration file is the [default config.yml](sciencebeam_parser/resources/default_config/config.yml).

Paths may point to local or remote files. Remote files are downloaded and cached locally (urls are assumed to be versioned).

You may override config values using environment variables.
Environment variables should start with `SCIENCEBEAM_PARSER__`. After that `__` is used as a section separator.
For example `SCIENCEBEAM_PARSER__LOGGING__HANDLERS__LOG_FILE__LEVEL` would override `logging.handlers.log_file.level`.

Generally, resources and models are loaded on demand, depending on the `preload_on_startup` configuration option (`SCIENCEBEAM_PARSER__PRELOAD_ON_STARTUP` environment variable).
Models will be loaded "eagerly" at startup, by setting the configuration option to `true`.

### Run tests (linting, pytest, etc.)

```bash
make dev-test
```

### Start the server

```bash
make dev-start
```

Run the server in debug mode (including auto-reload and debug logging):

```bash
make dev-debug
```

Run the server with auto reload but no debug logging:

```bash
make dev-start-no-debug-logging-auto-reload
```

By default the server will start on port 8080 and you can find the API docs under `/api/docs`, e.g.:

[http://localhost:8080/api/docs](http://localhost:8080/api/docs)

### Submit a sample document to the server

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/pdfalto"
```

### Submit a sample document to the header model

The following output formats are supported:

| output_format | description |
| ------------- | ----- |
| raw_data | generated data (without using the model) |
| data | generated data with predicted labels |
| xml | using simple xml elements for predicted labels |
| json | json of prediction |

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/models/header?first_page=1&last_page=1&output_format=xml"
```

### Submit a sample document to the name-header api

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/models/name-header?first_page=1&last_page=1&output_format=xml"
```

### GROBID compatible APIs

The following APIs are aiming to be compatible with selected endpoints of the
[GROBID's REST API](https://grobid.readthedocs.io/en/latest/Grobid-service/), for common use-cases.

#### Submit a sample document to the header document api

The `/processHeaderDocument` endpoint is similar to the `/processFulltextDocument`, but it will only contain front matter.
It still uses the same segmentation model, but it won't need to process a number of other models.

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processHeaderDocument?first_page=1&last_page=1"
```

The default response will be TEI XML (`application/tei+xml`).
The `Accept` HTTP request header may be used to request JATS, with the mime type `application/vnd.jats+xml`.

```bash
curl --fail --show-error \
    --header 'Accept: application/vnd.jats+xml' \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processHeaderDocument?first_page=1&last_page=1"
```

Regardless, the returned content type will be `application/xml`.

(BibTeX output is currently not supported)

#### Submit a sample document to the full text document api

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processFulltextDocument?first_page=1&last_page=1"
```

The default response will be TEI XML (`application/tei+xml`).
The `Accept` HTTP request header may be used to request JATS, with the mime type `application/vnd.jats+xml`.

```bash
curl --fail --show-error \
    --header 'Accept: application/vnd.jats+xml' \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processFulltextDocument?first_page=1&last_page=1"
```

Regardless, the returned content type will be `application/xml`.

#### Submit a sample document to the references api

The `/processReferences` endpoint is similar to the `/processFulltextDocument`, but it will only contain references.
It still uses the same segmentation model, but it won't need to process a number of other models.

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processReferences?first_page=1&last_page=100"
```

The default response will be TEI XML (`application/tei+xml`).
The `Accept` HTTP request header may be used to request JATS, with the mime type `application/vnd.jats+xml`.

```bash
curl --fail --show-error \
    --header 'Accept: application/vnd.jats+xml' \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processReferences?first_page=1&last_page=100"
```

Regardless, the returned content type will be `application/xml`.

#### Submit a sample document to the full text asset document api

The `processFulltextAssetDocument` is like `processFulltextDocument`. But instead of returning the TEI XML directly, it will contain a zip with the TEI XML document, along with other assets such as figure images.

```bash
curl --fail --show-error \
    --output "example-tei-xml-and-assets.zip" \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processFulltextAssetDocument?first_page=1&last_page=1"
```

The default response will be ZIP containing TEI XML (`application/tei+xml+zip`).
The `Accept` HTTP request header may be used to request a ZIP containing JATS,
with the mime type `application/vnd.jats+xml+zip`.

```bash
curl --fail --show-error \
    --header 'Accept: application/vnd.jats+xml+zip' \
    --output "example-jats-xml-and-assets.zip" \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/processFulltextAssetDocument?first_page=1&last_page=1"
```

Regardless, the returned content type will be `application/zip`.

### Submit a sample document to the `/convert` api

The `/convert` API is aiming to be a single endpoint for the conversion of PDF documents to a semantic representation.
By default it will return JATS XML.

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/convert?first_page=1&last_page=1"
```

The following section describe parameters to influence the response:

#### Using the `Accept` HTTP header parameter

The [Accept HTTP header](https://en.wikipedia.org/wiki/List_of_HTTP_header_fields)
may be used to request a different response type. e.g. `application/tei+xml` for TEI XML.

```bash
curl --fail --show-error \
    --header 'Accept: application/tei+xml' \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/convert?first_page=1&last_page=1"
```

Regardless, the returned content type will be `application/xml`.

The `/convert` endpoint can also be used for a Word* to PDF conversion
by specifying `application/pdf` as the desired response:

```bash
curl --fail --show-error --silent \
    --header 'Accept: application/pdf' \
    --form "file=@test-data/minimal-office-open.docx;filename=test-data/minimal-office-open.docx" \
    --output "example.pdf" \
    "http://localhost:8080/api/convert?first_page=1&last_page=1"
```

#### Using the `includes` request parameter

The `includes` request parameter may be used to specify the requested fields, in order to reduce the processing time.
e.g. `title,abstract` to requst the `title` and the `abstract` only. In that case fewer models will be used.
The output may still contain more fields than requested.

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-example.pdf;filename=test-data/minimal-example.pdf" \
    --silent "http://localhost:8080/api/convert?includes=title,abstract"
```

The currently supported fields are:

* `title`
* `abstract`
* `authors`
* `affiliations`
* `references`

Passing in any other values (no values), will behave as if no `includes` parameter was passed in.

### Word* support

All of the above APIs will also accept a Word* document instead of a PDF.

Formats that are supported:

* `.docx` (media type: `application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
* `.dotx` (media type: `application/vnd.openxmlformats-officedocument.wordprocessingml.template`)
* `.doc` (media type: `application/msword`)
* `.rtf` (media type: `application/rtf`)

The support is currently implemented by converting the document to PDF using [LibreOffice](https://www.libreoffice.org/).

Where no content type is provided, the content type is inferred from the file extension.

For example:

```bash
curl --fail --show-error \
    --form "file=@test-data/minimal-office-open.docx;filename=test-data/minimal-office-open.docx" \
    --silent "http://localhost:8080/api/convert?first_page=1&last_page=1"
```

### Docker Usage

```bash
docker pull ghcr.io/elifepathways/sciencebeam-parser
```

```bash
docker run --rm \
    -p 8070:8070 \
    ghcr.io/elifepathways/sciencebeam-parser
```

Note: Docker images with the tag suffix `-cv` include the dependencies required for the CV (Computer Vision) models (disabled by default).

```bash
docker run --rm \
    -p 8070:8070 \
    --env SCIENCEBEAM_PARSER__PROCESSORS__FULLTEXT__USE_CV_MODEL=true \
    --env SCIENCEBEAM_PARSER__PROCESSORS__FULLTEXT__USE_OCR_MODEL=true \
    ghcr.io/elifepathways/sciencebeam-parser:latest-cv
```

Non-release builds are available with the `_unstable` image suffix, e.g. `ghcr.io/elifepathways/sciencebeam-parser_unstable`.

## See also

* [Architecture](ARCHITECTURE.md)
* [Python API](doc/python_library.md)
* [Training](doc/training.md)
