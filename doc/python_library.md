# ScienceBeam Parser Python Library

ScienceBeam Parser allows you to parse scientific documents. It provides a REST API Service, as well as a Python API.

## Installation

```bash
pip install sciencebeam-parser[delft,cpu]
```

## CLI

### CLI: Start Server

```bash
python -m sciencebeam_parser.service.server --port=8080
```

The server will start to listen on port `8080`.

The [default config.yml](../sciencebeam_parser/resources/default_config/config.yml) defines what models to load.

You can find the API docs under `/api/docs`, e.g.:

[http://localhost:8080/api/docs](http://localhost:8080/api/docs)

## Python API

### Python API: Start Server

```python
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.service.server import create_app


config = AppConfig.load_yaml(DEFAULT_CONFIG_FILE)
app = create_app(config)
app.run(port=8080, host='127.0.0.1', threaded=True)
```

The server will start to listen on port `8080`.

### Python API: Parse Multiple Files

```python
from sciencebeam_parser.resources.default_config import DEFAULT_CONFIG_FILE
from sciencebeam_parser.config.config import AppConfig
from sciencebeam_parser.utils.media_types import MediaTypes
from sciencebeam_parser.app.parser import ScienceBeamParser


config = AppConfig.load_yaml(DEFAULT_CONFIG_FILE)

# the parser contains all of the models
sciencebeam_parser = ScienceBeamParser.from_config(config)

# a session provides a scope and temporary directory for intermediate files
# it is recommended to create a separate session for every document
with sciencebeam_parser.get_new_session() as session:
    session_source = session.get_source(
        'test-data/minimal-example.pdf',
        MediaTypes.PDF
    )
    converted_file = session_source.get_local_file_for_response_media_type(
        MediaTypes.TEI_XML
    )
    # Note: the converted file will be in the temporary directory of the session
    print('converted file:', converted_file)
```

## More Usage Examples

For more usage examples see
[sciencebeam-usage-examples](https://github.com/eLifePathways/sciencebeam-usage-examples).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eLifePathways/sciencebeam-usage-examples/HEAD?urlpath=tree/sciencebeam-parser)
