FROM ghcr.io/astral-sh/uv:python3.9-bookworm AS base


# shared between builder and runtime image
RUN apt-get update \
    && apt-get install --assume-yes --no-install-recommends \
        dumb-init \
        poppler-utils \
        libgl1 \
        build-essential \
        # install LibreOffice Write to convert Word to PDF
        # also install fonts and fontconfig to provide common fonts
        # or configuration to their alternatives
        libreoffice-writer \
        python3-uno \
        fonts-liberation \
        fonts-liberation2 \
        fonts-crosextra-carlito \
        fonts-crosextra-caladea \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

# set and check UNO_PATH, UNO_PYTHON_PATH and UNO_OFFICE_BINARY_PATH
ENV UNO_PATH=/usr/lib/python3/dist-packages
ENV UNO_PYTHON_PATH=/usr/local/bin/python3.9
ENV UNO_OFFICE_BINARY_PATH=/usr/lib/libreoffice/program/soffice.bin
RUN \
  echo "UNO_PATH: ${UNO_PATH}" \
  && ls -l "${UNO_PATH}" \
  && echo "UNO_PYTHON_PATH: ${UNO_PYTHON_PATH}" \
  && PYTHONPATH="${UNO_PATH}" "${UNO_PYTHON_PATH}" -c 'import uno, unohelper' \
  && echo "UNO_OFFICE_BINARY_PATH: ${UNO_OFFICE_BINARY_PATH}" \
  && ls -l "${UNO_OFFICE_BINARY_PATH}"

WORKDIR /opt/sciencebeam_parser

ENV VENV=/opt/venv
ENV VIRTUAL_ENV=${VENV} PYTHONUSERBASE=${VENV} PATH=${VENV}/bin:$PATH


# builder
FROM base AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        tesseract-ocr-eng \
        libtesseract-dev \
        libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --active --frozen \
    --no-dev \
    --extra cpu \
    --extra delft


# builder-cv
FROM builder AS builder-cv

RUN uv sync --active --frozen \
    --no-dev \
    --extra cpu \
    --extra delft \
    --extra cv

# Note: OCR requirements are not included in the cv builder image because of issues installing tesserocr
# COPY requirements.ocr.txt ./
# RUN pip install --disable-pip-version-check --no-warn-script-location \
#     -r requirements.cpu.txt \
#     -r requirements.cv.txt \
#     -r requirements.torch.txt \
#     -r requirements.ocr.txt


# dev image
FROM builder-cv AS dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --active --frozen \
    --dev \
    --extra cpu \
    --extra delft \
    --extra cv

COPY sciencebeam_parser ./sciencebeam_parser
COPY delft ./delft

COPY tests ./tests
COPY test-data ./test-data
COPY scripts/dev ./scripts/dev
COPY doc ./doc
COPY .flake8 .pylintrc README.md ./

# temporary workaround for tesserocr https://github.com/sirfz/tesserocr/issues/165
ENV LC_ALL=C


# python-dist-builder
FROM dev AS python-dist-builder

ARG python_package_version
RUN echo "Setting version to: $version" && \
    uv version "$python_package_version"
RUN python scripts/dev/update_readme.py \
    --source=./doc/python_library.md \
    --target=./doc/generated_python_library.md \
    --source-base-path=doc \
    --link-prefix=https://github.com/eLifePathways/sciencebeam-parser/blob/main
RUN uv build && \
    ls -l dist


# python-dist
FROM scratch AS python-dist

WORKDIR /dist

COPY --from=python-dist-builder /opt/sciencebeam_parser/dist /dist


# lint-flake8
FROM dev AS lint-flake8

RUN python -m flake8 sciencebeam_parser tests


# lint-pylint
FROM dev AS lint-pylint

RUN python -m pylint sciencebeam_parser tests


# lint-mypy
FROM dev AS lint-mypy

RUN python -m mypy --ignore-missing-imports sciencebeam_parser tests


# pytest
FROM dev AS pytest

RUN python -m pytest -p no:cacheprovider


# end-to-end-tests
FROM dev AS end-to-end-tests

RUN ./scripts/dev/start-and-run-end-to-end-tests.sh


# runtime image
FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv

COPY sciencebeam_parser ./sciencebeam_parser
COPY delft ./delft

COPY docker/entrypoint.sh ./docker/entrypoint.sh

ENV SCIENCEBEAM_DELFT_MAX_SEQUENCE_LENGTH=2000
ENV SCIENCEBEAM_DELFT_INPUT_WINDOW_STRIDE=1800

CMD [ "--port=8070", "--host=0.0.0.0" ]
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/opt/sciencebeam_parser/docker/entrypoint.sh"]


# runtime-cv image
FROM base AS runtime-cv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libtesseract-dev \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-cv /opt/venv /opt/venv

COPY sciencebeam_parser ./sciencebeam_parser
COPY delft ./delft

COPY docker/entrypoint.sh ./docker/entrypoint.sh

ENV SCIENCEBEAM_DELFT_MAX_SEQUENCE_LENGTH=2000
ENV SCIENCEBEAM_DELFT_INPUT_WINDOW_STRIDE=1800

# temporary workaround for tesserocr https://github.com/sirfz/tesserocr/issues/165
ENV LC_ALL=C

CMD [ "--port=8070", "--host=0.0.0.0" ]
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/opt/sciencebeam_parser/docker/entrypoint.sh"]
