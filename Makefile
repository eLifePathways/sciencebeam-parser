DOCKER_COMPOSE_DEV = docker compose
DOCKER_COMPOSE_CI = docker compose -f docker-compose.yml
DOCKER_COMPOSE = $(DOCKER_COMPOSE_DEV)

VENV = .venv
UV = VIRTUAL_ENV=$(VENV) uv
PIP = $(UV) pip
PYTHON = PATH=$(VENV)/bin:$$PATH $(VENV)/bin/python

ARGS =

NOT_SLOW_PYTEST_ARGS = -m 'not slow'

SCIENCEBEAM_PARSER_PORT = 8080

PDFALTO_CONVERT_API_URL = http://localhost:$(SCIENCEBEAM_PARSER_PORT)/api/pdfalto
EXAMPLE_PDF_DOCUMENT = test-data/minimal-example.pdf
EXAMPLE_DOCX_DOCUMENT = test-data/minimal-office-open.docx


SCIENCEBEAM_DELFT_MAX_SEQUENCE_LENGTH = 2000
SCIENCEBEAM_DELFT_INPUT_WINDOW_STRIDE = 1800
SCIENCEBEAM_DELFT_BATCH_SIZE = 1
SCIENCEBEAM_DELFT_STATEFUL = false


DOCKER_SCIENCEBEAM_PARSER_HOST = sciencebeam-parser
DOCKER_PDFALTO_CONVERT_API_URL = http://$(DOCKER_SCIENCEBEAM_PARSER_HOST):8070/api/pdfalto
DOCKER_CONVERT_API_URL = http://$(DOCKER_SCIENCEBEAM_PARSER_HOST):8070/api/convert
DOCKER_DEV_RUN = $(DOCKER_COMPOSE) run --rm sciencebeam-parser-dev
DOCKER_DEV_PYTHON = $(DOCKER_DEV_RUN) python


venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	$(UV) venv $(VENV)


dev-install:
	$(UV) sync --active --frozen \
		--dev \
		--extra cpu \
		--extra delft \
		--extra cv


dev-venv: venv-create dev-install


dev-flake8:
	$(PYTHON) -m flake8 sciencebeam_parser tests


dev-pylint:
	$(PYTHON) -m pylint sciencebeam_parser tests


dev-mypy:
	$(PYTHON) -m mypy --ignore-missing-imports sciencebeam_parser tests


dev-lint: dev-flake8 dev-pylint dev-mypy


dev-pytest:
	$(PYTHON) -m pytest -p no:cacheprovider $(ARGS)


dev-watch:
	$(PYTHON) -m pytest_watcher \
		--patterns='*.py,*.xsl' \
		--runner=$(VENV)/bin/python \
		. \
		-m pytest \
		$(NOT_SLOW_PYTEST_ARGS) \
		-p no:cacheprovider -p no:warnings $(ARGS)


dev-watch-slow:
	$(PYTHON) -m pytest_watcher \
		--patterns='*.py,*.xsl' \
		--runner=$(VENV)/bin/python \
		. \
		-m pytest \
		-p no:cacheprovider -p no:warnings $(ARGS)


dev-test: dev-lint dev-pytest


dev-start:
	SCIENCEBEAM_DELFT_MAX_SEQUENCE_LENGTH=$(SCIENCEBEAM_DELFT_MAX_SEQUENCE_LENGTH) \
	SCIENCEBEAM_DELFT_INPUT_WINDOW_STRIDE=$(SCIENCEBEAM_DELFT_INPUT_WINDOW_STRIDE) \
	SCIENCEBEAM_DELFT_BATCH_SIZE=$(SCIENCEBEAM_DELFT_BATCH_SIZE) \
	SCIENCEBEAM_DELFT_STATEFUL=$(SCIENCEBEAM_DELFT_STATEFUL) \
		$(PYTHON) -m sciencebeam_parser.service.server --port=$(SCIENCEBEAM_PARSER_PORT)


dev-start-debug:
	FLASK_ENV=development \
	SCIENCEBEAM_PARSER__LOGGING__HANDLERS__LOG_FILE__LEVEL=DEBUG \
	$(MAKE) dev-start


dev-start-no-debug-logging-auto-reload:
	$(PYTHON) -m uvicorn \
		sciencebeam_parser.service.server:create_app \
		--reload \
		--factory \
		--host 127.0.0.1 \
		--port $(SCIENCEBEAM_PARSER_PORT)


dev-start-no-debug-logging-auto-reload-with-cv-and-ocr:
	SCIENCEBEAM_PARSER__PROCESSORS__FULLTEXT__USE_CV_MODEL=true \
	SCIENCEBEAM_PARSER__PROCESSORS__FULLTEXT__USE_OCR_MODEL=true \
	$(MAKE) dev-start-no-debug-logging-auto-reload


dev-end-to-end:
	curl --fail --show-error \
		--form "file=@$(EXAMPLE_PDF_DOCUMENT);filename=$(EXAMPLE_PDF_DOCUMENT)" \
		--silent "$(PDFALTO_CONVERT_API_URL)" \
		> /dev/null


dev-script-start-and-run-end-to-end-tests:
	./scripts/dev/start-and-run-end-to-end-tests.sh


dev-script-end-to-end-tests:
	./scripts/dev/end-to-end-tests.sh


dev-build-python-readme:
	$(PYTHON) scripts/dev/update_readme.py \
		--source=./doc/python_library.md \
		--target=./doc/generated_python_library.md \
		--source-base-path=doc \
		--link-prefix=https://github.com/eLifePathways/sciencebeam-parser/blob/main


run:
	$(PYTHON) -m sciencebeam_parser $(ARGS)


docker-buildx-bake-build-all:
	docker buildx bake \
		--file docker-bake.hcl \
		--set python-dist.args.python_package_version="$(VERSION)" \
		lint-flake8 \
		lint-pylint \
		lint-mypy \
		pytest \
		end-to-end-tests \
		python-dist \
		sciencebeam-parser \
		sciencebeam-parser-cv


docker-buildx-python-dist:
	docker buildx build \
		--target python-dist \
		--output type=local,dest=./build/dist-export \
		--build-arg python_package_version="$(VERSION)" \
		--debug \
		.


docker-build-all:
	$(DOCKER_COMPOSE) build


docker-lint:
	$(MAKE) PYTHON="$(DOCKER_DEV_PYTHON)" dev-lint


docker-pytest:
	$(MAKE) PYTHON="$(DOCKER_DEV_PYTHON)" dev-pytest


docker-show-api-logs-and-fail:
	$(DOCKER_COMPOSE) logs "$(DOCKER_SCIENCEBEAM_PARSER_HOST)" && exit 1


docker-wait-for-api:
	$(DOCKER_COMPOSE) run --rm wait-for-it \
		"$(DOCKER_SCIENCEBEAM_PARSER_HOST):8070" \
		--timeout=30 \
		--strict \
		-- echo "ScienceBeam Parser API is up" \
		|| $(MAKE) docker-show-api-logs-and-fail


docker-start:
	$(DOCKER_COMPOSE) up -d


docker-stop:
	$(DOCKER_COMPOSE) down


docker-start-and-wait-for-api:
	$(MAKE) docker-start
	$(MAKE) docker-wait-for-api


docker-logs:
	$(DOCKER_COMPOSE) logs -f


docker-end-to-end-pdfalto: docker-start-and-wait-for-api
	$(DOCKER_DEV_RUN) curl --fail --show-error --silent \
		--form "file=@$(EXAMPLE_PDF_DOCUMENT);filename=$(EXAMPLE_PDF_DOCUMENT)" \
		--output /dev/null \
		"$(DOCKER_PDFALTO_CONVERT_API_URL)"


docker-end-to-end-doc-to-jats: docker-start-and-wait-for-api
	$(DOCKER_DEV_RUN) curl --fail --show-error --silent \
		--form "file=@$(EXAMPLE_DOCX_DOCUMENT);filename=$(EXAMPLE_DOCX_DOCUMENT)" \
		--output /dev/null \
		"$(DOCKER_CONVERT_API_URL)"


docker-end-to-end: docker-end-to-end-pdfalto docker-end-to-end-doc-to-jats

docker-end-to-end-cv:
	$(MAKE) DOCKER_SCIENCEBEAM_PARSER_HOST=sciencebeam-parser-cv docker-end-to-end


ci-build-all:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-build-all


ci-lint:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-lint


ci-pytest:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-pytest


ci-end-to-end:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-end-to-end
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-end-to-end-cv


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
