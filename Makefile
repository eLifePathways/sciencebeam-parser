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


GROBID_URL = http://localhost:8070
SCIENCEBEAM_PARSER_URL = http://localhost:$(SCIENCEBEAM_PARSER_PORT)

BENCHMARK_CONFIG ?= benchmarks/eval.yml
BENCHMARK_MODE ?= smoke
BENCHMARK_SPLIT ?= train
BENCHMARK_RUN ?= benchmarks/runs/$(BENCHMARK_SPLIT)
BENCHMARK_PARSER_URL ?= $(SCIENCEBEAM_PARSER_URL)

SHOW_FIELD ?=
SHOW_METHOD ?= edit_sim
SHOW_CORPUS ?= biorxiv
SHOW_LIMIT ?= 10
SHOW_RUN_A ?= $(BENCHMARK_RUN)
SHOW_RUN_B ?= $(shell python3 -c "import yaml; b=yaml.safe_load(open('benchmarks/eval.yml')).get('baselines',[]); print('benchmarks/runs/baselines/'+b[0]['tool']+'/'+b[0]['version']+'/$(BENCHMARK_SPLIT)') if b else print('')" 2>/dev/null)
SHOW_PARSER_URL ?=

COMPARE_MODEL ?= segmentation
COMPARE_DOC_ID ?= $(basename $(notdir $(COMPARE_PDF)))
COMPARE_DOC_DIR = .temp/compare-with-grobid/by-doc/$(COMPARE_DOC_ID)


.require-%:
	@if [ -z "$($(*))" ]; then \
		echo "Error: $* is required. Usage: make $(@:.require-%=%) $*=<value>"; \
		exit 1; \
	fi

venv-clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf "$(VENV)"; \
	fi


venv-create:
	$(UV) venv $(VENV)


dev-install:
	$(UV) sync --active --frozen \
		--group benchmark \
		--dev \
		--extra cpu \
		--extra delft \
		--extra cv


dev-venv: venv-create dev-install


dev-flake8:
	$(PYTHON) -m flake8 sciencebeam_parser tests benchmarks


dev-pylint:
	$(PYTHON) -m pylint sciencebeam_parser tests benchmarks


dev-mypy:
	$(PYTHON) -m mypy --ignore-missing-imports sciencebeam_parser tests benchmarks


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


dev-benchmark-predict:
	$(PYTHON) -m benchmarks.predict \
		--config $(BENCHMARK_CONFIG) \
		--mode $(BENCHMARK_MODE) \
		--split $(BENCHMARK_SPLIT) \
		--out $(BENCHMARK_RUN) \
		--parser-url $(BENCHMARK_PARSER_URL) \
		$(ARGS)


dev-benchmark-score:
	$(PYTHON) -m benchmarks.score \
		--config $(BENCHMARK_CONFIG) \
		--run $(BENCHMARK_RUN) \
		$(ARGS)


dev-benchmark-compare:
	$(PYTHON) -m benchmarks.report \
		$(ARGS)


dev-benchmark: dev-benchmark-predict dev-benchmark-score


dev-show-regressions: .require-SHOW_FIELD
	$(PYTHON) -m benchmarks.show_cases \
		--run-a $(SHOW_RUN_A) \
		--run-b $(SHOW_RUN_B) \
		--field $(SHOW_FIELD) \
		--method $(SHOW_METHOD) \
		--corpus $(SHOW_CORPUS) \
		--mode regression \
		--data benchmarks/data \
		--split $(BENCHMARK_SPLIT) \
		--limit $(SHOW_LIMIT) \
		$(if $(SHOW_PARSER_URL),--parser-url $(SHOW_PARSER_URL),)


dev-show-improvements: .require-SHOW_FIELD
	$(PYTHON) -m benchmarks.show_cases \
		--run-a $(SHOW_RUN_A) \
		--run-b $(SHOW_RUN_B) \
		--field $(SHOW_FIELD) \
		--method $(SHOW_METHOD) \
		--corpus $(SHOW_CORPUS) \
		--mode improvement \
		--data benchmarks/data \
		--split $(BENCHMARK_SPLIT) \
		--limit $(SHOW_LIMIT) \
		$(if $(SHOW_PARSER_URL),--parser-url $(SHOW_PARSER_URL),)


dev-benchmark-with-baselines:
	$(PYTHON) -m benchmarks.run_local \
		--config $(BENCHMARK_CONFIG) \
		--mode $(BENCHMARK_MODE) \
		--split $(BENCHMARK_SPLIT) \
		--runs benchmarks/runs \
		--parser-url $(BENCHMARK_PARSER_URL) \
		$(ARGS)


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


fetch-grobid-model-data: .require-COMPARE_PDF
	mkdir -p $(COMPARE_DOC_DIR)/grobid
	curl --fail --show-error \
		--form input=@$(COMPARE_PDF) \
		--form debugMode=true \
		--form models=$(COMPARE_MODEL) \
		$(GROBID_URL)/api/processFulltextDocument \
		| grep -v '^=== model:' \
		| tr '\t' ' ' \
		> $(COMPARE_DOC_DIR)/grobid/$(COMPARE_MODEL).data


fetch-parser-model-data: .require-COMPARE_PDF
	mkdir -p $(COMPARE_DOC_DIR)/sciencebeam-parser
	curl -X POST \
		--fail --show-error \
		-H 'accept: application/json' \
		-H 'Content-Type: multipart/form-data' \
		-F "input=@$(COMPARE_PDF);type=application/pdf" \
		'$(SCIENCEBEAM_PARSER_URL)/api/models/$(COMPARE_MODEL)?output_format=data' \
		> $(COMPARE_DOC_DIR)/sciencebeam-parser/$(COMPARE_MODEL).data
	curl --fail --show-error \
		'$(SCIENCEBEAM_PARSER_URL)/api/models/$(COMPARE_MODEL)/feature-names' \
		> $(COMPARE_DOC_DIR)/sciencebeam-parser/$(COMPARE_MODEL).feature_names.json


diff-model-data:
	$(PYTHON) scripts/compare_model_data.py \
		--sbeam=$(COMPARE_DOC_DIR)/sciencebeam-parser/$(COMPARE_MODEL).data \
		--feature-names=$(COMPARE_DOC_DIR)/sciencebeam-parser/$(COMPARE_MODEL).feature_names.json \
		--grobid=$(COMPARE_DOC_DIR)/grobid/$(COMPARE_MODEL).data \
		> $(COMPARE_DOC_DIR)/$(COMPARE_MODEL).diff
	@echo "diff written to $(COMPARE_DOC_DIR)/$(COMPARE_MODEL).diff"


compare-model-data: fetch-grobid-model-data fetch-parser-model-data diff-model-data


OVERRIDE ?=

dev-diff-feature-override: .require-COMPARE_DOC_ID .require-OVERRIDE
	$(PYTHON) scripts/diff_feature_override.py \
		--data=$(COMPARE_DOC_DIR)/sciencebeam-parser/$(COMPARE_MODEL).data \
		--feature-names=$(COMPARE_DOC_DIR)/sciencebeam-parser/$(COMPARE_MODEL).feature_names.json \
		--model=$(COMPARE_MODEL) \
		--override="$(OVERRIDE)"


ci-lint:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-lint


ci-pytest:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-pytest


ci-end-to-end:
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-end-to-end
	$(MAKE) DOCKER_COMPOSE="$(DOCKER_COMPOSE_CI)" docker-end-to-end-cv


ci-clean:
	$(DOCKER_COMPOSE_CI) down -v
