PYTHON := $(shell which python)
PROJ_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

CONFIG_DIR := $(PROJ_DIR)/configs
CONDA_ENV_PATH := $(CONFIG_DIR)/environment.yml
MYPY_CONFIG_PATH := $(CONFIG_DIR)/mypy.ini
FLAKE8_CONFIG_PATH := $(CONFIG_DIR)/flake8.ini
ISORT_CONFIG_PATH := $(CONFIG_DIR)/isort.cfg
BLACK_CONFIG_PATH := $(CONFIG_DIR)/black.toml

PYTHONPATH := $(PYTHONPATH):$(PROJ_DIR)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    freeze:      export dependencies to $(CONDA_ENV_PATH) (requires conda)"
	@echo "    install:     install dependencies from $(CONDA_ENV_PATH) (requires conda)"
	@echo "    test:        unit test all test_*.py or *_test.py files (requires pytest)"
	@echo "    type:        check the type of all files (requires mypy)"
	@echo "    lint:        check the style of all files (requires flake8)"
	@echo "    format:      check the format of all files (requires black)"
	@echo "    check:       run all of three checks above;"

freeze:
	@echo "Exporting conda environment packages to ${CONDA_ENV_PATH} ..."
	@conda env export | grep -Ev "^prefix: |^name: " > $(CONDA_ENV_PATH)

install:
	@echo "Installing dependencies from ${CONDA_ENV_PATH} ..."
	@conda env update --file ${CONDA_ENV_PATH}

test:
	@echo "Unit testing with pytest ..."
	@python -m pytest || true

type:
	@echo "Checking source code typing with mypy ..."
	@mypy src --config-file ${MYPY_CONFIG_PATH} || true

lint:
	@echo "Checking source code style with flake8 ...."
	@flake8 src --config ${FLAKE8_CONFIG_PATH} || true
	@echo "Checking unused code with vulture ...."
	@vulture --min-confidence 100 src || true

format:
	@echo "Sorting imports with isort ..."
	@isort --settings-path ${CONFIG_DIR} src || true
	@echo "Checking source code format with black ...."
	@black src --check --diff --config ${BLACK_CONFIG_PATH} || true
	@echo "Do you want to confirm these changes? [y/N] " && read ans &&\
     if [ $${ans:-'N'} = 'y' ]; then\
        black src --config ${BLACK_CONFIG_PATH};\
     fi

check:
	@$(MAKE) type
	@$(MAKE) lint
	@$(MAKE) format
