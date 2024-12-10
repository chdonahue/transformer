# Take a look at https://github.com/the-full-stack/conda-piptools for an example

# Oneshell means all lines in a recipe run in the same shell
.ONESHELL:

# Need to specify bash in order for conda activate to work
SHELL=/bin/bash

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

CONDA_ENV=transformer_env

all: conda-env-update setup-pre-commit install-dev setup-jupyter

# Create or update conda env
conda-env-update:
	conda env update --prune

# Install package in development mode
.PHONY: install-dev
install-dev:
	$(CONDA_ACTIVATE) $(CONDA_ENV)
	echo "Installing package from directory: $$(pwd)"
	pip install -e .

# Setup jupyter kernel
.PHONY: setup-jupyter
setup-jupyter:
	$(CONDA_ACTIVATE) $(CONDA_ENV)
	python -m ipykernel install --user --name $(CONDA_ENV) --display-name "Python ($(CONDA_ENV))"

# Install and setup pre-commit hooks
.PHONY: setup-pre-commit
setup-pre-commit:
	$(CONDA_ACTIVATE) $(CONDA_ENV)
	pip install pre-commit
	@if [ ! -f .pre-commit-config.yaml ]; then \
		echo "Creating .pre-commit-config.yaml"; \
		echo "repos:" > .pre-commit-config.yaml; \
		echo "-   repo: https://github.com/psf/black" >> .pre-commit-config.yaml; \
		echo "    rev: 23.3.0" >> .pre-commit-config.yaml; \
		echo "    hooks:" >> .pre-commit-config.yaml; \
		echo "    -   id: black" >> .pre-commit-config.yaml; \
		echo "        language_version: python3" >> .pre-commit-config.yaml; \
		echo "        args: [--line-length=110]" >> .pre-commit-config.yaml; \
	else \
		echo ".pre-commit-config.yaml already exists"; \
	fi
	pre-commit install



# # Compile exact pip packages
# pip-compile:
# 	$(CONDA_ACTIVATE) $(CONDA_ENV)
# 	pip-compile -v requirements/prod.in && pip-compile -v requirements/dev.in

# # Install pip packages
# pip-sync:
# 	$(CONDA_ACTIVATE) $(CONDA_ENV)
# 	pip-sync requirements/prod.txt requirements/dev.txt