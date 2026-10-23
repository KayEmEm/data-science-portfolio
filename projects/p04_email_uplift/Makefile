# Project 4: email campaign uplift. Run from this folder.
PYTHON ?= python
SMOKE_DIR := artifacts/smoke

.PHONY: install data experiment train evaluate observational report all smoke test lint clean

install:             ## this project, the shared common package and CausalML (may compile from source)
	$(PYTHON) -m pip install -e ../../common -e ".[dev,causalml]"

data:                ## download the Hillstrom CSV (0.4 MB), check its checksum and schema
	$(PYTHON) -m email_uplift.data

experiment:          ## A/B readout: randomization checks, effects, power, CUPED
	$(PYTHON) -m email_uplift.experiment

train:               ## uplift models on visits and spend; model selection on the validation split
	$(PYTHON) -m email_uplift.train

evaluate:            ## Qini curves and policy values on the test split
	$(PYTHON) -m email_uplift.evaluate

observational:       ## simulated observational study with DoWhy
	$(PYTHON) -m email_uplift.observational

report:              ## tables, figures and the readout memo under reports/
	$(PYTHON) -m email_uplift.report

all: data experiment train evaluate observational report

smoke:               ## the whole pipeline on synthetic customers; outputs stay under artifacts/smoke
	$(PYTHON) -m email_uplift.synthetic $(SMOKE_DIR)/customers.parquet --n 6000
	$(PYTHON) -m email_uplift.experiment --customers $(SMOKE_DIR)/customers.parquet --artifacts $(SMOKE_DIR) \
		--data-label synthetic
	$(PYTHON) -m email_uplift.train --customers $(SMOKE_DIR)/customers.parquet --artifacts $(SMOKE_DIR) \
		--quick --data-label synthetic
	$(PYTHON) -m email_uplift.evaluate --artifacts $(SMOKE_DIR)
	$(PYTHON) -m email_uplift.observational --customers $(SMOKE_DIR)/customers.parquet \
		--artifacts $(SMOKE_DIR) --data-label synthetic --replicates 2 --simulations 5
	$(PYTHON) -m email_uplift.report --artifacts $(SMOKE_DIR) --reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
