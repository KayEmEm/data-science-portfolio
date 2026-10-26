# Project 8: anomaly detection on UNSW-NB15 network flows. Run from this folder.
PYTHON ?= python
SOURCE ?=
SMOKE_DIR := artifacts/smoke
# Share of flows kept by `make sample`, stratified by attack type and split; outputs stay under artifacts/sample.
SAMPLE ?= 0.1
SAMPLE_DIR := artifacts/sample

.PHONY: install data train evaluate report all sample smoke test lint clean

install:             ## this project and the shared common package
	$(PYTHON) -m pip install -e ../../common -e ".[dev]"

data:                ## read the UNSW-NB15 CSV files from data/raw (or SOURCE=<folder>), check them, split
	$(PYTHON) -m flow_anomaly.data $(if $(SOURCE),--source $(SOURCE))

train:               ## fit every detector of every experiment and store its scores; resumes
	$(PYTHON) -m flow_anomaly.train

evaluate:            ## PR AUC, recall at 1 percent false alarm rate, per attack type, intervals
	$(PYTHON) -m flow_anomaly.evaluate

report:              ## tables, figures, triage example and results.json under reports/
	$(PYTHON) -m flow_anomaly.report

all: data train evaluate report

sample:              ## the whole pipeline on a seeded share SAMPLE of the flows (or SOURCE=<folder>)
	$(PYTHON) -m flow_anomaly.data $(if $(SOURCE),--source $(SOURCE)) --data-dir $(SAMPLE_DIR)/data --sample $(SAMPLE)
	$(PYTHON) -m flow_anomaly.train --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR)
	$(PYTHON) -m flow_anomaly.evaluate --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR) \
		--data-label "UNSW-NB15 sample $(SAMPLE)"
	$(PYTHON) -m flow_anomaly.report --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR) \
		--reports $(SAMPLE_DIR)/reports

smoke:               ## the whole pipeline on synthetic flows; outputs stay under artifacts/smoke
	$(PYTHON) -m flow_anomaly.synthetic $(SMOKE_DIR)/raw --rows 12000
	$(PYTHON) -m flow_anomaly.data --source $(SMOKE_DIR)/raw --data-dir $(SMOKE_DIR)/data --no-count-check
	$(PYTHON) -m flow_anomaly.train --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m flow_anomaly.evaluate --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--data-label synthetic --bootstrap 20
	$(PYTHON) -m flow_anomaly.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
