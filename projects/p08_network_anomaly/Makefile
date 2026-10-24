# Project 8: anomaly detection on UNSW-NB15 network flows. Run from this folder.
PYTHON ?= python
SOURCE ?=
SMOKE_DIR := artifacts/smoke

.PHONY: install data train evaluate report all smoke test lint clean

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

smoke:               ## the whole pipeline on synthetic flows; outputs stay under artifacts/smoke
	$(PYTHON) -m flow_anomaly.synthetic $(SMOKE_DIR)/raw --rows 12000
	$(PYTHON) -m flow_anomaly.data --source $(SMOKE_DIR)/raw --data-dir $(SMOKE_DIR)/data
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
