# Project 9: customer segments from Online Retail II. Run from this folder.
PYTHON ?= python
SOURCE ?=
SMOKE_DIR := artifacts/smoke
# Share of customers kept by `make sample`, each with all their order lines; outputs stay under artifacts/sample.
SAMPLE ?= 0.25
SAMPLE_DIR := artifacts/sample

.PHONY: install data train evaluate report all sample smoke test lint clean

install:             ## this project and the shared common package
	$(PYTHON) -m pip install -e ../../common -e ".[dev]"

data:                ## read online_retail_II.xlsx from data/raw (or SOURCE=<file, zip or folder>), check, clean
	$(PYTHON) -m customer_segments.data $(if $(SOURCE),--source $(SOURCE))

train:               ## customer table, k means grid, gap statistic, stability, DBSCAN, HDBSCAN; resumes
	$(PYTHON) -m customer_segments.train

evaluate:            ## selection, next quarter check against the RFM grid, profiles, UMAP map
	$(PYTHON) -m customer_segments.evaluate

report:              ## tables, figures, segment_map.html and results.json under reports/
	$(PYTHON) -m customer_segments.report

all: data train evaluate report

sample:              ## the whole pipeline on a seeded share SAMPLE of the customers (or SOURCE=<file, zip or folder>)
	$(PYTHON) -m customer_segments.data $(if $(SOURCE),--source $(SOURCE)) --data-dir $(SAMPLE_DIR)/data \
		--sample $(SAMPLE)
	$(PYTHON) -m customer_segments.train --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR)
	$(PYTHON) -m customer_segments.evaluate --artifacts $(SAMPLE_DIR)
	$(PYTHON) -m customer_segments.report --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR) \
		--reports $(SAMPLE_DIR)/reports --data-label "Online Retail II sample $(SAMPLE)"

smoke:               ## the whole pipeline on synthetic order lines; outputs stay under artifacts/smoke
	$(PYTHON) -m customer_segments.synthetic $(SMOKE_DIR)/raw --customers 600
	$(PYTHON) -m customer_segments.data --source $(SMOKE_DIR)/raw --data-dir $(SMOKE_DIR)/data
	$(PYTHON) -m customer_segments.train --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m customer_segments.evaluate --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m customer_segments.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--reports $(SMOKE_DIR)/reports --data-label synthetic

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
