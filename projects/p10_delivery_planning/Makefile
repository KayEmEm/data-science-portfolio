# Project 10: delivery routes and driver shifts on the Solomon VRPTW benchmark. Run from this folder.
PYTHON ?= python
SOURCE ?=
SMOKE_DIR := artifacts/smoke

.PHONY: install data train evaluate report planner all smoke test lint clean

install:             ## this project, the shared common package and the optional Gurobi package
	$(PYTHON) -m pip install -e ../../common -e ".[dev,gurobi]"

data:                ## download the 56 instances with best known solutions (or SOURCE=<folder or zip>), check them
	$(PYTHON) -m delivery_planning.data $(if $(SOURCE),--source $(SOURCE))

train:               ## every routing search: benchmark and fleet size study; resumes
	$(PYTHON) -m delivery_planning.train

evaluate:            ## gaps to best known, shift model with three solvers, fleet size costs; resumes
	$(PYTHON) -m delivery_planning.evaluate

report:              ## tables, figures, route maps and results.json under reports/
	$(PYTHON) -m delivery_planning.report

planner:             ## the Streamlit planner on the downloaded instances
	$(PYTHON) -m streamlit run planner/app.py

all: data train evaluate report

smoke:               ## the whole pipeline on synthetic instances; outputs stay under artifacts/smoke
	$(PYTHON) -m delivery_planning.synthetic $(SMOKE_DIR)/source
	$(PYTHON) -m delivery_planning.data --source $(SMOKE_DIR)/source --data-dir $(SMOKE_DIR)/data --sha any
	$(PYTHON) -m delivery_planning.train --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m delivery_planning.evaluate --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m delivery_planning.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--reports $(SMOKE_DIR)/reports --data-label synthetic

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests planner
	$(PYTHON) -m ruff format --check src tests planner

clean:
	rm -rf artifacts
