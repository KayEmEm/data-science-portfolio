# Project 11: illicit transactions in the Elliptic Bitcoin graph. Run from this folder.
PYTHON ?= python
SOURCE ?=
SMOKE_DIR := artifacts/smoke

.PHONY: install data network train evaluate report neo4j-export neo4j-load all smoke test lint clean

install:             ## this project and the shared common package
	$(PYTHON) -m pip install -e ../../common -e ".[dev]"

data:                ## read the Elliptic CSV files from data/raw (or SOURCE=<folder or zip>), check them
	$(PYTHON) -m illicit_graph.data $(if $(SOURCE),--source $(SOURCE))

network:             ## statistics, centrality and Louvain communities per time step (NetworkX)
	$(PYTHON) -m illicit_graph.network

train:               ## logistic regression, LightGBM, GCN and GraphSAGE; resumes
	$(PYTHON) -m illicit_graph.train

evaluate:            ## illicit F1 at the validation threshold, per time step, intervals
	$(PYTHON) -m illicit_graph.evaluate

report:              ## tables, figures and results.json under reports/
	$(PYTHON) -m illicit_graph.report

neo4j-export:        ## CSV files for neo4j/load.cypher under artifacts/neo4j
	$(PYTHON) -m illicit_graph.neo4j_export export

neo4j-load:          ## load into a running Neo4j server and run neo4j/queries.cypher (needs NEO4J_PASSWORD)
	$(PYTHON) -m illicit_graph.neo4j_export load

all: data network train evaluate report neo4j-export

smoke:               ## the whole pipeline on a synthetic graph; outputs stay under artifacts/smoke
	$(PYTHON) -m illicit_graph.synthetic $(SMOKE_DIR)/raw --nodes 12000
	$(PYTHON) -m illicit_graph.data --source $(SMOKE_DIR)/raw --data-dir $(SMOKE_DIR)/data
	$(PYTHON) -m illicit_graph.network --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --samples 32
	$(PYTHON) -m illicit_graph.train --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m illicit_graph.evaluate --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--data-label synthetic --bootstrap 20
	$(PYTHON) -m illicit_graph.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--reports $(SMOKE_DIR)/reports
	$(PYTHON) -m illicit_graph.neo4j_export export --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR)

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
