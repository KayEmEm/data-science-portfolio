# Project 7: movie recommendations on MovieLens 32M. Run from this folder.
PYTHON ?= python
SOURCE ?=
SMOKE_DIR := artifacts/smoke

.PHONY: install data train evaluate report all demo smoke test lint clean

install:             ## this project with the demo extra and the shared common package
	$(PYTHON) -m pip install -e ../../common -e ".[dev,demo]"

data:                ## download ml-32m.zip (or SOURCE=<zip>), check its MD5 and counts, write tables and the time split
	$(PYTHON) -m movie_recs.data $(if $(SOURCE),--source $(SOURCE))

train:               ## popularity, ALS (grid) and two tower for the validation and test stages, then the ranker; resumes
	$(PYTHON) -m movie_recs.train
	$(PYTHON) -m movie_recs.rank

evaluate:            ## recall, NDCG, coverage and novelty at 10; cold start users and movies
	$(PYTHON) -m movie_recs.evaluate

report:              ## tables, figures, demo lists and results.json under reports/
	$(PYTHON) -m movie_recs.report

all: data train evaluate report

demo:                ## Streamlit page: pick a user, see the history and the top 10 of each model
	$(PYTHON) -m streamlit run demo/app.py

smoke:               ## the whole pipeline on a synthetic archive; outputs stay under artifacts/smoke
	$(PYTHON) -m movie_recs.synthetic $(SMOKE_DIR)/ml-32m.zip
	$(PYTHON) -m movie_recs.data --source $(SMOKE_DIR)/ml-32m.zip --data-dir $(SMOKE_DIR)/data --no-count-check \
		--md5 any
	$(PYTHON) -m movie_recs.train --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m movie_recs.rank --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m movie_recs.evaluate --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --data-label synthetic
	$(PYTHON) -m movie_recs.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests demo
	$(PYTHON) -m ruff format --check src tests demo

clean:
	rm -rf artifacts
