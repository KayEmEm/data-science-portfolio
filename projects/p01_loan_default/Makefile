# Project 1: loan default risk. Run from this folder.
PYTHON ?= python
SMOKE_DIR := artifacts/smoke

.PHONY: install data features train evaluate all smoke test lint api docker clean

install:
	$(PYTHON) -m pip install -e ../../common -e ".[dev]"

data:            ## download from the UCI repository and convert to Parquet
	$(PYTHON) -m loan_default.data

features:
	$(PYTHON) -m loan_default.features

train:
	$(PYTHON) -m loan_default.train

evaluate:
	$(PYTHON) -m loan_default.evaluate

all: data features train evaluate

smoke:           ## full pipeline on a synthetic table; outputs stay under artifacts/smoke
	$(PYTHON) -m loan_default.synthetic $(SMOKE_DIR)/clients.parquet --n 3000
	$(PYTHON) -m loan_default.features --clients $(SMOKE_DIR)/clients.parquet --out $(SMOKE_DIR)/features.parquet
	$(PYTHON) -m loan_default.train --features $(SMOKE_DIR)/features.parquet --artifacts $(SMOKE_DIR) --quick --data-label synthetic
	$(PYTHON) -m loan_default.evaluate --artifacts $(SMOKE_DIR) --reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

api:
	$(PYTHON) -m uvicorn loan_default.api:app --host 0.0.0.0 --port 8000

docker:          ## build context is the repository root, because common/ is shared
	docker build -f Dockerfile -t loan-default-api ../..

clean:
	rm -rf artifacts
