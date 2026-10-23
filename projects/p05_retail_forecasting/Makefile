# Project 5: retail demand forecasting on M5. Run from this folder.
PYTHON ?= python
SMOKE_DIR := artifacts/smoke

.PHONY: install data train evaluate report all smoke test lint clean

install:             ## this project, the shared common package and Darts with PyTorch (about 1 GB)
	$(PYTHON) -m pip install -e ../../common -e ".[dev,neural]"

data:                ## download the M5 zip (50 MB), check it, write wide tables and demand tables per level
	$(PYTHON) -m retail_forecast.data

train:               ## forecasts of every model for the 4 backtest windows and the test window
	$(PYTHON) -m retail_forecast.backtest

evaluate:            ## WRMSSE, MAPE, reconciliation and interval coverage; model choice on the backtest
	$(PYTHON) -m retail_forecast.evaluate

report:              ## tables, figures and results.json under reports/
	$(PYTHON) -m retail_forecast.report

all: data train evaluate report

smoke:               ## the whole pipeline on a synthetic archive; outputs stay under artifacts/smoke
	$(PYTHON) -m retail_forecast.synthetic $(SMOKE_DIR)/m5_synthetic.zip
	$(PYTHON) -m retail_forecast.data --source $(SMOKE_DIR)/m5_synthetic.zip --data-dir $(SMOKE_DIR)/data \
		--expected-series 0
	$(PYTHON) -m retail_forecast.backtest --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m retail_forecast.evaluate --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick \
		--data-label synthetic
	$(PYTHON) -m retail_forecast.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
