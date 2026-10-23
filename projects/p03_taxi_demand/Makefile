# Project 3: New York taxi demand, exploratory analysis and dashboard. Run from this folder.
PYTHON ?= python
YEAR ?= 2024
SMOKE_DIR := artifacts/smoke

.PHONY: install data ingest build report notebook dashboard screenshots all smoke test lint clean

install:
	$(PYTHON) -m pip install -e ../../common -e ".[dev,notebook,screenshots]"

data:                ## TLC trip files for one year (about 650 MB), the zone lookup and NOAA weather
	$(PYTHON) -m taxi_demand.data --year $(YEAR)

ingest:              ## files downloaded by hand: make ingest SOURCES="~/Downloads/taxi"
	$(PYTHON) -m taxi_demand.ingest $(SOURCES)
	$(PYTHON) -m taxi_demand.data --year $(YEAR) --skip-download

build:               ## DuckDB warehouse, quality audit and dashboard tables
	$(PYTHON) -m taxi_demand.warehouse --year $(YEAR)

report:              ## query library results, key numbers and figures under reports/
	$(PYTHON) -m taxi_demand.report

notebook:            ## generate and execute notebooks/analysis.ipynb
	$(PYTHON) -m taxi_demand.notebook

dashboard:           ## http://localhost:8501
	$(PYTHON) -m streamlit run dashboard/app.py

screenshots:         ## needs Playwright Chromium (playwright install chromium) or CHROMIUM_PATH
	$(PYTHON) -m taxi_demand.screenshots

all: data build report notebook

smoke:               ## the whole pipeline on synthetic files; outputs stay under artifacts/smoke
	$(PYTHON) -m taxi_demand.synthetic $(SMOKE_DIR)/raw
	$(PYTHON) -m taxi_demand.data --raw-dir $(SMOKE_DIR)/raw --skip-download
	$(PYTHON) -m taxi_demand.warehouse --raw-dir $(SMOKE_DIR)/raw --warehouse $(SMOKE_DIR)/taxi.duckdb \
		--marts $(SMOKE_DIR)/marts
	$(PYTHON) -m taxi_demand.report --warehouse $(SMOKE_DIR)/taxi.duckdb --reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests dashboard dags
	$(PYTHON) -m ruff format --check src tests dashboard dags

clean:
	rm -rf artifacts
