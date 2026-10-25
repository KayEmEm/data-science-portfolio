# Project 13: adversarial robustness of the models of projects 1, 6 and 8. Run from this folder.
# `make sample` (SAMPLE=0.2) gives the committed results; `make all` runs on the full data.
PYTHON ?= python
SAMPLE ?= 0.2
SAMPLE_DIR := artifacts/sample
SMOKE_DIR := artifacts/smoke
WEIGHTS ?= data/raw/detector/best.pt
CREDIT_SOURCE ?= data/raw/credit
FLOWS_SOURCE ?= data/raw/flows
IMAGES_SOURCE ?= data/raw/images
SOURCES := --credit-source $(CREDIT_SOURCE) --flows-source $(FLOWS_SOURCE) --images-source $(IMAGES_SOURCE)

.PHONY: install data attack report all sample smoke test lint clean

install:             ## this project, the shared common package and the three target projects
	$(PYTHON) -m pip install -e ../../common -e ../p01_loan_default -e ../p06_road_damage \
		-e ../p08_network_anomaly -e ".[dev]"

data:                ## the full data of each part, read by the loaders of projects 1, 6 and 8
	$(PYTHON) -m adv_robustness.data $(SOURCES)

attack:              ## every attack and defense; results under artifacts/<part>/results.json
	$(PYTHON) -m adv_robustness.credit
	$(PYTHON) -m adv_robustness.flows
	$(PYTHON) -m adv_robustness.images --weights $(WEIGHTS)

report:              ## scorecard, defense table, figures and results.json under reports/
	$(PYTHON) -m adv_robustness.report

all: data attack report

sample:              ## the whole pipeline on a seeded share SAMPLE of each data set; writes reports/
	$(PYTHON) -m adv_robustness.data $(SOURCES) --data-dir $(SAMPLE_DIR)/data --sample $(SAMPLE)
	$(PYTHON) -m adv_robustness.credit --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR)
	$(PYTHON) -m adv_robustness.flows --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR)
	$(PYTHON) -m adv_robustness.images --data-dir $(SAMPLE_DIR)/data --artifacts $(SAMPLE_DIR) --weights $(WEIGHTS)
	$(PYTHON) -m adv_robustness.report --artifacts $(SAMPLE_DIR) --data-label "sample $(SAMPLE)"

smoke:               ## the whole pipeline on synthetic sources and an untrained detector; outputs stay under artifacts/smoke
	$(PYTHON) -m adv_robustness.synthetic $(SMOKE_DIR)/raw
	$(PYTHON) -m adv_robustness.data --data-dir $(SMOKE_DIR)/data --block 4 \
		--credit-source $(SMOKE_DIR)/raw/credit --flows-source $(SMOKE_DIR)/raw/flows \
		--images-source $(SMOKE_DIR)/raw/images
	$(PYTHON) -m adv_robustness.credit --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m adv_robustness.flows --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m adv_robustness.images --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--weights $(SMOKE_DIR)/raw/detector/untrained.pt --imgsz 160 --quick
	$(PYTHON) -m adv_robustness.report --artifacts $(SMOKE_DIR) --reports $(SMOKE_DIR)/reports --data-label synthetic

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
