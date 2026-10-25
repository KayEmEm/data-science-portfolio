# Project 12: inventory control with reinforcement learning. Run from this folder.
# SAMPLE=0.1 runs every step on a sample: one M5 store, and that fraction of the training steps,
# seeds and scenarios. Without it, the full run applies.
PYTHON ?= python
SOURCE ?=
SAMPLE ?= 1
JOBS ?= 4
SMOKE_DIR := artifacts/smoke
RUN := --sample $(SAMPLE)

.PHONY: install data baselines validate train evaluate report all smoke test lint clean

install:             ## this project and the shared common package
	$(PYTHON) -m pip install -e ../../common -e ".[dev]"

data:                ## download the M5 zip (50 MB) or read SOURCE=<zip>, select three products, fit demand
	$(PYTHON) -m inventory_rl.data $(RUN) $(if $(SOURCE),--source $(SOURCE))

baselines:           ## tune base stock (R, S) and (s, S) by simulation
	$(PYTHON) -m inventory_rl.baselines $(RUN)

validate:            ## exact optimum by dynamic programming (slow product, fixed lead time) against the simulator
	$(PYTHON) -m inventory_rl.dp $(RUN)

train:               ## PPO and SAC per product and seed, and the reward shaping study; resumes
	$(PYTHON) -m inventory_rl.train $(RUN) --jobs $(JOBS)

evaluate:            ## every policy on the test scenarios and on a replay of the real test year
	$(PYTHON) -m inventory_rl.evaluate $(RUN)

report:              ## tables, figures, the animation and results.json under reports/
	$(PYTHON) -m inventory_rl.report
	$(PYTHON) -m inventory_rl.animate

all: data baselines validate train evaluate report

smoke:               ## the whole pipeline on a synthetic archive with tiny runs; outputs stay under artifacts/smoke
	$(PYTHON) -m inventory_rl.synthetic $(SMOKE_DIR)/m5_synthetic.zip
	$(PYTHON) -m inventory_rl.data --source $(SMOKE_DIR)/m5_synthetic.zip --data-dir $(SMOKE_DIR)/data
	$(PYTHON) -m inventory_rl.baselines --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m inventory_rl.dp --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick
	$(PYTHON) -m inventory_rl.train --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick --jobs 2
	$(PYTHON) -m inventory_rl.evaluate --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) --quick \
		--data-label synthetic
	$(PYTHON) -m inventory_rl.report --data-dir $(SMOKE_DIR)/data --artifacts $(SMOKE_DIR) \
		--reports $(SMOKE_DIR)/reports --validation-episodes 4
	$(PYTHON) -m inventory_rl.animate --artifacts $(SMOKE_DIR) --reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
