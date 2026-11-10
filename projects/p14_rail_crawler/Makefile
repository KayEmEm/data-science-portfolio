# Project 14: rail crawler. Run from this folder.
# CONFIG selects the settings file; ARGS passes extra options, for example ARGS="--max-pages 3".
PYTHON ?= python
CONFIG ?= cobweb.yaml
ARGS ?=
SMOKE_DIR := artifacts/smoke

.PHONY: install browser crawl resume catch-up fill-gaps gaps dry-run smoke sample test lint clean

install:             ## this project and Chromium for Playwright
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m playwright install chromium

browser:             ## Chromium and its system libraries (Linux, needs sudo); for a fresh machine or CI
	$(PYTHON) -m playwright install --with-deps chromium

crawl:               ## crawl the series in CONFIG
	$(PYTHON) -m rail_spider $(CONFIG) $(ARGS)

resume:              ## continue from the page saved in state_file
	$(PYTHON) -m rail_spider $(CONFIG) --resume $(ARGS)

catch-up:            ## walk from the entry URL past saved positions, fill gaps, continue
	$(PYTHON) -m rail_spider $(CONFIG) --catch-up $(ARGS)

fill-gaps:           ## walk from the entry URL and fill only the gaps
	$(PYTHON) -m rail_spider $(CONFIG) --fill-gaps $(ARGS)

gaps:                ## list the gaps in register_file and download_dir; no network
	$(PYTHON) -m rail_spider $(CONFIG) --gaps

dry-run:             ## log the extracted links without downloading
	$(PYTHON) -m rail_spider $(CONFIG) --dry-run $(ARGS)

smoke:               ## crawl the synthetic demo site headless; outputs stay under artifacts/smoke
	$(PYTHON) -m rail_spider.smoke $(SMOKE_DIR)

sample: smoke        ## for the Colab profiles: the smoke crawl; SAMPLE has no effect here

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	rm -rf artifacts
