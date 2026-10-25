# Project 2: question answering over company filings. Run from this folder.
PYTHON ?= python
SMOKE_DIR := artifacts/smoke
# Share of filings kept by `make sample`, each with all its questions; outputs stay under artifacts/sample.
SAMPLE ?= 0.25
SAMPLE_DIR := artifacts/sample
# Paid steps stop before their spending reaches this many US dollars per run.
MAX_USD ?= 5
CLAUDE_MODEL ?= claude-opus-5-5
OLLAMA_MODEL ?= qwen2.5:7b
CONTEXTS ?= artifacts/contexts/doc_dense_k8.parquet

.PHONY: install data parse index evaluate-retrieval sections baseline estimate answer-claude answer-ollama \
	grade report all sample smoke test lint demo clean

install:
	$(PYTHON) -m pip install -e ../../common -e ".[dev,demo]"

data:                ## FinanceBench questions and the 84 filings they cite (about 300 MB of PDFs)
	$(PYTHON) -m filings_qa.data

parse:               ## PDF pages to text, pages to passages
	$(PYTHON) -m filings_qa.parse

index:               ## BM25 and Chroma (MiniLM) indexes; the dense index takes about 90 minutes on 4 CPU cores
	$(PYTHON) -m filings_qa.retrieval

evaluate-retrieval:  ## recall at k per retriever, and the passages each question receives
	$(PYTHON) -m filings_qa.evaluate_retrieval

sections:            ## 10-K section classifier
	$(PYTHON) -m filings_qa.sections

baseline:            ## extractive answers, no language model
	$(PYTHON) -m filings_qa.generation --generator extractive --contexts $(CONTEXTS)

estimate:            ## cost estimate for one Claude run, no call made
	$(PYTHON) -m filings_qa.generation --generator claude --model $(CLAUDE_MODEL) --contexts $(CONTEXTS) --estimate

answer-claude:       ## paid: needs ANTHROPIC_API_KEY
	$(PYTHON) -m filings_qa.generation --generator claude --model $(CLAUDE_MODEL) --contexts $(CONTEXTS) --max-usd $(MAX_USD)

answer-ollama:       ## needs a running Ollama with the model pulled
	$(PYTHON) -m filings_qa.generation --generator ollama --model $(OLLAMA_MODEL) --contexts $(CONTEXTS)

grade:               ## paid: the Claude judge grades every run that has no grades yet
	$(PYTHON) -m filings_qa.scoring --max-usd $(MAX_USD)

report:
	$(PYTHON) -m filings_qa.report

all: data parse index evaluate-retrieval sections baseline report

sample:              ## the whole free pipeline on a seeded share SAMPLE of the filings, with the real models
	$(PYTHON) -m filings_qa.data --sample $(SAMPLE) --out-dir $(SAMPLE_DIR)/data
	$(PYTHON) -m filings_qa.parse --documents $(SAMPLE_DIR)/data/documents.parquet \
		--pages $(SAMPLE_DIR)/data/pages.parquet --chunks $(SAMPLE_DIR)/data/chunks.parquet
	$(PYTHON) -m filings_qa.retrieval --chunks $(SAMPLE_DIR)/data/chunks.parquet --index-dir $(SAMPLE_DIR)/index
	$(PYTHON) -m filings_qa.evaluate_retrieval --questions $(SAMPLE_DIR)/data/questions.parquet \
		--chunks $(SAMPLE_DIR)/data/chunks.parquet --index-dir $(SAMPLE_DIR)/index --reports $(SAMPLE_DIR)/reports \
		--contexts $(SAMPLE_DIR)/contexts
	$(PYTHON) -m filings_qa.sections --pages $(SAMPLE_DIR)/data/pages.parquet --index-dir $(SAMPLE_DIR)/index \
		--reports $(SAMPLE_DIR)/reports
	$(PYTHON) -m filings_qa.generation --generator extractive --contexts $(SAMPLE_DIR)/contexts/doc_dense_k8.parquet \
		--questions $(SAMPLE_DIR)/data/questions.parquet --runs-dir $(SAMPLE_DIR)/runs
	$(PYTHON) -m filings_qa.report --runs-dir $(SAMPLE_DIR)/runs --questions $(SAMPLE_DIR)/data/questions.parquet \
		--reports $(SAMPLE_DIR)/reports

smoke:               ## the whole free pipeline on a synthetic corpus; outputs stay under artifacts/smoke
	$(PYTHON) -m filings_qa.synthetic $(SMOKE_DIR)/data
	$(PYTHON) -m filings_qa.parse --skip-pdfs --pages $(SMOKE_DIR)/data/pages.parquet --chunks $(SMOKE_DIR)/data/chunks.parquet
	$(PYTHON) -m filings_qa.retrieval --chunks $(SMOKE_DIR)/data/chunks.parquet --index-dir $(SMOKE_DIR)/index --hashing-embeddings
	$(PYTHON) -m filings_qa.evaluate_retrieval --questions $(SMOKE_DIR)/data/questions.parquet \
		--chunks $(SMOKE_DIR)/data/chunks.parquet --index-dir $(SMOKE_DIR)/index --reports $(SMOKE_DIR)/reports \
		--contexts $(SMOKE_DIR)/contexts --hashing-embeddings
	$(PYTHON) -m filings_qa.generation --generator extractive --contexts $(SMOKE_DIR)/contexts/doc_hybrid_k8.parquet \
		--questions $(SMOKE_DIR)/data/questions.parquet --runs-dir $(SMOKE_DIR)/runs
	$(PYTHON) -m filings_qa.report --runs-dir $(SMOKE_DIR)/runs --questions $(SMOKE_DIR)/data/questions.parquet \
		--reports $(SMOKE_DIR)/reports

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

demo:                ## after make index; add ARGS="--claude-model claude-opus-5-5" for Claude answers (paid)
	$(PYTHON) -m filings_qa.demo $(ARGS)

clean:
	rm -rf artifacts
