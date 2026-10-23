"""Paths, source locations, prices and run settings for project 2."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = RAW_DIR / "pdfs"
QUESTIONS_PATH = DATA_DIR / "questions.parquet"
DOCUMENTS_PATH = DATA_DIR / "documents.parquet"
PAGES_PATH = DATA_DIR / "pages.parquet"
CHUNKS_PATH = DATA_DIR / "chunks.parquet"
INDEX_DIR = DATA_DIR / "index"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# FinanceBench open sample (Patronus AI), CC BY-NC 4.0. The files are read from
# a fixed commit so that a later change in the repository cannot change results.
SOURCE_COMMIT = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
SOURCE_BASE = f"https://raw.githubusercontent.com/patronus-ai/financebench/{SOURCE_COMMIT}"
QUESTIONS_FILE = "financebench_open_source.jsonl"
DOCUMENTS_FILE = "financebench_document_information.jsonl"

SEED = 42

REQUIRED_QUESTION_FIELDS = [
    "financebench_id",
    "company",
    "doc_name",
    "question_type",
    "question_reasoning",
    "question",
    "answer",
    "evidence",
]
REQUIRED_DOCUMENT_FIELDS = ["doc_name", "company", "doc_type", "doc_period"]

# Anthropic list prices in USD per million tokens (input, output), from the
# model table of the Claude API documentation, 2026-10. Cache reads and
# batch discounts are not used by this project.
PRICES_USD_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# The API endpoint is fixed here rather than read from ANTHROPIC_BASE_URL, so
# that a base URL set for another tool never receives this project's key.
ANTHROPIC_API_URL = "https://api.anthropic.com"
API_KEY_ENV = "ANTHROPIC_API_KEY"
OLLAMA_URL_ENV = "OLLAMA_HOST"
OLLAMA_DEFAULT_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ChunkSettings:
    """Chunks stay inside one page, so every passage cites exactly one page.

    ``words`` is close to the 256 token input limit of the MiniLM embedding
    model, so the dense index sees the whole chunk.
    """

    words: int = 180
    overlap: int = 40


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 8
    candidates: int = 50
    rrf_k: int = 60
