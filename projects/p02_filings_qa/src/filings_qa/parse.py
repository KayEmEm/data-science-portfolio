"""Extract the text of every filing page, and split pages into passages for retrieval."""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from filings_qa.config import CHUNKS_PATH, DOCUMENTS_PATH, PAGES_PATH, PDF_DIR, ChunkSettings

_SPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def clean_text(text: str) -> str:
    """Collapse runs of spaces and blank lines; keep single line breaks, which mark table rows."""
    text = _SPACE.sub(" ", text.replace("\r", "\n"))
    lines = [line.strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def read_pdf(path: Path) -> list[str]:
    """Return the cleaned text of each page; page ``i`` of the list is zero based page ``i``."""
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(clean_text(page.extract_text() or ""))
        except Exception:  # a damaged page must not stop the whole filing
            pages.append("")
    return pages


def _pages_of(args: tuple[str, Path]) -> list[dict]:
    doc_name, path = args
    return [{"doc_name": doc_name, "page": i, "text": t} for i, t in enumerate(read_pdf(path))]


def parse_documents(documents: pd.DataFrame, pdf_dir: Path, workers: int = 4) -> pd.DataFrame:
    jobs = [(name, pdf_dir / f"{name}.pdf") for name in documents["doc_name"]]
    missing = [str(p) for _, p in jobs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} filings are missing, for example {missing[0]}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for pages in pool.map(_pages_of, jobs) for row in pages]
    pages = pd.DataFrame(rows, columns=["doc_name", "page", "text"])
    return pages.merge(documents[["doc_name", "company", "doc_type", "doc_period"]], on="doc_name")


def chunk_page(text: str, settings: ChunkSettings) -> list[str]:
    """Split one page into word windows that overlap by ``settings.overlap`` words."""
    words = text.split()
    if not words:
        return []
    starts = [0]
    while starts[-1] + settings.words < len(words):
        starts.append(starts[-1] + settings.words - settings.overlap)
    return [" ".join(words[s : s + settings.words]) for s in starts]


def build_chunks(pages: pd.DataFrame, settings: ChunkSettings | None = None) -> pd.DataFrame:
    """Return one row per passage with a stable id ``<doc_name>#<page>#<n>``."""
    settings = settings or ChunkSettings()
    rows = []
    for page in pages.itertuples(index=False):
        for n, text in enumerate(chunk_page(page.text, settings)):
            rows.append(
                {
                    "chunk_id": f"{page.doc_name}#{page.page}#{n}",
                    "doc_name": page.doc_name,
                    "page": page.page,
                    "company": page.company,
                    "doc_type": page.doc_type,
                    "doc_period": page.doc_period,
                    "text": text,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--pages", type=Path, default=PAGES_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--skip-pdfs", action="store_true", help="chunk an existing pages file")
    args = parser.parse_args()

    if args.skip_pdfs:
        pages = pd.read_parquet(args.pages)
    else:
        pages = parse_documents(pd.read_parquet(args.documents), args.pdf_dir)
        args.pages.parent.mkdir(parents=True, exist_ok=True)
        pages.to_parquet(args.pages, index=False)
    chunks = build_chunks(pages)
    chunks.to_parquet(args.chunks, index=False)
    empty = (pages["text"].str.len() == 0).mean()
    print(
        f"{len(pages):,} pages ({empty:.1%} without text) and {len(chunks):,} passages "
        f"written to {args.chunks.parent}"
    )


if __name__ == "__main__":
    main()
