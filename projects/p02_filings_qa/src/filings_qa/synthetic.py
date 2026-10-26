"""A small synthetic corpus in the shape of the real one, for tests and the smoke run.

Each company has one annual report of a few pages. One page per report states
its revenue and one states its capital expenditure; the other pages hold
filler text. Every question asks for one of these figures. The numbers are
random and say nothing about any real company.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common.config import parse_args

from filings_qa.config import SEED

COMPANIES = ["Alder Works", "Birch Labs", "Cedar Freight", "Dogwood Foods", "Elm Energy", "Fir Retail"]
FILLER = (
    "The company operates in several markets and reviews its strategy each year. "
    "Management discusses risks, competition, regulation and liquidity in this report. "
)


def build(n_pages: int = 6, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``questions``, ``documents`` and ``pages`` frames with the columns of the real files."""
    rng = np.random.default_rng(seed)
    questions, documents, pages = [], [], []
    for i, company in enumerate(COMPANIES):
        doc_name = f"{company.split()[0].upper()}_2022_10K"
        documents.append(
            {
                "doc_name": doc_name,
                "company": company,
                "doc_type": "10K",
                "doc_period": 2022,
                "gics_sector": "Test",
            }
        )
        revenue, capex = int(rng.integers(1_000, 90_000)), int(rng.integers(100, 9_000))
        revenue_page, capex_page = 1, n_pages - 2
        for page in range(n_pages):
            text = FILLER * 3
            if page == revenue_page:
                text += f"Total net revenue of {company} for fiscal 2022 was ${revenue:,} million. "
            if page == capex_page:
                text += (
                    f"Capital expenditures (purchases of property and equipment) were ${capex:,} million. "
                )
            pages.append({"doc_name": doc_name, "page": page, "text": text + FILLER})
        for kind, value, page in (
            ("total net revenue", revenue, revenue_page),
            ("capital expenditure", capex, capex_page),
        ):
            questions.append(
                {
                    "question_id": f"syn_{i}_{kind.split()[-1]}",
                    "company": company,
                    "doc_name": doc_name,
                    "question_type": "metrics-generated",
                    "question_reasoning": "Information extraction",
                    "question": f"What was the FY2022 {kind} of {company} in USD millions?",
                    "answer": f"${value}.00",
                    "evidence_text": f"{kind} {value}",
                    "evidence_pages": [f"{doc_name}#{page}"],
                }
            )
    documents = pd.DataFrame(documents)
    pages = pd.DataFrame(pages).merge(
        documents[["doc_name", "company", "doc_type", "doc_period"]], on="doc_name"
    )
    return pd.DataFrame(questions), documents, pages


def write(out_dir: Path) -> None:
    questions, documents, pages = build()
    out_dir.mkdir(parents=True, exist_ok=True)
    questions.to_parquet(out_dir / "questions.parquet", index=False)
    documents.to_parquet(out_dir / "documents.parquet", index=False)
    pages.to_parquet(out_dir / "pages.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    write(parse_args(parser, __file__).out_dir)


if __name__ == "__main__":
    main()
