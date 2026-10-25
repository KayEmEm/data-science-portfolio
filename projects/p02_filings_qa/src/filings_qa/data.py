"""Download the FinanceBench open sample and the filings it cites, check them, store as Parquet."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
from portfolio_common import sample

from filings_qa.config import (
    DOCUMENTS_FILE,
    DOCUMENTS_PATH,
    PDF_DIR,
    QUESTIONS_FILE,
    QUESTIONS_PATH,
    RAW_DIR,
    REQUIRED_DOCUMENT_FIELDS,
    REQUIRED_QUESTION_FIELDS,
    SOURCE_BASE,
)


class SchemaError(ValueError):
    """A source file lacks a required field or holds an inconsistent record."""


def fetch(url: str, target: Path) -> Path:
    """Download ``url`` to ``target`` unless the file is already present."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(target)
    return target


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_questions(records: list[dict]) -> pd.DataFrame:
    """Check the question records and flatten each evidence list into page references.

    ``evidence_pages`` holds ``"<doc_name>#<page>"`` strings with zero based
    page numbers, as in the source file.
    """
    rows = []
    for record in records:
        missing = [f for f in REQUIRED_QUESTION_FIELDS if f not in record]
        if missing:
            raise SchemaError(f"question {record.get('financebench_id')} lacks {missing}")
        if not record["evidence"]:
            raise SchemaError(f"question {record['financebench_id']} has no evidence")
        # The README of the source names the field ``evidence_doc_name``; the
        # file itself uses ``doc_name``. Both spellings are accepted.
        pages = sorted(
            {
                f"{e.get('evidence_doc_name', e.get('doc_name'))}#{int(e['evidence_page_num'])}"
                for e in record["evidence"]
            }
        )
        rows.append(
            {
                "question_id": str(record["financebench_id"]),
                "company": record["company"],
                "doc_name": record["doc_name"],
                "question_type": record["question_type"],
                "question_reasoning": record["question_reasoning"] or "",
                "question": record["question"],
                "answer": str(record["answer"]),
                "evidence_text": "\n\n".join(e["evidence_text"] for e in record["evidence"]),
                "evidence_pages": pages,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["question_id"].duplicated().any():
        raise SchemaError("duplicate question ids")
    return frame


def normalize_documents(records: list[dict]) -> pd.DataFrame:
    for record in records:
        missing = [f for f in REQUIRED_DOCUMENT_FIELDS if f not in record]
        if missing:
            raise SchemaError(f"document {record.get('doc_name')} lacks {missing}")
    frame = pd.DataFrame(records)
    # The source lists FOOTLOCKER_2023_annualreport twice, with the same link
    # and two periods. A repeated name is accepted only when it points to the
    # same file; the first record is kept.
    repeated = frame[frame["doc_name"].duplicated(keep=False)]
    if (repeated.groupby("doc_name")["doc_link"].nunique() > 1).any():
        raise SchemaError("one document name points to two different files")
    frame = frame.drop_duplicates("doc_name", keep="first")
    frame = frame[REQUIRED_DOCUMENT_FIELDS + ["gics_sector"]].reset_index(drop=True)
    frame["doc_type"] = frame["doc_type"].str.upper()
    return frame


def documents_in_scope(questions: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Return the documents that the questions or their evidence cite."""
    cited = set(questions["doc_name"])
    cited |= {ref.split("#")[0] for refs in questions["evidence_pages"] for ref in refs}
    unknown = cited - set(documents["doc_name"])
    if unknown:
        raise SchemaError(f"questions cite unknown documents {sorted(unknown)}")
    return documents[documents["doc_name"].isin(cited)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pdfs", action="store_true", help="download the question files only")
    parser.add_argument("--out-dir", type=Path, default=QUESTIONS_PATH.parent)
    sample.add_arguments(parser, "filings, each with all the questions on it")
    args = parser.parse_args()

    questions = normalize_questions(
        read_jsonl(fetch(f"{SOURCE_BASE}/data/{QUESTIONS_FILE}", RAW_DIR / QUESTIONS_FILE))
    )
    n_filings = questions["doc_name"].nunique()
    questions = questions[sample.select(questions["doc_name"], args.sample, args.sample_seed)]
    questions = questions.reset_index(drop=True)
    documents = documents_in_scope(
        questions,
        normalize_documents(
            read_jsonl(fetch(f"{SOURCE_BASE}/data/{DOCUMENTS_FILE}", RAW_DIR / DOCUMENTS_FILE))
        ),
    )
    if not args.skip_pdfs:
        for i, name in enumerate(documents["doc_name"], start=1):
            fetch(f"{SOURCE_BASE}/pdfs/{name}.pdf", PDF_DIR / f"{name}.pdf")
            print(f"\r{i}/{len(documents)} filings", end="", flush=True)
        print()
    sample.record(
        args.out_dir, args.sample, args.sample_seed, "filing", n_filings, questions["doc_name"].nunique()
    )
    questions.to_parquet(args.out_dir / QUESTIONS_PATH.name, index=False)
    documents.to_parquet(args.out_dir / DOCUMENTS_PATH.name, index=False)
    print(
        f"{len(questions)} questions on {len(documents)} filings of "
        f"{documents['company'].nunique()} companies written to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
