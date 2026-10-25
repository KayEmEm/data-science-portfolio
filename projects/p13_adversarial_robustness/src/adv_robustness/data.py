"""Read the data of the four target projects with their own loaders, draw the sample, and store it.

* credit: the UCI spreadsheet of project 1 (or a client table in its schema), the feature SQL of
  project 1, a sample of clients stratified by default;
* flows: the UNSW-NB15 files of project 8, its split, a sample of flows stratified by attack type and
  split;
* images: the RDD2022 country zips of project 6, its block split, a sample of blocks;
* filings: the FinanceBench questions and filings of project 2, a sample of filings with all their
  questions, the pages and passages of project 2, and the BIPIA text attacks.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from portfolio_common import sample

from adv_robustness.config import (
    BIPIA_BASE,
    BIPIA_FILES,
    CREDIT_SOURCE,
    DATA_DIR,
    FLOWS_SOURCE,
    IMAGE_COUNTRIES,
    IMAGES_SOURCE,
    PARTS,
)


class SourceError(FileNotFoundError):
    """A source folder lacks the files a part needs."""


def credit(source: Path, out: Path, share: float, seed: int) -> dict:
    from loan_default.config import ID_COL, TARGET_COL
    from loan_default.data import find_spreadsheet, normalize, read_spreadsheet
    from loan_default.features import build_features

    out.mkdir(parents=True, exist_ok=True)
    tables = sorted(source.glob("*.parquet"))
    if tables:
        clients = normalize(pd.read_parquet(tables[0]).rename(columns={"CLIENT_ID": "ID"}))
    else:
        raw = out / "raw"
        raw.mkdir(exist_ok=True)
        for path in source.iterdir():
            if path.suffix.lower() in {".zip", ".xls", ".xlsx"}:
                shutil.copy2(path, raw / path.name)
        try:
            clients = read_spreadsheet(find_spreadsheet(raw))
        except FileNotFoundError as error:
            raise SourceError(f"no UCI zip, spreadsheet or client table in {source}") from error
    n_full = len(clients)
    clients = clients[sample.select(clients[ID_COL], share, seed, strata=clients[TARGET_COL])]
    clients = clients.reset_index(drop=True)
    clients.to_parquet(out / "clients.parquet", index=False)
    build_features(out / "clients.parquet").to_parquet(out / "features.parquet", index=False)
    return sample.record(out, share, seed, "client", n_full, len(clients))


def flows(source: Path, out: Path, share: float, seed: int) -> dict:
    from flow_anomaly.data import prepare, write

    if not source.exists():
        raise SourceError(f"no UNSW-NB15 files in {source}")
    table, report = prepare(source)
    n_full = len(table)
    strata = table["attack_cat"].astype(str) + "_" + table["split"].astype(str)
    table = table[sample.select(pd.Series(np.arange(n_full)), share, seed, strata=strata)]
    write(table.reset_index(drop=True), report, out)
    return sample.record(out, share, seed, "flow", n_full, len(table))


def images(source: Path, out: Path, share: float, seed: int, block: int | None = None) -> dict:
    from road_damage.config import SPLIT_BLOCK
    from road_damage.data import build

    if not any(source.glob("*.zip")) and not any(source.iterdir()):
        raise SourceError(f"no RDD2022 zips in {source}")
    counts = build(source, out, IMAGE_COUNTRIES, block or SPLIT_BLOCK, share, seed)
    (out / "counts.json").write_text(json.dumps(counts, indent=2))
    return json.loads((out / sample.RECORD_NAME).read_text())


def bipia(source: Path | None, out: Path) -> dict[str, dict[str, list[str]]]:
    """The BIPIA text attacks per split: a category name and its attack instructions."""
    from filings_qa.data import fetch

    attacks = {}
    for split, name in BIPIA_FILES.items():
        local = source / name if source is not None else None
        path = local if local is not None and local.exists() else fetch(f"{BIPIA_BASE}/{name}", out / name)
        attacks[split] = json.loads(Path(path).read_text(encoding="utf-8"))
    return attacks


def filings(source: Path | None, out: Path, share: float, seed: int) -> dict:
    """``source`` holds the parquet files of the project 2 synthetic corpus; None downloads the real one."""
    from filings_qa.config import DOCUMENTS_FILE, QUESTIONS_FILE, SOURCE_BASE
    from filings_qa.data import documents_in_scope, fetch, normalize_documents, normalize_questions, read_jsonl
    from filings_qa.parse import build_chunks, parse_documents

    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    if source is not None and (source / "pages.parquet").exists():
        questions = pd.read_parquet(source / "questions.parquet")
        documents = pd.read_parquet(source / "documents.parquet")
        pages = pd.read_parquet(source / "pages.parquet")
    else:
        questions = normalize_questions(
            read_jsonl(fetch(f"{SOURCE_BASE}/data/{QUESTIONS_FILE}", raw / QUESTIONS_FILE))
        )
        documents = normalize_documents(
            read_jsonl(fetch(f"{SOURCE_BASE}/data/{DOCUMENTS_FILE}", raw / DOCUMENTS_FILE))
        )
        pages = None
    n_filings = questions["doc_name"].nunique()
    questions = questions[sample.select(questions["doc_name"], share, seed)].reset_index(drop=True)
    # Only the filing each question names is attacked and searched, as in the per filing retrieval
    # of project 2.
    documents = documents[documents["doc_name"].isin(set(questions["doc_name"]))].reset_index(drop=True)
    if pages is None:
        documents = documents_in_scope(questions.assign(evidence_pages=[[]] * len(questions)), documents)
        for i, name in enumerate(documents["doc_name"], start=1):
            fetch(f"{SOURCE_BASE}/pdfs/{name}.pdf", raw / "pdfs" / f"{name}.pdf")
            print(f"\r{i}/{len(documents)} filings", end="", flush=True)
        print()
        pages = parse_documents(documents, raw / "pdfs")
    else:
        pages = pages[pages["doc_name"].isin(set(documents["doc_name"]))]
    questions.to_parquet(out / "questions.parquet", index=False)
    documents.to_parquet(out / "documents.parquet", index=False)
    build_chunks(pages).to_parquet(out / "chunks.parquet", index=False)
    (out / "bipia.json").write_text(json.dumps(bipia(source, raw), indent=2))
    return sample.record(out, share, seed, "filing", n_filings, questions["doc_name"].nunique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--credit-source", type=Path, default=CREDIT_SOURCE)
    parser.add_argument("--flows-source", type=Path, default=FLOWS_SOURCE)
    parser.add_argument("--images-source", type=Path, default=IMAGES_SOURCE)
    parser.add_argument(
        "--filings-source", type=Path, default=None, help="synthetic corpus folder; default: download"
    )
    parser.add_argument("--block", type=int, default=None, help="images per split block (project 6: 25)")
    sample.add_arguments(parser, "clients, flows, image blocks and filings, each stratified as in its project")
    args = parser.parse_args()
    steps = {
        "credit": lambda d: credit(args.credit_source, d, args.sample, args.sample_seed),
        "flows": lambda d: flows(args.flows_source, d, args.sample, args.sample_seed),
        "images": lambda d: images(args.images_source, d, args.sample, args.sample_seed, args.block),
        "filings": lambda d: filings(args.filings_source, d, args.sample, args.sample_seed),
    }
    for part in args.parts:
        record = steps[part](args.data_dir / part)
        print(f"{part}: kept {record['units_after']:,} of {record['units_before']:,} (unit: {record['unit']})")


if __name__ == "__main__":
    main()
