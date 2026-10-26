"""Score the retrievers on the FinanceBench evidence pages and store the passages each question receives.

A retrieved passage counts as relevant when its page is one of the evidence
pages of the question. Two settings are scored: ``doc`` searches only the
filing the question names; ``corpus`` searches all filings.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.config import parse_args
from portfolio_common.plotting import apply_style

from filings_qa.config import (
    ARTIFACTS_DIR,
    CHUNKS_PATH,
    INDEX_DIR,
    QUESTIONS_PATH,
    REPORTS_DIR,
    RetrievalSettings,
)
from filings_qa.retrieval import Retriever, embedding_function, load_retrievers

CUTOFFS = (1, 3, 5, 8, 10, 20, 50)
SETTINGS = ("doc", "corpus")
CONTEXTS_DIR = ARTIFACTS_DIR / "contexts"


def page_of(chunk_id: str) -> str:
    """``<doc>#<page>#<n>`` to ``<doc>#<page>``."""
    return chunk_id.rsplit("#", 1)[0]


def first_relevant_rank(chunk_ids: list[str], evidence_pages: set[str]) -> int | None:
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if page_of(chunk_id) in evidence_pages:
            return rank
    return None


def rank_all(questions: pd.DataFrame, retriever: Retriever, setting: str, depth: int) -> pd.DataFrame:
    rows = []
    for q in questions.itertuples(index=False):
        hits = retriever.search(q.question, depth, q.doc_name if setting == "doc" else None)
        chunk_ids = [chunk_id for chunk_id, _ in hits]
        rows.append(
            {
                "question_id": q.question_id,
                "retriever": retriever.name,
                "setting": setting,
                "chunk_ids": chunk_ids,
                "first_relevant_rank": first_relevant_rank(chunk_ids, set(q.evidence_pages)),
                "doc_rank": next(
                    (i for i, c in enumerate(chunk_ids, start=1) if c.split("#")[0] == q.doc_name), None
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(ranks: pd.DataFrame) -> pd.DataFrame:
    """Recall at each cutoff (share of questions with an evidence page in the top k) and MRR."""
    rows = []
    for (setting, retriever), group in ranks.groupby(["setting", "retriever"], sort=False):
        first = group["first_relevant_rank"].astype("float").to_numpy()
        row = {"setting": setting, "retriever": retriever, "questions": len(group)}
        for k in CUTOFFS:
            row[f"recall@{k}"] = float(np.mean(np.nan_to_num(first, nan=np.inf) <= k))
        row["mrr"] = float(np.mean(np.where(np.isnan(first), 0.0, 1.0 / np.nan_to_num(first, nan=1.0))))
        if setting == "corpus":
            doc = group["doc_rank"].astype("float").to_numpy()
            row["right_filing@8"] = float(np.mean(np.nan_to_num(doc, nan=np.inf) <= 8))
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_ranks(questions: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    """The passages of the evidence pages, in page order: the upper bound for retrieval."""
    by_page = chunks.groupby(chunks["chunk_id"].map(page_of))["chunk_id"].apply(list)
    rows = []
    for q in questions.itertuples(index=False):
        ids = [c for page in q.evidence_pages for c in by_page.get(page, [])]
        rows.append({"question_id": q.question_id, "retriever": "oracle", "setting": "doc", "chunk_ids": ids})
    return pd.DataFrame(rows)


def write_contexts(ranks: pd.DataFrame, chunks: pd.DataFrame, k: int, out_dir: Path) -> list[Path]:
    """Store the top ``k`` passages per question, with their text, one file per retriever and setting.

    The generation step reads only these files, so it can run on another
    machine without the PDFs or the indexes.
    """
    text = chunks.set_index("chunk_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for (setting, retriever), group in ranks.groupby(["setting", "retriever"], sort=False):
        records = []
        # The oracle keeps every passage of the evidence pages, however many.
        depth = None if retriever == "oracle" else k
        for row in group.itertuples(index=False):
            for position, chunk_id in enumerate(row.chunk_ids[:depth], start=1):
                c = text.loc[chunk_id]
                records.append(
                    {
                        "question_id": row.question_id,
                        "position": position,
                        "chunk_id": chunk_id,
                        "company": c["company"],
                        "doc_name": c["doc_name"],
                        "doc_type": c["doc_type"],
                        "doc_period": int(c["doc_period"]),
                        "page": int(c["page"]),
                        "text": c["text"],
                    }
                )
        path = out_dir / (
            f"{setting}_{retriever}.parquet" if depth is None else f"{setting}_{retriever}_k{k}.parquet"
        )
        pd.DataFrame(records).to_parquet(path, index=False)
        paths.append(path)
    return paths


def plot_recall(summary: pd.DataFrame, path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, setting in zip(axes, SETTINGS, strict=True):
        for row in summary[summary["setting"] == setting].itertuples(index=False):
            values = [getattr(row, f"recall_{k}") for k in CUTOFFS]
            ax.plot(CUTOFFS, values, marker="o", label=row.retriever)
        ax.set_xscale("log")
        ax.set_xticks(CUTOFFS, [str(k) for k in CUTOFFS])
        ax.set_xlabel("passages retrieved (k)")
        ax.set_title("one filing (doc)" if setting == "doc" else "all filings (corpus)")
    axes[0].set_ylabel("questions with an evidence page in top k")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--contexts", type=Path, default=CONTEXTS_DIR)
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--hashing-embeddings", action="store_true", help="no model download (smoke runs)")
    args = parse_args(parser, __file__)

    settings = RetrievalSettings()
    questions = pd.read_parquet(args.questions)
    chunks = pd.read_parquet(args.chunks)
    retrievers = load_retrievers(args.index_dir, embedding_function(args.hashing_embeddings))
    ranks = pd.concat(
        [
            rank_all(questions, retrievers[name], setting, max(CUTOFFS))
            for setting in SETTINGS
            for name in args.retrievers
        ],
        ignore_index=True,
    )
    summary = summarize(ranks)

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "figures").mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.reports / "retrieval_results.csv", index=False)
    (args.reports / "retrieval_results.md").write_text(
        summary.round(3).to_markdown(index=False) + "\n", encoding="utf-8"
    )
    plot_recall(
        summary.rename(columns=lambda c: c.replace("@", "_")),
        args.reports / "figures" / "retrieval_recall.png",
    )
    paths = write_contexts(
        pd.concat([ranks, oracle_ranks(questions, chunks)], ignore_index=True),
        chunks,
        settings.top_k,
        args.contexts,
    )
    print(summary.round(3).to_string(index=False))
    print(json.dumps([str(p) for p in paths], indent=1))


if __name__ == "__main__":
    main()
