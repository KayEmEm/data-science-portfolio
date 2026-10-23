import pandas as pd

from filings_qa.evaluate_retrieval import oracle_ranks, rank_all, write_contexts
from filings_qa.generation import ExtractiveGenerator, run
from filings_qa.report import summarize
from filings_qa.scoring import free_scores
from filings_qa.sections import label_pages


def test_retrieval_to_report_on_the_synthetic_corpus(corpus, retrievers, tmp_path):
    questions, _, _, chunks = corpus
    ranks = pd.concat([rank_all(questions, retrievers["hybrid"], "doc", 50), oracle_ranks(questions, chunks)])
    paths = write_contexts(ranks, chunks, 8, tmp_path / "contexts")
    assert sorted(p.name for p in paths) == ["doc_hybrid_k8.parquet", "doc_oracle.parquet"]

    contexts = pd.read_parquet(tmp_path / "contexts" / "doc_oracle.parquet")
    results = run(
        ExtractiveGenerator(),
        questions,
        contexts,
        tmp_path / "runs" / "oracle.jsonl",
        {"run": "oracle", "contexts": "doc_oracle", "generator": "extractive", "model": "none"},
        float("inf"),
    )
    summary = summarize(free_scores(results, questions))
    row = summary.iloc[0]
    assert row["questions"] == len(questions)
    assert row["cites an evidence page"] == 1.0
    assert row["figure match (metrics questions)"] == 1.0


def _page(doc, page, text):
    return {"doc_name": doc, "company": "A", "doc_type": "10K", "page": page, "text": text}


def test_section_labels_follow_headings_and_skip_contents_pages():
    body = " ".join(["word"] * 60)
    pages = pd.DataFrame(
        [
            _page("A", 0, "Item 1. Business\nItem 1A. Risk Factors\nItem 7. MD&A\nItem 8. Financial"),
            _page("A", 1, "Item 1A. Risk Factors\n" + body),
            _page("A", 2, body),
            _page("A", 3, "Item 7. Management's Discussion\n" + body),
            _page("A", 4, body),
            _page("A", 5, "Item 15. Exhibits\n" + body),
            _page("A", 6, body),
            _page("A", 7, "short"),
        ]
    )
    labeled = label_pages(pages)
    assert labeled[["page", "section"]].values.tolist() == [[2, "risk factors"], [4, "management discussion"]]
