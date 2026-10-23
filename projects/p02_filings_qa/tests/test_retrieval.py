import pytest

from filings_qa.evaluate_retrieval import first_relevant_rank, page_of, rank_all, summarize
from filings_qa.retrieval import BM25Retriever, reciprocal_rank_fusion


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid"])
def test_each_retriever_finds_the_evidence_page_in_the_named_filing(corpus, retrievers, name):
    questions = corpus[0]
    for q in questions.itertuples(index=False):
        hits = retrievers[name].search(q.question, 3, q.doc_name)
        assert hits, q.question
        assert all(chunk_id.startswith(q.doc_name + "#") for chunk_id, _ in hits)
        assert first_relevant_rank([c for c, _ in hits], set(q.evidence_pages)) is not None


def test_bm25_search_across_filings_ranks_the_company_first(corpus, retrievers):
    questions = corpus[0]
    q = questions.iloc[0]
    top_id, _ = retrievers["bm25"].search(q["question"], 1)[0]
    assert top_id.split("#")[0] == q["doc_name"]


def test_bm25_returns_nothing_for_unknown_words_or_filings(retrievers):
    assert retrievers["bm25"].search("zzzz qqqq", 5) == []
    assert retrievers["bm25"].search("revenue", 5, "NO_SUCH_DOC") == []


def test_bm25_survives_a_save_and_load(corpus, retrievers, tmp_path):
    path = tmp_path / "bm25.joblib"
    retrievers["bm25"].save(path)
    loaded = BM25Retriever.load(path)
    query = corpus[0].iloc[1]["question"]
    assert loaded.search(query, 5) == retrievers["bm25"].search(query, 5)


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[("a", 9), ("b", 8)], [("b", 1.0), ("c", 0.5)]], k=3, rrf_k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_recall_summary_counts_ranks_against_cutoffs(corpus, retrievers):
    questions = corpus[0]
    ranks = rank_all(questions, retrievers["bm25"], "doc", 10)
    summary = summarize(ranks)
    row = summary.iloc[0]
    assert row["questions"] == len(questions)
    assert 0 <= row["recall@1"] <= row["recall@3"] <= row["recall@50"] <= 1
    assert page_of("A_2022_10K#7#2") == "A_2022_10K#7"
