import pytest

from filings_qa.parse import build_chunks
from filings_qa.retrieval import BM25Retriever, DenseRetriever, HashingEmbedding, HybridRetriever
from filings_qa.synthetic import build


@pytest.fixture(scope="session")
def corpus():
    questions, documents, pages = build()
    return questions, documents, pages, build_chunks(pages)


@pytest.fixture(scope="session")
def retrievers(corpus):
    chunks = corpus[3]
    bm25 = BM25Retriever.build(chunks)
    dense = DenseRetriever.build(chunks, None, HashingEmbedding())
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}
