"""Lexical (BM25), dense (Chroma with MiniLM embeddings) and hybrid passage retrieval.

Every retriever answers ``search(query, k, doc_name=None)`` with a ranked list
of ``(chunk_id, score)``. ``doc_name`` limits the search to one filing, which
is the setting of an analyst who has already opened the right report.
"""

import argparse
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from chromadb.api.types import EmbeddingFunction
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from filings_qa.config import CHUNKS_PATH, INDEX_DIR, RetrievalSettings

Hit = tuple[str, float]
COLLECTION = "filings"
_TOKEN = re.compile(r"[a-z0-9]+")


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int, doc_name: str | None = None) -> list[Hit]: ...


def index_text(chunks: pd.DataFrame) -> pd.Series:
    """Prefix each passage with its company, period and form.

    Filing pages rarely repeat the company name, so without this header a
    search across all filings cannot tell one company's revenue table from
    another's.
    """
    header = chunks["company"] + " " + chunks["doc_period"].astype(str) + " " + chunks["doc_type"]
    return header + ". " + chunks["text"]


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


class BM25Retriever:
    """Okapi BM25 over a sparse term matrix, with the term weights computed once at build time."""

    name = "bm25"

    def __init__(self, chunk_ids: np.ndarray, doc_names: np.ndarray, vocabulary: dict, weights):
        self.chunk_ids = chunk_ids
        self.doc_names = doc_names
        self.vocabulary = vocabulary
        self.weights = weights.tocsc()
        self._rows_by_doc = pd.Series(np.arange(len(doc_names))).groupby(doc_names).apply(np.asarray)

    @classmethod
    def build(cls, chunks: pd.DataFrame, k1: float = 1.5, b: float = 0.75) -> "BM25Retriever":
        vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = vectorizer.fit_transform(index_text(chunks)).tocsr().astype(np.float64)
        n_docs = tf.shape[0]
        length = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * length / length.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        weights = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape)
        return cls(
            chunks["chunk_id"].to_numpy(), chunks["doc_name"].to_numpy(), vectorizer.vocabulary_, weights
        )

    def search(self, query: str, k: int, doc_name: str | None = None) -> list[Hit]:
        columns = sorted({self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary})
        if not columns:
            return []
        if doc_name is not None and doc_name not in self._rows_by_doc.index:
            return []
        scores = np.asarray(self.weights[:, columns].sum(axis=1)).ravel()
        rows = self._rows_by_doc[doc_name] if doc_name is not None else np.arange(len(scores))
        candidate = scores[rows]
        top = rows[np.argsort(-candidate, kind="stable")[:k]]
        return [(str(self.chunk_ids[i]), float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.chunk_ids, self.doc_names, self.vocabulary, self.weights.tocsr()), path)

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        return cls(*joblib.load(path))


class HashingEmbedding(EmbeddingFunction):
    """Feature hashing of word counts into a fixed number of dimensions.

    A stand in for MiniLM in tests and smoke runs: it needs no model download.
    It finds passages that share words with the query, nothing more.
    """

    def __init__(self, dimensions: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dimensions = dimensions
        self._vectorizer = HashingVectorizer(n_features=dimensions, tokenizer=tokenize, token_pattern=None)

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        matrix = self._vectorizer.transform(input).toarray().astype(np.float32)
        matrix[np.linalg.norm(matrix, axis=1) == 0, 0] = 1.0
        return list(matrix)

    @staticmethod
    def name() -> str:
        return "filings_qa_hashing"

    def get_config(self) -> dict:
        return {"dimensions": self.dimensions}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbedding":
        return HashingEmbedding(config["dimensions"])

    def is_legacy(self) -> bool:
        return False


def embedding_function(hashing: bool = False):
    return HashingEmbedding() if hashing else default_embedding_function()


def default_embedding_function():
    """MiniLM L6 v2 as an ONNX model, fetched by Chroma from its own download location."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    return ONNXMiniLM_L6_V2()


class DenseRetriever:
    """Cosine similarity search in a persistent Chroma collection."""

    name = "dense"

    def __init__(self, collection):
        self.collection = collection

    @staticmethod
    def _client(path: Path | None):
        import chromadb

        return chromadb.EphemeralClient() if path is None else chromadb.PersistentClient(path=str(path))

    @classmethod
    def build(
        cls,
        chunks: pd.DataFrame,
        path: Path | None,
        embedding_function: Callable | None = None,
        batch_size: int = 512,
        progress: bool = False,
    ) -> "DenseRetriever":
        client = cls._client(path)
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
        collection = client.create_collection(
            COLLECTION,
            embedding_function=embedding_function or default_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
        texts = index_text(chunks).tolist()
        for start in range(0, len(chunks), batch_size):
            part = chunks.iloc[start : start + batch_size]
            collection.add(
                ids=part["chunk_id"].tolist(),
                documents=texts[start : start + batch_size],
                metadatas=[
                    {"doc_name": d, "page": int(p)}
                    for d, p in zip(part["doc_name"], part["page"], strict=True)
                ],
            )
            if progress:
                print(
                    f"\r{min(start + batch_size, len(chunks)):,}/{len(chunks):,} passages", end="", flush=True
                )
        if progress:
            print()
        return cls(collection)

    @classmethod
    def load(cls, path: Path, embedding_function: Callable | None = None) -> "DenseRetriever":
        client = cls._client(path)
        return cls(
            client.get_collection(
                COLLECTION, embedding_function=embedding_function or default_embedding_function()
            )
        )

    def search(self, query: str, k: int, doc_name: str | None = None) -> list[Hit]:
        result = self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"doc_name": doc_name} if doc_name is not None else None,
        )
        return [(i, 1.0 - float(d)) for i, d in zip(result["ids"][0], result["distances"][0], strict=True)]


def reciprocal_rank_fusion(rankings: Sequence[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    """Merge ranked lists by the sum of ``1 / (rrf_k + rank)``; ties keep the order of first appearance."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])[:k]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], settings: RetrievalSettings | None = None):
        self.retrievers = retrievers
        self.settings = settings or RetrievalSettings()

    def search(self, query: str, k: int, doc_name: str | None = None) -> list[Hit]:
        rankings = [r.search(query, self.settings.candidates, doc_name) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k, self.settings.rrf_k)


def load_retrievers(index_dir: Path = INDEX_DIR, embedding_function: Callable | None = None) -> dict:
    """Return the BM25, dense and hybrid retrievers built by ``make index``."""
    bm25 = BM25Retriever.load(index_dir / "bm25.joblib")
    dense = DenseRetriever.load(index_dir / "chroma", embedding_function)
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BM25 and dense indexes over all passages.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--skip-dense", action="store_true", help="build the BM25 index only")
    parser.add_argument("--hashing-embeddings", action="store_true", help="no model download (smoke runs)")
    args = parser.parse_args()

    chunks = pd.read_parquet(args.chunks)
    BM25Retriever.build(chunks).save(args.index_dir / "bm25.joblib")
    print(f"BM25 index over {len(chunks):,} passages written")
    if not args.skip_dense:
        DenseRetriever.build(
            chunks, args.index_dir / "chroma", embedding_function(args.hashing_embeddings), progress=True
        )
        print(f"Chroma collection written to {args.index_dir / 'chroma'}")


if __name__ == "__main__":
    main()
