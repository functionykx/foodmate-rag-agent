from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.neural_retrieval import BGEEmbeddingIndex
from src.utils import PROJECT_ROOT, load_restaurants


class RestaurantRetriever:
    """TF-IDF, dense embedding fallback, and hybrid retrieval.

    `mode` can be:
    - `tfidf`: sparse character n-gram lexical retrieval
    - `embedding`: local dense embedding fallback using TF-IDF + SVD
    - `hybrid`: weighted merge of TF-IDF and dense scores
    - `bge`: real BGE embedding retrieval with FAISS when available

    If sentence-transformers/FAISS are not installed, `bge` falls back to the
    local SVD dense vectors and marks `vector_backend` as `svd_fallback`.
    """

    def __init__(self, df: pd.DataFrame | None = None, mode: str | None = None):
        self.df = df if df is not None else load_restaurants()
        self.mode = (mode or os.getenv("FOODMATE_RETRIEVER_MODE", "hybrid")).lower()

        docs = self.df["document"].tolist()
        self.tfidf_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)

        n_docs, n_features = self.tfidf_matrix.shape
        n_components = max(1, min(128, n_docs - 1, n_features - 1))
        self.embedding_svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.embedding_matrix = self.embedding_svd.fit_transform(self.tfidf_matrix)
        self.embedding_matrix = normalize(self.embedding_matrix)
        self.vector_backend = "sklearn"
        self.faiss_index = None
        try:
            import faiss  # type: ignore

            vectors = self.embedding_matrix.astype("float32")
            self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss_index.add(vectors)
            self.vector_backend = "faiss"
        except Exception:
            self.faiss_index = None
        self.bge_index = BGEEmbeddingIndex(self.df) if self.mode == "bge" else None

    def _tfidf_scores(self, query: str) -> np.ndarray:
        query_vec = self.tfidf_vectorizer.transform([query])
        return cosine_similarity(query_vec, self.tfidf_matrix).ravel()

    def _embedding_scores(self, query: str) -> np.ndarray:
        query_vec = self.tfidf_vectorizer.transform([query])
        query_embedding = self.embedding_svd.transform(query_vec)
        query_embedding = normalize(query_embedding)
        if self.faiss_index is not None:
            scores, idx = self.faiss_index.search(query_embedding.astype("float32"), len(self.df))
            dense_scores = np.zeros(len(self.df), dtype=float)
            dense_scores[idx[0]] = scores[0]
            return dense_scores
        return cosine_similarity(query_embedding, self.embedding_matrix).ravel()

    @staticmethod
    def _scale(scores: np.ndarray) -> np.ndarray:
        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score - min_score < 1e-12:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def search(self, query: str, top_k: int = 20) -> pd.DataFrame:
        tfidf_scores = self._tfidf_scores(query)
        embedding_scores = self._embedding_scores(query)

        if self.mode == "bge" and self.bge_index is not None and self.bge_index.available():
            result = self.bge_index.search(query, top_k=top_k)
            source_idx = result["source_index"].astype(int).to_numpy()
            result["tfidf_score"] = tfidf_scores[source_idx]
            result["embedding_score"] = result["bge_score"]
            result["retriever_mode"] = self.mode
            return result.reset_index(drop=True)
        if self.mode == "tfidf":
            scores = tfidf_scores
        elif self.mode == "embedding":
            scores = embedding_scores
        elif self.mode == "bge":
            scores = embedding_scores
        else:
            scores = 0.55 * self._scale(tfidf_scores) + 0.45 * self._scale(embedding_scores)

        idx = scores.argsort()[::-1][:top_k]
        result = self.df.iloc[idx].copy()
        result["source_index"] = idx
        result["semantic_similarity"] = scores[idx]
        result["tfidf_score"] = tfidf_scores[idx]
        result["embedding_score"] = embedding_scores[idx]
        result["retriever_mode"] = self.mode
        result["vector_backend"] = "svd_fallback" if self.mode == "bge" else self.vector_backend
        if self.mode == "bge" and self.bge_index is not None:
            result["embedding_model"] = self.bge_index.status.model_name
            result["embedding_error"] = self.bge_index.status.error
        return result.reset_index(drop=True)

    def save(self, directory=PROJECT_ROOT / "vector_store") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.tfidf_vectorizer, directory / "tfidf_vectorizer.joblib")
        joblib.dump(self.tfidf_matrix, directory / "tfidf_matrix.joblib")
        joblib.dump(self.embedding_svd, directory / "embedding_svd.joblib")
        joblib.dump(self.embedding_matrix, directory / "embedding_matrix.joblib")
        self.df.to_csv(directory / "restaurants_index.csv", index=False)
