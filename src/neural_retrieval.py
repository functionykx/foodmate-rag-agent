from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_BGE_MODEL = "BAAI/bge-base-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _scale(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score - min_score < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


@dataclass
class NeuralBackendStatus:
    enabled: bool
    backend: str
    model_name: str
    error: str = ""


class BGEEmbeddingIndex:
    """Real BGE embedding retrieval with optional FAISS acceleration.

    The class is intentionally optional: if sentence-transformers or the model
    is unavailable, callers can keep using their lexical/SVD fallback.
    """

    def __init__(self, df: pd.DataFrame, model_name: str | None = None):
        self.df = df
        self.model_name = model_name or os.getenv("FOODMATE_BGE_MODEL", DEFAULT_BGE_MODEL)
        self.model = None
        self.embeddings = None
        self.faiss_index = None
        self.status = NeuralBackendStatus(False, "unavailable", self.model_name)

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            docs = self.df["document"].astype(str).tolist()
            self.model = SentenceTransformer(self.model_name)
            self.embeddings = self.model.encode(
                docs,
                batch_size=int(os.getenv("FOODMATE_BGE_BATCH_SIZE", "32")),
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            self.status = NeuralBackendStatus(True, "sentence-transformers", self.model_name)

            try:
                import faiss  # type: ignore

                self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.faiss_index.add(self.embeddings)
                self.status = NeuralBackendStatus(True, "faiss", self.model_name)
            except Exception:
                self.faiss_index = None
        except Exception as exc:
            self.status = NeuralBackendStatus(False, "fallback", self.model_name, str(exc))

    def available(self) -> bool:
        return bool(self.status.enabled and self.model is not None and self.embeddings is not None)

    def scores(self, query: str) -> np.ndarray:
        if not self.available():
            raise RuntimeError(self.status.error or "BGE embedding backend is unavailable.")
        query_embedding = self.model.encode(  # type: ignore[union-attr]
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        if self.faiss_index is not None:
            raw_scores, idx = self.faiss_index.search(query_embedding, len(self.df))
            scores = np.zeros(len(self.df), dtype=float)
            scores[idx[0]] = raw_scores[0]
            return scores
        return np.matmul(self.embeddings, query_embedding[0])

    def search(self, query: str, top_k: int) -> pd.DataFrame:
        scores = self.scores(query)
        idx = scores.argsort()[::-1][:top_k]
        result = self.df.iloc[idx].copy()
        result["source_index"] = idx
        result["bge_score"] = scores[idx]
        result["semantic_similarity"] = _scale(scores)[idx]
        result["vector_backend"] = self.status.backend
        result["embedding_model"] = self.status.model_name
        return result.reset_index(drop=True)


class CrossEncoderRanker:
    """Optional CrossEncoder reranker for Top30 candidates."""

    _MODEL_CACHE = {}
    _STATUS_CACHE = {}

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("FOODMATE_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        self.model = None
        self.status = NeuralBackendStatus(False, "unavailable", self.model_name)
        if self.model_name in self._MODEL_CACHE:
            self.model = self._MODEL_CACHE[self.model_name]
            self.status = self._STATUS_CACHE[self.model_name]
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self.model = CrossEncoder(self.model_name)
            self.status = NeuralBackendStatus(True, "sentence-transformers-cross-encoder", self.model_name)
            self._MODEL_CACHE[self.model_name] = self.model
            self._STATUS_CACHE[self.model_name] = self.status
        except Exception as exc:
            self.status = NeuralBackendStatus(False, "fallback", self.model_name, str(exc))
            self._MODEL_CACHE[self.model_name] = None
            self._STATUS_CACHE[self.model_name] = self.status

    def available(self) -> bool:
        return bool(self.status.enabled and self.model is not None)

    def rerank(self, query: str, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates.copy()
        if "semantic_similarity" in candidates.columns:
            base_semantic = pd.to_numeric(candidates["semantic_similarity"], errors="coerce").fillna(0.0)
        elif "semantic_score" in candidates.columns:
            base_semantic = pd.to_numeric(candidates["semantic_score"], errors="coerce").fillna(0.0)
        else:
            base_semantic = pd.Series(0.0, index=candidates.index, dtype=float)
        if not self.available():
            result = candidates.copy()
            result["semantic_similarity"] = base_semantic.to_numpy(dtype=float)
            result["cross_encoder_score"] = result["semantic_similarity"]
            result["cross_encoder_backend"] = self.status.backend
            result["cross_encoder_model"] = self.status.model_name
            return result

        pairs = [(query, str(row["document"])) for _, row in candidates.iterrows()]
        raw_scores = np.asarray(self.model.predict(pairs), dtype=float)  # type: ignore[union-attr]
        scaled = _scale(raw_scores)
        result = candidates.copy()
        result["cross_encoder_raw_score"] = raw_scores
        result["cross_encoder_score"] = scaled
        result["cross_encoder_backend"] = self.status.backend
        result["cross_encoder_model"] = self.status.model_name
        result["semantic_similarity"] = 0.7 * scaled + 0.3 * base_semantic.to_numpy(dtype=float)
        return result.sort_values("semantic_similarity", ascending=False).reset_index(drop=True)
