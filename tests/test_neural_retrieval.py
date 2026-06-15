from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.neural_retrieval import CrossEncoderRanker, NeuralBackendStatus


class FakeCrossEncoder:
    def predict(self, pairs):
        return np.arange(len(pairs), dtype=float)


class CrossEncoderRankerTest(unittest.TestCase):
    def test_rerank_without_semantic_similarity_column(self):
        ranker = CrossEncoderRanker.__new__(CrossEncoderRanker)
        ranker.model_name = "fake-reranker"
        ranker.model = FakeCrossEncoder()
        ranker.status = NeuralBackendStatus(True, "test", ranker.model_name)
        candidates = pd.DataFrame(
            {
                "title": ["房源A", "房源B"],
                "document": ["两室近地铁", "一室精装修"],
                "semantic_score": [0.2, 0.4],
            }
        )

        result = ranker.rerank("两人合租", candidates)

        self.assertEqual(len(result), 2)
        self.assertIn("semantic_similarity", result.columns)
        self.assertIn("cross_encoder_score", result.columns)
        self.assertTrue(result["semantic_similarity"].notna().all())

    def test_fallback_creates_row_aligned_semantic_column(self):
        ranker = CrossEncoderRanker.__new__(CrossEncoderRanker)
        ranker.model_name = "missing-reranker"
        ranker.model = None
        ranker.status = NeuralBackendStatus(False, "fallback", ranker.model_name)
        candidates = pd.DataFrame(
            {
                "title": ["房源A", "房源B"],
                "document": ["两室近地铁", "一室精装修"],
            }
        )

        result = ranker.rerank("预算2000", candidates)

        self.assertEqual(result["semantic_similarity"].tolist(), [0.0, 0.0])
        self.assertEqual(result["cross_encoder_score"].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
