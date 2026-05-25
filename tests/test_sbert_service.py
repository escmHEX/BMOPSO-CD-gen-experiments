from __future__ import annotations

import unittest

import numpy as np

from sbert_service import SbertSimilarityService


class FakeSentenceTransformer:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.texts = texts
        self.convert_to_numpy = convert_to_numpy
        self.normalize_embeddings = normalize_embeddings
        return np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=float)


class StubSbertSimilarityService(SbertSimilarityService):
    def __init__(self) -> None:
        super().__init__()
        self.fake_model = FakeSentenceTransformer()

    def _model(self, model_name):
        self.model_name = model_name
        return self.fake_model


class SbertSimilarityServiceTests(unittest.TestCase):
    def test_rejects_unsupported_model(self):
        service = SbertSimilarityService()
        with self.assertRaises(ValueError):
            service.calculate_pair({
                "model": "unsupported-model",
                "textA": "flooded housing",
                "textB": "residential flood damage",
            })

    def test_legacy_xenova_alias_uses_python_backend(self):
        service = StubSbertSimilarityService()
        result = service.calculate_pair({
            "model": "Xenova/all-MiniLM-L6-v2",
            "textA": "flooded housing",
            "textB": "residential flood damage",
        })

        self.assertEqual(result["model"], "all-MiniLM-L6-v2")
        self.assertEqual(result["sourceModel"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(result["backend"], "python-sentence-transformers")

    def test_requires_two_texts(self):
        service = SbertSimilarityService()
        with self.assertRaises(ValueError):
            service.calculate_pair({
                "model": "all-MiniLM-L6-v2",
                "textA": "flooded housing",
                "textB": "",
            })

    def test_calculates_with_backend_sentence_transformer_contract(self):
        service = StubSbertSimilarityService()
        result = service.calculate_pair({
            "model": "all-MiniLM-L6-v2",
            "textA": "flooded housing",
            "textB": "residential flood damage",
        })

        self.assertEqual(service.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(service.fake_model.texts, ["flooded housing", "residential flood damage"])
        self.assertTrue(service.fake_model.convert_to_numpy)
        self.assertTrue(service.fake_model.normalize_embeddings)
        self.assertEqual(result["backend"], "python-sentence-transformers")
        self.assertEqual(result["dimension"], 3)
        self.assertAlmostEqual(result["similarity"], 0.5)
        self.assertAlmostEqual(result["distance"], 0.5)


if __name__ == "__main__":
    unittest.main()
