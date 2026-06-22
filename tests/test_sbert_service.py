from __future__ import annotations

import unittest

from sbert_service import project_embeddings_2d, SbertSimilarityService


class FakeEmbeddingVector:
    def __init__(self, values):
        self.values = list(values)

    def __matmul__(self, other):
        return sum(left * right for left, right in zip(self.values, other.values))

    def tolist(self):
        return list(self.values)


class FakeEmbeddingMatrix:
    def __init__(self, rows):
        self.rows = [FakeEmbeddingVector(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0].values) if self.rows else 0)

    def __getitem__(self, index):
        return self.rows[index]


class FakeSentenceTransformer:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.texts = texts
        self.convert_to_numpy = convert_to_numpy
        self.normalize_embeddings = normalize_embeddings
        return FakeEmbeddingMatrix([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])


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

    def test_projects_embeddings_with_pca(self):
        result = project_embeddings_2d(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ],
            "pca",
        )

        self.assertEqual(result["method"], "pca")
        self.assertEqual(result["effectiveMethod"], "pca")
        self.assertEqual(len(result["coordinates"]), 3)
        self.assertTrue(all(len(point) == 2 for point in result["coordinates"]))

    def test_projection_falls_back_to_pca_for_small_tsne_inputs(self):
        result = project_embeddings_2d(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "tsne",
        )

        self.assertEqual(result["method"], "tsne")
        self.assertEqual(result["effectiveMethod"], "pca")
        self.assertIn("se uso PCA", result["warnings"][0])

    def test_projection_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            project_embeddings_2d([[1.0, 0.0], [0.0, 1.0]], "mds")


if __name__ == "__main__":
    unittest.main()
