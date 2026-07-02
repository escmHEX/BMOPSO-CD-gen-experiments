from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import server
from reference_text_selection.service import (
    DEFAULT_REFERENCE_TEXT_SELECTION_CONFIG,
    ReferenceTextSelectionService,
    read_json,
    select_reference_text,
    write_json,
)


class FakeEmbeddingService:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, list[str]]] = []

    def encode_texts(self, model_name: str, texts: list[str]):
        self.calls.append((model_name, texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=float), {
            "embeddingModel": model_name,
            "sourceModel": model_name,
            "embeddingTexts": len(texts),
            "embeddingWallClockSeconds": 0.01,
        }

    def source_model_name(self, model_name: str) -> str:
        if model_name != "all-MiniLM-L6-v2":
            raise ValueError(f"Modelo SBERT no soportado: {model_name}")
        return "sentence-transformers/all-MiniLM-L6-v2"


def write_tsv(path: Path, rows: list[tuple[str, str]], header: str = "tweetId\ttexto\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = header + "".join(f"{tweet_id}\t{text}\n" for tweet_id, text in rows)
    path.write_text(content, encoding="utf-8")


class ReferenceTextSelectionAlgorithmTests(unittest.TestCase):
    def test_defaults_match_reference_strategy(self):
        self.assertEqual(
            DEFAULT_REFERENCE_TEXT_SELECTION_CONFIG,
            {
                "datasetPath": "data/external/covid19_tweets/hydrated/10k_data.tsv",
                "sampleSize": 1000,
                "clusterCount": 4,
                "minWords": 20,
                "maxWords": 80,
                "seed": 42,
                "semanticWeight": 0.7,
                "embeddingModel": "all-MiniLM-L6-v2",
            },
        )

    def test_selects_majority_cluster_central_text_and_preserves_original_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.tsv"
            texts = {
                "t1": "alpha beta gamma",
                "t2": "alpha beta gamma delta",
                "t3": "alpha beta gamma delta epsilon zeta eta theta",
                "t4": "omega psi chi",
            }
            write_tsv(dataset, list(texts.items()))
            embeddings = FakeEmbeddingService(
                {
                    texts["t1"]: [1.0, 0.0],
                    texts["t2"]: [0.994937, 0.100499],
                    texts["t3"]: [0.979804, -0.19996],
                    texts["t4"]: [-1.0, 0.0],
                }
            )

            result = select_reference_text(
                {
                    "datasetPath": str(dataset),
                    "sampleSize": 4,
                    "clusterCount": 2,
                    "minWords": 1,
                    "maxWords": 10,
                    "seed": 7,
                    "semanticWeight": 0.7,
                    "embeddingModel": "all-MiniLM-L6-v2",
                },
                root=root,
                embedding_service=embeddings,
            )

        self.assertEqual(result["selectedTweetId"], "t1")
        self.assertEqual(result["selectedText"], texts["t1"])
        self.assertEqual(result["selectedOriginalIndex"], 1)
        self.assertEqual(result["filteredCount"], 4)
        self.assertEqual(result["sampleCount"], 4)
        self.assertEqual(result["majorityClusterSize"], 3)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertEqual(result["embeddingCost"]["embeddingTexts"], 4)
        self.assertEqual(result["rankedCandidates"][0]["tweetId"], "t1")

    def test_tie_breaks_by_original_dataset_index_after_score_and_distances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.tsv"
            first = "same length text"
            second = "same other text"
            write_tsv(dataset, [("first", first), ("second", second)])
            embeddings = FakeEmbeddingService(
                {
                    first: [1.0, 0.0],
                    second: [1.0, 0.0],
                }
            )

            result = select_reference_text(
                {
                    "datasetPath": str(dataset),
                    "sampleSize": 2,
                    "clusterCount": 1,
                    "minWords": 1,
                    "maxWords": 10,
                    "seed": 99,
                    "semanticWeight": 0.7,
                    "embeddingModel": "all-MiniLM-L6-v2",
                },
                root=root,
                embedding_service=embeddings,
            )

        self.assertEqual(result["selectedTweetId"], "first")
        self.assertEqual(result["selectedOriginalIndex"], 1)

    def test_missing_tsv_raises_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings = FakeEmbeddingService({})

            with self.assertRaisesRegex(ValueError, "No existe el TSV"):
                select_reference_text(
                    {
                        "datasetPath": "missing.tsv",
                        "sampleSize": 2,
                        "clusterCount": 1,
                        "minWords": 1,
                        "maxWords": 10,
                        "seed": 1,
                        "semanticWeight": 0.7,
                        "embeddingModel": "all-MiniLM-L6-v2",
                    },
                    root=root,
                    embedding_service=embeddings,
                )

    def test_missing_required_columns_raise_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.tsv"
            write_tsv(dataset, [("1", "hello world")], header="id\ttext\n")
            embeddings = FakeEmbeddingService({})

            with self.assertRaisesRegex(ValueError, "tweetId.*texto"):
                select_reference_text(
                    {
                        "datasetPath": str(dataset),
                        "sampleSize": 2,
                        "clusterCount": 1,
                        "minWords": 1,
                        "maxWords": 10,
                        "seed": 1,
                        "semanticWeight": 0.7,
                        "embeddingModel": "all-MiniLM-L6-v2",
                    },
                    root=root,
                    embedding_service=embeddings,
                )

    def test_empty_filtered_set_raises_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.tsv"
            write_tsv(dataset, [("1", "too short")])
            embeddings = FakeEmbeddingService({})

            with self.assertRaisesRegex(ValueError, "No existen textos"):
                select_reference_text(
                    {
                        "datasetPath": str(dataset),
                        "sampleSize": 2,
                        "clusterCount": 1,
                        "minWords": 5,
                        "maxWords": 10,
                        "seed": 1,
                        "semanticWeight": 0.7,
                        "embeddingModel": "all-MiniLM-L6-v2",
                    },
                    root=root,
                    embedding_service=embeddings,
                )

    def test_sample_smaller_than_cluster_count_raises_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.tsv"
            write_tsv(dataset, [("1", "one valid text"), ("2", "another valid text")])
            embeddings = FakeEmbeddingService({})

            with self.assertRaisesRegex(ValueError, "clusterCount"):
                select_reference_text(
                    {
                        "datasetPath": str(dataset),
                        "sampleSize": 2,
                        "clusterCount": 3,
                        "minWords": 1,
                        "maxWords": 10,
                        "seed": 1,
                        "semanticWeight": 0.7,
                        "embeddingModel": "all-MiniLM-L6-v2",
                    },
                    root=root,
                    embedding_service=embeddings,
                )


class ReferenceTextSelectionServiceTests(unittest.TestCase):
    def test_service_persists_completed_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.tsv"
            first = "alpha beta gamma"
            second = "alpha beta delta"
            write_tsv(dataset, [("first", first), ("second", second)])
            service = ReferenceTextSelectionService(
                root,
                embedding_service=FakeEmbeddingService({first: [1.0, 0.0], second: [0.98, 0.2]}),
            )

            run = service.start_run(
                {
                    "datasetPath": str(dataset),
                    "sampleSize": 2,
                    "clusterCount": 1,
                    "minWords": 1,
                    "maxWords": 10,
                    "seed": 42,
                    "semanticWeight": 0.7,
                    "embeddingModel": "all-MiniLM-L6-v2",
                }
            )
            deadline = time.monotonic() + 5
            while run["status"] in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.05)
                run = service.get_run(run["runId"])

            self.assertEqual(run["status"], "completed")
            run_dir = root / "runs" / "reference-text-selection" / run["runId"]
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "selected_reference_text.json").exists())
            self.assertTrue((run_dir / "ranked_candidates.json").exists())
            self.assertTrue((run_dir / "sampled_candidates.json").exists())
            self.assertTrue((run_dir / "sample_embeddings.npy").exists())
            sampled_candidates = read_json(run_dir / "sampled_candidates.json")
            sample_embeddings = np.load(run_dir / "sample_embeddings.npy")
            self.assertEqual(len(sampled_candidates), run["sampleCount"])
            self.assertEqual(sample_embeddings.shape[0], run["sampleCount"])
            self.assertEqual([item["originalIndex"] for item in sampled_candidates], [1, 2])
            self.assertEqual(sum(1 for item in sampled_candidates if item["isSelected"]), 1)

    def test_service_projects_persisted_sample_embeddings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset.tsv"
            first = "alpha beta gamma"
            second = "alpha beta delta"
            third = "omega psi chi"
            write_tsv(dataset, [("first", first), ("second", second), ("third", third)])
            service = ReferenceTextSelectionService(
                root,
                embedding_service=FakeEmbeddingService(
                    {
                        first: [1.0, 0.0, 0.0],
                        second: [0.98, 0.2, 0.0],
                        third: [-1.0, 0.0, 0.0],
                    }
                ),
            )

            run = service.start_run(
                {
                    "datasetPath": str(dataset),
                    "sampleSize": 3,
                    "clusterCount": 2,
                    "minWords": 1,
                    "maxWords": 10,
                    "seed": 42,
                    "semanticWeight": 0.7,
                    "embeddingModel": "all-MiniLM-L6-v2",
                }
            )
            deadline = time.monotonic() + 5
            while run["status"] in {"queued", "running"} and time.monotonic() < deadline:
                time.sleep(0.05)
                run = service.get_run(run["runId"])

            payload = service.get_run_embedding_projection(run["runId"], method="pca")

            self.assertEqual(payload["runId"], run["runId"])
            self.assertEqual(payload["method"], "pca")
            self.assertEqual(payload["effectiveMethod"], "pca")
            self.assertEqual(payload["embeddingModel"], "all-MiniLM-L6-v2")
            self.assertEqual(payload["sampleCount"], 3)
            self.assertEqual(len(payload["points"]), 3)
            selected_points = [point for point in payload["points"] if point["isSelected"]]
            self.assertEqual(len(selected_points), 1)
            self.assertEqual(selected_points[0]["tweetId"], run["selectedTweetId"])
            self.assertIn("clusterSizes", payload)
            self.assertTrue(
                (root / "runs" / "reference-text-selection" / run["runId"] / "embedding_projection_pca.json").exists()
            )

    def test_projection_returns_none_for_missing_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ReferenceTextSelectionService(Path(temp_dir), embedding_service=FakeEmbeddingService({}))

            self.assertIsNone(service.get_run_embedding_projection("missing-run", method="pca"))

    def test_projection_requires_persisted_sample_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "reference-text-selection" / "old-run"
            run_dir.mkdir(parents=True)
            write_json(
                run_dir / "summary.json",
                {
                    "runId": "old-run",
                    "runDir": str(run_dir),
                    "status": "completed",
                    "config": {"embeddingModel": "all-MiniLM-L6-v2"},
                },
            )
            service = ReferenceTextSelectionService(root, embedding_service=FakeEmbeddingService({}))

            with self.assertRaisesRegex(ValueError, "artefactos de proyeccion"):
                service.get_run_embedding_projection("old-run", method="pca")


class ReferenceTextSelectionServerRouteTests(unittest.TestCase):
    def test_defaults_route_returns_service_defaults(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            reference_text_selection_path_parts=lambda: ["defaults"],
            reference_text_selection_service=SimpleNamespace(default_config=lambda: {"sampleSize": 1000}),
            send_json=lambda status, payload: sent.append((status, payload)),
        )

        server.ToolPortalHandler.handle_reference_text_selection_get(fake_handler)

        self.assertEqual(sent, [(200, {"defaults": {"sampleSize": 1000}})])

    def test_start_run_route_returns_accepted_run(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            reference_text_selection_path_parts=lambda: ["runs"],
            reference_text_selection_service=SimpleNamespace(start_run=lambda payload: {"runId": "run-1"}),
            read_json_body=lambda: {"sampleSize": 1000},
            send_json=lambda status, payload: sent.append((status, payload)),
        )

        server.ToolPortalHandler.handle_reference_text_selection_post(fake_handler)

        self.assertEqual(sent, [(202, {"runId": "run-1"})])

    def test_get_run_route_returns_run(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            reference_text_selection_path_parts=lambda: ["runs", "run-1"],
            reference_text_selection_service=SimpleNamespace(get_run=lambda run_id: {"runId": run_id}),
            send_json=lambda status, payload: sent.append((status, payload)),
        )

        server.ToolPortalHandler.handle_reference_text_selection_get(fake_handler)

        self.assertEqual(sent, [(200, {"runId": "run-1"})])

    def test_embedding_projection_route_returns_projection_payload(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            path="/api/reference-text-selection/runs/run-1/embedding-projection?method=pca",
            reference_text_selection_path_parts=lambda: ["runs", "run-1", "embedding-projection"],
            reference_text_selection_service=SimpleNamespace(
                get_run_embedding_projection=lambda run_id, method: {"runId": run_id, "method": method}
            ),
            send_json=lambda status, payload: sent.append((status, payload)),
        )

        server.ToolPortalHandler.handle_reference_text_selection_get(fake_handler)

        self.assertEqual(sent, [(200, {"runId": "run-1", "method": "pca"})])

    def test_embedding_projection_route_returns_400_for_invalid_method(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            path="/api/reference-text-selection/runs/run-1/embedding-projection?method=mds",
            reference_text_selection_path_parts=lambda: ["runs", "run-1", "embedding-projection"],
            reference_text_selection_service=SimpleNamespace(
                get_run_embedding_projection=lambda run_id, method: (_ for _ in ()).throw(
                    ValueError("method debe ser pca, tsne o umap.")
                )
            ),
            send_json=lambda status, payload: sent.append((status, payload)),
        )

        server.ToolPortalHandler.handle_reference_text_selection_get(fake_handler)

        self.assertEqual(sent, [(400, {"error": "method debe ser pca, tsne o umap."})])

    def test_embedding_projection_route_returns_500_for_unexpected_errors(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            path="/api/reference-text-selection/runs/run-1/embedding-projection?method=pca",
            reference_text_selection_path_parts=lambda: ["runs", "run-1", "embedding-projection"],
            reference_text_selection_service=SimpleNamespace(
                get_run_embedding_projection=lambda run_id, method: (_ for _ in ()).throw(RuntimeError("boom"))
            ),
            send_json=lambda status, payload: sent.append((status, payload)),
        )

        server.ToolPortalHandler.handle_reference_text_selection_get(fake_handler)

        self.assertEqual(sent, [(500, {"error": "boom"})])


class ReferenceTextSelectionFrontendSmokeTests(unittest.TestCase):
    def test_reference_text_selection_nav_is_after_proposal_comparator(self):
        html = Path("LLM/index.html").read_text(encoding="utf-8")

        proposal_index = html.index('data-tool="proposalComparator"')
        selection_index = html.index('data-tool="referenceTextSelection"')
        embedding_index = html.index('data-tool="embeddingCalculator"')

        self.assertLess(proposal_index, selection_index)
        self.assertLess(selection_index, embedding_index)

    def test_reference_text_selection_ui_ids_exist(self):
        html = Path("LLM/index.html").read_text(encoding="utf-8")
        for element_id in (
            "referenceTextSelection",
            "referenceSelectionDatasetPath",
            "referenceSelectionSampleSize",
            "referenceSelectionClusterCount",
            "referenceSelectionMinWords",
            "referenceSelectionMaxWords",
            "referenceSelectionSeed",
            "referenceSelectionSemanticWeight",
            "runReferenceSelectionButton",
            "saveSelectedReferenceButton",
            "referenceSelectionResultText",
            "referenceSelectionCandidatesBody",
            "referenceSelectionRunId",
            "referenceSelectionResumeRunId",
            "resumeReferenceSelectionButton",
            "referenceSelectionProjectionMethod",
            "referenceSelectionProjectionStatus",
            "referenceSelectionProjectionChart",
        ):
            self.assertIn(f'id="{element_id}"', html)

        app = Path("LLM/app.js").read_text(encoding="utf-8")
        self.assertIn('const REFERENCE_TEXT_SELECTION_API = "/api/reference-text-selection";', app)
        self.assertIn("runReferenceTextSelection", app)
        self.assertIn("renderReferenceTextSelectionProjection", app)


if __name__ == "__main__":
    unittest.main()
