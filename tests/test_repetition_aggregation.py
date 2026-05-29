from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines import comparator as comparator_module
from baselines.bootstrap import install_evolmd_bertscore_guard
from baselines.comparator import ComparatorService, PROPOSALS, aggregate_proposal_repetitions
from baselines.comparator import mark_non_dominated
from initial_population.comparison import InitialPopulationComparisonService
from turbulence_comparison.service import aggregate_turbulence_repetitions


class RepetitionAggregationTests(unittest.TestCase):
    def test_comparator_config_accepts_seed_and_repetitions(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "seed": 123,
                "repetitionsK": 4,
            }
        )
        self.assertEqual(parsed["seed"], 123)
        self.assertEqual(parsed["repetitionsK"], 4)

    def test_initial_population_config_accepts_repetitions(self):
        service = InitialPopulationComparisonService(Path("."))
        payload = service.default_config()
        payload["selectedStrategies"] = ["hybrid-semantic-v7"]
        payload["repetitionsK"] = 3
        parsed = service._read_config(payload)
        self.assertEqual(parsed["repetitionsK"], 3)

    def test_comparator_aggregates_metrics_across_repetitions(self):
        proposal = PROPOSALS[0]
        results = [
            {
                "status": "completed",
                "repetitionIndex": 1,
                "repetitionSeed": 10,
                "rows": [{"rank": 1, "status": "ok"}],
                "metrics": {
                    "totalRows": 10,
                    "completedRows": 10,
                    "objectiveNames": ["fitness"],
                    "bestObjectiveVector": [0.6],
                    "nonDominatedRows": 0,
                    "hypervolume": 0.2,
                    "spread": 0.4,
                    "postHocDiagnostic": True,
                    "bestDiagnosticObjectiveVector": [0.6, 0.5],
                    "postHocNonDominatedRows": 2,
                },
                "cost": {"llmCalls": 2, "llmClientWallClockSeconds": 1.0, "totalTokens": 20},
            },
            {
                "status": "completed",
                "repetitionIndex": 2,
                "repetitionSeed": 11,
                "rows": [{"rank": 1, "status": "ok"}],
                "metrics": {
                    "totalRows": 8,
                    "completedRows": 8,
                    "objectiveNames": ["fitness"],
                    "bestObjectiveVector": [0.8],
                    "nonDominatedRows": 0,
                    "hypervolume": 0.4,
                    "spread": 0.2,
                    "postHocDiagnostic": True,
                    "bestDiagnosticObjectiveVector": [0.8, 0.7],
                    "postHocNonDominatedRows": 4,
                },
                "cost": {"llmCalls": 3, "llmClientWallClockSeconds": 2.0, "totalTokens": 30},
            },
        ]
        aggregated = aggregate_proposal_repetitions(proposal, Path("out"), results, 2)
        self.assertEqual(aggregated["completedRepetitions"], 2)
        self.assertAlmostEqual(aggregated["metrics"]["completedRows"], 9.0)
        self.assertEqual(aggregated["metrics"]["bestObjectiveLabel"], "[0.700000]")
        self.assertEqual(aggregated["cost"]["llmCalls"], 5)
        self.assertEqual(aggregated["rows"][0]["repetitionSeed"], 10)

    def test_evolmd_bertscore_guard_assigns_zero_to_empty_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_dir = Path(temp_dir) / "metrics"
            metrics_dir.mkdir()
            (metrics_dir / "__init__.py").write_text("", encoding="utf-8")
            (metrics_dir / "bert.py").write_text(
                "\n".join(
                    [
                        "def bertscore_individuos(individuos, ref_text, model_type, lang='en'):",
                        "    if any(not item.get('generated_data') for item in individuos):",
                        "        raise RuntimeError('empty candidate reached BERTScore')",
                        "    for item in individuos:",
                        "        item['fitness'] = 0.7",
                        "    return individuos",
                    ]
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, temp_dir)
            previous_metrics = sys.modules.pop("metrics", None)
            previous_bert = sys.modules.pop("metrics.bert", None)
            try:
                install_evolmd_bertscore_guard()
                import metrics.bert as bert_module

                rows = [{"generated_data": ""}, {"generated_data": "valid text"}]
                result = bert_module.bertscore_individuos(
                    rows,
                    "reference text",
                    "bert-base-uncased",
                )
                self.assertIs(result, rows)
                self.assertEqual(rows[0]["fitness"], 0.0)
                self.assertEqual(rows[0]["fitness_status"], "empty_generated_data")
                self.assertEqual(rows[1]["fitness"], 0.7)
                self.assertEqual(rows[1]["fitness_status"], "ok")
            finally:
                sys.path.remove(temp_dir)
                sys.modules.pop("metrics", None)
                sys.modules.pop("metrics.bert", None)
                if previous_metrics is not None:
                    sys.modules["metrics"] = previous_metrics
                if previous_bert is not None:
                    sys.modules["metrics.bert"] = previous_bert

    def test_non_dominated_excludes_invalid_rows(self):
        rows = [
            {"status": "empty_generated_data", "objectiveVector": [1.0, 1.0]},
            {"status": "ok", "objectiveVector": [0.8, 0.8]},
            {"status": "ok", "objectiveVector": [0.6, 0.6]},
        ]
        mark_non_dominated(rows)
        self.assertFalse(rows[0]["nonDominated"])
        self.assertTrue(rows[1]["nonDominated"])
        self.assertFalse(rows[2]["nonDominated"])

    def test_evolmd_posthoc_diagnostics_excludes_invalid_rows(self):
        service = ComparatorService(Path("."))
        rows = [
            {
                "status": "ok",
                "objectiveVector": [0.8],
                "generatedText": "valid text",
            },
            {
                "status": "empty_generated_data",
                "objectiveVector": [0.0],
                "generatedText": "",
            },
        ]

        with patch.object(comparator_module, "calculate_posthoc_semantic_diversity", return_value=[0.25]) as mocked:
            service._attach_evolmd_posthoc_diagnostics(rows)

        mocked.assert_called_once_with(["valid text"])
        self.assertEqual(rows[0]["diagnosticObjectiveVector"], [0.8, 0.25])
        self.assertTrue(rows[0]["postHocNonDominated"])
        self.assertNotIn("diagnosticObjectiveVector", rows[1])
        self.assertFalse(rows[1]["postHocNonDominated"])

    def test_mo_metrics_exclude_invalid_rows_with_high_objectives(self):
        service = ComparatorService(Path("."))
        proposal = PROPOSALS[1]
        rows = service._normalize_rows(
            proposal,
            [
                {"generated_data": "", "objetivos": [1.0, 1.0]},
                {"generated_data": "valid text", "objetivos": [0.5, 0.5]},
            ],
            top_k=10,
        )
        invalid = next(row for row in rows if row["status"] != "ok")
        valid = next(row for row in rows if row["status"] == "ok")
        self.assertFalse(invalid["nonDominated"])
        self.assertTrue(valid["nonDominated"])

        metrics = service._summarize_rows(proposal, rows, Path("out"))
        self.assertEqual(metrics["completedRows"], 1)
        self.assertAlmostEqual(metrics["hypervolume"], 0.375)

    def test_turbulence_aggregates_rates_and_final_deltas(self):
        config = {
            "strategies": ["llm"],
            "individualCount": 2,
            "seed": 7,
            "repetitionsK": 2,
            "finalFidelityMin": 0.0,
            "finalFidelityMax": 1.0,
            "kCandidates": 5,
            "operatorParallelism": 1,
            "generationParallelism": 1,
            "turbulenceMinSimilarity": 0.65,
            "turbulenceMaxSimilarity": 0.9,
            "embeddingModel": "fake",
            "distilbertModel": "fake",
        }
        repetitions = [
            {
                "movementCount": 2,
                "configSummary": {"repetitionIndex": 1, "seed": 7},
                "baseline": {"completedRows": 2, "averageFidelity": 0.5, "averageDiversity": 0.3, "rows": []},
                "strategies": [
                    {
                        "strategyId": "llm",
                        "operatorMetrics": {
                            "moves": 2,
                            "successes": 1,
                            "coverageCount": 2,
                            "similarityCount": 1,
                            "similaritySum": 0.7,
                            "operatorCumulativeSeconds": 1.0,
                            "operatorWallClockSeconds": 1.0,
                            "operatorParallelism": 1,
                            "cost": {"llmCalls": 2},
                        },
                        "finalMetrics": {"completedRows": 2, "averageFidelity": 0.6, "averageDiversity": 0.4, "rows": []},
                        "generationCost": {},
                        "finalEmbeddingCost": {},
                        "movementRows": [{"success": True}],
                        "finalRows": [],
                    }
                ],
            },
            {
                "movementCount": 2,
                "configSummary": {"repetitionIndex": 2, "seed": 8},
                "baseline": {"completedRows": 2, "averageFidelity": 0.7, "averageDiversity": 0.5, "rows": []},
                "strategies": [
                    {
                        "strategyId": "llm",
                        "operatorMetrics": {
                            "moves": 2,
                            "successes": 2,
                            "coverageCount": 2,
                            "similarityCount": 2,
                            "similaritySum": 1.5,
                            "operatorCumulativeSeconds": 1.0,
                            "operatorWallClockSeconds": 1.0,
                            "operatorParallelism": 1,
                            "cost": {"llmCalls": 2},
                        },
                        "finalMetrics": {"completedRows": 2, "averageFidelity": 0.8, "averageDiversity": 0.6, "rows": []},
                        "generationCost": {},
                        "finalEmbeddingCost": {},
                        "movementRows": [{"success": True}],
                        "finalRows": [],
                    }
                ],
            },
        ]
        aggregated = aggregate_turbulence_repetitions(repetitions, config)
        strategy = aggregated["strategies"][0]
        self.assertEqual(aggregated["completedRepetitions"], 2)
        self.assertAlmostEqual(aggregated["baseline"]["averageFidelity"], 0.6)
        self.assertAlmostEqual(strategy["operatorMetrics"]["successRate"], 0.75)
        self.assertAlmostEqual(strategy["operatorMetrics"]["averageSimilarity"], 2.2 / 3)
        self.assertAlmostEqual(strategy["finalMetrics"]["averageFidelity"], 0.7)
        self.assertAlmostEqual(strategy["deltas"]["averageFidelityDelta"], 0.1)
        self.assertEqual(strategy["repetitions"][0]["repetitionIndex"], 1)
        self.assertEqual(strategy["repetitions"][1]["repetitionSeed"], 8)


if __name__ == "__main__":
    unittest.main()
