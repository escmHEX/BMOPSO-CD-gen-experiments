from __future__ import annotations

import unittest
from pathlib import Path

from baselines.comparator import ComparatorService, PROPOSALS, aggregate_proposal_repetitions
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
