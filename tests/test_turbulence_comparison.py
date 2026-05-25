from __future__ import annotations

import unittest
from pathlib import Path

from turbulence_comparison.operators import (
    CandidateRecord,
    DistilBertOperator,
    LinguisticAnalyzer,
    MutableUnit,
    TurbulenceOperator,
    WordNetPPDBOperator,
    candidate_validation_reason,
)
from turbulence_comparison.service import (
    TurbulenceComparisonService,
    create_movement_plan,
    final_metric_deltas,
    parse_candidates,
    recommend_strategy,
    summarize_operator_rows,
    unparsed_candidate_diagnostics,
)
from turbulence_comparison.service import LlmTurbulenceOperator


class FakeAnalyzer:
    def __init__(self, units):
        self.units = units

    def extract_units(self, text, include_phrases=True):
        if include_phrases:
            return self.units
        return [unit for unit in self.units if unit.kind == "word"]


class FakeRanker:
    def __init__(self, similarities):
        self._similarities = similarities

    def similarities(self, reference, candidates):
        return self._similarities[: len(candidates)], {
            "embeddingModel": "fake",
            "embeddingTexts": len(candidates) + 1,
            "embeddingWallClockSeconds": 0.01,
        }


class FakeWordNet:
    def __init__(self, values):
        self.values = values

    def lookup(self, lemma, pos=None):
        return list(self.values)


class FakePPDB:
    available = False

    def lookup(self, text):
        return []


class FakeDistilBert:
    def __init__(self, values):
        self.values = values

    def predict(self, masked_text, top_k):
        return self.values[:top_k]


class FakeLlmClient:
    def __init__(self, text="1) request urgent shelter aid"):
        self.text = text
        self.calls = []

    def call(self, user_prompt, **kwargs):
        self.calls.append((user_prompt, kwargs))
        return {
            "text": self.text,
            "elapsedSeconds": 0.05,
            "promptTokens": 4,
            "completionTokens": 5,
            "totalTokens": 9,
        }


class DummyOperator(TurbulenceOperator):
    pass


def config():
    return {
        "kCandidates": 5,
        "minWords": 1,
        "maxWords": 8,
        "turbulenceMinSimilarity": 0.55,
        "turbulenceMaxSimilarity": 0.9,
    }


def llm_config():
    payload = config()
    payload.update(
        {
            "llmTurbulenceSystemPrompt": "System {component_name}",
            "llmTurbulencePromptTemplate": "Generate {num_candidates}: {current_component}",
            "componentName": "action",
            "componentDefinition": "intent",
            "otherComponents": "role = resident",
            "referenceText": "flooded streets",
            "lmStudio": {"temperature": 0.4, "topP": 0.9, "maxTokens": 100},
        }
    )
    return payload


class TurbulenceComparisonTests(unittest.TestCase):
    def test_extracts_modifiable_units(self):
        analyzer = LinguisticAnalyzer()
        units = analyzer.extract_units("local resident reports flooded housing", include_phrases=True)
        texts = {unit.text.lower() for unit in units}
        self.assertIn("resident", texts)
        self.assertIn("housing", texts)
        self.assertTrue(all(unit.start < unit.end for unit in units))

    def test_candidate_validation_rejects_copies_and_prompt_markers(self):
        self.assertEqual(
            candidate_validation_reason("request urgent shelter help", "request urgent shelter help", 1, 8),
            "literal_copy_current",
        )
        self.assertEqual(candidate_validation_reason("```json", "request urgent shelter help", 1, 8), "prompt_marker")
        self.assertEqual(candidate_validation_reason("**Flooded Housing**: explanation", "flooded housing", 1, 8), "prompt_marker")
        self.assertEqual(candidate_validation_reason("Based on the context, here are values", "flooded housing", 1, 8), "prompt_marker")
        self.assertEqual(candidate_validation_reason("aid", "request urgent shelter help", 2, 8), "too_short")

    def test_parse_candidates_uses_numbered_lines_only(self):
        raw = "Based on the given context, here are values:\n1) flooded homes\n- evacuation route\n2) shelter options"
        self.assertEqual(parse_candidates(raw, 5), ["flooded homes", "shelter options"])
        diagnostics = unparsed_candidate_diagnostics(raw, 5)
        self.assertEqual(diagnostics[0].reason, "not_numbered_line")
        self.assertEqual(diagnostics[0].candidate, "Based on the given context, here are values:")

    def test_missing_llm_turbulence_system_prompt_falls_back_to_default(self):
        service = TurbulenceComparisonService(Path("."))
        config_payload = service.default_config()
        config_payload["llmTurbulenceSystemPrompt"] = ""
        config_payload["strategies"] = ["llm"]
        parsed = service._read_config(config_payload)
        self.assertIn("prompt-component rewriting module", parsed["llmTurbulenceSystemPrompt"])

    def test_mocked_sbert_ranking_selects_valid_candidate(self):
        operator = DummyOperator()
        scored, embedding_cost = operator.score_candidates(
            "request urgent shelter help",
            [
                CandidateRecord("request urgent shelter aid", "wordnet"),
                CandidateRecord("request urgent shelter support", "wordnet"),
            ],
            config(),
            FakeRanker([0.72, 0.95]),
        )
        selected = operator.select_candidate(scored)
        self.assertEqual(selected.candidate, "request urgent shelter aid")
        self.assertEqual(scored[1].reason, "too_close_to_current")
        self.assertEqual(embedding_cost["embeddingTexts"], 3)

    def test_wordnet_strategy_runs_without_ppdb_index(self):
        unit = MutableUnit("help", "help", "NOUN", 23, 27, "word")
        operator = WordNetPPDBOperator(
            FakeAnalyzer([unit]),
            FakeRanker([0.72, 0.91]),
            FakePPDB(),
            wordnet=FakeWordNet(["aid", "assistance"]),
        )
        result = operator.apply("request urgent shelter help", config())
        self.assertEqual(result.output, "request urgent shelter aid")
        self.assertTrue(result.coverage)
        self.assertEqual(result.cost["wordnetQueries"], 1)
        self.assertFalse(result.cost["ppdbAvailable"])

    def test_ppdb_unavailable_counts_attempts_not_queries(self):
        unit = MutableUnit("help", "help", "NOUN", 23, 27, "word")
        operator = WordNetPPDBOperator(
            FakeAnalyzer([unit]),
            FakeRanker([]),
            FakePPDB(),
            wordnet=FakeWordNet([]),
        )
        result = operator.apply("request urgent shelter help", config())
        self.assertEqual(result.cost["ppdbLookupAttempts"], 1)
        self.assertEqual(result.cost["ppdbQueries"], 0)
        self.assertFalse(result.coverage)

    def test_distilbert_strategy_filters_by_similarity_range(self):
        unit = MutableUnit("help", "help", "NOUN", 23, 27, "word")
        operator = DistilBertOperator(
            FakeAnalyzer([unit]),
            FakeRanker([0.94, 0.66]),
            FakeDistilBert(["support", "aid"]),
        )
        result = operator.apply("request urgent shelter help", config())
        self.assertEqual(result.output, "request urgent shelter aid")
        self.assertEqual(result.candidates[0].reason, "too_close_to_current")
        self.assertEqual(result.cost["distilbertInferences"], 1)

    def test_llm_strategy_reads_nested_lm_studio_config(self):
        client = FakeLlmClient()
        operator = LlmTurbulenceOperator(FakeRanker([0.72]), client)
        result = operator.apply("request urgent shelter help", llm_config())
        self.assertEqual(result.output, "request urgent shelter aid")
        self.assertEqual(client.calls[0][1]["system_prompt"], "System action")
        self.assertEqual(client.calls[0][1]["temperature"], 0.4)
        self.assertEqual(client.calls[0][1]["top_p"], 0.9)
        self.assertEqual(client.calls[0][1]["max_tokens"], 100)
        self.assertEqual(result.diagnostics["llm"]["systemPrompt"], "System action")
        self.assertEqual(result.diagnostics["llm"]["userPrompt"], "Generate 5: request urgent shelter help")
        self.assertEqual(result.diagnostics["llm"]["temperature"], 0.4)

    def test_llm_turbulence_selects_uniform_valid_candidate_from_seed(self):
        client = FakeLlmClient("1) request urgent shelter aid\n2) request urgent shelter support")
        payload = llm_config()
        payload["kCandidates"] = 2
        payload["selectionSeed"] = 1
        operator = LlmTurbulenceOperator(FakeRanker([0.72, 0.73]), client)
        result = operator.apply("request urgent shelter help", payload)
        self.assertEqual(result.output, "request urgent shelter aid")

    def test_shared_movement_plan_is_deterministic(self):
        individuals = [
            {"id": "ind-1", "rol": "a", "topico": "b", "accion": "c"},
            {"id": "ind-2", "rol": "d", "topico": "e", "accion": "f"},
        ]
        first = create_movement_plan(individuals, seed=7)
        second = create_movement_plan(individuals, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(individuals))
        self.assertEqual({movement.individual_id for movement in first}, {"ind-1", "ind-2"})
        self.assertTrue(all(movement.component_key in {"rol", "topico", "accion"} for movement in first))

    def test_recommendation_uses_fidelity_filter_then_cost(self):
        results = [
            {
                "strategyId": "expensive",
                "displayName": "Expensive",
                "relativeOperatorCost": 100,
                "operatorMetrics": {"successRate": 1.0, "operatorAverageSeconds": 1},
                "finalMetrics": {"averageFidelity": 0.6},
            },
            {
                "strategyId": "cheap",
                "displayName": "Cheap",
                "relativeOperatorCost": 5,
                "operatorMetrics": {"successRate": 0.5, "operatorAverageSeconds": 1},
                "finalMetrics": {"averageFidelity": 0.7},
            },
        ]
        chosen = recommend_strategy(results, {"finalFidelityMin": 0.25, "finalFidelityMax": 0.99})
        self.assertEqual(chosen["strategyId"], "cheap")

    def test_recommendation_does_not_fallback_to_invalid_final_metrics(self):
        chosen = recommend_strategy(
            [
                {
                    "strategyId": "cheap",
                    "displayName": "Cheap",
                    "relativeOperatorCost": 1,
                    "operatorMetrics": {"successRate": 1.0, "operatorAverageSeconds": 0.1},
                    "finalMetrics": {"averageFidelity": None},
                }
            ],
            {"finalFidelityMin": 0.25, "finalFidelityMax": 0.99},
        )
        self.assertIsNone(chosen["strategyId"])
        self.assertFalse(chosen["eligibleByFidelity"])

    def test_operator_metrics_and_final_deltas(self):
        rows = [
            {"success": True, "coverage": True, "elapsedSeconds": 0.2},
            {"success": False, "coverage": True, "elapsedSeconds": 0.4},
        ]
        costs = [
            {"wordnetQueries": 2, "embeddingTexts": 3, "embeddingWallClockSeconds": 0.05},
            {"wordnetQueries": 1, "embeddingTexts": 2, "embeddingWallClockSeconds": 0.04},
        ]
        summary = summarize_operator_rows(rows, costs, operator_wall_clock_seconds=0.45, operator_parallelism=2)
        deltas = final_metric_deltas(
            {"averageFidelity": 0.5, "averageDiversity": 0.2},
            {"averageFidelity": 0.65, "averageDiversity": 0.25},
        )
        self.assertEqual(summary["successRate"], 0.5)
        self.assertEqual(summary["coverageRate"], 1.0)
        self.assertAlmostEqual(summary["operatorCumulativeSeconds"], 0.6)
        self.assertAlmostEqual(summary["operatorWallClockSeconds"], 0.45)
        self.assertEqual(summary["operatorParallelism"], 2)
        self.assertEqual(summary["cost"]["wordnetQueries"], 3)
        self.assertAlmostEqual(deltas["averageFidelityDelta"], 0.15)
        self.assertAlmostEqual(deltas["averageDiversityDelta"], 0.05)


if __name__ == "__main__":
    unittest.main()
