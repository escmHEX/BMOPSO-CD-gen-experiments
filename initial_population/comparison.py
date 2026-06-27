from __future__ import annotations

import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baselines.comparator import (
    PROPOSAL_BY_ID,
    proposal_python_executable,
    proposal_python_path_entries,
    resolve_repository,
)
from initial_population.common_metrics import attach_common_metrics, finite_float, format_duration
from initial_population.service import InitialPopulationService


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

HYBRID_PROGRESS_PREFIX = "__INITIAL_POPULATION_PROGRESS__"
COMPARISON_PROGRESS_PREFIX = "__INITIAL_POPULATION_COMPARISON_PROGRESS__"
DEFAULT_REFERENCE_TEXT = "Our action center has been updated with more information about restaurant shutdowns and disaster financing options for SMBs."
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
INITIAL_BASELINE_PROPOSAL_IDS = {
    "evolmd-initial": "evolmd",
    "evolmd-mo-initial": "evolmd-mo",
}


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    display_name: str
    description: str
    runtime: str
    objective_names: tuple[str, ...]
    repository_path: str | None = None
    runner: str = ""


STRATEGIES: tuple[StrategyDefinition, ...] = (
    StrategyDefinition(
        strategy_id="hybrid-semantic-v7",
        display_name="Hybrid semantic initialization v7",
        description="Anchors, semantic pools, stratified prompts, prompt-space diversity and generated-text SBERT fidelity.",
        runtime="Ollama OpenAI-compatible",
        objective_names=("semantic_fidelity", "semantic_diversity"),
        runner="hybrid",
    ),
    StrategyDefinition(
        strategy_id="evolmd-initial",
        display_name="EVOLMD inicial",
        description="EVOLMD original up to initial population, data generation and initial BERTScore evaluation.",
        runtime="Ollama",
        objective_names=("fitness",),
        repository_path="baselines/external/evolmd",
        runner="baseline",
    ),
    StrategyDefinition(
        strategy_id="evolmd-mo-initial",
        display_name="EVOLMD-MO inicial",
        description="EVOLMD-MO original up to initial population, data generation and initial SBERT/diversity evaluation.",
        runtime="Ollama",
        objective_names=("fidelity_sbert", "diversity_individual"),
        repository_path="baselines/external/evolmd-mo",
        runner="baseline",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: OSError | None = None
    for attempt in range(6):
        temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            temp_path.write_text(encoded, encoding="utf-8")
            temp_path.replace(path)
            return
        except OSError as error:
            last_error = error
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def read_json_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def status_label(status: str) -> str:
    return {
        STATUS_QUEUED: "En cola",
        STATUS_RUNNING: "Ejecutando",
        STATUS_COMPLETED: "Completada",
        STATUS_FAILED: "Fallida",
        STATUS_CANCELLED: "Cancelada",
    }.get(status, status)


def interval_union_seconds(calls: list[dict[str, Any]]) -> float | None:
    intervals: list[tuple[float, float]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        start = call.get("startedSeconds")
        finish = call.get("finishedSeconds")
        if start is None or finish is None:
            continue
        start_float = finite_float(start)
        finish_float = finite_float(finish)
        if finish_float > start_float:
            intervals.append((start_float, finish_float))
    if not intervals:
        return None
    intervals.sort(key=lambda item: item[0])
    total = 0.0
    current_start, current_finish = intervals[0]
    for start, finish in intervals[1:]:
        if start <= current_finish:
            current_finish = max(current_finish, finish)
            continue
        total += current_finish - current_start
        current_start, current_finish = start, finish
    total += current_finish - current_start
    return total


def summarize_cost(cost: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(cost or {})
    wall_clock = enriched.get("wallClockSeconds") or enriched.get("processWallClockSeconds")
    llm_seconds = enriched.get("llmClientWallClockSeconds")
    wall_clock_float = finite_float(wall_clock)
    final_individuals = finite_float(enriched.get("finalIndividuals"))
    if final_individuals > 0 and enriched.get("wallClockPerFinalIndividualSeconds") is None:
        enriched["wallClockPerFinalIndividualSeconds"] = wall_clock_float / final_individuals
    if llm_seconds is not None and wall_clock is not None and enriched.get("estimatedSequentialWallClockSeconds") is None:
        llm_seconds_float = finite_float(llm_seconds)
        concurrent = finite_float(enriched.get("llmConcurrentWallClockSeconds"), min(llm_seconds_float, wall_clock_float))
        sequential = wall_clock_float + max(0.0, llm_seconds_float - concurrent)
        savings = max(0.0, sequential - wall_clock_float)
        enriched["estimatedSequentialWallClockSeconds"] = sequential
        enriched["parallelismSavingsSeconds"] = savings
        enriched["parallelismSavingsPercent"] = (savings / sequential * 100.0) if sequential > 0 else None
    if enriched.get("estimatedSequentialWallClockSeconds") is not None:
        sequential = finite_float(enriched.get("estimatedSequentialWallClockSeconds"))
        if sequential < wall_clock_float:
            enriched["estimatedSequentialWallClockSeconds"] = wall_clock_float
            enriched["parallelismSavingsSeconds"] = 0.0
            enriched["parallelismSavingsPercent"] = 0.0
    for key, label_key in (
        ("wallClockSeconds", "wallClockLabel"),
        ("processWallClockSeconds", "processWallClockLabel"),
        ("estimatedSequentialWallClockSeconds", "estimatedSequentialWallClockLabel"),
        ("parallelismSavingsSeconds", "parallelismSavingsLabel"),
        ("llmClientWallClockSeconds", "llmClientWallClockLabel"),
        ("llmAverageCallSeconds", "llmAverageCallLabel"),
        ("embeddingWallClockSeconds", "embeddingWallClockLabel"),
        ("commonEmbeddingWallClockSeconds", "commonEmbeddingWallClockLabel"),
        ("wallClockPerFinalIndividualSeconds", "wallClockPerFinalIndividualLabel"),
    ):
        enriched[label_key] = format_duration(enriched.get(key))
    return enriched


def empty_cost() -> dict[str, Any]:
    return {
        "wallClockSeconds": 0.0,
        "processWallClockSeconds": 0.0,
        "llmCalls": 0,
        "llmSuccessfulCalls": 0,
        "llmFailedCalls": 0,
        "llmClientWallClockSeconds": 0.0,
        "llmAverageCallSeconds": None,
        "promptTokens": 0,
        "completionTokens": 0,
        "totalTokens": 0,
        "embeddingWallClockSeconds": 0.0,
        "commonEmbeddingWallClockSeconds": 0.0,
        "finalIndividuals": 0,
    }


def average_present(values: list[Any]) -> float | None:
    numbers = [finite_float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def average_vector(vectors: list[Any]) -> list[float]:
    normalized = [
        [finite_float(value) for value in vector]
        for vector in vectors
        if isinstance(vector, list) and vector
    ]
    if not normalized:
        return []
    width = min(len(vector) for vector in normalized)
    return [sum(vector[index] for vector in normalized) / len(normalized) for index in range(width)]


def vector_label(vector: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]" if vector else "--"


def aggregate_initial_costs(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = empty_cost()
    for result in results:
        cost = result.get("cost") or {}
        for key, value in cost.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                totals[key] = finite_float(totals.get(key)) + float(value)
    return summarize_cost(totals)


def aggregate_common_metric_costs(results: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [result.get("commonMetricCost") or {} for result in results]
    seconds = sum(finite_float(cost.get("embeddingWallClockSeconds")) for cost in costs)
    texts = sum(int(finite_float(cost.get("embeddingTexts"))) for cost in costs)
    return {
        "embeddingModel": next((cost.get("embeddingModel") for cost in costs if cost.get("embeddingModel")), None),
        "embeddingTexts": texts,
        "embeddingWallClockSeconds": seconds,
        "embeddingWallClockLabel": format_duration(seconds),
    }


def aggregate_initial_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_list = [result.get("metrics") or {} for result in results]
    vector = average_vector([metrics.get("bestObjectiveVector") for metrics in metrics_list])
    hypervolume = average_present([metrics.get("hypervolume") for metrics in metrics_list])
    spread = average_present([metrics.get("spread") for metrics in metrics_list])
    return {
        "commonObjectiveNames": next((metrics.get("commonObjectiveNames") for metrics in metrics_list if metrics.get("commonObjectiveNames")), []),
        "completedRows": average_present([metrics.get("completedRows") for metrics in metrics_list]),
        "totalRows": average_present([metrics.get("totalRows") for metrics in metrics_list]),
        "nonDominatedRows": average_present([metrics.get("nonDominatedRows") for metrics in metrics_list]),
        "bestObjectiveVector": vector,
        "bestObjectiveLabel": vector_label(vector),
        "bestFidelity": average_present([metrics.get("bestFidelity") for metrics in metrics_list]),
        "bestDiversity": average_present([metrics.get("bestDiversity") for metrics in metrics_list]),
        "hypervolume": hypervolume,
        "hypervolumeLabel": f"{hypervolume:.6f}" if hypervolume is not None else "No aplica",
        "spread": spread,
        "spreadLabel": f"{spread:.6f}" if spread is not None else "No aplica",
        "moConvention": "Metricas ponderadas como promedio sobre repeticiones K con semillas distintas.",
    }


def aggregate_initial_strategy_repetitions(
    strategy: StrategyDefinition,
    results: list[dict[str, Any]],
    repetitions_k: int,
) -> dict[str, Any]:
    completed = [result for result in results if result.get("status") == STATUS_COMPLETED]
    if results:
        first_dir = Path(str(results[0].get("outputDir") or "."))
        base_dir = str(first_dir.parent if first_dir.name.startswith("rep-") else first_dir)
    else:
        base_dir = ""
    if not completed:
        failure = results[-1] if results else {}
        return {
            "strategyId": strategy.strategy_id,
            "displayName": strategy.display_name,
            "runtime": strategy.runtime,
            "status": STATUS_FAILED,
            "outputDir": base_dir,
            "rows": [],
            "metrics": {
                "totalRows": 0,
                "completedRows": 0,
                "nonDominatedRows": 0,
                "bestObjectiveVector": [],
                "bestObjectiveLabel": "--",
                "hypervolumeLabel": "No aplica",
                "spreadLabel": "No aplica",
            },
            "nativeMetrics": {},
            "cost": aggregate_initial_costs(results),
            "commonMetricCost": aggregate_common_metric_costs(results),
            "artifacts": {},
            "repetitionsK": repetitions_k,
            "completedRepetitions": 0,
            "repetitions": results,
            "error": failure.get("error") or "Todas las repeticiones fallaron.",
        }

    rows: list[dict[str, Any]] = []
    for result in completed:
        repetition_index = result.get("repetitionIndex")
        repetition_seed = result.get("repetitionSeed")
        for row in result.get("rows") or []:
            if isinstance(row, dict):
                rows.append({**row, "repetitionIndex": repetition_index, "repetitionSeed": repetition_seed})

    metrics = aggregate_initial_metrics(completed)
    metrics["outputDir"] = base_dir
    return {
        "strategyId": strategy.strategy_id,
        "displayName": strategy.display_name,
        "runtime": strategy.runtime,
        "status": STATUS_COMPLETED,
        "outputDir": base_dir,
        "rows": rows,
        "metrics": metrics,
        "nativeMetrics": aggregate_initial_metrics([
            {"metrics": result.get("nativeMetrics") or {}}
            for result in completed
        ]),
        "cost": aggregate_initial_costs(completed),
        "commonMetricCost": aggregate_common_metric_costs(completed),
        "artifacts": completed[-1].get("artifacts") or {},
        "repetitionsK": repetitions_k,
        "completedRepetitions": len(completed),
        "repetitions": results,
        "error": None if len(completed) == repetitions_k else f"{repetitions_k - len(completed)} repeticion(es) fallaron.",
    }


class InitialPopulationComparisonService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_root = root / "runs" / "initial-population-comparison"
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._hybrid_service = InitialPopulationService(root)

    def list_strategies(self) -> list[dict[str, Any]]:
        return [
            {
                "strategyId": strategy.strategy_id,
                "displayName": strategy.display_name,
                "description": strategy.description,
                "runtime": strategy.runtime,
                "objectiveNames": list(strategy.objective_names),
                "available": self._strategy_available(strategy),
                "repositoryPath": strategy.repository_path,
            }
            for strategy in STRATEGIES
        ]

    def default_config(self) -> dict[str, Any]:
        hybrid = self._hybrid_service.default_config()
        hybrid["strategyId"] = "hybrid-semantic-v7"
        return {
            "referenceText": DEFAULT_REFERENCE_TEXT,
            "selectedStrategies": [strategy.strategy_id for strategy in STRATEGIES],
            "n": 10,
            "topK": 10,
            "seed": 42,
            "repetitionsK": 1,
            "commonEmbeddingModel": DEFAULT_EMBEDDING_MODEL,
            "hybrid": hybrid,
            "baselines": {
                "model": "llama3",
                "bertModel": "bert-base-uncased",
                "timeoutMinutes": 60,
                "tempPrompts": 0.9,
                "tempKeywords": 0.3,
                "tempGeneration": 0.7,
            },
        }

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._read_config(payload)
        run_id = self._new_run_id()
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", config)

        run = {
            "runId": run_id,
            "status": STATUS_QUEUED,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "startedAtEpoch": None,
            "runDir": str(run_dir),
            "config": config,
            "logs": [],
            "strategies": [],
            "strategyStates": {
                strategy.strategy_id: self._initial_strategy_state(strategy)
                for strategy in self._selected_definitions(config)
            },
            "progress": {
                "percent": 0,
                "detail": "Comparacion en cola.",
                "elapsedSeconds": 0,
                "elapsedLabel": "0s",
                "remainingSeconds": None,
                "remainingLabel": "No disponible",
            },
            "costSummary": self._summarize_costs([], 0.0),
            "error": None,
            "cancelRequested": False,
            "activeProcess": None,
        }
        with self._lock:
            self._runs[run_id] = run
            self._write_summary_unlocked(run)

        thread = threading.Thread(target=self._run_worker, args=(run_id,), daemon=True)
        thread.start()
        return self._public_run(run)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                if run.get("status") == STATUS_RUNNING:
                    self._refresh_run_progress_unlocked(run)
                    self._write_summary_unlocked(run)
                return self._public_run(run)

        summary_path = self.runs_root / run_id / "summary.json"
        if summary_path.exists():
            return read_json_or_default(summary_path, None)
        return None

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            run["cancelRequested"] = True
            run["updatedAt"] = utc_now()
            process = run.get("activeProcess")
            self._append_log_unlocked(run, "system", "Cancel requested. Terminating active strategy process.")
            self._mark_queued_as_cancelled_unlocked(run)

        if process is not None:
            self._terminate_process(process)

        with self._lock:
            self._write_summary_unlocked(run)
            return self._public_run(run)

    def _read_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.default_config()
        reference_text = str(payload.get("referenceText", defaults["referenceText"])).strip()
        if not reference_text:
            raise ValueError("referenceText is required.")
        selected = payload.get("selectedStrategies", defaults["selectedStrategies"])
        if not isinstance(selected, list):
            raise ValueError("selectedStrategies must be a list.")
        selected_ids = [str(item).strip() for item in selected if str(item).strip()]
        known = {strategy.strategy_id for strategy in STRATEGIES}
        unknown = [strategy_id for strategy_id in selected_ids if strategy_id not in known]
        if unknown:
            raise ValueError(f"Unsupported strategy id(s): {', '.join(unknown)}.")
        selected_ids = [strategy_id for strategy_id in selected_ids if self._strategy_available(self._definition(strategy_id))]
        if not selected_ids:
            raise ValueError("Select at least one available strategy.")

        n = self._int_between(payload.get("n", defaults["n"]), "n", 1, 500)
        top_k = min(self._int_between(payload.get("topK", defaults["topK"]), "topK", 1, 500), n)
        seed_value = payload.get("seed", defaults["seed"])
        seed = self._int_between(seed_value, "seed", 0, 2_147_483_647) if seed_value is not None else None
        repetitions_k = self._int_between(payload.get("repetitionsK", defaults["repetitionsK"]), "repetitionsK", 1, 30)
        common_embedding = self._safe_text(payload.get("commonEmbeddingModel", defaults["commonEmbeddingModel"]), "commonEmbeddingModel")

        hybrid_payload = payload.get("hybrid") if isinstance(payload.get("hybrid"), dict) else defaults["hybrid"]
        hybrid_payload = dict(hybrid_payload)
        hybrid_payload.update({
            "strategyId": "hybrid-semantic-v7",
            "referenceText": reference_text,
            "n": n,
            "topK": top_k,
            "seed": seed if seed is not None else hybrid_payload.get("seed", defaults["hybrid"]["seed"]),
            "embeddingModel": common_embedding,
        })
        hybrid_config = self._hybrid_service._read_config(hybrid_payload)

        baselines_payload = payload.get("baselines") if isinstance(payload.get("baselines"), dict) else {}
        baselines_defaults = defaults["baselines"]
        baseline_config = {
            "model": self._safe_text(baselines_payload.get("model", baselines_defaults["model"]), "baselines.model"),
            "bertModel": self._safe_text(baselines_payload.get("bertModel", baselines_defaults["bertModel"]), "baselines.bertModel"),
            "timeoutMinutes": self._int_between(baselines_payload.get("timeoutMinutes", baselines_defaults["timeoutMinutes"]), "baselines.timeoutMinutes", 1, 1440),
            "tempPrompts": self._float_between(baselines_payload.get("tempPrompts", baselines_defaults["tempPrompts"]), "baselines.tempPrompts", 0.0, 2.0),
            "tempKeywords": self._float_between(baselines_payload.get("tempKeywords", baselines_defaults["tempKeywords"]), "baselines.tempKeywords", 0.0, 2.0),
            "tempGeneration": self._float_between(baselines_payload.get("tempGeneration", baselines_defaults["tempGeneration"]), "baselines.tempGeneration", 0.0, 2.0),
        }
        return {
            "referenceText": reference_text,
            "selectedStrategies": selected_ids,
            "n": n,
            "topK": top_k,
            "seed": seed,
            "repetitionsK": repetitions_k,
            "commonEmbeddingModel": common_embedding,
            "hybrid": hybrid_config,
            "baselines": baseline_config,
        }

    def _run_worker(self, run_id: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = STATUS_RUNNING
            run["startedAtEpoch"] = time.time()
            run["updatedAt"] = utc_now()
            self._append_log_unlocked(run, "system", "Initial-population comparison started.")
            self._write_summary_unlocked(run)

        try:
            for strategy in self._selected_definitions(run["config"]):
                with self._lock:
                    if run.get("cancelRequested"):
                        self._mark_queued_as_cancelled_unlocked(run)
                        break
                result = self._execute_strategy_repeated(run, strategy)
                with self._lock:
                    run["strategies"].append(result)
                    state = run["strategyStates"].setdefault(strategy.strategy_id, self._initial_strategy_state(strategy))
                    state["status"] = result["status"]
                    state["stageLabel"] = status_label(result["status"])
                    state["progress"] = 1.0
                    state["updatedAt"] = utc_now()
                    self._refresh_run_progress_unlocked(run)
                    self._refresh_cost_summary_unlocked(run)
                    self._write_summary_unlocked(run)

            with self._lock:
                failed = [item for item in run["strategies"] if item["status"] == STATUS_FAILED]
                if run.get("cancelRequested"):
                    run["status"] = STATUS_CANCELLED
                elif failed and len(failed) == len(run["strategies"]):
                    run["status"] = STATUS_FAILED
                    run["error"] = "All selected strategies failed."
                elif failed:
                    run["status"] = STATUS_FAILED
                    run["error"] = "Strategy failures: " + ", ".join(item["displayName"] for item in failed)
                else:
                    run["status"] = STATUS_COMPLETED
                run["updatedAt"] = utc_now()
                self._refresh_run_progress_unlocked(run)
                self._refresh_cost_summary_unlocked(run)
                self._append_log_unlocked(run, "system", f"Initial-population comparison finished with status {run['status']}.")
                self._write_summary_unlocked(run)
        except Exception as error:
            with self._lock:
                run["status"] = STATUS_FAILED
                run["error"] = str(error)
                run["updatedAt"] = utc_now()
                self._refresh_run_progress_unlocked(run)
                self._refresh_cost_summary_unlocked(run)
                self._append_log_unlocked(run, "system", f"Unexpected comparison error: {error}")
                self._write_summary_unlocked(run)

    def _execute_strategy_repeated(self, run: dict[str, Any], strategy: StrategyDefinition) -> dict[str, Any]:
        repetitions_k = int(run["config"].get("repetitionsK") or 1)
        if repetitions_k <= 1:
            return self._execute_strategy(run, strategy, 0, 1)

        results: list[dict[str, Any]] = []
        for repetition_index in range(repetitions_k):
            with self._lock:
                if run.get("cancelRequested"):
                    break
                self._set_strategy_state_unlocked(
                    run,
                    strategy.strategy_id,
                    STATUS_RUNNING,
                    f"Repeticion {repetition_index + 1}/{repetitions_k}",
                    repetition_index / repetitions_k,
                )
            result = self._execute_strategy(run, strategy, repetition_index, repetitions_k)
            result["repetitionIndex"] = repetition_index + 1
            result["repetitionSeed"] = self._repetition_seed(run["config"].get("seed"), repetition_index)
            results.append(result)

        return aggregate_initial_strategy_repetitions(strategy, results, repetitions_k)

    def _execute_strategy(
        self,
        run: dict[str, Any],
        strategy: StrategyDefinition,
        repetition_index: int = 0,
        repetition_count: int = 1,
    ) -> dict[str, Any]:
        with self._lock:
            label = "Preparando ejecucion" if repetition_count <= 1 else f"Preparando ejecucion {repetition_index + 1}/{repetition_count}"
            self._set_strategy_state_unlocked(run, strategy.strategy_id, STATUS_RUNNING, label, 0.01)

        execution_config = self._config_for_repetition(run["config"], repetition_index)
        strategy_dir = Path(run["runDir"]) / strategy.strategy_id
        if repetition_count > 1:
            strategy_dir = strategy_dir / f"rep-{repetition_index + 1:03d}"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        if strategy.runner == "hybrid":
            process_result = self._run_hybrid_process(run, strategy, strategy_dir, execution_config)
        else:
            process_result = self._run_baseline_process(run, strategy, strategy_dir, execution_config)

        result = read_json_or_default(strategy_dir / "result.json", {})
        if run.get("cancelRequested"):
            return self._cancelled_result(strategy, strategy_dir, process_result)
        if process_result.get("timedOut"):
            return self._failed_result(strategy, strategy_dir, f"Process timed out after {process_result['timeoutLabel']}.", process_result)
        if int(process_result.get("returnCode") or 0) != 0 and not result:
            return self._failed_result(strategy, strategy_dir, f"Process exited with code {process_result.get('returnCode')}.", process_result)
        if not isinstance(result, dict) or result.get("status") != STATUS_COMPLETED:
            error = result.get("error") if isinstance(result, dict) else None
            return self._failed_result(strategy, strategy_dir, self._error_label(error) if error else "Strategy did not produce a completed result.", process_result, result)

        try:
            normalized = self._normalize_strategy_result(run, strategy, strategy_dir, result, process_result, execution_config)
            write_json(strategy_dir / "normalized_result.json", normalized)
            return normalized
        except Exception as error:
            return self._failed_result(strategy, strategy_dir, f"Could not normalize strategy result: {error}", process_result, result)

    def _run_hybrid_process(self, run: dict[str, Any], strategy: StrategyDefinition, strategy_dir: Path, execution_config: dict[str, Any]) -> dict[str, Any]:
        config_path = strategy_dir / "hybrid_config.json"
        write_json(config_path, execution_config["hybrid"])
        command = [
            sys.executable,
            "-m",
            "initial_population.runner",
            "--config",
            str(config_path),
            "--output-dir",
            str(strategy_dir),
        ]
        return self._run_process(run, strategy.strategy_id, command, self.root, execution_config["hybrid"]["timeoutSeconds"], "second(s)", None)

    def _run_baseline_process(self, run: dict[str, Any], strategy: StrategyDefinition, strategy_dir: Path, execution_config: dict[str, Any]) -> dict[str, Any]:
        assert strategy.repository_path is not None
        repository_dir = self._baseline_repository_dir(strategy)
        baseline_config = {
            **execution_config["baselines"],
            "strategyId": strategy.strategy_id,
            "displayName": strategy.display_name,
            "referenceText": execution_config["referenceText"],
            "n": execution_config["n"],
            "seed": execution_config["seed"],
            "repositoryDir": str(repository_dir),
        }
        config_path = strategy_dir / "baseline_config.json"
        write_json(config_path, baseline_config)
        cost_metrics_path = strategy_dir / "ollama_cost_metrics.json"
        command = [
            self._baseline_python_executable(strategy, repository_dir),
            str(self.root / "baselines" / "bootstrap.py"),
            str(self.root / "initial_population" / "baseline_runner.py"),
            "--config",
            str(config_path),
            "--output-dir",
            str(strategy_dir),
        ]
        timeout_seconds = execution_config["baselines"]["timeoutMinutes"] * 60
        return self._run_process(
            run,
            strategy.strategy_id,
            command,
            self.root,
            timeout_seconds,
            "minute(s)",
            cost_metrics_path,
            self._baseline_python_path_entries(strategy),
        )

    def _baseline_proposal(self, strategy: StrategyDefinition):
        proposal_id = INITIAL_BASELINE_PROPOSAL_IDS.get(strategy.strategy_id)
        return PROPOSAL_BY_ID.get(proposal_id or "")

    def _baseline_repository_dir(self, strategy: StrategyDefinition) -> Path:
        proposal = self._baseline_proposal(strategy)
        if proposal is not None:
            return resolve_repository(self.root, proposal)
        assert strategy.repository_path is not None
        return self.root / strategy.repository_path

    def _baseline_python_executable(self, strategy: StrategyDefinition, repository_dir: Path) -> str:
        proposal = self._baseline_proposal(strategy)
        if proposal is not None:
            return proposal_python_executable(self.root, repository_dir, proposal)
        return sys.executable

    def _baseline_python_path_entries(self, strategy: StrategyDefinition) -> tuple[str, ...]:
        proposal = self._baseline_proposal(strategy)
        if proposal is None:
            return ()
        return tuple(proposal_python_path_entries(self.root, proposal))

    def _run_process(
        self,
        run: dict[str, Any],
        strategy_id: str,
        command: list[str],
        cwd: Path,
        timeout_seconds: int,
        timeout_unit_label: str,
        cost_metrics_path: Path | None,
        python_path_entries: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        if python_path_entries:
            environment["PYTHONPATH"] = os.pathsep.join(
                [*python_path_entries, environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
        if cost_metrics_path is not None:
            environment["BASELINE_COST_METRICS_PATH"] = str(cost_metrics_path)
            environment["BASELINE_PRELOAD_MODULES"] = "torch"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            creationflags=creationflags,
            start_new_session=sys.platform != "win32",
        )
        with self._lock:
            run["activeProcess"] = process

        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout_seconds
        reader_done = False
        timed_out = False
        return_code: int | None = None
        termination_requested = False

        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                line = ""

            if line is None:
                reader_done = True
            elif line:
                self._handle_process_line(run, strategy_id, line)

            if run.get("cancelRequested") and process.poll() is None and not termination_requested:
                self._append_log(run, strategy_id, "Cancellation requested; terminating process tree.")
                self._terminate_process(process)
                termination_requested = True

            if process.poll() is None and time.monotonic() >= deadline and not termination_requested:
                self._append_log(run, strategy_id, f"Timeout reached after {timeout_seconds} seconds.")
                self._terminate_process(process)
                termination_requested = True
                timed_out = True
                return_code = 124
                break

            if reader_done and process.poll() is not None:
                return_code = process.returncode
                break

        reader.join(timeout=2)
        with self._lock:
            if run.get("activeProcess") is process:
                run["activeProcess"] = None
        return {
            "returnCode": return_code,
            "processWallClockSeconds": time.perf_counter() - started,
            "timedOut": timed_out,
            "timeoutSeconds": timeout_seconds,
            "timeoutLabel": f"{timeout_seconds if timeout_unit_label.startswith('second') else timeout_seconds // 60} {timeout_unit_label}",
            "costMetricsPath": str(cost_metrics_path) if cost_metrics_path else None,
        }

    def _normalize_strategy_result(
        self,
        run: dict[str, Any],
        strategy: StrategyDefinition,
        strategy_dir: Path,
        result: dict[str, Any],
        process_result: dict[str, Any],
        execution_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = execution_config or run["config"]
        rows = [dict(row) for row in result.get("rows", []) if isinstance(row, dict)]
        if strategy.strategy_id == "hybrid-semantic-v7":
            rows = self._normalize_hybrid_rows(rows)
            native_metrics = result.get("metrics") or {}
            artifacts = self._read_hybrid_artifacts(strategy_dir)
            native_cost = summarize_cost(result.get("cost") or {})
            native_cost["runtime"] = "Ollama OpenAI-compatible"
        else:
            native_metrics = result.get("nativeMetrics") or {}
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
            native_cost = self._build_baseline_cost(strategy_dir, process_result, result)
            native_cost["runtime"] = "Ollama"

        rows, common_metrics, common_cost = attach_common_metrics(
            rows,
            config["referenceText"],
            config["commonEmbeddingModel"],
        )
        rows.sort(
            key=lambda row: (
                1 if row.get("commonNonDominated") else 0,
                sum(finite_float(value) for value in (row.get("commonObjectiveVector") or [])),
                -finite_float(row.get("rank"), 0),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, start=1):
            row["comparisonRank"] = rank
            row["strategyId"] = strategy.strategy_id
            row["displayName"] = strategy.display_name

        native_cost["processWallClockSeconds"] = process_result.get("processWallClockSeconds")
        native_cost["wallClockSeconds"] = process_result.get("processWallClockSeconds")
        native_cost["returnCode"] = process_result.get("returnCode")
        native_cost["timedOut"] = process_result.get("timedOut")
        native_cost["finalIndividuals"] = len([row for row in rows if row.get("status") == "ok"])
        native_cost["commonEmbeddingWallClockSeconds"] = common_cost.get("embeddingWallClockSeconds")
        for key in (
            "estimatedSequentialWallClockSeconds",
            "estimatedSequentialWallClockLabel",
            "parallelismSavingsSeconds",
            "parallelismSavingsLabel",
            "parallelismSavingsPercent",
        ):
            native_cost.pop(key, None)
        native_cost = summarize_cost(native_cost)

        return {
            "strategyId": strategy.strategy_id,
            "displayName": strategy.display_name,
            "runtime": strategy.runtime,
            "status": STATUS_COMPLETED,
            "outputDir": result.get("outputDir") or str(strategy_dir),
            "rows": rows,
            "metrics": {
                **common_metrics,
                "totalRows": len(rows),
                "outputDir": result.get("outputDir") or str(strategy_dir),
            },
            "nativeMetrics": native_metrics,
            "cost": native_cost,
            "commonMetricCost": common_cost,
            "artifacts": artifacts,
            "error": None,
        }

    def _config_for_repetition(self, config: dict[str, Any], repetition_index: int) -> dict[str, Any]:
        seed = self._repetition_seed(config.get("seed"), repetition_index)
        hybrid = dict(config["hybrid"])
        hybrid["seed"] = seed if seed is not None else hybrid.get("seed")
        return {
            **config,
            "seed": seed,
            "hybrid": hybrid,
            "baselines": dict(config["baselines"]),
        }

    def _repetition_seed(self, seed: Any, repetition_index: int) -> int | None:
        if seed is None:
            return None
        return (int(seed) + int(repetition_index)) % 2_147_483_648

    def _normalize_hybrid_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            native_vector = row.get("objectiveVector") if isinstance(row.get("objectiveVector"), list) else []
            normalized.append({
                **row,
                "sourceIndex": row.get("sourceIndex", index),
                "rank": row.get("rank", index),
                "role": str(row.get("role") or ""),
                "topic": str(row.get("topic") or ""),
                "action": str(row.get("action") or ""),
                "prompt": str(row.get("prompt") or ""),
                "generatedText": str(row.get("generatedText") or ""),
                "nativeObjectiveVector": native_vector,
                "nativeObjectiveLabel": row.get("objectiveLabel") or "--",
                "nativeObjectiveNames": row.get("objectiveNames") or ["semantic_fidelity", "semantic_diversity"],
                "nativeNonDominated": bool(row.get("nonDominated")),
            })
        return normalized

    def _read_hybrid_artifacts(self, strategy_dir: Path) -> dict[str, Any]:
        anchors = read_json_or_default(strategy_dir / "anchors.json", {})
        pools = read_json_or_default(strategy_dir / "pools.json", {})
        expansion = read_json_or_default(strategy_dir / "pool_expansion.json", None)
        return {
            "semanticArtifacts": {
                "anchors": anchors if isinstance(anchors, dict) else {},
                "pools": pools if isinstance(pools, dict) else {},
                "expansion": expansion,
            },
            "runtimeConfig": {
                "runtime": "Ollama OpenAI-compatible",
            },
        }

    def _build_baseline_cost(self, strategy_dir: Path, process_result: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        metrics_path = process_result.get("costMetricsPath")
        llm_payload = read_json_or_default(Path(metrics_path), {}) if metrics_path else {}
        summary = llm_payload.get("summary") if isinstance(llm_payload, dict) else {}
        summary = summary if isinstance(summary, dict) else {}
        calls = llm_payload.get("calls") if isinstance(llm_payload, dict) else []
        calls = calls if isinstance(calls, list) else []
        llm_calls = int(finite_float(summary.get("totalCalls")))
        llm_seconds = finite_float(summary.get("clientWallClockSeconds"))
        concurrent = interval_union_seconds(calls)
        runtime = result.get("runtime") if isinstance(result.get("runtime"), dict) else {}
        final_individuals = len([row for row in result.get("rows", []) if isinstance(row, dict) and row.get("status") == "ok"])
        prompt_tokens = int(finite_float(summary.get("promptEvalCount")))
        completion_tokens = int(finite_float(summary.get("evalCount")))
        native_embedding_seconds = finite_float(runtime.get("initial_eval_sec")) if result.get("strategyId") == "evolmd-mo-initial" else 0.0
        return summarize_cost({
            "wallClockSeconds": process_result.get("processWallClockSeconds"),
            "processWallClockSeconds": process_result.get("processWallClockSeconds"),
            "llmCalls": llm_calls,
            "llmSuccessfulCalls": int(finite_float(summary.get("successfulCalls"))),
            "llmFailedCalls": int(finite_float(summary.get("failedCalls"))),
            "llmClientWallClockSeconds": llm_seconds,
            "llmConcurrentWallClockSeconds": concurrent,
            "llmAverageCallSeconds": llm_seconds / llm_calls if llm_calls else None,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
            "ollamaTotalDurationSeconds": finite_float(summary.get("ollamaTotalDurationSeconds")),
            "embeddingWallClockSeconds": native_embedding_seconds,
            "runtimeBreakdown": runtime,
            "metricsPath": metrics_path,
            "finalIndividuals": final_individuals,
            "llmCallsPerFinalIndividual": llm_calls / final_individuals if final_individuals else None,
        })

    def _failed_result(
        self,
        strategy: StrategyDefinition,
        strategy_dir: Path,
        message: str,
        process_result: dict[str, Any] | None = None,
        raw_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cost = empty_cost()
        if process_result:
            cost["processWallClockSeconds"] = process_result.get("processWallClockSeconds")
            cost["wallClockSeconds"] = process_result.get("processWallClockSeconds")
            cost["returnCode"] = process_result.get("returnCode")
            cost["timedOut"] = process_result.get("timedOut")
        cost = summarize_cost(cost)
        return {
            "strategyId": strategy.strategy_id,
            "displayName": strategy.display_name,
            "runtime": strategy.runtime,
            "status": STATUS_FAILED,
            "outputDir": str(strategy_dir),
            "rows": [],
            "metrics": {
                "totalRows": 0,
                "completedRows": 0,
                "nonDominatedRows": 0,
                "bestObjectiveVector": [],
                "bestObjectiveLabel": "--",
                "hypervolumeLabel": "No aplica",
                "spreadLabel": "No aplica",
                "outputDir": str(strategy_dir),
            },
            "nativeMetrics": raw_result.get("nativeMetrics", {}) if isinstance(raw_result, dict) else {},
            "cost": cost,
            "commonMetricCost": {},
            "artifacts": raw_result.get("artifacts", {}) if isinstance(raw_result, dict) else {},
            "error": message,
        }

    def _cancelled_result(
        self,
        strategy: StrategyDefinition,
        strategy_dir: Path,
        process_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self._failed_result(strategy, strategy_dir, "Execution was cancelled.", process_result)
        result["status"] = STATUS_CANCELLED
        result["cost"]["cancelled"] = True
        return result

    def _handle_process_line(self, run: dict[str, Any], strategy_id: str, line: str) -> None:
        message = ANSI_RE.sub("", line).strip()
        if not message:
            return
        for prefix in (HYBRID_PROGRESS_PREFIX, COMPARISON_PROGRESS_PREFIX):
            if message.startswith(prefix):
                raw = message.removeprefix(prefix)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    self._append_log(run, strategy_id, message)
                    return
                with self._lock:
                    percent = finite_float(payload.get("percent")) / 100.0
                    self._set_strategy_state_unlocked(
                        run,
                        strategy_id,
                        STATUS_RUNNING,
                        str(payload.get("detail") or "Ejecutando."),
                        percent,
                        payload,
                    )
                    self._write_summary_unlocked(run)
                return
        self._append_log(run, strategy_id, message)

    def _set_strategy_state_unlocked(
        self,
        run: dict[str, Any],
        strategy_id: str,
        status: str,
        stage_label: str,
        progress: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        strategy = self._definition(strategy_id)
        state = run["strategyStates"].setdefault(strategy_id, self._initial_strategy_state(strategy))
        state["status"] = status
        state["stageLabel"] = stage_label
        if progress is not None:
            state["progress"] = max(0.0, min(0.99, progress if status == STATUS_RUNNING else progress))
        if payload:
            state["stage"] = payload.get("stage")
            state["stageIndex"] = payload.get("stageIndex")
            state["stageTotal"] = payload.get("stageTotal")
            state["completed"] = payload.get("completed")
            state["total"] = payload.get("total")
        state["updatedAt"] = utc_now()
        run["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)

    def _refresh_run_progress_unlocked(self, run: dict[str, Any]) -> None:
        states = list((run.get("strategyStates") or {}).values())
        if not states:
            return
        progress_sum = sum(float(state.get("progress") or 0.0) for state in states)
        percent = int(max(0, min(100, round(100 * progress_sum / len(states)))))
        active = next((state for state in states if state.get("status") == STATUS_RUNNING), None)
        queued = sum(1 for state in states if state.get("status") == STATUS_QUEUED)
        completed = sum(1 for state in states if state.get("status") == STATUS_COMPLETED)
        failed = sum(1 for state in states if state.get("status") == STATUS_FAILED)
        cancelled = sum(1 for state in states if state.get("status") == STATUS_CANCELLED)
        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        remaining = elapsed * (100 - percent) / percent if run.get("status") == STATUS_RUNNING and 0 < percent < 100 else None
        if active:
            detail = f"{active.get('displayName')}: {active.get('stageLabel', 'Ejecutando')}"
        elif run.get("cancelRequested"):
            detail = "Cancelacion solicitada; esperando cierre del proceso activo."
        else:
            detail = f"{completed} completada(s), {failed} fallida(s), {cancelled} cancelada(s), {queued} en cola."
        run["progress"] = {
            "percent": percent,
            "detail": detail,
            "elapsedSeconds": elapsed,
            "elapsedLabel": format_duration(elapsed),
            "remainingSeconds": remaining,
            "remainingLabel": format_duration(remaining),
            "activeStrategyId": active.get("strategyId") if active else None,
            "activeStrategyName": active.get("displayName") if active else None,
            "queuedStrategies": queued,
            "completedStrategies": completed,
            "failedStrategies": failed,
            "cancelledStrategies": cancelled,
            "totalStrategies": len(states),
        }

    def _refresh_cost_summary_unlocked(self, run: dict[str, Any]) -> None:
        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        run["costSummary"] = self._summarize_costs(run.get("strategies") or [], elapsed)

    def _summarize_costs(self, strategies: list[dict[str, Any]], run_elapsed_seconds: float) -> dict[str, Any]:
        costs = [strategy.get("cost") or empty_cost() for strategy in strategies]
        llm_calls = sum(int(cost.get("llmCalls") or 0) for cost in costs)
        llm_seconds = sum(finite_float(cost.get("llmClientWallClockSeconds")) for cost in costs)
        prompt_tokens = sum(int(cost.get("promptTokens") or 0) for cost in costs)
        completion_tokens = sum(int(cost.get("completionTokens") or 0) for cost in costs)
        native_embedding = sum(finite_float(cost.get("embeddingWallClockSeconds")) for cost in costs)
        common_embedding = sum(finite_float((strategy.get("commonMetricCost") or {}).get("embeddingWallClockSeconds")) for strategy in strategies)
        return summarize_cost({
            "runWallClockSeconds": run_elapsed_seconds,
            "runWallClockLabel": format_duration(run_elapsed_seconds),
            "strategyWallClockSecondsSum": sum(finite_float(cost.get("processWallClockSeconds")) for cost in costs),
            "llmCalls": llm_calls,
            "llmSuccessfulCalls": sum(int(cost.get("llmSuccessfulCalls") or 0) for cost in costs),
            "llmFailedCalls": sum(int(cost.get("llmFailedCalls") or 0) for cost in costs),
            "llmClientWallClockSeconds": llm_seconds,
            "llmAverageCallSeconds": llm_seconds / llm_calls if llm_calls else None,
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
            "embeddingWallClockSeconds": native_embedding,
            "commonEmbeddingWallClockSeconds": common_embedding,
            "completedStrategies": sum(1 for strategy in strategies if strategy.get("status") == STATUS_COMPLETED),
            "totalStrategiesWithResults": len(strategies),
        })

    def _mark_queued_as_cancelled_unlocked(self, run: dict[str, Any]) -> None:
        for state in (run.get("strategyStates") or {}).values():
            if state.get("status") == STATUS_QUEUED:
                state["status"] = STATUS_CANCELLED
                state["stageLabel"] = "Cancelada antes de iniciar"
                state["progress"] = 1.0
                state["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)

    def _append_log(self, run: dict[str, Any], strategy_id: str, message: str) -> None:
        with self._lock:
            self._append_log_unlocked(run, strategy_id, message)
            self._write_summary_unlocked(run)

    def _append_log_unlocked(self, run: dict[str, Any], strategy_id: str, message: str) -> None:
        clean = ANSI_RE.sub("", str(message)).strip()
        if not clean:
            return
        run["logs"].append({"at": utc_now(), "strategyId": strategy_id, "message": clean})
        run["logs"] = run["logs"][-300:]
        run["updatedAt"] = utc_now()

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()

    def _strategy_available(self, strategy: StrategyDefinition) -> bool:
        if strategy.repository_path is None:
            return True
        return (self.root / strategy.repository_path).exists()

    def _selected_definitions(self, config: dict[str, Any]) -> list[StrategyDefinition]:
        order = {strategy.strategy_id: strategy for strategy in STRATEGIES}
        return [order[strategy_id] for strategy_id in config["selectedStrategies"] if strategy_id in order]

    def _definition(self, strategy_id: str) -> StrategyDefinition:
        for strategy in STRATEGIES:
            if strategy.strategy_id == strategy_id:
                return strategy
        raise ValueError(f"Unknown strategy id: {strategy_id}")

    def _initial_strategy_state(self, strategy: StrategyDefinition) -> dict[str, Any]:
        return {
            "strategyId": strategy.strategy_id,
            "displayName": strategy.display_name,
            "status": STATUS_QUEUED,
            "stageLabel": "En cola",
            "progress": 0.0,
            "updatedAt": utc_now(),
        }

    def _int_between(self, value: Any, label: str, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be an integer.") from None
        if number < minimum or number > maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")
        return number

    def _float_between(self, value: Any, label: str, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be numeric.") from None
        if not math.isfinite(number) or number < minimum or number > maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")
        return number

    def _safe_text(self, value: Any, label: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{label} is required.")
        return text

    def _new_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    def _error_label(self, error: Any) -> str:
        if isinstance(error, dict):
            stage = f"{error.get('stage')}: " if error.get("stage") else ""
            return f"{stage}{error.get('message') or 'Strategy failed.'}"
        return str(error or "Strategy failed.")

    def _public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in run.items()
            if key not in {"activeProcess", "startedAtEpoch"}
        }

    def _write_summary_unlocked(self, run: dict[str, Any]) -> None:
        write_json(Path(run["runDir"]) / "summary.json", self._public_run(run))
