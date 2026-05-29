from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from initial_population.common_metrics import attach_common_metrics, finite_float, format_duration
from llm_studio import LmStudioClient
from turbulence_comparison.operators import (
    CandidateRecord,
    DistilBertOperator,
    DistilBertProvider,
    EmbeddingRanker,
    LinguisticAnalyzer,
    OperatorResult,
    PPDBIndex,
    ScoredCandidate,
    TurbulenceOperator,
    WordNetPPDBOperator,
    dedupe,
)


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

DEFAULT_LM_STUDIO = "http://127.0.0.1:1234"
DEFAULT_API_MODE = "native"
DEFAULT_MODEL = "Qwen3.5-2B"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DISTILBERT_MODEL = "distilbert/distilbert-base-uncased"
REFERENCE_TEXT = "streets are flooded and families are asking for shelter after heavy rain"
DEFAULT_PPDB_SOURCE_PATH = "data/external/ppdb/ppdb-2.0-s-all"
DEFAULT_PPDB_INDEX_PATH = "data/turbulence/ppdb_index.json"

GENERATION_SYSTEM_PROMPT = """You are a plain-text generator for natural-disaster scenario messages. You will receive one text-generation instruction from the user. Follow the instruction and generate exactly one final text message.

Output rules:
- Return plain text only.
- Return the dataset content itself, not a prompt, explanation, title, label, list, code, or metadata.
- Do not describe the task.
- Do not add unsolicited safety advice.
- Do not use quotation marks, hashtags, URLs, usernames, placeholders, tags, or special markers.
- Limit the message to between 1 and 4 sentences.
- Return only the final message."""

FINAL_PROMPT_TEMPLATE = (
    "Generate a short natural-disaster scenario message using the following semantic components: "
    "role = {role}; topic = {topic}; action = {action}. "
    "The generated message must follow the role, address the topic, and satisfy the action."
)

TURBULENCE_SYSTEM_PROMPT = """You are a prompt-component rewriting module.

Task:
Generate very close rewrites of one semantic prompt component.
The rewrite must preserve the same meaning and the same component type.

General rules:
- Return only rewritten component candidates.
- Each candidate must have 2-8 words.
- Do not copy the current component exactly.
- Do not change the main meaning.
- Keep most words from the current component in the same order.
- Replace only 1 word if possible, maximum 2 words.
- Do not use context to add new information.
- Candidates must be distinct from each other.
- Do not explain.
- Do not use quotes.
- Do not add titles, labels, comments, or extra text.
- Return exactly the number of candidates requested by the user.
- Use numbered lines with this format:
1) rewritten candidate
2) rewritten candidate
3) rewritten candidate
..."""

TURBULENCE_USER_PROMPT_TEMPLATE = """Number of candidates: {num_candidates}

Component type: {component_name}
Component definition: {component_definition}
Current component value: {current_component}

Context:
{other_components}"""

PSO_COMPONENTS = (
    {"key": "rol", "promptName": "role", "definition": "speaker identity or perspective"},
    {"key": "topico", "promptName": "topic", "definition": "subject or event focus"},
    {"key": "accion", "promptName": "action", "definition": "communicative intent or requested operation"},
)

STRATEGIES = {
    "llm": "LLM actual",
    "wordnet-ppdb-sbert": "WordNet+PPDB+SBERT",
    "distilbert-sbert": "DistilBERT+SBERT",
}


@dataclass(frozen=True)
class MovementPlan:
    id: str
    number: int
    individual_id: str
    component_key: str
    component_name: str
    component_definition: str
    random_value: float


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


def render_template(template: str, variables: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = variables.get(match.group(1), match.group(0))
        return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)

    return re.sub(r"\{([A-Za-z0-9_]+)\}", replace, template)


def clean_generated_text(raw: str) -> str:
    text = re.sub(r"```[A-Za-z]*\n?", "", raw)
    return text.replace("```", "").strip().strip("\"'`").strip()


def parse_candidates(raw_output: str, expected_count: int) -> list[str]:
    cleaned = raw_output.replace("```", "").strip()
    candidates: list[str] = []
    for line in cleaned.splitlines():
        value = line.strip()
        numbered = re.match(r"^\s*\d+\s*[\).\]:-]\s*(.+?)\s*$", value)
        if not numbered:
            continue
        value = numbered.group(1)
        value = re.sub(r"^candidate\s*\d*\s*[:.-]\s*", "", value, flags=re.IGNORECASE)
        value = value.strip().strip("\"'`<>").strip()
        if value:
            candidates.append(value)
    return dedupe(candidates)[:expected_count]


def unparsed_candidate_diagnostics(raw_output: str, expected_count: int) -> list[ScoredCandidate]:
    cleaned = raw_output.replace("```", "").strip()
    diagnostics: list[ScoredCandidate] = []
    for line in cleaned.splitlines():
        value = line.strip().strip("\"'`<>").strip()
        if not value or re.match(r"^\s*\d+\s*[\).\]:-]\s*", value):
            continue
        diagnostics.append(
            ScoredCandidate(
                candidate=value,
                source="llm_raw",
                similarity=0.0,
                diversity=0.0,
                reason="not_numbered_line",
                is_valid=False,
            )
        )
    return diagnostics[:expected_count]


class LlmTurbulenceOperator(TurbulenceOperator):
    strategy_id = "llm"
    display_name = "LLM actual"

    def __init__(self, ranker: EmbeddingRanker, client: LmStudioClient) -> None:
        self.ranker = ranker
        self.client = client

    def apply(self, current: str, config: dict[str, Any]) -> OperatorResult:
        started = time.perf_counter()
        lm_config = config["lmStudio"]
        variables = {
            "num_candidates": config["kCandidates"],
            "component_name": config["componentName"],
            "component_definition": config["componentDefinition"],
            "current_component": current,
            "other_components": config["otherComponents"],
            "reference_text": config["referenceText"],
        }
        system_prompt = render_template(
            config.get("llmTurbulenceSystemPrompt", ""),
            variables,
        )
        prompt = render_template(
            config["llmTurbulencePromptTemplate"],
            variables,
        )
        response = self.client.call(
            prompt,
            system_prompt=system_prompt,
            temperature=float(lm_config["temperature"]),
            top_p=float(lm_config["topP"]),
            max_tokens=int(lm_config["maxTokens"]),
        )
        raw_candidates = parse_candidates(response["text"], int(config["kCandidates"]))
        parse_diagnostics = unparsed_candidate_diagnostics(response["text"], int(config["kCandidates"]))
        records = [CandidateRecord(candidate=candidate, source="llm") for candidate in raw_candidates]
        scored, embedding_cost = self.score_candidates(current, records, config, self.ranker)
        selected = self.select_candidate(scored, int(config.get("selectionSeed", 0)))
        elapsed = time.perf_counter() - started
        cost = {
            "llmCalls": 1,
            "llmClientWallClockSeconds": response["elapsedSeconds"],
            "promptTokens": response["promptTokens"],
            "completionTokens": response["completionTokens"],
            "totalTokens": response["totalTokens"],
            "wordnetQueries": 0,
            "ppdbQueries": 0,
            "ppdbLookupAttempts": 0,
            "ppdbAvailable": False,
            "distilbertInferences": 0,
            **embedding_cost,
        }
        return OperatorResult(
            output=selected.candidate if selected else current,
            selected=selected,
            candidates=scored + parse_diagnostics,
            raw_candidate_count=len(raw_candidates),
            coverage=bool(raw_candidates),
            elapsed_seconds=elapsed,
            cost=cost,
            status="ok" if selected else "no_valid_candidate",
            diagnostics={
                "llm": {
                    "model": lm_config.get("model", getattr(self.client, "model", "")),
                    "apiMode": lm_config.get("apiMode", getattr(self.client, "api_mode", "")),
                    "temperature": float(lm_config["temperature"]),
                    "topP": float(lm_config["topP"]),
                    "maxTokens": int(lm_config["maxTokens"]),
                    "systemPrompt": system_prompt,
                    "userPrompt": prompt,
                },
            },
        )

    def select_candidate(self, scored: list[ScoredCandidate], seed: int = 0) -> ScoredCandidate | None:
        valid = [candidate for candidate in scored if candidate.is_valid]
        if not valid:
            return None
        return random.Random(seed).choice(valid)


class TurbulenceComparisonService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_root = root / "runs" / "turbulence-comparison"
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def default_config(self) -> dict[str, Any]:
        return {
            "strategies": ["llm", "wordnet-ppdb-sbert", "distilbert-sbert"],
            "individualCount": 5,
            "seed": 42,
            "repetitionsK": 1,
            "kCandidates": 5,
            "turbulenceMinSimilarity": 0.65,
            "turbulenceMaxSimilarity": 0.9,
            "minWords": 2,
            "maxWords": 8,
            "embeddingModel": DEFAULT_EMBEDDING_MODEL,
            "distilbertModel": DEFAULT_DISTILBERT_MODEL,
            "referenceText": REFERENCE_TEXT,
            "finalFidelityMin": 0.25,
            "finalFidelityMax": 0.99,
            "operatorParallelism": 1,
            "generationParallelism": 1,
            "usePpdb": True,
            "ppdbSourcePath": DEFAULT_PPDB_SOURCE_PATH,
            "ppdbIndexPath": DEFAULT_PPDB_INDEX_PATH,
            "lmStudio": {
                "baseUrl": DEFAULT_LM_STUDIO,
                "apiMode": DEFAULT_API_MODE,
                "model": DEFAULT_MODEL,
                "temperature": 0.35,
                "topP": 0.95,
                "maxTokens": 220,
                "timeoutSeconds": 120,
            },
            "llmTurbulenceSystemPrompt": TURBULENCE_SYSTEM_PROMPT,
            "llmTurbulencePromptTemplate": TURBULENCE_USER_PROMPT_TEMPLATE,
            "generationSystemPrompt": GENERATION_SYSTEM_PROMPT,
            "finalPromptTemplate": FINAL_PROMPT_TEMPLATE,
        }

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._read_config(payload)
        run_id = self._new_run_id()
        run_dir = self.runs_root / run_id
        run = {
            "runId": run_id,
            "status": STATUS_QUEUED,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "runDir": str(run_dir),
            "config": config,
            "logs": [],
            "progress": {
                "stage": "queued",
                "message": "Corrida en cola.",
                "percent": 0,
                "completed": 0,
                "total": 1,
            },
            "result": None,
            "error": None,
        }
        write_json(run_dir / "config.json", config)
        with self._lock:
            self._runs[run_id] = run
        thread = threading.Thread(target=self._run_worker, args=(run_id,), daemon=True)
        thread.start()
        return self._public_run(run)

    def ppdb_status(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._ppdb_status(payload or {})

    def prepare_ppdb(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_path = self._ppdb_source_path(payload)
        index_path = self._ppdb_index_path(payload)
        if not source_path.exists():
            raise ValueError(f"PPDB completo no encontrado en {source_path}. Descarga el dataset y deja el archivo en esa ruta.")
        from scripts.prepare_ppdb_index import prepare_ppdb_index

        summary = prepare_ppdb_index(
            self.root,
            source_path,
            index_path,
            int(clamp_number(payload.get("limitPerKey", 30), 1, 500)),
        )
        return {
            "prepared": True,
            "summary": summary,
            "ppdb": self._ppdb_status({"ppdbSourcePath": str(source_path), "ppdbIndexPath": str(index_path)}),
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
        if run:
            return self._public_run(run)
        summary_path = self.runs_root / run_id / "summary.json"
        if not summary_path.exists():
            return None
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return self.get_run(run_id)
            if run["status"] in {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}:
                return self._public_run(run)
            run["status"] = STATUS_CANCELLED
            run["updatedAt"] = utc_now()
            self._log(run, "Cancelacion solicitada.")
            return self._public_run(run)

    def _run_worker(self, run_id: str) -> None:
        run = self._require_run(run_id)
        started = time.perf_counter()
        try:
            self._set_status(run, STATUS_RUNNING, "preparing", "Preparando movimientos compartidos.", 2)
            result = self._execute(run)
            if self._is_cancelled(run):
                self._set_status(run, STATUS_CANCELLED, "cancelled", "Corrida cancelada.", 100)
            else:
                result["runWallClockSeconds"] = time.perf_counter() - started
                result["runWallClockLabel"] = format_duration(result["runWallClockSeconds"])
                with self._lock:
                    run["result"] = result
                completion_message = (
                    "Comparacion completada con advertencias; revisa logs y filas con error."
                    if result.get("warnings")
                    else "Comparacion completada."
                )
                self._set_status(run, STATUS_COMPLETED, "completed", completion_message, 100)
        except Exception as error:
            with self._lock:
                run["error"] = str(error)
            self._set_status(run, STATUS_FAILED, "failed", str(error), 100)
        finally:
            public = self._public_run(run)
            write_json(Path(run["runDir"]) / "summary.json", public)

    def _execute(self, run: dict[str, Any]) -> dict[str, Any]:
        config = run["config"]
        repetitions_k = int(config.get("repetitionsK") or 1)
        if repetitions_k <= 1:
            return self._execute_single(run, config, 1, config["seed"])

        repetitions: list[dict[str, Any]] = []
        for repetition_index in range(repetitions_k):
            if self._is_cancelled(run):
                break
            repetition_seed = (int(config["seed"]) + repetition_index) % 2_147_483_648
            repetition_config = {**config, "seed": repetition_seed}
            self._set_progress(
                run,
                "repetition",
                f"Ejecutando repeticion {repetition_index + 1}/{repetitions_k} con semilla {repetition_seed}.",
                100 * repetition_index / repetitions_k,
                repetition_index,
                repetitions_k,
            )
            repetition = self._execute_single(run, repetition_config, repetition_index + 1, repetition_seed)
            repetitions.append(repetition)
            write_json(Path(run["runDir"]) / f"repetition-{repetition_index + 1:03d}.json", repetition)

        result = aggregate_turbulence_repetitions(repetitions, config)
        write_json(Path(run["runDir"]) / "result.json", result)
        return result

    def _execute_single(
        self,
        run: dict[str, Any],
        config: dict[str, Any],
        repetition_index: int,
        repetition_seed: int,
    ) -> dict[str, Any]:
        individuals = self._load_individuals(config["individualCount"])
        movements = create_movement_plan(individuals, config["seed"])
        self._log(
            run,
            f"Repeticion {repetition_index}: {len(movements)} movimiento(s), semilla {repetition_seed}.",
        )
        self._set_progress(run, "baseline_generation", "Generando baseline sin turbulencia.", 8, 0, max(1, len(movements)))

        generation_client = self._lm_client(config)
        baseline_positions = [
            {"id": individual["id"], "rol": individual["rol"], "topico": individual["topico"], "accion": individual["accion"]}
            for individual in individuals
        ]
        baseline_rows, baseline_generation_cost = self._generate_final_texts(
            baseline_positions,
            config,
            generation_client,
            run,
            label="baseline",
        )
        baseline_rows, baseline_metrics, baseline_embedding_cost = attach_common_metrics(
            baseline_rows,
            config["referenceText"],
            config["embeddingModel"],
        )
        baseline_summary = summarize_final_metrics(baseline_rows, baseline_metrics)
        baseline_summary["generationCost"] = baseline_generation_cost
        baseline_summary["embeddingCost"] = baseline_embedding_cost
        if baseline_summary["completedRows"] == 0:
            self._log(
                run,
                "Baseline sin textos finales válidos; las métricas finales quedan no disponibles hasta corregir la generación LM Studio.",
            )

        strategy_results = []
        for index, strategy_id in enumerate(config["strategies"], start=1):
            if self._is_cancelled(run):
                break
            self._set_progress(
                run,
                strategy_id,
                f"Ejecutando estrategia {STRATEGIES[strategy_id]}.",
                12 + (70 * (index - 1) / max(1, len(config["strategies"]))),
                index - 1,
                len(config["strategies"]),
            )
            strategy_result = self._execute_strategy(
                strategy_id,
                individuals,
                movements,
                config,
                generation_client,
                run,
            )
            strategy_result["deltas"] = final_metric_deltas(baseline_summary, strategy_result["finalMetrics"])
            strategy_results.append(strategy_result)
            write_json(Path(run["runDir"]) / f"{strategy_id}.json", strategy_result)

        recommendation = recommend_strategy(strategy_results, config)
        self._set_progress(run, "summarizing", "Calculando recomendación.", 94, len(strategy_results), len(config["strategies"]))
        warnings = comparison_warnings(baseline_summary, strategy_results, recommendation)
        result = {
            "movementCount": len(movements),
            "individualCount": len(individuals),
            "ppdb": self._ppdb_status(config),
            "baseline": baseline_summary,
            "strategies": strategy_results,
            "recommendation": recommendation,
            "warnings": warnings,
            "configSummary": {
                "seed": config["seed"],
                "repetitionIndex": repetition_index,
                "movementPolicy": "one_uniform_component_per_individual",
                "kCandidates": config["kCandidates"],
                "operatorParallelism": config["operatorParallelism"],
                "generationParallelism": config["generationParallelism"],
                "turbulenceRange": [config["turbulenceMinSimilarity"], config["turbulenceMaxSimilarity"]],
                "embeddingModel": config["embeddingModel"],
                "distilbertModel": config["distilbertModel"],
                "usePpdb": config["usePpdb"],
                "ppdbSourcePath": config["ppdbSourcePath"],
                "ppdbIndexPath": config["ppdbIndexPath"],
            },
        }
        write_json(Path(run["runDir"]) / "result.json", result)
        return result

    def _execute_strategy(
        self,
        strategy_id: str,
        individuals: list[dict[str, Any]],
        movements: list[MovementPlan],
        config: dict[str, Any],
        generation_client: LmStudioClient,
        run: dict[str, Any],
    ) -> dict[str, Any]:
        ranker = EmbeddingRanker(config["embeddingModel"])
        operator = self._operator(strategy_id, ranker, generation_client, config)
        positions = {
            individual["id"]: {"rol": individual["rol"], "topico": individual["topico"], "accion": individual["accion"]}
            for individual in individuals
        }
        rows: list[dict[str, Any]] = []
        costs: list[dict[str, Any]] = []
        operator_parallelism = min(max(1, int(config["operatorParallelism"])), max(1, len(movements)))

        def apply_movement(movement: MovementPlan) -> dict[str, Any]:
            if self._is_cancelled(run):
                return {"movement": movement, "row": None, "cost": None, "selected": None}
            position = positions[movement.individual_id]
            current = position[movement.component_key]
            other_components = pso_other_components(position, movement.component_key)
            operator_config = {
                **config,
                "componentName": movement.component_name,
                "componentDefinition": movement.component_definition,
                "otherComponents": other_components,
                "selectionSeed": movement_selection_seed(config["seed"], strategy_id, movement.number),
                "unitSelectionSeed": unit_selection_seed(config["seed"], movement.number),
            }
            try:
                result = operator.apply(current, operator_config)
            except Exception as error:
                row = movement_error_row(movement, current, other_components, str(error))
                return {"movement": movement, "row": row, "cost": None, "selected": None, "error": error}
            row = movement_result_row(movement, current, other_components, result)
            cost = result.cost | {"operatorApplicationSeconds": result.elapsed_seconds}
            selected = result.selected.candidate if result.selected else None
            return {"movement": movement, "row": row, "cost": cost, "selected": selected}

        operator_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=operator_parallelism, thread_name_prefix=f"turbulence-{strategy_id}") as executor:
            futures = {executor.submit(apply_movement, movement): movement for movement in movements}
            for future in as_completed(futures):
                payload = future.result()
                row = payload.get("row")
                if row is None:
                    continue
                movement = payload["movement"]
                rows.append(row)
                if payload.get("cost"):
                    costs.append(payload["cost"])
                if payload.get("selected"):
                    positions[movement.individual_id][movement.component_key] = payload["selected"]
                if payload.get("error"):
                    self._log(
                        run,
                        f"{STRATEGIES[strategy_id]} movimiento {movement.number} fallo: {short_error(payload['error'])}",
                    )
        operator_wall_clock_seconds = time.perf_counter() - operator_started
        rows.sort(key=lambda row: int(row.get("movementNumber") or 0))

        final_positions = [{"id": individual_id, **position} for individual_id, position in positions.items()]
        final_rows, generation_cost = self._generate_final_texts(final_positions, config, generation_client, run, label=strategy_id)
        final_rows, final_metrics, final_embedding_cost = attach_common_metrics(final_rows, config["referenceText"], config["embeddingModel"])
        final_summary = summarize_final_metrics(final_rows, final_metrics)
        if final_summary["completedRows"] == 0:
            self._log(
                run,
                f"{STRATEGIES[strategy_id]} sin textos finales válidos; fidelidad y diversidad no disponibles.",
            )
        operator_summary = summarize_operator_rows(rows, costs, operator_wall_clock_seconds, operator_parallelism)
        strategy_result = {
            "strategyId": strategy_id,
            "displayName": STRATEGIES[strategy_id],
            "status": "completed",
            "operatorMetrics": operator_summary,
            "finalMetrics": final_summary,
            "generationCost": generation_cost,
            "finalEmbeddingCost": final_embedding_cost,
            "movementRows": rows[:300],
            "finalRows": final_rows,
        }
        strategy_result["warnings"] = strategy_warnings(rows, final_rows)
        strategy_result["relativeOperatorCost"] = relative_operator_cost(operator_summary)
        return strategy_result

    def _generate_final_texts(
        self,
        positions: list[dict[str, Any]],
        config: dict[str, Any],
        client: LmStudioClient,
        run: dict[str, Any],
        label: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(positions)
        calls: list[dict[str, Any]] = []
        parallelism = min(max(1, int(config["generationParallelism"])), max(1, len(positions)))

        def worker(index: int, position: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
            prompt = render_final_prompt(position, config)
            response = client.call(
                prompt,
                system_prompt=config["generationSystemPrompt"],
                temperature=float(config["lmStudio"]["temperature"]),
                top_p=float(config["lmStudio"]["topP"]),
                max_tokens=int(config["lmStudio"]["maxTokens"]),
            )
            row = {
                "id": f"{label}-{position['id']}",
                "individualId": position["id"],
                "role": position["rol"],
                "topic": position["topico"],
                "action": position["accion"],
                "prompt": prompt,
                "generatedText": clean_generated_text(response["text"]),
                "status": "ok",
            }
            return index, row, response

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="turbulence-generation") as executor:
            futures = {executor.submit(worker, index, position): index for index, position in enumerate(positions)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    _, row, call = future.result()
                    results[index] = row
                    calls.append(call)
                except Exception as error:
                    position = positions[index]
                    results[index] = {
                        "id": f"{label}-{position['id']}",
                        "individualId": position["id"],
                        "role": position["rol"],
                        "topic": position["topico"],
                        "action": position["accion"],
                        "prompt": render_final_prompt(position, config),
                        "generatedText": "",
                        "status": "generation_error",
                        "error": str(error),
                    }
        elapsed = time.perf_counter() - started
        failed_rows = [row for row in results if row and row.get("status") == "generation_error"]
        if failed_rows:
            self._log(
                run,
                f"Generacion final {label}: {len(failed_rows)}/{len(positions)} texto(s) fallaron. Primer error: {short_error(failed_rows[0].get('error'))}",
            )
        cost = {
            "llmCalls": len(calls),
            "llmClientWallClockSeconds": sum(float(call.get("elapsedSeconds") or 0.0) for call in calls),
            "wallClockSeconds": elapsed,
            "promptTokens": sum(int(call.get("promptTokens") or 0) for call in calls),
            "completionTokens": sum(int(call.get("completionTokens") or 0) for call in calls),
            "totalTokens": sum(int(call.get("totalTokens") or 0) for call in calls),
            "wallClockLabel": format_duration(elapsed),
        }
        return [row for row in results if row is not None], cost

    def _operator(
        self,
        strategy_id: str,
        ranker: EmbeddingRanker,
        client: LmStudioClient,
        config: dict[str, Any],
    ) -> TurbulenceOperator:
        if strategy_id == "llm":
            return LlmTurbulenceOperator(ranker, client)
        analyzer = LinguisticAnalyzer()
        if strategy_id == "wordnet-ppdb-sbert":
            ppdb = PPDBIndex(self._ppdb_index_path(config)) if config.get("usePpdb", True) else None
            return WordNetPPDBOperator(analyzer, ranker, ppdb)
        if strategy_id == "distilbert-sbert":
            return DistilBertOperator(analyzer, ranker, DistilBertProvider(config["distilbertModel"]))
        raise ValueError(f"Estrategia no soportada: {strategy_id}")

    def _read_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.default_config()
        strategies = payload.get("strategies", defaults["strategies"])
        if not isinstance(strategies, list):
            raise ValueError("strategies debe ser una lista.")
        selected = [str(item) for item in strategies if str(item) in STRATEGIES]
        if not selected:
            raise ValueError("Selecciona al menos una estrategia valida.")
        min_similarity = clamp_number(payload.get("turbulenceMinSimilarity", defaults["turbulenceMinSimilarity"]), -1, 1)
        max_similarity = clamp_number(payload.get("turbulenceMaxSimilarity", defaults["turbulenceMaxSimilarity"]), -1, 1)
        if min_similarity > max_similarity:
            raise ValueError("La similitud minima de turbulencia no puede ser mayor que la maxima.")
        min_words = int(clamp_number(payload.get("minWords", defaults["minWords"]), 1, 40))
        max_words = int(clamp_number(payload.get("maxWords", defaults["maxWords"]), 1, 80))
        if min_words > max_words:
            raise ValueError("El minimo de palabras no puede ser mayor que el maximo.")
        final_min = clamp_number(payload.get("finalFidelityMin", defaults["finalFidelityMin"]), -1, 1)
        final_max = clamp_number(payload.get("finalFidelityMax", defaults["finalFidelityMax"]), -1, 1)
        if final_min > final_max:
            raise ValueError("La fidelidad final minima no puede ser mayor que la maxima.")
        lm_payload = payload.get("lmStudio") if isinstance(payload.get("lmStudio"), dict) else {}
        api_mode = str(lm_payload.get("apiMode", defaults["lmStudio"]["apiMode"]))
        if api_mode not in {"native", "openai"}:
            raise ValueError("apiMode debe ser native u openai.")
        llm_turbulence_system_prompt = str(
            payload.get("llmTurbulenceSystemPrompt", defaults["llmTurbulenceSystemPrompt"]) or ""
        ).strip()
        if not llm_turbulence_system_prompt.strip():
            llm_turbulence_system_prompt = defaults["llmTurbulenceSystemPrompt"]
        return {
            "strategies": selected,
            "individualCount": int(clamp_number(payload.get("individualCount", defaults["individualCount"]), 1, 200)),
            "seed": int(clamp_number(payload.get("seed", defaults["seed"]), 0, 2_147_483_647)),
            "repetitionsK": int(clamp_number(payload.get("repetitionsK", defaults["repetitionsK"]), 1, 30)),
            "kCandidates": int(clamp_number(payload.get("kCandidates", defaults["kCandidates"]), 1, 30)),
            "turbulenceMinSimilarity": min_similarity,
            "turbulenceMaxSimilarity": max_similarity,
            "minWords": min_words,
            "maxWords": max_words,
            "embeddingModel": safe_text(payload.get("embeddingModel", defaults["embeddingModel"]), "embeddingModel"),
            "distilbertModel": safe_text(payload.get("distilbertModel", defaults["distilbertModel"]), "distilbertModel"),
            "referenceText": safe_text(payload.get("referenceText", defaults["referenceText"]), "referenceText"),
            "finalFidelityMin": final_min,
            "finalFidelityMax": final_max,
            "operatorParallelism": int(clamp_number(payload.get("operatorParallelism", defaults["operatorParallelism"]), 1, 8)),
            "generationParallelism": int(clamp_number(payload.get("generationParallelism", defaults["generationParallelism"]), 1, 8)),
            "usePpdb": read_bool(payload.get("usePpdb", defaults["usePpdb"])),
            "ppdbSourcePath": safe_path_text(payload.get("ppdbSourcePath") or defaults["ppdbSourcePath"], "ppdbSourcePath"),
            "ppdbIndexPath": safe_path_text(payload.get("ppdbIndexPath") or defaults["ppdbIndexPath"], "ppdbIndexPath"),
            "lmStudio": {
                "baseUrl": safe_text(lm_payload.get("baseUrl", defaults["lmStudio"]["baseUrl"]), "baseUrl").rstrip("/"),
                "apiMode": api_mode,
                "model": safe_text(lm_payload.get("model", defaults["lmStudio"]["model"]), "model"),
                "temperature": clamp_number(lm_payload.get("temperature", defaults["lmStudio"]["temperature"]), 0, 2),
                "topP": clamp_number(lm_payload.get("topP", defaults["lmStudio"]["topP"]), 0, 1),
                "maxTokens": int(clamp_number(lm_payload.get("maxTokens", defaults["lmStudio"]["maxTokens"]), 8, 4096)),
                "timeoutSeconds": clamp_number(lm_payload.get("timeoutSeconds", defaults["lmStudio"]["timeoutSeconds"]), 5, 1200),
            },
            "llmTurbulencePromptTemplate": safe_text(
                payload.get("llmTurbulencePromptTemplate", defaults["llmTurbulencePromptTemplate"]),
                "llmTurbulencePromptTemplate",
            ),
            "llmTurbulenceSystemPrompt": llm_turbulence_system_prompt,
            "generationSystemPrompt": safe_text(payload.get("generationSystemPrompt", defaults["generationSystemPrompt"]), "generationSystemPrompt"),
            "finalPromptTemplate": safe_text(payload.get("finalPromptTemplate", defaults["finalPromptTemplate"]), "finalPromptTemplate"),
        }

    def _lm_client(self, config: dict[str, Any]) -> LmStudioClient:
        lm = config["lmStudio"]
        return LmStudioClient(lm["baseUrl"], lm["apiMode"], lm["model"], float(lm["timeoutSeconds"]))

    def _load_individuals(self, count: int) -> list[dict[str, Any]]:
        path = self.root / "LLM" / "data" / "pso-individuals.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        individuals = payload.get("individuals")
        if not isinstance(individuals, list):
            raise ValueError("LLM/data/pso-individuals.json no contiene individuals.")
        return individuals[:count]

    def _resolve_local_path(self, value: str | Path) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def _ppdb_source_path(self, payload: dict[str, Any] | None = None) -> Path:
        payload = payload or {}
        explicit = payload.get("ppdbSourcePath")
        source_text = safe_path_text(explicit or DEFAULT_PPDB_SOURCE_PATH, "ppdbSourcePath")
        source_path = self._resolve_local_path(source_text)
        if explicit is None and not source_path.exists() and source_path.suffix != ".gz":
            gz_path = Path(f"{source_path}.gz")
            if gz_path.exists():
                return gz_path
        return source_path

    def _ppdb_index_path(self, payload: dict[str, Any] | None = None) -> Path:
        payload = payload or {}
        return self._resolve_local_path(safe_path_text(payload.get("ppdbIndexPath") or DEFAULT_PPDB_INDEX_PATH, "ppdbIndexPath"))

    def _ppdb_status(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        if not read_bool(payload.get("usePpdb", True)):
            return {
                "enabled": False,
                "available": False,
                "path": str(self._ppdb_index_path(payload)),
                "message": "PPDB desactivado; la estrategia WordNet+PPDB+SBERT usará solo WordNet.",
                "source": {},
                "index": {},
            }
        source_path = self._ppdb_source_path(payload)
        index_path = self._ppdb_index_path(payload)
        source = {
            "path": str(source_path),
            "exists": source_path.exists(),
            "sizeBytes": source_path.stat().st_size if source_path.exists() else None,
            "sizeLabel": format_bytes(source_path.stat().st_size) if source_path.exists() else "No disponible",
            "message": "PPDB completo encontrado." if source_path.exists() else "PPDB completo no encontrado.",
        }
        if not index_path.exists():
            return {
                "enabled": True,
                "available": False,
                "path": str(index_path),
                "message": "Índice compacto PPDB no preparado.",
                "source": source,
                "index": {
                    "path": str(index_path),
                    "exists": False,
                    "available": False,
                    "message": "Prepara el índice desde el PPDB completo local.",
                },
            }
        try:
            index = PPDBIndex(index_path)
        except Exception as error:
            return {
                "enabled": True,
                "available": False,
                "path": str(index_path),
                "message": f"Índice PPDB inválido: {error}",
                "source": source,
                "index": {
                    "path": str(index_path),
                    "exists": True,
                    "available": False,
                    "message": str(error),
                },
            }
        return {
            "enabled": True,
            "available": index.available,
            "path": str(index_path),
            "entries": len(index.entries),
            "message": f"Índice compacto PPDB disponible con {len(index.entries)} clave(s).",
            "source": source,
            "index": {
                "path": str(index_path),
                "exists": True,
                "available": index.available,
                "entries": len(index.entries),
                "message": "Índice compacto cargado correctamente.",
            },
        }

    def _require_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
        if not run:
            raise KeyError(run_id)
        return run

    def _new_run_id(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

    def _set_status(self, run: dict[str, Any], status: str, stage: str, message: str, percent: float) -> None:
        with self._lock:
            run["status"] = status
            run["updatedAt"] = utc_now()
            run["progress"] = {**run.get("progress", {}), "stage": stage, "message": message, "percent": percent}
            run["logs"].append({"time": utc_now(), "message": message})

    def _set_progress(self, run: dict[str, Any], stage: str, message: str, percent: float, completed: int, total: int) -> None:
        with self._lock:
            run["updatedAt"] = utc_now()
            run["progress"] = {
                "stage": stage,
                "message": message,
                "percent": percent,
                "completed": completed,
                "total": total,
            }
            run["logs"].append({"time": utc_now(), "message": message})

    def _log(self, run: dict[str, Any], message: str) -> None:
        with self._lock:
            run["logs"].append({"time": utc_now(), "message": message})
            run["updatedAt"] = utc_now()

    def _is_cancelled(self, run: dict[str, Any]) -> bool:
        with self._lock:
            return run.get("status") == STATUS_CANCELLED

    def _public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        payload = dict(run)
        payload["logs"] = list(run.get("logs", []))[-80:]
        return payload


def create_movement_plan(individuals: list[dict[str, Any]], seed: int) -> list[MovementPlan]:
    rng = random.Random(seed)
    movements: list[MovementPlan] = []
    for individual in individuals:
        random_value = rng.random()
        component_index = min(int(random_value * len(PSO_COMPONENTS)), len(PSO_COMPONENTS) - 1)
        component = PSO_COMPONENTS[component_index]
        number = len(movements) + 1
        movements.append(
            MovementPlan(
                id=f"mov-{number:04d}",
                number=number,
                individual_id=str(individual["id"]),
                component_key=component["key"],
                component_name=component["promptName"],
                component_definition=component["definition"],
                random_value=random_value,
            )
        )
    return movements


def movement_selection_seed(seed: int, strategy_id: str, movement_number: int) -> int:
    strategy_offset = sum((index + 1) * ord(char) for index, char in enumerate(strategy_id))
    return (int(seed) * 1_000_003 + strategy_offset * 9_176 + int(movement_number)) & 0xFFFFFFFF


def unit_selection_seed(seed: int, movement_number: int) -> int:
    return (int(seed) * 1_000_003 + int(movement_number) * 97_531) & 0xFFFFFFFF


def pso_other_components(position: dict[str, str], component_key: str) -> str:
    return "; ".join(
        f"{component['promptName']} = {position[component['key']]}"
        for component in PSO_COMPONENTS
        if component["key"] != component_key
    )


def render_final_prompt(position: dict[str, Any], config: dict[str, Any]) -> str:
    return render_template(
        config["finalPromptTemplate"],
        {
            "role": position["rol"],
            "topic": position["topico"],
            "action": position["accion"],
            "reference_text": config["referenceText"],
        },
    )


def short_error(error: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(error or "")).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def movement_result_row(movement: MovementPlan, current: str, other_components: str, result: OperatorResult) -> dict[str, Any]:
    selected = result.selected
    return {
        "movementNumber": movement.number,
        "individualId": movement.individual_id,
        "componentName": movement.component_name,
        "current": current,
        "otherComponents": other_components,
        "selectedCandidate": selected.candidate if selected else "",
        "selectedSource": selected.source if selected else "",
        "similarity": selected.similarity if selected else None,
        "diversity": selected.diversity if selected else None,
        "status": result.status,
        "success": bool(selected),
        "coverage": result.coverage,
        "rawCandidateCount": result.raw_candidate_count,
        "elapsedSeconds": result.elapsed_seconds,
        "candidates": [candidate_to_dict(candidate) for candidate in result.candidates[:20]],
        "diagnostics": result.diagnostics or {},
    }


def movement_error_row(movement: MovementPlan, current: str, other_components: str, error: str) -> dict[str, Any]:
    return {
        "movementNumber": movement.number,
        "individualId": movement.individual_id,
        "componentName": movement.component_name,
        "current": current,
        "otherComponents": other_components,
        "selectedCandidate": "",
        "selectedSource": "",
        "similarity": None,
        "diversity": None,
        "status": "error",
        "success": False,
        "coverage": False,
        "rawCandidateCount": 0,
        "elapsedSeconds": 0,
        "error": error,
        "candidates": [],
    }


def candidate_to_dict(candidate: ScoredCandidate) -> dict[str, Any]:
    return {
        "candidate": candidate.candidate,
        "source": candidate.source,
        "similarity": candidate.similarity,
        "diversity": candidate.diversity,
        "reason": candidate.reason,
        "valid": candidate.is_valid,
    }


def strategy_warnings(movement_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    operator_errors = [row for row in movement_rows if row.get("status") == "error"]
    final_errors = [row for row in final_rows if row.get("status") == "generation_error"]
    if operator_errors:
        warnings.append(
            f"{len(operator_errors)} movimiento(s) fallaron en el operador; primer error: {short_error(operator_errors[0].get('error'))}"
        )
    if final_errors:
        warnings.append(
            f"{len(final_errors)} texto(s) finales fallaron en generación LM Studio; primer error: {short_error(final_errors[0].get('error'))}"
        )
    return warnings


def comparison_warnings(
    baseline_summary: dict[str, Any],
    strategy_results: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if baseline_summary.get("completedRows") == 0:
        warnings.append("Baseline sin textos finales válidos; no se pueden calcular deltas finales.")
    for result in strategy_results:
        warnings.extend(str(message) for message in result.get("warnings") or [])
    if recommendation.get("strategyId") is None:
        warnings.append(str(recommendation.get("message") or "Sin recomendación disponible."))
    return warnings


def weighted_average(items: list[dict[str, Any]], value_key: str, weight_key: str = "completedRows") -> float | None:
    weighted = [
        (finite_float(item.get(value_key)), finite_float(item.get(weight_key)))
        for item in items
        if item.get(value_key) is not None and finite_float(item.get(weight_key)) > 0
    ]
    if not weighted:
        values = [finite_float(item.get(value_key)) for item in items if item.get(value_key) is not None]
        return sum(values) / len(values) if values else None
    total_weight = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / total_weight if total_weight else None


def aggregate_final_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    completed_rows = sum(int(finite_float(summary.get("completedRows"))) for summary in summaries)
    all_rows = [
        row
        for summary in summaries
        for row in (summary.get("rows") or [])
        if isinstance(row, dict)
    ]
    hypervolume = weighted_average(summaries, "hypervolume")
    spread = weighted_average(summaries, "spread")
    return {
        "completedRows": completed_rows,
        "averageFidelity": weighted_average(summaries, "averageFidelity"),
        "averageDiversity": weighted_average(summaries, "averageDiversity"),
        "bestFidelity": max((finite_float(summary.get("bestFidelity"), -2.0) for summary in summaries if summary.get("bestFidelity") is not None), default=None),
        "bestDiversity": max((finite_float(summary.get("bestDiversity"), -2.0) for summary in summaries if summary.get("bestDiversity") is not None), default=None),
        "hypervolume": hypervolume,
        "spread": spread,
        "rows": all_rows,
        "repetitions": len(summaries),
    }


def aggregate_generation_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "llmCalls": sum(int(finite_float(cost.get("llmCalls"))) for cost in costs),
        "llmClientWallClockSeconds": sum(finite_float(cost.get("llmClientWallClockSeconds")) for cost in costs),
        "wallClockSeconds": sum(finite_float(cost.get("wallClockSeconds")) for cost in costs),
        "promptTokens": sum(int(finite_float(cost.get("promptTokens"))) for cost in costs),
        "completionTokens": sum(int(finite_float(cost.get("completionTokens"))) for cost in costs),
        "totalTokens": sum(int(finite_float(cost.get("totalTokens"))) for cost in costs),
    }
    result["wallClockLabel"] = format_duration(result["wallClockSeconds"])
    return result


def aggregate_embedding_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = sum(finite_float(cost.get("embeddingWallClockSeconds")) for cost in costs)
    texts = sum(int(finite_float(cost.get("embeddingTexts"))) for cost in costs)
    return {
        "embeddingModel": next((cost.get("embeddingModel") for cost in costs if cost.get("embeddingModel")), None),
        "embeddingTexts": texts,
        "embeddingWallClockSeconds": seconds,
        "embeddingWallClockLabel": format_duration(seconds),
    }


def aggregate_operator_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    moves = sum(int(finite_float(metrics.get("moves"))) for metrics in metrics_list)
    successes = sum(int(finite_float(metrics.get("successes"))) for metrics in metrics_list)
    coverage = sum(int(finite_float(metrics.get("coverageCount"))) for metrics in metrics_list)
    cumulative = sum(finite_float(metrics.get("operatorCumulativeSeconds")) for metrics in metrics_list)
    wall = sum(finite_float(metrics.get("operatorWallClockSeconds")) for metrics in metrics_list)
    cost = aggregate_costs([metrics.get("cost") or {} for metrics in metrics_list])
    average = cumulative / moves if moves else None
    similarity_count = sum(int(finite_float(metrics.get("similarityCount"))) for metrics in metrics_list)
    similarity_sum = sum(finite_float(metrics.get("similaritySum")) for metrics in metrics_list)
    if similarity_count == 0:
        similarity_values = [
            finite_float(metrics.get("averageSimilarity"))
            for metrics in metrics_list
            if metrics.get("averageSimilarity") is not None
        ]
        average_similarity = sum(similarity_values) / len(similarity_values) if similarity_values else None
        similarity_sum = sum(similarity_values)
        similarity_count = len(similarity_values)
    else:
        average_similarity = similarity_sum / similarity_count
    return {
        "moves": moves,
        "successes": successes,
        "successRate": successes / moves if moves else 0.0,
        "coverageCount": coverage,
        "coverageRate": coverage / moves if moves else 0.0,
        "similarityCount": similarity_count,
        "similaritySum": similarity_sum,
        "averageSimilarity": average_similarity,
        "operatorParallelism": max((int(finite_float(metrics.get("operatorParallelism"))) for metrics in metrics_list), default=1),
        "operatorCumulativeSeconds": cumulative,
        "operatorCumulativeLabel": format_duration(cumulative),
        "operatorWallClockSeconds": wall,
        "operatorWallClockLabel": format_duration(wall),
        "operatorAverageSeconds": average,
        "operatorAverageLabel": format_duration(average),
        "cost": cost,
    }


def aggregate_turbulence_repetitions(repetitions: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    completed = [repetition for repetition in repetitions if repetition.get("strategies")]
    if not completed:
        return {
            "movementCount": 0,
            "individualCount": config["individualCount"],
            "ppdb": {},
            "baseline": {"completedRows": 0, "averageFidelity": None, "averageDiversity": None, "rows": []},
            "strategies": [],
            "recommendation": {"strategyId": None, "message": "Sin repeticiones completadas."},
            "warnings": ["Sin repeticiones completadas."],
            "repetitionsK": config.get("repetitionsK", 1),
            "completedRepetitions": 0,
            "repetitions": repetitions,
        }

    baseline = aggregate_final_summaries([repetition.get("baseline") or {} for repetition in completed])
    baseline["generationCost"] = aggregate_generation_costs([
        (repetition.get("baseline") or {}).get("generationCost") or {}
        for repetition in completed
    ])
    baseline["embeddingCost"] = aggregate_embedding_costs([
        (repetition.get("baseline") or {}).get("embeddingCost") or {}
        for repetition in completed
    ])

    strategy_results: list[dict[str, Any]] = []
    for strategy_id in config["strategies"]:
        per_repetition = []
        for repetition in completed:
            repetition_meta = repetition.get("configSummary") or {}
            for strategy in repetition.get("strategies", []):
                if strategy.get("strategyId") == strategy_id:
                    per_repetition.append({
                        **strategy,
                        "repetitionIndex": repetition_meta.get("repetitionIndex"),
                        "repetitionSeed": repetition_meta.get("seed"),
                    })
        if not per_repetition:
            continue
        movement_rows = []
        final_rows = []
        warnings = []
        for repetition, strategy in [
            (repetition, strategy)
            for repetition in completed
            for strategy in repetition.get("strategies", [])
            if strategy.get("strategyId") == strategy_id
        ]:
            repetition_meta = repetition.get("configSummary") or {}
            for row in strategy.get("movementRows") or []:
                movement_rows.append({**row, "repetitionIndex": repetition_meta.get("repetitionIndex"), "repetitionSeed": repetition_meta.get("seed")})
            for row in strategy.get("finalRows") or []:
                final_rows.append({**row, "repetitionIndex": repetition_meta.get("repetitionIndex"), "repetitionSeed": repetition_meta.get("seed")})
            warnings.extend(str(warning) for warning in strategy.get("warnings") or [])

        final_metrics = aggregate_final_summaries([strategy.get("finalMetrics") or {} for strategy in per_repetition])
        strategy_result = {
            "strategyId": strategy_id,
            "displayName": STRATEGIES[strategy_id],
            "status": "completed",
            "operatorMetrics": aggregate_operator_metrics([strategy.get("operatorMetrics") or {} for strategy in per_repetition]),
            "finalMetrics": final_metrics,
            "generationCost": aggregate_generation_costs([strategy.get("generationCost") or {} for strategy in per_repetition]),
            "finalEmbeddingCost": aggregate_embedding_costs([strategy.get("finalEmbeddingCost") or {} for strategy in per_repetition]),
            "movementRows": movement_rows[:300],
            "finalRows": final_rows,
            "warnings": warnings,
            "repetitionsK": config.get("repetitionsK", 1),
            "completedRepetitions": len(per_repetition),
            "repetitions": per_repetition,
        }
        strategy_result["deltas"] = final_metric_deltas(baseline, final_metrics)
        strategy_result["relativeOperatorCost"] = relative_operator_cost(strategy_result["operatorMetrics"])
        strategy_results.append(strategy_result)

    recommendation = recommend_strategy(strategy_results, config)
    warnings = comparison_warnings(baseline, strategy_results, recommendation)
    return {
        "movementCount": sum(int(repetition.get("movementCount") or 0) for repetition in completed),
        "individualCount": config["individualCount"],
        "ppdb": completed[-1].get("ppdb") or {},
        "baseline": baseline,
        "strategies": strategy_results,
        "recommendation": recommendation,
        "warnings": warnings,
        "repetitionsK": config.get("repetitionsK", 1),
        "completedRepetitions": len(completed),
        "repetitions": repetitions,
        "configSummary": {
            "seed": config["seed"],
            "repetitionsK": config.get("repetitionsK", 1),
            "seeds": [(int(config["seed"]) + index) % 2_147_483_648 for index in range(int(config.get("repetitionsK") or 1))],
            "movementPolicy": "one_uniform_component_per_individual",
            "kCandidates": config["kCandidates"],
            "operatorParallelism": config["operatorParallelism"],
            "generationParallelism": config["generationParallelism"],
            "turbulenceRange": [config["turbulenceMinSimilarity"], config["turbulenceMaxSimilarity"]],
            "embeddingModel": config["embeddingModel"],
            "distilbertModel": config["distilbertModel"],
            "usePpdb": config.get("usePpdb", True),
        },
    }


def summarize_operator_rows(
    rows: list[dict[str, Any]],
    costs: list[dict[str, Any]],
    operator_wall_clock_seconds: float | None = None,
    operator_parallelism: int = 1,
) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("success"))
    coverage = sum(1 for row in rows if row.get("coverage"))
    cumulative_seconds = sum(finite_float(row.get("elapsedSeconds")) for row in rows)
    wall_clock_seconds = operator_wall_clock_seconds if operator_wall_clock_seconds is not None else cumulative_seconds
    aggregate_cost = aggregate_costs(costs)
    average_seconds = cumulative_seconds / total if total else None
    similarity_values = [
        finite_float(row.get("similarity"))
        for row in rows
        if row.get("similarity") is not None
    ]
    similarity_sum = sum(similarity_values)
    similarity_count = len(similarity_values)
    return {
        "moves": total,
        "successes": successes,
        "successRate": successes / total if total else 0.0,
        "coverageCount": coverage,
        "coverageRate": coverage / total if total else 0.0,
        "similarityCount": similarity_count,
        "similaritySum": similarity_sum,
        "averageSimilarity": similarity_sum / similarity_count if similarity_count else None,
        "operatorParallelism": operator_parallelism,
        "operatorCumulativeSeconds": cumulative_seconds,
        "operatorCumulativeLabel": format_duration(cumulative_seconds),
        "operatorWallClockSeconds": wall_clock_seconds,
        "operatorWallClockLabel": format_duration(wall_clock_seconds),
        "operatorAverageSeconds": average_seconds,
        "operatorAverageLabel": format_duration(average_seconds),
        "cost": aggregate_cost,
    }


def aggregate_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = {
        "llmCalls",
        "llmClientWallClockSeconds",
        "promptTokens",
        "completionTokens",
        "totalTokens",
        "wordnetQueries",
        "ppdbQueries",
        "ppdbLookupAttempts",
        "distilbertInferences",
        "embeddingTexts",
        "embeddingWallClockSeconds",
        "operatorApplicationSeconds",
        "operatorWallClockSeconds",
    }
    result = {key: 0.0 for key in numeric_keys}
    result["ppdbAvailable"] = any(bool(cost.get("ppdbAvailable")) for cost in costs)
    for cost in costs:
        for key in numeric_keys:
            result[key] += finite_float(cost.get(key))
    for key in ("llmCalls", "promptTokens", "completionTokens", "totalTokens", "wordnetQueries", "ppdbQueries", "ppdbLookupAttempts", "distilbertInferences", "embeddingTexts"):
        result[key] = int(result[key])
    result["embeddingWallClockLabel"] = format_duration(result["embeddingWallClockSeconds"])
    result["llmClientWallClockLabel"] = format_duration(result["llmClientWallClockSeconds"])
    result["operatorApplicationLabel"] = format_duration(result["operatorApplicationSeconds"])
    result["operatorWallClockLabel"] = format_duration(result["operatorWallClockSeconds"])
    return result


def summarize_final_metrics(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("commonFidelity") is not None]
    avg_fidelity = sum(finite_float(row.get("commonFidelity")) for row in completed) / len(completed) if completed else None
    avg_diversity = sum(finite_float(row.get("commonDiversity")) for row in completed) / len(completed) if completed else None
    return {
        "completedRows": len(completed),
        "averageFidelity": avg_fidelity,
        "averageDiversity": avg_diversity,
        "bestFidelity": metrics.get("bestFidelity"),
        "bestDiversity": metrics.get("bestDiversity"),
        "hypervolume": metrics.get("hypervolume"),
        "spread": metrics.get("spread"),
        "rows": rows,
    }


def final_metric_deltas(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    fidelity = current.get("averageFidelity")
    baseline_fidelity = baseline.get("averageFidelity")
    diversity = current.get("averageDiversity")
    baseline_diversity = baseline.get("averageDiversity")
    return {
        "averageFidelityDelta": fidelity - baseline_fidelity if fidelity is not None and baseline_fidelity is not None else None,
        "averageDiversityDelta": diversity - baseline_diversity if diversity is not None and baseline_diversity is not None else None,
    }


def relative_operator_cost(operator_metrics: dict[str, Any]) -> float:
    cost = operator_metrics.get("cost") or {}
    return (
        finite_float(cost.get("llmCalls")) * 1000.0
        + finite_float(cost.get("distilbertInferences")) * 10.0
        + finite_float(cost.get("embeddingTexts")) * 1.0
        + finite_float(cost.get("wordnetQueries")) * 0.2
        + finite_float(cost.get("ppdbQueries")) * 0.2
        + finite_float(operator_metrics.get("operatorCumulativeSeconds", operator_metrics.get("operatorWallClockSeconds"))) * 5.0
    )


def recommend_strategy(strategy_results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    eligible = [
        result
        for result in strategy_results
        if result["finalMetrics"].get("averageFidelity") is not None
        and config["finalFidelityMin"] <= result["finalMetrics"]["averageFidelity"] <= config["finalFidelityMax"]
    ]
    if not strategy_results:
        return {"strategyId": None, "message": "Sin resultados suficientes para recomendar."}
    if not eligible:
        return {
            "strategyId": None,
            "eligibleByFidelity": False,
            "message": "Ninguna estrategia tiene fidelidad final disponible dentro del rango configurado.",
        }
    winner = sorted(
        eligible,
        key=lambda result: (
            result.get("relativeOperatorCost", 0.0),
            -finite_float(result["operatorMetrics"].get("successRate")),
            finite_float(result["operatorMetrics"].get("operatorAverageSeconds"), 1e9),
        ),
    )[0]
    return {
        "strategyId": winner["strategyId"],
        "displayName": winner["displayName"],
        "eligibleByFidelity": bool(eligible),
        "message": (
            f"{winner['displayName']} combina menor costo relativo, "
            f"tasa de éxito {winner['operatorMetrics']['successRate']:.2%} "
            f"y fidelidad final promedio {winner['finalMetrics'].get('averageFidelity') or 0:.6f}."
        ),
    }


def clamp_number(value: Any, minimum: float, maximum: float) -> float:
    number = finite_float(value, minimum)
    return max(minimum, min(maximum, number))


def safe_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} es obligatorio.")
    return text


def read_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def safe_path_text(value: Any, field: str) -> str:
    text = safe_text(value, field)
    if text.startswith(("http://", "https://")):
        raise ValueError(f"{field} debe ser una ruta local, no una URL.")
    return text


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "No disponible"
    number = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while number >= 1024 and unit_index < len(units) - 1:
        number /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(number)} {units[unit_index]}"
    return f"{number:.2f} {units[unit_index]}"
