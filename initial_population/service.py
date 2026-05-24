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
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from initial_population.runner import DOMAIN_DEFAULT, PROGRESS_PREFIX


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

DEFAULT_REFERENCE_TEXT = (
    "Our action center has been updated with more information about restaurant shutdowns "
    "and disaster financing options for SMBs."
)
DEFAULT_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_LM_STUDIO = "http://127.0.0.1:1234"
DEFAULT_API_MODE = "native"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


ANCHOR_SYSTEM_PROMPT = """You are an information extraction module for a prompt optimization algorithm.

Task:
Extract short semantic anchors from the reference text.

Output format:
Return only valid JSON with exactly these keys:
{
  "entities": ["..."],
  "topics": ["..."],
  "actions": ["..."],
  "constraints": ["..."]
}

Rules:
- Use the reference text as the main source.
- Use the general domain only as context.
- Do not invent specific facts that are not supported by the reference text.
- Keep each item short.
- Use 1 to 5 items per list.
- Do not include explanations, markdown, numbering, or extra keys."""

ANCHOR_USER_PROMPT = """Reference text:
\"\"\"{reference_text}\"\"\"

General domain:
{domain}"""

POOL_SYSTEM_PROMPT = """You generate one pool of semantic components for a prompt optimization algorithm.

Task:
Generate only the requested component type: roles, topics, or actions.

Output format:
Return only valid JSON with exactly this structure:
{"items": ["...", "..."]}

Rules:
- The reference text and the general domain are the main source.
- Semantic anchors are only supporting hints.
- Every item must be associated with the reference text and the general domain.
- Do not simply copy the semantic anchors.
- Keep each item short.
- Do not include explanations, markdown, numbering, or extra keys."""

POOL_USER_PROMPT = """Reference text:
\"\"\"{reference_text}\"\"\"

General domain:
{domain}

Component to generate:
{component_type}

Required number of items:
{q_component}

Semantic anchors for support:
{anchors}"""

EXPANSION_SYSTEM_PROMPT = """You expand one existing pool of semantic components.

Task:
Generate additional items for the requested component type.

Output format:
Return only valid JSON with exactly this structure:
{"items": ["...", "..."]}

Rules:
- The reference text and the general domain are the main source.
- Semantic anchors are only supporting hints.
- Do not repeat existing items.
- Do not paraphrase existing items too closely.
- Keep each item short.
- Do not include explanations, markdown, numbering, or extra keys."""

EXPANSION_USER_PROMPT = """Reference text:
\"\"\"{reference_text}\"\"\"

General domain:
{domain}

Component to expand:
{component_type}

Required number of new items:
{q_extra}

Existing items to avoid:
{existing_items}

Semantic anchors for support:
{anchors}"""

GENERATION_SYSTEM_PROMPT = """You are a plain-text generator for natural-disaster scenario messages. You will receive one text-generation instruction from the user. Follow the instruction and generate exactly one final text message.

Output rules:
- Return plain text only.
- Return the dataset content itself, not a prompt, explanation, title, label, list, code, or metadata.
- Do not describe the task.
- Do not add unsolicited safety advice.
- Do not use quotation marks, hashtags, URLs, usernames, placeholders, tags, or special markers.
- Limit the message to between 1 and 4 sentences.
- Return only the final message."""

GENERATION_USER_PROMPT = "{prompt}"

PROMPT_TEMPLATE = (
    "Generate a short natural-disaster scenario message using the following semantic components: "
    "role = {role}; topic = {topic}; action = {action}. "
    "The generated message must follow the role, address the topic, and satisfy the action."
)

DEFAULT_STAGE_CONFIGS = {
    "anchors": {
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "topP": 0.95,
        "topK": 40,
        "maxTokens": 400,
        "systemPrompt": ANCHOR_SYSTEM_PROMPT,
        "userPrompt": ANCHOR_USER_PROMPT,
    },
    "roles": {
        "model": DEFAULT_MODEL,
        "temperature": 0.6,
        "topP": 0.95,
        "topK": 40,
        "maxTokens": 500,
        "systemPrompt": POOL_SYSTEM_PROMPT,
        "userPrompt": POOL_USER_PROMPT,
    },
    "topics": {
        "model": DEFAULT_MODEL,
        "temperature": 0.35,
        "topP": 0.95,
        "topK": 40,
        "maxTokens": 500,
        "systemPrompt": POOL_SYSTEM_PROMPT,
        "userPrompt": POOL_USER_PROMPT,
    },
    "actions": {
        "model": DEFAULT_MODEL,
        "temperature": 0.5,
        "topP": 0.95,
        "topK": 40,
        "maxTokens": 500,
        "systemPrompt": POOL_SYSTEM_PROMPT,
        "userPrompt": POOL_USER_PROMPT,
    },
    "expansion": {
        "model": DEFAULT_MODEL,
        "temperature": 0.6,
        "topP": 0.95,
        "topK": 40,
        "maxTokens": 500,
        "systemPrompt": EXPANSION_SYSTEM_PROMPT,
        "userPrompt": EXPANSION_USER_PROMPT,
    },
    "generation": {
        "model": DEFAULT_MODEL,
        "temperature": 0.75,
        "topP": 0.95,
        "topK": 40,
        "maxTokens": 180,
        "systemPrompt": GENERATION_SYSTEM_PROMPT,
        "userPrompt": GENERATION_USER_PROMPT,
    },
}


@dataclass(frozen=True)
class InitialPopulationStrategy:
    strategy_id: str
    display_name: str
    description: str
    objective_names: tuple[str, ...]


STRATEGIES: tuple[InitialPopulationStrategy, ...] = (
    InitialPopulationStrategy(
        strategy_id="hybrid-semantic-v7",
        display_name="Hybrid semantic initialization v7",
        description="Anchors, semantic pools, stratified prompts, SBERT diversity, real generated-text fidelity.",
        objective_names=("semantic_fidelity", "semantic_diversity"),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "No disponible"
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


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


def summarize_cost(cost: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(cost or {})
    wall_clock = enriched.get("wallClockSeconds") or enriched.get("processWallClockSeconds")
    llm_seconds = enriched.get("llmClientWallClockSeconds")
    final_individuals = finite_float(enriched.get("finalIndividuals"))
    wall_clock_float = finite_float(wall_clock)
    if (
        enriched.get("wallClockPerFinalIndividualSeconds") is None
        and wall_clock is not None
        and final_individuals > 0
    ):
        enriched["wallClockPerFinalIndividualSeconds"] = wall_clock_float / final_individuals
    if wall_clock is not None and llm_seconds is not None and enriched.get("estimatedSequentialWallClockSeconds") is None:
        llm_seconds_float = finite_float(llm_seconds)
        concurrent_llm = enriched.get("llmConcurrentWallClockSeconds")
        concurrent_llm_float = finite_float(concurrent_llm, min(llm_seconds_float, wall_clock_float))
        sequential = wall_clock_float + max(0.0, llm_seconds_float - concurrent_llm_float)
        savings = max(0.0, sequential - wall_clock_float)
        enriched["estimatedSequentialWallClockSeconds"] = sequential
        enriched["parallelismSavingsSeconds"] = savings
        enriched["parallelismSavingsPercent"] = (savings / sequential * 100.0) if sequential > 0 else None
    for key, label_key in (
        ("wallClockSeconds", "wallClockLabel"),
        ("estimatedSequentialWallClockSeconds", "estimatedSequentialWallClockLabel"),
        ("parallelismSavingsSeconds", "parallelismSavingsLabel"),
        ("llmClientWallClockSeconds", "llmClientWallClockLabel"),
        ("llmConcurrentWallClockSeconds", "llmConcurrentWallClockLabel"),
        ("llmAverageCallSeconds", "llmAverageCallLabel"),
        ("embeddingWallClockSeconds", "embeddingWallClockLabel"),
        ("embeddingAverageBatchSeconds", "embeddingAverageBatchLabel"),
        ("wallClockPerFinalIndividualSeconds", "wallClockPerFinalIndividualLabel"),
    ):
        enriched[label_key] = format_duration(enriched.get(key))
    return enriched


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


class InitialPopulationService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_root = root / "runs" / "initial-population"
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def list_strategies(self) -> list[dict[str, Any]]:
        return [
            {
                "strategyId": strategy.strategy_id,
                "displayName": strategy.display_name,
                "description": strategy.description,
                "objectiveNames": list(strategy.objective_names),
                "available": True,
            }
            for strategy in STRATEGIES
        ]

    def default_config(self) -> dict[str, Any]:
        return {
            "strategyId": "hybrid-semantic-v7",
            "referenceText": DEFAULT_REFERENCE_TEXT,
            "domain": DOMAIN_DEFAULT,
            "n": 10,
            "topK": 10,
            "timeoutSeconds": 120,
            "generationParallelism": 1,
            "seed": 42,
            "embeddingModel": DEFAULT_EMBEDDING_MODEL,
            "lmStudio": {
                "baseUrl": DEFAULT_LM_STUDIO,
                "apiMode": DEFAULT_API_MODE,
            },
            "stages": DEFAULT_STAGE_CONFIGS,
            "promptTemplate": PROMPT_TEMPLATE,
            "validation": {
                "rolesMaxWords": 6,
                "topicsMaxWords": 8,
                "actionsMaxWords": 6,
                "generatedMinWords": 1,
                "generatedMaxWords": 120,
            },
        }

    def list_lm_studio_models(self, base_url: str | None, api_mode: str | None) -> dict[str, Any]:
        base = (base_url or DEFAULT_LM_STUDIO).rstrip("/")
        mode = api_mode if api_mode in {"native", "openai"} else DEFAULT_API_MODE
        endpoint = "/api/v1/models" if mode == "native" else "/v1/models"
        request = urllib.request.Request(f"{base}{endpoint}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ValueError(f"LM Studio returned HTTP {error.code}.") from error
        except Exception as error:
            raise ValueError(f"Could not query LM Studio models: {error}") from error

        models: list[str] = []
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
                elif isinstance(item, str):
                    models.append(item)
        elif isinstance(payload, list):
            models = [str(item.get("id") if isinstance(item, dict) else item) for item in payload]

        return {
            "baseUrl": base,
            "apiMode": mode,
            "models": sorted(set(model for model in models if model)),
            "raw": payload,
        }

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._read_config(payload)
        run_id = self._new_run_id()
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        write_json(config_path, config)

        run = {
            "runId": run_id,
            "status": STATUS_QUEUED,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "startedAtEpoch": None,
            "runDir": str(run_dir),
            "config": config,
            "logs": [],
            "progress": {
                "percent": 0,
                "detail": "Corrida en cola.",
                "stage": "queued",
                "elapsedSeconds": 0,
                "elapsedLabel": "0s",
                "remainingSeconds": None,
                "remainingLabel": "No disponible",
            },
            "strategy": None,
            "rows": [],
            "metrics": {},
            "cost": {},
            "semanticArtifacts": {},
            "error": None,
            "cancelRequested": False,
            "activeProcess": None,
            "lastSummaryWriteEpoch": 0.0,
        }
        with self._lock:
            self._runs[run_id] = run
            self._write_summary_unlocked(run)

        thread = threading.Thread(target=self._run_worker, args=(run_id, config_path), daemon=True)
        thread.start()
        return self._public_run(run)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                if run.get("status") == STATUS_RUNNING:
                    if not self._sync_finished_result_unlocked(run):
                        self._refresh_running_state_unlocked(run)
                        self._write_summary_unlocked(run)
                return self._public_run(run)
        summary_path = self.runs_root / run_id / "summary.json"
        if summary_path.exists():
            summary = read_json_or_default(summary_path, None)
            result = read_json_or_default(self.runs_root / run_id / "result.json", {})
            if (
                isinstance(summary, dict)
                and isinstance(result, dict)
                and result.get("status") in {STATUS_COMPLETED, STATUS_FAILED}
            ):
                if self._summary_needs_result_merge(summary, result):
                    summary = self._merge_result_into_summary(summary, result)
                    write_json(summary_path, summary)
            if isinstance(summary, dict):
                summary["cost"] = summarize_cost(summary.get("cost") or {})
                summary["semanticArtifacts"] = self._read_semantic_artifacts(self.runs_root / run_id)
            return summary
        return None

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            if run.get("status") == STATUS_RUNNING and self._sync_finished_result_unlocked(run):
                return self._public_run(run)
            run["cancelRequested"] = True
            run["updatedAt"] = utc_now()
            process = run.get("activeProcess")
            self._append_log_unlocked(run, "system", "Cancel requested. Terminating active process.")
            current_progress = run.get("progress") if isinstance(run.get("progress"), dict) else {}
            self._refresh_progress_unlocked(
                run,
                "cancel",
                "Cancelacion solicitada; cerrando subproceso.",
                finite_float(current_progress.get("percent")),
                current_progress,
            )

        if process is not None:
            self._terminate_process(process)

        with self._lock:
            self._write_summary_unlocked(run)
            return self._public_run(run)

    def _read_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.default_config()
        strategy_id = str(payload.get("strategyId") or defaults["strategyId"]).strip()
        if strategy_id not in {"hybrid-semantic-v6", "hybrid-semantic-v7"}:
            raise ValueError("Unsupported initial population strategy.")
        strategy_id = "hybrid-semantic-v7"

        reference_text = str(payload.get("referenceText", defaults["referenceText"])).strip()
        if not reference_text:
            raise ValueError("referenceText is required.")

        config = {
            "strategyId": strategy_id,
            "referenceText": reference_text,
            "domain": str(payload.get("domain") or defaults["domain"]).strip() or DOMAIN_DEFAULT,
            "n": self._int_between(payload.get("n", defaults["n"]), "n", 1, 500),
            "topK": self._int_between(payload.get("topK", defaults["topK"]), "topK", 1, 500),
            "timeoutSeconds": self._int_between(payload.get("timeoutSeconds", defaults["timeoutSeconds"]), "timeoutSeconds", 5, 3600),
            "generationParallelism": self._int_between(payload.get("generationParallelism", defaults["generationParallelism"]), "generationParallelism", 1, 16),
            "seed": self._int_between(payload.get("seed", defaults["seed"]), "seed", 0, 2_147_483_647),
            "embeddingModel": self._safe_text(payload.get("embeddingModel", defaults["embeddingModel"]), "embeddingModel"),
            "lmStudio": self._read_lm_studio_config(payload.get("lmStudio") or defaults["lmStudio"]),
            "stages": self._read_stage_configs(payload.get("stages") or {}),
            "promptTemplate": self._safe_text(payload.get("promptTemplate", defaults["promptTemplate"]), "promptTemplate"),
            "validation": self._read_validation(payload.get("validation") or defaults["validation"]),
        }
        config["topK"] = min(config["topK"], config["n"])
        return config

    def _read_lm_studio_config(self, value: Any) -> dict[str, Any]:
        payload = value if isinstance(value, dict) else {}
        base_url = self._safe_text(payload.get("baseUrl", DEFAULT_LM_STUDIO), "LM Studio base URL").rstrip("/")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LM Studio base URL must be an http(s) URL.")
        api_mode = str(payload.get("apiMode", DEFAULT_API_MODE)).strip()
        if api_mode not in {"native", "openai"}:
            raise ValueError("LM Studio apiMode must be native or openai.")
        return {"baseUrl": base_url, "apiMode": api_mode}

    def _read_stage_configs(self, payload: dict[str, Any]) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        for stage_name, defaults in DEFAULT_STAGE_CONFIGS.items():
            stage_payload = payload.get(stage_name) if isinstance(payload, dict) else {}
            stage_payload = stage_payload if isinstance(stage_payload, dict) else {}
            stages[stage_name] = {
                "model": self._safe_text(stage_payload.get("model", defaults["model"]), f"{stage_name}.model"),
                "temperature": self._float_between(stage_payload.get("temperature", defaults["temperature"]), f"{stage_name}.temperature", 0.0, 2.0),
                "topP": self._float_between(stage_payload.get("topP", defaults["topP"]), f"{stage_name}.topP", 0.0, 1.0),
                "topK": self._int_between(stage_payload.get("topK", defaults["topK"]), f"{stage_name}.topK", 0, 500),
                "maxTokens": self._int_between(stage_payload.get("maxTokens", defaults["maxTokens"]), f"{stage_name}.maxTokens", 8, 4096),
                "systemPrompt": str(stage_payload.get("systemPrompt", defaults["systemPrompt"])),
                "userPrompt": str(stage_payload.get("userPrompt", defaults["userPrompt"])),
            }
        return stages

    def _read_validation(self, payload: dict[str, Any]) -> dict[str, int]:
        defaults = self.default_config()["validation"]
        validation = {
            "rolesMaxWords": self._int_between(payload.get("rolesMaxWords", defaults["rolesMaxWords"]), "rolesMaxWords", 1, 20),
            "topicsMaxWords": self._int_between(payload.get("topicsMaxWords", defaults["topicsMaxWords"]), "topicsMaxWords", 1, 30),
            "actionsMaxWords": self._int_between(payload.get("actionsMaxWords", defaults["actionsMaxWords"]), "actionsMaxWords", 1, 20),
            "generatedMinWords": self._int_between(payload.get("generatedMinWords", defaults["generatedMinWords"]), "generatedMinWords", 1, 500),
            "generatedMaxWords": self._int_between(payload.get("generatedMaxWords", defaults["generatedMaxWords"]), "generatedMaxWords", 1, 500),
        }
        if validation["generatedMinWords"] > validation["generatedMaxWords"]:
            raise ValueError("generatedMinWords cannot be greater than generatedMaxWords.")
        return validation

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

    def _run_worker(self, run_id: str, config_path: Path) -> None:
        try:
            with self._lock:
                run = self._runs[run_id]
                run["status"] = STATUS_RUNNING
                run["startedAtEpoch"] = time.time()
                run["updatedAt"] = utc_now()
                self._append_log_unlocked(run, "system", "Initial population run started.")
                self._write_summary_unlocked(run)

            run_dir = Path(run["runDir"])
            command = [
                sys.executable,
                "-m",
                "initial_population.runner",
                "--config",
                str(config_path),
                "--output-dir",
                str(run_dir),
            ]
            process_cost = self._run_process(run, command, self.root)
            result = read_json_or_default(run_dir / "result.json", {})

            with self._lock:
                previous_progress = self._latest_artifact_progress(run_dir, dict(run.get("progress") or {}))
                has_result = isinstance(result, dict) and result.get("status") in {STATUS_COMPLETED, STATUS_FAILED}
                timed_out_without_result = bool(process_cost["timedOut"] and not has_result)
                if run.get("cancelRequested"):
                    run["status"] = STATUS_CANCELLED
                    run["error"] = "Run cancelled."
                elif result.get("status") == STATUS_COMPLETED:
                    run["status"] = STATUS_COMPLETED
                elif result.get("status") == STATUS_FAILED:
                    run["status"] = STATUS_FAILED
                    run["error"] = result.get("error")
                elif timed_out_without_result:
                    run["status"] = STATUS_FAILED
                    run["error"] = self._build_timeout_error(run, process_cost, previous_progress)
                elif int(process_cost["returnCode"]) != 0:
                    run["status"] = STATUS_FAILED
                    run["error"] = f"Process exited with code {process_cost['returnCode']}."
                else:
                    run["status"] = STATUS_FAILED
                    run["error"] = "Runner did not produce a valid result."

                if result:
                    self._apply_result_unlocked(run, result)
                else:
                    run["rows"] = self._read_partial_generated_rows(run_dir)
                    run["metrics"] = self._partial_metrics(run["rows"])
                    run["cost"] = summarize_cost(self._read_artifact_cost(run_dir, run["rows"]))

                run["cost"]["processWallClockSeconds"] = process_cost["processWallClockSeconds"]
                run["cost"]["processWallClockLabel"] = format_duration(process_cost["processWallClockSeconds"])
                if run["cost"].get("wallClockSeconds") is None:
                    run["cost"]["wallClockSeconds"] = process_cost["processWallClockSeconds"]
                run["cost"]["returnCode"] = process_cost["returnCode"]
                run["cost"]["timedOut"] = timed_out_without_result
                run["cost"] = summarize_cost(run["cost"])
                run["updatedAt"] = utc_now()
                final_detail = self._error_message(run["error"]) if run["status"] == STATUS_FAILED and run.get("error") else status_label(run["status"])
                final_percent = 100 if run["status"] in {STATUS_COMPLETED, STATUS_CANCELLED} else finite_float(previous_progress.get("percent"), 100)
                final_stage = "timeout" if timed_out_without_result else run["status"]
                self._refresh_progress_unlocked(run, final_stage, final_detail, final_percent, previous_progress)
                self._append_log_unlocked(run, "system", f"Initial population run finished with status {run['status']}.")
                self._write_summary_unlocked(run)
        except Exception as error:
            with self._lock:
                run = self._runs.get(run_id)
                if not run:
                    return
                run["status"] = STATUS_FAILED
                run["error"] = f"Initial population worker failed: {error}"
                self._refresh_running_state_unlocked(run)
                self._refresh_progress_unlocked(run, STATUS_FAILED, run["error"], 100)
                self._append_log_unlocked(run, "system", run["error"])
                self._write_summary_unlocked(run)

    def _read_artifact_cost(self, run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
        llm_calls = read_json_or_default(run_dir / "llm_calls.json", [])
        embedding_batches = read_json_or_default(run_dir / "embedding_batches.json", [])
        generated_texts = read_json_or_default(run_dir / "generated_texts.json", [])
        population = read_json_or_default(run_dir / "population.json", [])
        llm_calls = llm_calls if isinstance(llm_calls, list) else []
        embedding_batches = embedding_batches if isinstance(embedding_batches, list) else []
        generated_texts = generated_texts if isinstance(generated_texts, list) else []
        population = population if isinstance(population, list) else []
        llm_seconds = sum(finite_float(call.get("elapsedSeconds")) for call in llm_calls if isinstance(call, dict))
        embedding_seconds = sum(finite_float(batch.get("elapsedSeconds")) for batch in embedding_batches if isinstance(batch, dict))
        final_count = len(population)
        if not final_count and rows and any(isinstance(row, dict) and row.get("objectiveVector") for row in rows):
            final_count = len(rows)
        generated_count = len(generated_texts)
        llm_concurrent_seconds = interval_union_seconds(llm_calls)
        return {
            "llmCalls": len(llm_calls),
            "llmSuccessfulCalls": sum(1 for call in llm_calls if isinstance(call, dict) and call.get("status") == "ok"),
            "llmFailedCalls": sum(1 for call in llm_calls if isinstance(call, dict) and call.get("status") == "error"),
            "llmClientWallClockSeconds": llm_seconds,
            "llmConcurrentWallClockSeconds": llm_concurrent_seconds,
            "llmAverageCallSeconds": llm_seconds / len(llm_calls) if llm_calls else None,
            "embeddingBatches": len(embedding_batches),
            "embeddingTexts": sum(int(finite_float(batch.get("textCount"))) for batch in embedding_batches if isinstance(batch, dict)),
            "embeddingWallClockSeconds": embedding_seconds,
            "embeddingAverageBatchSeconds": embedding_seconds / len(embedding_batches) if embedding_batches else None,
            "promptTokens": sum(int(finite_float(call.get("promptTokens"))) for call in llm_calls if isinstance(call, dict)),
            "completionTokens": sum(int(finite_float(call.get("completionTokens"))) for call in llm_calls if isinstance(call, dict)),
            "totalTokens": sum(int(finite_float(call.get("totalTokens"))) for call in llm_calls if isinstance(call, dict)),
            "generatedTexts": generated_count,
            "finalIndividuals": final_count,
            "llmCallsPerFinalIndividual": len(llm_calls) / final_count if final_count else None,
            "llmCallsPerGeneratedText": len(llm_calls) / generated_count if generated_count else None,
            "wallClockPerFinalIndividualSeconds": None,
        }

    def _run_process(self, run: dict[str, Any], command: list[str], cwd: Path) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        started = time.perf_counter()
        timed_out = False
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
                for output_line in process.stdout:
                    output_queue.put(output_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        deadline = time.monotonic() + run["config"]["timeoutSeconds"]
        reader_done = False
        termination_requested = False
        last_tick = time.monotonic()
        process_finished_at: float | None = None

        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                line = ""

            if line is None:
                reader_done = True
            elif line:
                self._handle_process_line(run, line)

            now = time.monotonic()
            current_return_code = process.poll()
            if current_return_code is not None and process_finished_at is None:
                process_finished_at = now
            if now - last_tick >= 1.0 and current_return_code is None:
                with self._lock:
                    if run.get("status") == STATUS_RUNNING:
                        self._refresh_running_state_unlocked(run)
                        self._write_summary_unlocked(run)
                last_tick = now

            if run.get("cancelRequested") and current_return_code is None and not termination_requested:
                self._append_log(run, "system", "Cancellation requested; terminating process tree.")
                self._terminate_process(process)
                termination_requested = True

            if current_return_code is None and now >= deadline and not termination_requested:
                self._append_log(run, "system", f"Timeout reached after {run['config']['timeoutSeconds']} second(s).")
                self._terminate_process(process)
                termination_requested = True
                timed_out = True
                return_code = 124
                break

            if reader_done and current_return_code is not None:
                return_code = current_return_code
                break

            if current_return_code is not None and process_finished_at is not None and now - process_finished_at > 2.0:
                return_code = current_return_code
                break

        reader.join(timeout=2)
        with self._lock:
            if run.get("activeProcess") is process:
                run["activeProcess"] = None
        return {
            "returnCode": return_code,
            "processWallClockSeconds": time.perf_counter() - started,
            "timedOut": timed_out,
        }

    def _handle_process_line(self, run: dict[str, Any], line: str) -> None:
        message = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
        if not message:
            return
        if message.startswith(PROGRESS_PREFIX):
            raw_payload = message.removeprefix(PROGRESS_PREFIX)
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                self._append_log(run, "runner", message)
                return
            with self._lock:
                self._refresh_progress_unlocked(
                    run,
                    str(payload.get("stage") or "runner"),
                    str(payload.get("detail") or "Ejecutando."),
                    finite_float(payload.get("percent")),
                    payload,
                )
                self._refresh_running_artifacts_unlocked(run)
                self._write_summary_throttled_unlocked(run)
            return
        self._append_log(run, "runner", message)

    def _refresh_progress_unlocked(
        self,
        run: dict[str, Any],
        stage: str,
        detail: str,
        percent: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        remaining = None
        safe_percent = int(max(0, min(100, round(percent))))
        if run.get("status") == STATUS_RUNNING and 0 < safe_percent < 100:
            remaining = elapsed * (100 - safe_percent) / safe_percent
        run["progress"] = {
            "percent": safe_percent,
            "detail": detail,
            "stage": stage,
            "stageIndex": (payload or {}).get("stageIndex"),
            "stageTotal": (payload or {}).get("stageTotal"),
            "completed": (payload or {}).get("completed"),
            "total": (payload or {}).get("total"),
            "elapsedSeconds": elapsed,
            "elapsedLabel": format_duration(elapsed),
            "remainingSeconds": remaining,
            "remainingLabel": format_duration(remaining),
        }
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

    def _append_log(self, run: dict[str, Any], source: str, message: str) -> None:
        with self._lock:
            self._append_log_unlocked(run, source, message)
            self._write_summary_throttled_unlocked(run)

    def _append_log_unlocked(self, run: dict[str, Any], source: str, message: str) -> None:
        if not message:
            return
        run["logs"].append({"at": utc_now(), "source": source, "message": message})
        run["logs"] = run["logs"][-300:]
        run["updatedAt"] = utc_now()

    def _sync_finished_result_unlocked(self, run: dict[str, Any]) -> bool:
        result_path = Path(run["runDir"]) / "result.json"
        result = read_json_or_default(result_path, {})
        if not isinstance(result, dict) or result.get("status") not in {STATUS_COMPLETED, STATUS_FAILED}:
            return False
        previous_status = run.get("status")
        run["status"] = result["status"]
        run["error"] = result.get("error") if result["status"] == STATUS_FAILED else None
        self._apply_result_unlocked(run, result)
        progress = self._latest_artifact_progress(Path(run["runDir"]), run.get("progress") or {})
        if run["status"] == STATUS_COMPLETED:
            final_percent = 100
            detail = status_label(run["status"])
        else:
            final_percent = finite_float(progress.get("percent"))
            detail = self._error_message(run["error"]) if run.get("error") else status_label(run["status"])
        self._refresh_progress_unlocked(run, run["status"], detail, final_percent, progress)
        if previous_status == STATUS_RUNNING:
            self._append_log_unlocked(run, "system", f"Recovered final status from result.json: {run['status']}.")
        self._write_summary_unlocked(run)
        return True

    def _apply_result_unlocked(self, run: dict[str, Any], result: dict[str, Any]) -> None:
        run_dir = Path(run["runDir"])
        rows = result.get("rows") or []
        if not rows and result.get("status") == STATUS_FAILED:
            rows = self._read_partial_generated_rows(run_dir)
        run["strategy"] = {
            "strategyId": result.get("strategyId"),
            "displayName": result.get("displayName"),
            "status": result.get("status"),
            "outputDir": result.get("outputDir"),
        }
        run["rows"] = rows
        run["metrics"] = result.get("metrics") or self._partial_metrics(rows)
        run["cost"] = summarize_cost(result.get("cost") or self._read_artifact_cost(run_dir, run["rows"]))
        run["semanticArtifacts"] = self._read_semantic_artifacts(run_dir)
        artifact_logs = self._read_artifact_logs(run_dir)
        if artifact_logs:
            system_logs = [entry for entry in run.get("logs", []) if isinstance(entry, dict) and entry.get("source") == "system"]
            run["logs"] = (system_logs + artifact_logs)[-300:]
        run["updatedAt"] = utc_now()

    def _refresh_running_state_unlocked(self, run: dict[str, Any]) -> None:
        progress = run.get("progress") if isinstance(run.get("progress"), dict) else {}
        progress = self._latest_artifact_progress(Path(run["runDir"]), progress)
        self._refresh_progress_unlocked(
            run,
            str(progress.get("stage") or STATUS_RUNNING),
            str(progress.get("detail") or status_label(STATUS_RUNNING)),
            finite_float(progress.get("percent")),
            progress,
        )
        self._refresh_running_artifacts_unlocked(run)

    def _refresh_running_artifacts_unlocked(self, run: dict[str, Any]) -> None:
        run_dir = Path(run["runDir"])
        rows = run.get("rows") if isinstance(run.get("rows"), list) else []
        cost = self._read_artifact_cost(run_dir, rows)
        started_at = run.get("startedAtEpoch")
        if started_at:
            elapsed = max(0.0, time.time() - started_at)
            cost["wallClockSeconds"] = elapsed
            cost["processWallClockSeconds"] = elapsed
        run["cost"] = summarize_cost(cost)
        run["semanticArtifacts"] = self._read_semantic_artifacts(run_dir)

    def _latest_artifact_progress(self, run_dir: Path, fallback: dict[str, Any]) -> dict[str, Any]:
        logs = read_json_or_default(run_dir / "logs.json", [])
        if not isinstance(logs, list):
            return fallback
        for entry in reversed(logs):
            if not isinstance(entry, dict):
                continue
            stage = str(entry.get("stage") or "")
            message = str(entry.get("message") or "").strip()
            if not stage or not message:
                continue
            if stage == "generation":
                match = re.search(r"Generated text\s+(\d+)\s*/\s*(\d+)", message)
                if match:
                    completed = int(match.group(1))
                    total = int(match.group(2))
                    percent = 65 + (20 * completed / max(1, total))
                    return {
                        **fallback,
                        "stage": "generation",
                        "detail": message,
                        "percent": percent,
                        "stageIndex": 8,
                        "stageTotal": 10,
                        "completed": completed,
                        "total": total,
                    }
            if stage in {"objectives", "done"}:
                return {
                    **fallback,
                    "stage": stage,
                    "detail": message,
                    "percent": 88 if stage == "objectives" else 100,
                    "stageIndex": 9 if stage == "objectives" else 10,
                    "stageTotal": 10,
                }
        return fallback

    def _merge_result_into_summary(self, summary: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        merged = dict(summary)
        run_dir = Path(str(merged.get("runDir") or ""))
        rows = result.get("rows") or []
        if not rows and result.get("status") == STATUS_FAILED and run_dir.exists():
            rows = self._read_partial_generated_rows(run_dir)
        merged["status"] = result.get("status")
        merged["strategy"] = {
            "strategyId": result.get("strategyId"),
            "displayName": result.get("displayName"),
            "status": result.get("status"),
            "outputDir": result.get("outputDir"),
        }
        merged["rows"] = rows
        merged["metrics"] = result.get("metrics") or self._partial_metrics(rows)
        if run_dir.exists():
            merged["cost"] = summarize_cost(result.get("cost") or self._read_artifact_cost(run_dir, rows))
            merged["semanticArtifacts"] = self._read_semantic_artifacts(run_dir)
            artifact_logs = self._read_artifact_logs(run_dir)
            if artifact_logs:
                system_logs = [
                    entry
                    for entry in merged.get("logs", [])
                    if isinstance(entry, dict) and entry.get("source") == "system"
                ]
                merged["logs"] = (system_logs + artifact_logs)[-300:]
        else:
            merged["cost"] = summarize_cost(result.get("cost") or {})
            merged["semanticArtifacts"] = {}
        merged["error"] = result.get("error") if result.get("status") == STATUS_FAILED else None
        progress = self._latest_artifact_progress(run_dir, summary.get("progress") or {}) if run_dir.exists() else {}
        completed = result.get("status") == STATUS_COMPLETED
        merged["progress"] = {
            "percent": 100 if completed else int(max(0, min(100, round(finite_float(progress.get("percent")))))),
            "detail": status_label(str(result.get("status") or "")) if completed else self._error_message(result.get("error")),
            "stage": result.get("status"),
            "stageIndex": None,
            "stageTotal": None,
            "completed": None,
            "total": None,
            "elapsedSeconds": None,
            "elapsedLabel": "No disponible",
            "remainingSeconds": None,
            "remainingLabel": "No disponible",
        }
        merged["updatedAt"] = utc_now()
        return merged

    def _build_timeout_error(
        self,
        run: dict[str, Any],
        process_cost: dict[str, Any],
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        run_dir = Path(run["runDir"])
        cost = self._read_artifact_cost(run_dir, [])
        selected = read_json_or_default(run_dir / "selected_prompts.json", [])
        selected_count = len(selected) if isinstance(selected, list) else None
        stage = str(progress.get("stage") or "runner")
        detail = str(progress.get("detail") or "subproceso activo")
        completed = progress.get("completed")
        total = progress.get("total")
        message = f"Timeout reached after {run['config']['timeoutSeconds']} second(s) during {stage}."
        return {
            "stage": stage,
            "message": message,
            "details": {
                "detail": detail,
                "timeoutSeconds": run["config"]["timeoutSeconds"],
                "elapsedSeconds": process_cost.get("processWallClockSeconds"),
                "completed": completed,
                "total": total,
                "llmCalls": cost.get("llmCalls"),
                "generatedTexts": cost.get("generatedTexts"),
                "selectedPrompts": selected_count,
                "suggestion": "Increase timeoutSeconds or generationParallelism, or reduce n.",
            },
        }

    def _error_message(self, error: Any) -> str:
        if isinstance(error, dict):
            stage = f"{error.get('stage')}: " if error.get("stage") else ""
            message = str(error.get("message") or "Run failed.")
            details = error.get("details") if isinstance(error.get("details"), dict) else {}
            completed = details.get("completed")
            total = details.get("total")
            count_text = f" Completed {completed}/{total}." if completed is not None and total is not None else ""
            suggestion = f" {details.get('suggestion')}" if details.get("suggestion") else ""
            return f"{stage}{message}{count_text}{suggestion}".strip()
        return str(error or "Run failed.")

    def _read_semantic_artifacts(self, run_dir: Path) -> dict[str, Any]:
        anchors = read_json_or_default(run_dir / "anchors.json", {})
        pools = read_json_or_default(run_dir / "pools.json", {})
        expansion = read_json_or_default(run_dir / "pool_expansion.json", None)
        anchors = anchors if isinstance(anchors, dict) else {}
        pools = pools if isinstance(pools, dict) else {}
        components = pools.get("components") if isinstance(pools.get("components"), dict) else {}
        normalized_components: dict[str, Any] = {}
        for component_name in ("roles", "topics", "actions"):
            component = components.get(component_name) if isinstance(components, dict) else None
            if not isinstance(component, dict):
                normalized_components[component_name] = {
                    "component": component_name,
                    "requested": None,
                    "items": [],
                    "rawItems": [],
                    "discarded": None,
                }
                continue
            items = component.get("items") if isinstance(component.get("items"), list) else []
            raw_items = component.get("rawItems") if isinstance(component.get("rawItems"), list) else []
            normalized_components[component_name] = {
                "component": component_name,
                "requested": component.get("requested"),
                "items": [str(item) for item in items],
                "rawItems": [str(item) for item in raw_items],
                "discarded": component.get("discarded"),
            }
        targets = pools.get("targets") if isinstance(pools.get("targets"), dict) else {}
        return {
            "anchors": {
                key: [str(item) for item in value] if isinstance(value, list) else []
                for key, value in anchors.items()
            },
            "pools": {
                "targets": targets,
                "components": normalized_components,
            },
            "expansion": expansion,
        }

    def _read_partial_generated_rows(self, run_dir: Path) -> list[dict[str, Any]]:
        generated = read_json_or_default(run_dir / "generated_texts.json", [])
        if not isinstance(generated, list):
            return []
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(generated, start=1):
            if not isinstance(row, dict):
                continue
            partial = dict(row)
            partial["rank"] = partial.get("rank") or partial.get("selectionRank") or index
            partial["status"] = partial.get("status") or "generated_partial"
            partial.setdefault("objectiveLabel", "--")
            partial.setdefault("objectiveVector", [])
            partial.setdefault("nonDominated", False)
            rows.append(partial)
        return rows

    def _partial_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "totalRows": len(rows),
            "completedRows": 0,
            "partialRows": len(rows),
            "nonDominatedRows": 0,
            "bestObjectiveVector": [],
            "bestObjectiveLabel": "--",
            "hypervolumeLabel": "No aplica",
            "spreadLabel": "No aplica",
        }

    def _summary_needs_result_merge(self, summary: dict[str, Any], result: dict[str, Any]) -> bool:
        if summary.get("status") != result.get("status"):
            return True
        if result.get("rows") and not summary.get("rows"):
            return True
        if result.get("metrics") and not summary.get("metrics"):
            return True
        if result.get("cost") and not summary.get("cost"):
            return True

        run_dir = Path(str(summary.get("runDir") or ""))
        if not run_dir.exists():
            return False
        artifact_logs = self._read_artifact_logs(run_dir)
        summary_logs = summary.get("logs") if isinstance(summary.get("logs"), list) else []
        if not artifact_logs:
            return False
        last_message = artifact_logs[-1]["message"]
        return not any(
            isinstance(entry, dict) and entry.get("message") == last_message
            for entry in summary_logs[-10:]
        )

    def _read_artifact_logs(self, run_dir: Path) -> list[dict[str, Any]]:
        raw_logs = read_json_or_default(run_dir / "logs.json", [])
        if not isinstance(raw_logs, list):
            return []
        logs: list[dict[str, Any]] = []
        for entry in raw_logs:
            if not isinstance(entry, dict):
                continue
            stage = str(entry.get("stage") or "runner")
            message = str(entry.get("message") or "").strip()
            if not message:
                continue
            logs.append({
                "at": str(entry.get("at") or utc_now()),
                "source": "runner",
                "message": f"[{stage}] {message}",
            })
        return logs[-300:]

    def _public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in run.items()
            if key not in {"activeProcess", "startedAtEpoch", "lastSummaryWriteEpoch"}
        }

    def _write_summary_throttled_unlocked(self, run: dict[str, Any], interval_seconds: float = 1.0) -> None:
        now = time.monotonic()
        last_write = finite_float(run.get("lastSummaryWriteEpoch"))
        if now - last_write < interval_seconds:
            return
        self._write_summary_unlocked(run)

    def _write_summary_unlocked(self, run: dict[str, Any]) -> None:
        run["lastSummaryWriteEpoch"] = time.monotonic()
        write_json(Path(run["runDir"]) / "summary.json", self._public_run(run))
