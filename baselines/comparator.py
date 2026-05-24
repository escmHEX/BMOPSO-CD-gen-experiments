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
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STAGE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+(.+)$")
GENERATION_RE = re.compile(r"Generaci[oó]n\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
PERCENT_RE = re.compile(r"(\d{1,3})%")
PROPOSAL_TOTALS = {proposal_id: total for proposal_id, total in (("evolmd", 6), ("evolmd-mo", 5))}
POSTHOC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_POSTHOC_MODEL: Any = None
_POSTHOC_MODEL_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def objective_label(vector: list[float]) -> str:
    if not vector:
        return "--"
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def clean_log_message(message: str) -> str:
    return ANSI_RE.sub("", message).strip()


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


def comparator_status_message(status: str) -> str:
    return {
        STATUS_QUEUED: "En cola",
        STATUS_RUNNING: "Ejecutando",
        STATUS_COMPLETED: "Completada",
        STATUS_FAILED: "Fallida",
        STATUS_CANCELLED: "Cancelada",
    }.get(status, status)


def is_dominated(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    vector = row.get("objectiveVector") or []
    if len(vector) < 2:
        return False
    for other in rows:
        if other is row:
            continue
        other_vector = other.get("objectiveVector") or []
        if len(other_vector) != len(vector):
            continue
        if all(a >= b for a, b in zip(other_vector, vector)) and any(
            a > b for a, b in zip(other_vector, vector)
        ):
            return True
    return False


def mark_non_dominated(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["nonDominated"] = len(row.get("objectiveVector") or []) >= 2 and not is_dominated(row, rows)


def is_diagnostic_dominated(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    vector = row.get("diagnosticObjectiveVector") or []
    if len(vector) < 2:
        return False
    for other in rows:
        if other is row:
            continue
        other_vector = other.get("diagnosticObjectiveVector") or []
        if len(other_vector) != len(vector):
            continue
        if all(a >= b for a, b in zip(other_vector, vector)) and any(
            a > b for a, b in zip(other_vector, vector)
        ):
            return True
    return False


def mark_posthoc_non_dominated(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["postHocNonDominated"] = (
            len(row.get("diagnosticObjectiveVector") or []) >= 2
            and not is_diagnostic_dominated(row, rows)
        )


def normalized_mo_point(row: dict[str, Any]) -> tuple[float, float] | None:
    vector = row.get("objectiveVector") or []
    if len(vector) < 2:
        return None
    fidelity = finite_float(vector[0])
    diversity = finite_float(vector[1])
    normalized_fidelity = clamp((fidelity + 1.0) / 2.0, 0.0, 1.0)
    normalized_diversity = clamp(diversity, 0.0, 1.0)
    return normalized_fidelity, normalized_diversity


def normalized_posthoc_point(row: dict[str, Any]) -> tuple[float, float] | None:
    vector = row.get("diagnosticObjectiveVector") or []
    if len(vector) < 2:
        return None
    return clamp(finite_float(vector[0]), 0.0, 1.0), clamp(finite_float(vector[1]), 0.0, 1.0)


def posthoc_embedding_model() -> Any:
    global _POSTHOC_MODEL
    with _POSTHOC_MODEL_LOCK:
        if _POSTHOC_MODEL is None:
            import torch  # noqa: F401
            from sentence_transformers import SentenceTransformer

            _POSTHOC_MODEL = SentenceTransformer(POSTHOC_EMBEDDING_MODEL)
        return _POSTHOC_MODEL


def calculate_posthoc_semantic_diversity(generated_texts: list[str]) -> list[float]:
    if len(generated_texts) <= 1:
        return [0.0] * len(generated_texts)

    import torch
    from sentence_transformers import util

    texts = [text if text.strip() else "[texto vacio]" for text in generated_texts]
    embeddings = posthoc_embedding_model().encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    similarity_matrix = util.cos_sim(embeddings, embeddings)
    scores: list[float] = []
    for index in range(len(texts)):
        sum_similarity = torch.sum(similarity_matrix[index]) - 1.0
        average_similarity = sum_similarity / (len(texts) - 1)
        scores.append(clamp(1.0 - float(average_similarity), 0.0, 1.0))
    return scores


def pareto_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points), key=lambda item: (item[0], item[1]))
    front: list[tuple[float, float]] = []
    for point in unique:
        if not any(
            other != point
            and other[0] >= point[0]
            and other[1] >= point[1]
            and (other[0] > point[0] or other[1] > point[1])
            for other in unique
        ):
            front.append(point)
    return sorted(front, key=lambda item: item[0])


def calculate_hypervolume(points: list[tuple[float, float]]) -> float | None:
    front = pareto_points(points)
    if len(front) < 1:
        return None

    collapsed: dict[float, float] = {}
    for x_value, y_value in front:
        collapsed[x_value] = max(collapsed.get(x_value, 0.0), y_value)

    ordered = sorted(collapsed.items())
    hv = 0.0
    previous_x = 0.0
    for x_value, y_value in ordered:
        if x_value > previous_x:
            hv += (x_value - previous_x) * y_value
            previous_x = x_value
    return clamp(hv, 0.0, 1.0)


def calculate_spread(points: list[tuple[float, float]]) -> float | None:
    front = pareto_points(points)
    if len(front) < 3:
        return None

    distances = [
        math.dist(front[index - 1], front[index])
        for index in range(1, len(front))
    ]
    mean_distance = sum(distances) / len(distances)
    if mean_distance <= 0:
        return 0.0

    absolute_deviation = sum(abs(distance - mean_distance) for distance in distances)
    return absolute_deviation / (len(distances) * mean_distance)


def latest_child_directory(path: Path) -> Path | None:
    if not path.exists():
        return None
    children = [child for child in path.iterdir() if child.is_dir()]
    if not children:
        return None
    return max(children, key=lambda child: child.stat().st_mtime)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def read_json_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return default


def label_from_seconds(seconds: Any) -> str:
    return format_duration(finite_float(seconds, -1.0))


def parse_runtime_file(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    runtime: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        runtime[key.strip()] = finite_float(value.strip())
    return runtime


def empty_cost_metrics() -> dict[str, Any]:
    return {
        "processWallClockSeconds": 0.0,
        "processWallClockLabel": "0s",
        "algorithmRuntimeSeconds": None,
        "algorithmRuntimeLabel": "No disponible",
        "llmCalls": 0,
        "llmSuccessfulCalls": 0,
        "llmFailedCalls": 0,
        "llmClientWallClockSeconds": 0.0,
        "llmClientWallClockLabel": "0s",
        "llmAverageCallSeconds": None,
        "llmAverageCallLabel": "No disponible",
        "ollamaTotalDurationSeconds": 0.0,
        "ollamaTotalDurationLabel": "0s",
        "promptEvalCount": 0,
        "evalCount": 0,
        "totalTokens": 0,
        "returnCode": None,
        "timedOut": False,
        "cancelled": False,
        "runtimeBreakdown": {},
    }


def build_cost_metrics(
    process_cost: dict[str, Any],
    llm_payload: dict[str, Any],
    output_dir: Path | None,
    cancelled: bool,
) -> dict[str, Any]:
    summary = llm_payload.get("summary") if isinstance(llm_payload, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    runtime = parse_runtime_file(output_dir / "runtime.txt") if output_dir else {}
    algorithm_runtime = runtime.get("total_sec")
    llm_calls = int(finite_float(summary.get("totalCalls")))
    llm_client_seconds = finite_float(summary.get("clientWallClockSeconds"))
    average_call_seconds = llm_client_seconds / llm_calls if llm_calls > 0 else None
    cost = empty_cost_metrics()
    cost.update(
        {
            "processWallClockSeconds": finite_float(process_cost.get("processWallClockSeconds")),
            "processWallClockLabel": label_from_seconds(process_cost.get("processWallClockSeconds")),
            "algorithmRuntimeSeconds": algorithm_runtime,
            "algorithmRuntimeLabel": label_from_seconds(algorithm_runtime),
            "llmCalls": llm_calls,
            "llmSuccessfulCalls": int(finite_float(summary.get("successfulCalls"))),
            "llmFailedCalls": int(finite_float(summary.get("failedCalls"))),
            "llmClientWallClockSeconds": llm_client_seconds,
            "llmClientWallClockLabel": label_from_seconds(llm_client_seconds),
            "llmAverageCallSeconds": average_call_seconds,
            "llmAverageCallLabel": label_from_seconds(average_call_seconds),
            "ollamaTotalDurationSeconds": finite_float(summary.get("ollamaTotalDurationSeconds")),
            "ollamaTotalDurationLabel": label_from_seconds(summary.get("ollamaTotalDurationSeconds")),
            "promptEvalCount": int(finite_float(summary.get("promptEvalCount"))),
            "evalCount": int(finite_float(summary.get("evalCount"))),
            "totalTokens": int(finite_float(summary.get("totalTokens"))),
            "returnCode": process_cost.get("returnCode"),
            "timedOut": bool(process_cost.get("timedOut")),
            "cancelled": cancelled,
            "runtimeBreakdown": runtime,
            "metricsPath": process_cost.get("metricsPath"),
        }
    )
    return cost


def summarize_costs(proposals: list[dict[str, Any]], run_elapsed_seconds: float) -> dict[str, Any]:
    costs = [proposal.get("cost") or empty_cost_metrics() for proposal in proposals]
    llm_calls = sum(int(cost.get("llmCalls") or 0) for cost in costs)
    llm_client_seconds = sum(finite_float(cost.get("llmClientWallClockSeconds")) for cost in costs)
    process_seconds_sum = sum(finite_float(cost.get("processWallClockSeconds")) for cost in costs)
    algorithm_seconds_sum = sum(
        finite_float(cost.get("algorithmRuntimeSeconds"))
        for cost in costs
        if cost.get("algorithmRuntimeSeconds") is not None
    )
    prompt_tokens = sum(int(cost.get("promptEvalCount") or 0) for cost in costs)
    completion_tokens = sum(int(cost.get("evalCount") or 0) for cost in costs)
    average_call_seconds = llm_client_seconds / llm_calls if llm_calls > 0 else None
    return {
        "runWallClockSeconds": run_elapsed_seconds,
        "runWallClockLabel": label_from_seconds(run_elapsed_seconds),
        "proposalWallClockSecondsSum": process_seconds_sum,
        "proposalWallClockSumLabel": label_from_seconds(process_seconds_sum),
        "algorithmRuntimeSecondsSum": algorithm_seconds_sum,
        "algorithmRuntimeSumLabel": label_from_seconds(algorithm_seconds_sum),
        "llmCalls": llm_calls,
        "llmSuccessfulCalls": sum(int(cost.get("llmSuccessfulCalls") or 0) for cost in costs),
        "llmFailedCalls": sum(int(cost.get("llmFailedCalls") or 0) for cost in costs),
        "llmClientWallClockSeconds": llm_client_seconds,
        "llmClientWallClockLabel": label_from_seconds(llm_client_seconds),
        "llmAverageCallSeconds": average_call_seconds,
        "llmAverageCallLabel": label_from_seconds(average_call_seconds),
        "ollamaTotalDurationSeconds": sum(finite_float(cost.get("ollamaTotalDurationSeconds")) for cost in costs),
        "ollamaTotalDurationLabel": label_from_seconds(sum(finite_float(cost.get("ollamaTotalDurationSeconds")) for cost in costs)),
        "promptEvalCount": prompt_tokens,
        "evalCount": completion_tokens,
        "totalTokens": prompt_tokens + completion_tokens,
    }


@dataclass(frozen=True)
class ProposalDefinition:
    proposal_id: str
    display_name: str
    repository_path: str
    description: str
    objective_names: tuple[str, ...]
    result_file: str
    single_objective: bool
    preload_modules: tuple[str, ...] = ()


PROPOSALS: tuple[ProposalDefinition, ...] = (
    ProposalDefinition(
        proposal_id="evolmd",
        display_name="EVOLMD",
        repository_path="baselines/external/evolmd",
        description="Genetic prompt evolution baseline with BERTScore fitness.",
        objective_names=("fitness",),
        result_file="data_final_evaluada.json",
        single_objective=True,
        preload_modules=("torch",),
    ),
    ProposalDefinition(
        proposal_id="evolmd-mo",
        display_name="EVOLMD-MO",
        repository_path="baselines/external/evolmd-mo",
        description="NSGA-II prompt evolution baseline with SBERT fidelity and diversity.",
        objective_names=("fidelity_sbert", "diversity_individual"),
        result_file="pareto_front.json",
        single_objective=False,
        preload_modules=("torch",),
    ),
)


class ComparatorService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_root = root / "runs" / "comparator"
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def list_proposals(self) -> list[dict[str, Any]]:
        return [
            {
                "proposalId": proposal.proposal_id,
                "displayName": proposal.display_name,
                "description": proposal.description,
                "repositoryPath": proposal.repository_path,
                "objectiveNames": list(proposal.objective_names),
                "singleObjective": proposal.single_objective,
                "available": (self.root / proposal.repository_path / "main.py").exists(),
            }
            for proposal in PROPOSALS
        ]

    def _initial_proposal_state(self, proposal: ProposalDefinition) -> dict[str, Any]:
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "status": STATUS_QUEUED,
            "stageLabel": "En cola",
            "stageIndex": 0,
            "stageTotal": PROPOSAL_TOTALS.get(proposal.proposal_id, 1),
            "generationIndex": None,
            "generationTotal": None,
            "progress": 0.0,
            "logs": 0,
            "updatedAt": utc_now(),
        }

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._read_config(payload)
        run_id = self._new_run_id()
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run = {
            "runId": run_id,
            "status": STATUS_QUEUED,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "startedAtEpoch": None,
            "runDir": str(run_dir),
            "config": config,
            "logs": [],
            "proposals": [],
            "proposalStates": {
                proposal.proposal_id: self._initial_proposal_state(proposal)
                for proposal in PROPOSALS
            },
            "progress": {
                "percent": 0,
                "detail": "Corrida en cola.",
                "elapsedSeconds": 0,
                "elapsedLabel": "0s",
                "remainingSeconds": None,
                "remainingLabel": "No disponible",
            },
            "costSummary": summarize_costs([], 0.0),
            "error": None,
            "cancelRequested": False,
            "activeProcesses": {},
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
                return self._public_run(run)

        summary_path = self.runs_root / run_id / "summary.json"
        if summary_path.exists():
            return read_json(summary_path)
        return None

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            run["cancelRequested"] = True
            run["updatedAt"] = utc_now()
            processes = list((run.get("activeProcesses") or {}).values())
            self._append_log_unlocked(run, "system", "Cancel requested. Terminating active baseline processes.")
            self._mark_queued_as_cancelled_unlocked(run)

        for process in processes:
            self._terminate_process(process)

        with self._lock:
            self._write_summary_unlocked(run)
            return self._public_run(run)

    def _read_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        reference_text = str(payload.get("referenceText", "")).strip()
        if not reference_text:
            raise ValueError("referenceText is required.")

        return {
            "referenceText": reference_text,
            "topK": self._int_between(payload.get("topK", 10), "topK", 1, 200),
            "n": self._int_between(payload.get("n", 10), "n", 1, 500),
            "generaciones": self._int_between(payload.get("generaciones", 3), "generaciones", 0, 500),
            "model": self._safe_model_name(payload.get("model", "llama3")),
            "k": self._int_between(payload.get("k", 3), "k", 1, 100),
            "probCrossover": self._float_between(payload.get("probCrossover", 0.8), "probCrossover", 0.0, 1.0),
            "probMutacion": self._float_between(payload.get("probMutacion", 0.05), "probMutacion", 0.0, 1.0),
            "proposalParallelism": self._int_between(payload.get("proposalParallelism", 1), "proposalParallelism", 1, 8),
            "timeoutMinutes": self._int_between(payload.get("timeoutMinutes", 60), "timeoutMinutes", 1, 1440),
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

    def _safe_model_name(self, value: Any) -> str:
        model = str(value or "").strip()
        if not model:
            raise ValueError("model is required.")
        return model

    def _new_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    def _run_worker(self, run_id: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = STATUS_RUNNING
            run["startedAtEpoch"] = time.time()
            run["updatedAt"] = utc_now()
            self._append_log_unlocked(run, "system", "Comparator run started.")
            self._write_summary_unlocked(run)

        try:
            parallelism = min(run["config"]["proposalParallelism"], len(PROPOSALS))
            if parallelism <= 1:
                self._run_proposals_sequential(run)
            else:
                self._run_proposals_parallel(run, parallelism)

            with self._lock:
                self._sort_proposals_unlocked(run)
                failed = [item for item in run["proposals"] if item["status"] == STATUS_FAILED]
                if run["cancelRequested"]:
                    run["status"] = STATUS_CANCELLED
                elif failed and len(failed) == len(PROPOSALS):
                    run["status"] = STATUS_FAILED
                    run["error"] = "All proposal executions failed."
                elif failed:
                    run["status"] = STATUS_FAILED
                    failed_names = ", ".join(item["displayName"] for item in failed)
                    run["error"] = f"Proposal executions failed: {failed_names}."
                else:
                    run["status"] = STATUS_COMPLETED
                run["updatedAt"] = utc_now()
                self._refresh_cost_summary_unlocked(run)
                self._append_log_unlocked(run, "system", f"Comparator run finished with status {run['status']}.")
                self._write_summary_unlocked(run)
        except Exception as error:
            with self._lock:
                run["status"] = STATUS_FAILED
                run["error"] = str(error)
                run["updatedAt"] = utc_now()
                self._refresh_cost_summary_unlocked(run)
                self._append_log_unlocked(run, "system", f"Unexpected error: {error}")
                self._write_summary_unlocked(run)

    def _run_proposals_sequential(self, run: dict[str, Any]) -> None:
        for proposal in PROPOSALS:
            with self._lock:
                if run["cancelRequested"]:
                    run["status"] = STATUS_CANCELLED
                    self._append_log_unlocked(run, "system", "Run cancelled before next proposal.")
                    self._write_summary_unlocked(run)
                    return

            result = self._execute_proposal(run, proposal)
            with self._lock:
                self._record_proposal_result_unlocked(run, result)
                self._write_summary_unlocked(run)

    def _run_proposals_parallel(self, run: dict[str, Any], parallelism: int) -> None:
        with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="comparator") as executor:
            futures: dict[Future[dict[str, Any]], ProposalDefinition] = {
                executor.submit(self._execute_proposal, run, proposal): proposal
                for proposal in PROPOSALS
            }

            for future in as_completed(futures):
                proposal = futures[future]
                if future.cancelled():
                    result = self._cancelled_result(proposal, Path(run["runDir"]) / proposal.proposal_id)
                else:
                    try:
                        result = future.result()
                    except Exception as error:
                        result = self._failed_result(
                            proposal,
                            Path(run["runDir"]) / proposal.proposal_id,
                            f"Unexpected proposal error: {error}",
                        )

                with self._lock:
                    self._record_proposal_result_unlocked(run, result)
                    if run["cancelRequested"]:
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        self._mark_queued_as_cancelled_unlocked(run)
                    self._write_summary_unlocked(run)

    def _record_proposal_result_unlocked(self, run: dict[str, Any], result: dict[str, Any]) -> None:
        run["proposals"].append(result)
        state = run["proposalStates"].setdefault(
            result["proposalId"],
            {
                "proposalId": result["proposalId"],
                "displayName": result.get("displayName", result["proposalId"]),
            },
        )
        state["status"] = result["status"]
        state["stageLabel"] = comparator_status_message(result["status"])
        state["progress"] = 1.0
        state["updatedAt"] = utc_now()
        run["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)
        self._refresh_cost_summary_unlocked(run)

    def _sort_proposals_unlocked(self, run: dict[str, Any]) -> None:
        order = {proposal.proposal_id: index for index, proposal in enumerate(PROPOSALS)}
        run["proposals"].sort(key=lambda item: order.get(item.get("proposalId"), len(order)))

    def _mark_queued_as_cancelled_unlocked(self, run: dict[str, Any]) -> None:
        for state in (run.get("proposalStates") or {}).values():
            if state.get("status") == STATUS_QUEUED:
                state["status"] = STATUS_CANCELLED
                state["stageLabel"] = "Cancelada antes de iniciar"
                state["progress"] = 1.0
                state["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)

    def _refresh_cost_summary_unlocked(self, run: dict[str, Any]) -> None:
        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        run["costSummary"] = summarize_costs(run.get("proposals") or [], elapsed)

    def _execute_proposal(self, run: dict[str, Any], proposal: ProposalDefinition) -> dict[str, Any]:
        with self._lock:
            if run.get("cancelRequested"):
                return self._cancelled_result(proposal, Path(run["runDir"]) / proposal.proposal_id)
            self._set_proposal_state_unlocked(
                run,
                proposal.proposal_id,
                STATUS_RUNNING,
                "Preparando directorio de salida",
                0.01,
            )

        proposal_dir = Path(run["runDir"]) / proposal.proposal_id
        proposal_dir.mkdir(parents=True, exist_ok=True)
        reference_path = proposal_dir / "reference.txt"
        reference_path.write_text(run["config"]["referenceText"], encoding="utf-8")
        output_base = proposal_dir / "exec"
        output_base.mkdir(parents=True, exist_ok=True)
        cost_metrics_path = proposal_dir / "cost_metrics.json"

        repository_dir = self.root / proposal.repository_path
        if not (repository_dir / "main.py").exists():
            return self._failed_result(
                proposal,
                proposal_dir,
                f"Submodule is missing main.py at {repository_dir}.",
            )

        command = [
            sys.executable,
            str(self.root / "baselines" / "bootstrap.py"),
            "main.py",
            "--n",
            str(run["config"]["n"]),
            "--generaciones",
            str(run["config"]["generaciones"]),
            "--k",
            str(run["config"]["k"]),
            "--prob-crossover",
            str(run["config"]["probCrossover"]),
            "--prob-mutacion",
            str(run["config"]["probMutacion"]),
            "--model",
            run["config"]["model"],
            "--outdir-base",
            str(output_base),
            "--texto-referencia",
            str(reference_path),
        ]

        self._append_log(run, proposal.proposal_id, "Starting baseline process.")
        process_cost = self._run_process(
            run,
            proposal.proposal_id,
            command,
            repository_dir,
            proposal.preload_modules,
            cost_metrics_path,
        )
        return_code = int(process_cost["returnCode"])
        llm_payload = read_json_or_default(cost_metrics_path, {})

        if run.get("cancelRequested"):
            cost = build_cost_metrics(process_cost, llm_payload, None, True)
            return self._cancelled_result(proposal, proposal_dir, cost)
        if return_code != 0:
            message = (
                f"Process timed out after {run['config']['timeoutMinutes']} minute(s)."
                if return_code == 124
                else f"Process exited with code {return_code}. Check dependencies, Ollama, and model availability."
            )
            cost = build_cost_metrics(process_cost, llm_payload, None, False)
            return self._failed_result(
                proposal,
                proposal_dir,
                message,
                cost,
            )

        output_dir = latest_child_directory(output_base)
        if not output_dir:
            cost = build_cost_metrics(process_cost, llm_payload, None, False)
            return self._failed_result(proposal, proposal_dir, "No output directory was created.", cost)

        result_path = output_dir / proposal.result_file
        if not result_path.exists():
            cost = build_cost_metrics(process_cost, llm_payload, output_dir, False)
            return self._failed_result(
                proposal,
                proposal_dir,
                f"Expected result file was not found: {result_path.name}.",
                cost,
            )

        try:
            rows = self._normalize_rows(proposal, read_json(result_path), run["config"]["topK"])
            metrics = self._summarize_rows(proposal, rows, output_dir)
            cost = build_cost_metrics(process_cost, llm_payload, output_dir, False)
            return {
                "proposalId": proposal.proposal_id,
                "displayName": proposal.display_name,
                "status": STATUS_COMPLETED,
                "outputDir": str(output_dir),
                "rows": rows[: run["config"]["topK"]],
                "metrics": metrics,
                "cost": cost,
                "error": None,
            }
        except Exception as error:
            cost = build_cost_metrics(process_cost, llm_payload, output_dir, False)
            return self._failed_result(proposal, proposal_dir, f"Could not normalize output: {error}", cost)

    def _set_proposal_state_unlocked(
        self,
        run: dict[str, Any],
        proposal_id: str,
        status: str,
        stage_label: str,
        progress: float | None = None,
    ) -> None:
        state = run["proposalStates"].setdefault(
            proposal_id,
            {
                "proposalId": proposal_id,
                "displayName": proposal_id,
                "stageTotal": PROPOSAL_TOTALS.get(proposal_id, 1),
            },
        )
        state["status"] = status
        state["stageLabel"] = stage_label
        if progress is not None:
            state["progress"] = clamp(progress, 0.0, 1.0)
        state["updatedAt"] = utc_now()
        run["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)

    def _apply_log_progress_unlocked(self, run: dict[str, Any], proposal_id: str, message: str) -> None:
        if proposal_id == "system":
            self._refresh_run_progress_unlocked(run)
            return

        state = run["proposalStates"].get(proposal_id)
        if not state:
            return

        state["logs"] = int(state.get("logs") or 0) + 1
        state["status"] = STATUS_RUNNING
        state["updatedAt"] = utc_now()

        stage_match = STAGE_RE.match(message)
        if stage_match:
            stage_index = int(stage_match.group(1))
            stage_total = int(stage_match.group(2))
            label = stage_match.group(3).strip()
            state["stageIndex"] = stage_index
            state["stageTotal"] = stage_total
            state["stageLabel"] = label
            state["generationIndex"] = None
            state["generationTotal"] = None
            state["progress"] = clamp((stage_index - 1) / max(stage_total, 1), 0.0, 0.98)
            self._refresh_run_progress_unlocked(run)
            return

        generation_match = GENERATION_RE.search(message)
        if generation_match:
            generation_index = int(generation_match.group(1))
            generation_total = int(generation_match.group(2))
            stage_total = max(int(state.get("stageTotal") or PROPOSAL_TOTALS.get(proposal_id, 1)), 1)
            stage_index = max(int(state.get("stageIndex") or stage_total), 1)
            base = clamp((stage_index - 1) / stage_total, 0.0, 0.98)
            state["generationIndex"] = generation_index
            state["generationTotal"] = generation_total
            state["stageLabel"] = f"Generacion {generation_index}/{generation_total}"
            state["progress"] = clamp(base + (generation_index / max(generation_total, 1)) / stage_total, 0.0, 0.98)
            self._refresh_run_progress_unlocked(run)
            return

        percent_match = PERCENT_RE.search(message)
        if percent_match:
            percent = clamp(float(percent_match.group(1)) / 100.0, 0.0, 1.0)
            stage_total = max(int(state.get("stageTotal") or PROPOSAL_TOTALS.get(proposal_id, 1)), 1)
            stage_index = max(int(state.get("stageIndex") or 1), 1)
            base = clamp((stage_index - 1) / stage_total, 0.0, 0.98)
            state["progress"] = max(float(state.get("progress") or 0.0), clamp(base + percent / stage_total, 0.0, 0.98))

        if not state.get("stageLabel") or state.get("stageLabel") == "En cola":
            state["stageLabel"] = message[:120]
        self._refresh_run_progress_unlocked(run)

    def _refresh_run_progress_unlocked(self, run: dict[str, Any]) -> None:
        states = list((run.get("proposalStates") or {}).values())
        if not states:
            return

        progress_sum = sum(float(state.get("progress") or 0.0) for state in states)
        percent = int(round(100 * progress_sum / len(states)))
        percent = int(clamp(percent, 0, 100))

        active = next((state for state in states if state.get("status") == STATUS_RUNNING), None)
        queued = sum(1 for state in states if state.get("status") == STATUS_QUEUED)
        completed = sum(1 for state in states if state.get("status") == STATUS_COMPLETED)
        failed = sum(1 for state in states if state.get("status") == STATUS_FAILED)
        cancelled = sum(1 for state in states if state.get("status") == STATUS_CANCELLED)

        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        remaining = None
        if run.get("status") == STATUS_RUNNING and percent > 0 and percent < 100:
            remaining = elapsed * (100 - percent) / percent

        if active:
            detail = f"{active.get('displayName', active.get('proposalId'))}: {active.get('stageLabel', 'Ejecutando')}"
        elif run.get("cancelRequested"):
            detail = "Cancelacion solicitada; esperando cierre de procesos activos."
        else:
            detail = f"{completed} completada(s), {failed} fallida(s), {cancelled} cancelada(s), {queued} en cola."

        run["progress"] = {
            "percent": percent,
            "detail": detail,
            "elapsedSeconds": elapsed,
            "elapsedLabel": format_duration(elapsed),
            "remainingSeconds": remaining,
            "remainingLabel": format_duration(remaining),
            "activeProposalId": active.get("proposalId") if active else None,
            "activeProposalName": active.get("displayName") if active else None,
            "queuedProposals": queued,
            "completedProposals": completed,
            "failedProposals": failed,
            "cancelledProposals": cancelled,
            "totalProposals": len(states),
        }

    def _run_process(
        self,
        run: dict[str, Any],
        proposal_id: str,
        command: list[str],
        cwd: Path,
        preload_modules: tuple[str, ...],
        cost_metrics_path: Path,
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["BASELINE_COST_METRICS_PATH"] = str(cost_metrics_path)
        if preload_modules:
            environment["BASELINE_PRELOAD_MODULES"] = ",".join(preload_modules)
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        process_started = time.perf_counter()
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
            run["activeProcesses"][proposal_id] = process

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
        timeout_seconds = run["config"]["timeoutMinutes"] * 60
        deadline = time.monotonic() + timeout_seconds
        reader_done = False
        termination_requested = False

        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                line = ""

            if line is None:
                reader_done = True
            elif line:
                message = clean_log_message(line)
                if message:
                    self._append_log(run, proposal_id, message)

            if run.get("cancelRequested") and process.poll() is None and not termination_requested:
                self._append_log(run, proposal_id, "Cancellation requested; terminating process tree.")
                self._terminate_process(process)
                termination_requested = True

            if process.poll() is None and time.monotonic() >= deadline and not termination_requested:
                self._append_log(run, proposal_id, f"Timeout reached after {run['config']['timeoutMinutes']} minute(s).")
                self._terminate_process(process)
                termination_requested = True
                timed_out = True
                return_code = 124
                break

            if reader_done and process.poll() is not None:
                return_code = process.returncode
                break

            if line is None and process.poll() is None:
                continue

        reader.join(timeout=2)
        with self._lock:
            if run.get("activeProcesses", {}).get(proposal_id) is process:
                run["activeProcesses"].pop(proposal_id, None)
        return {
            "returnCode": return_code,
            "processWallClockSeconds": time.perf_counter() - process_started,
            "timedOut": timed_out,
            "metricsPath": str(cost_metrics_path),
        }

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

    def _normalize_rows(
        self,
        proposal: ProposalDefinition,
        raw_rows: Any,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_rows, list):
            raise ValueError("Expected a JSON list.")

        rows = [
            self._normalize_evolmd_row(proposal, row, index)
            if proposal.single_objective
            else self._normalize_evolmd_mo_row(proposal, row, index)
            for index, row in enumerate(raw_rows, start=1)
            if isinstance(row, dict)
        ]
        if proposal.single_objective:
            self._attach_evolmd_posthoc_diagnostics(rows)

        mark_non_dominated(rows)
        if proposal.single_objective:
            rows.sort(key=lambda row: row["objectiveVector"][0], reverse=True)
        else:
            rows.sort(
                key=lambda row: (
                    1 if row.get("nonDominated") else 0,
                    sum(row.get("objectiveVector") or []),
                ),
                reverse=True,
            )

        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row["shownInTopK"] = rank <= top_k
        return rows

    def _attach_evolmd_posthoc_diagnostics(self, rows: list[dict[str, Any]]) -> None:
        scores = calculate_posthoc_semantic_diversity([row.get("generatedText") or "" for row in rows])
        for row, diversity_score in zip(rows, scores):
            diagnostic_vector = [finite_float(row["objectiveVector"][0]), diversity_score]
            row["diagnosticObjectiveVector"] = diagnostic_vector
            row["diagnosticObjectiveLabel"] = objective_label(diagnostic_vector)
            row["diagnosticObjectiveNames"] = ["fitness", "semantic_diversity_posthoc"]
            row["postHocDiagnostics"] = {
                "semanticDiversity": diversity_score,
                "semanticDiversityModel": POSTHOC_EMBEDDING_MODEL,
                "note": "Diagnostic only; EVOLMD selection remains single-objective.",
            }
        mark_posthoc_non_dominated(rows)

    def _normalize_evolmd_row(
        self,
        proposal: ProposalDefinition,
        row: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        vector = [finite_float(row.get("fitness"))]
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "sourceIndex": index,
            "rank": index,
            "generatedText": str(row.get("generated_data") or ""),
            "prompt": str(row.get("prompt") or ""),
            "objectiveVector": vector,
            "objectiveLabel": objective_label(vector),
            "objectiveNames": list(proposal.objective_names),
            "status": "ok" if row.get("generated_data") else "empty_generated_data",
            "nonDominated": False,
            "raw": {
                "role": row.get("role"),
                "topic": row.get("topic"),
                "keywords": row.get("keywords"),
            },
        }

    def _normalize_evolmd_mo_row(
        self,
        proposal: ProposalDefinition,
        row: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        objetivos = row.get("objetivos") if isinstance(row.get("objetivos"), list) else []
        metrics_detail = row.get("metrics_detail") if isinstance(row.get("metrics_detail"), dict) else {}
        vector = [
            finite_float(objetivos[0] if len(objetivos) > 0 else metrics_detail.get("fidelity_sbert")),
            finite_float(objetivos[1] if len(objetivos) > 1 else metrics_detail.get("diversity_individual")),
        ]
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "sourceIndex": index,
            "rank": index,
            "generatedText": str(row.get("generated_data") or ""),
            "prompt": str(row.get("prompt") or ""),
            "objectiveVector": vector,
            "objectiveLabel": objective_label(vector),
            "objectiveNames": list(proposal.objective_names),
            "status": "ok" if row.get("generated_data") else "empty_generated_data",
            "nonDominated": False,
            "raw": {
                "role": row.get("role"),
                "topic": row.get("topic"),
                "keywords": row.get("keywords"),
                "metricsDetail": metrics_detail,
            },
        }

    def _summarize_rows(
        self,
        proposal: ProposalDefinition,
        rows: list[dict[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        completed = [row for row in rows if row.get("status") == "ok"]
        best_vector = completed[0]["objectiveVector"] if completed else []
        metrics: dict[str, Any] = {
            "totalRows": len(rows),
            "completedRows": len(completed),
            "objectiveNames": list(proposal.objective_names),
            "bestObjectiveVector": best_vector,
            "bestObjectiveLabel": objective_label(best_vector),
            "nonDominatedRows": sum(1 for row in rows if row.get("nonDominated")),
            "hypervolume": None,
            "hypervolumeLabel": "No aplica",
            "spread": None,
            "spreadLabel": "No aplica",
            "outputDir": str(output_dir),
        }

        if proposal.single_objective:
            diagnostic_rows = [row for row in rows if row.get("diagnosticObjectiveVector")]
            diagnostic_points = [
                point
                for row in diagnostic_rows
                if row.get("postHocNonDominated")
                for point in [normalized_posthoc_point(row)]
                if point is not None
            ]
            hypervolume = calculate_hypervolume(diagnostic_points)
            spread = calculate_spread(diagnostic_points)
            best_diagnostic = (
                max(
                    diagnostic_rows,
                    key=lambda row: sum(row.get("diagnosticObjectiveVector") or []),
                ).get("diagnosticObjectiveVector")
                if diagnostic_rows
                else []
            )
            metrics["postHocDiagnostic"] = True
            metrics["diagnosticObjectiveNames"] = ["fitness", "semantic_diversity_posthoc"]
            metrics["bestDiagnosticObjectiveVector"] = best_diagnostic
            metrics["bestDiagnosticObjectiveLabel"] = objective_label(best_diagnostic)
            metrics["postHocNonDominatedRows"] = sum(1 for row in rows if row.get("postHocNonDominated"))
            metrics["hypervolume"] = hypervolume
            metrics["hypervolumeLabel"] = f"{hypervolume:.6f}" if hypervolume is not None else "No aplica"
            metrics["spread"] = spread
            metrics["spreadLabel"] = f"{spread:.6f}" if spread is not None else "No aplica"
            metrics["moConvention"] = (
                "Post-hoc diagnostic only; EVOLMD optimized fitness as a single objective. "
                "Semantic diversity is 1 - average cosine similarity over SBERT embeddings. "
                "HV uses [fitness, diversity] in [0, 1] with reference point [0, 0]; "
                "spread is normalized consecutive-distance deviation over the diagnostic non-dominated set."
            )
        else:
            points = [
                point
                for row in rows
                if row.get("nonDominated")
                for point in [normalized_mo_point(row)]
                if point is not None
            ]
            hypervolume = calculate_hypervolume(points)
            spread = calculate_spread(points)
            metrics["hypervolume"] = hypervolume
            metrics["hypervolumeLabel"] = f"{hypervolume:.6f}" if hypervolume is not None else "No aplica"
            metrics["spread"] = spread
            metrics["spreadLabel"] = f"{spread:.6f}" if spread is not None else "No aplica"
            metrics["moConvention"] = (
                "Maximization; fidelity normalized with (fidelity + 1) / 2; "
                "diversity clamped to [0, 1]; HV reference point [0, 0]; "
                "spread is normalized consecutive-distance deviation, lower is better."
            )

        return metrics

    def _failed_result(
        self,
        proposal: ProposalDefinition,
        proposal_dir: Path,
        message: str,
        cost: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "status": STATUS_FAILED,
            "outputDir": str(proposal_dir),
            "rows": [],
            "metrics": {
                "totalRows": 0,
                "completedRows": 0,
                "objectiveNames": list(proposal.objective_names),
                "bestObjectiveVector": [],
                "bestObjectiveLabel": "--",
                "nonDominatedRows": 0,
                "hypervolume": None,
                "hypervolumeLabel": "No aplica",
                "spread": None,
                "spreadLabel": "No aplica",
                "outputDir": str(proposal_dir),
            },
            "cost": cost or empty_cost_metrics(),
            "error": message,
        }

    def _cancelled_result(
        self,
        proposal: ProposalDefinition,
        proposal_dir: Path,
        cost: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self._failed_result(proposal, proposal_dir, "Execution was cancelled.", cost)
        result["status"] = STATUS_CANCELLED
        result["cost"]["cancelled"] = True
        return result

    def _append_log(self, run: dict[str, Any], proposal_id: str, message: str) -> None:
        with self._lock:
            self._append_log_unlocked(run, proposal_id, message)
            self._write_summary_unlocked(run)

    def _append_log_unlocked(self, run: dict[str, Any], proposal_id: str, message: str) -> None:
        clean_message = clean_log_message(message)
        if not clean_message:
            return
        run["logs"].append(
            {
                "at": utc_now(),
                "proposalId": proposal_id,
                "message": clean_message,
            }
        )
        run["logs"] = run["logs"][-250:]
        run["updatedAt"] = utc_now()
        self._apply_log_progress_unlocked(run, proposal_id, clean_message)

    def _public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in run.items()
            if key not in {"activeProcesses", "startedAtEpoch"}
        }

    def _write_summary_unlocked(self, run: dict[str, Any]) -> None:
        write_json(Path(run["runDir"]) / "summary.json", self._public_run(run))
