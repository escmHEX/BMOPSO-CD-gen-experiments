from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_studio import LmStudioClient, LmStudioHttpError
from sbert_service import shared_sbert_service


PROGRESS_PREFIX = "__INITIAL_POPULATION_PROGRESS__"
DOMAIN_DEFAULT = "Natural-disaster and emergency scenario messages."
GENERIC_VALUES = {
    "unknown",
    "other",
    "misc",
    "general",
    "generic role",
    "generic topic",
    "generic action",
}


class StrategyExecutionError(Exception):
    def __init__(self, stage: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


@dataclass
class LlmCallResult:
    text: str
    payload: dict[str, Any]
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RunRecorder:
    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.config = config
        self.started = time.perf_counter()
        self.logs: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.embedding_batches: list[dict[str, Any]] = []
        self._write_lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "config.json", config)

    def log(self, stage: str, message: str) -> None:
        entry = {"at": utc_now(), "stage": stage, "message": message}
        self.logs.append(entry)
        print(f"[{stage}] {message}", flush=True)
        write_json(self.output_dir / "logs.json", self.logs)

    def progress(
        self,
        stage: str,
        detail: str,
        percent: float,
        stage_index: int,
        stage_total: int,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        payload = {
            "stage": stage,
            "detail": detail,
            "percent": int(max(0, min(100, round(percent)))),
            "stageIndex": stage_index,
            "stageTotal": stage_total,
            "completed": completed,
            "total": total,
            "elapsedSeconds": time.perf_counter() - self.started,
        }
        print(f"{PROGRESS_PREFIX}{json.dumps(payload, ensure_ascii=False)}", flush=True)
        self.log(stage, detail)

    def add_llm_call(self, call: dict[str, Any]) -> None:
        with self._write_lock:
            call["index"] = len(self.llm_calls) + 1
            self.llm_calls.append(call)
            write_json(self.output_dir / "llm_calls.json", self.llm_calls)

    def add_embedding_batch(self, batch: dict[str, Any]) -> None:
        with self._write_lock:
            batch["index"] = len(self.embedding_batches) + 1
            self.embedding_batches.append(batch)
            write_json(self.output_dir / "embedding_batches.json", self.embedding_batches)

    def cost_summary(self, final_individuals: int, generated_texts: int) -> dict[str, Any]:
        wall_clock = time.perf_counter() - self.started
        llm_success = [call for call in self.llm_calls if call.get("status") == "ok"]
        llm_failures = [call for call in self.llm_calls if call.get("status") == "error"]
        llm_seconds = sum(float(call.get("elapsedSeconds") or 0.0) for call in self.llm_calls)
        llm_concurrent_seconds = interval_union_seconds(self.llm_calls)
        if llm_concurrent_seconds is None:
            llm_concurrent_seconds = min(llm_seconds, wall_clock)
        estimated_sequential_wall = wall_clock + max(0.0, llm_seconds - llm_concurrent_seconds)
        parallelism_savings = max(0.0, estimated_sequential_wall - wall_clock)
        embedding_seconds = sum(float(batch.get("elapsedSeconds") or 0.0) for batch in self.embedding_batches)
        llm_calls = len(self.llm_calls)
        final_count = max(final_individuals, 0)
        generated_count = max(generated_texts, 0)
        return {
            "wallClockSeconds": wall_clock,
            "estimatedSequentialWallClockSeconds": estimated_sequential_wall,
            "parallelismSavingsSeconds": parallelism_savings,
            "parallelismSavingsPercent": (parallelism_savings / estimated_sequential_wall * 100.0) if estimated_sequential_wall > 0 else None,
            "llmCalls": llm_calls,
            "llmSuccessfulCalls": len(llm_success),
            "llmFailedCalls": len(llm_failures),
            "llmClientWallClockSeconds": llm_seconds,
            "llmConcurrentWallClockSeconds": llm_concurrent_seconds,
            "llmAverageCallSeconds": llm_seconds / llm_calls if llm_calls else None,
            "embeddingBatches": len(self.embedding_batches),
            "embeddingTexts": sum(int(batch.get("textCount") or 0) for batch in self.embedding_batches),
            "embeddingWallClockSeconds": embedding_seconds,
            "embeddingAverageBatchSeconds": embedding_seconds / len(self.embedding_batches) if self.embedding_batches else None,
            "promptTokens": sum(int(call.get("promptTokens") or 0) for call in self.llm_calls),
            "completionTokens": sum(int(call.get("completionTokens") or 0) for call in self.llm_calls),
            "totalTokens": sum(int(call.get("totalTokens") or 0) for call in self.llm_calls),
            "generatedTexts": generated_count,
            "finalIndividuals": final_count,
            "llmCallsPerFinalIndividual": llm_calls / final_count if final_count else None,
            "llmCallsPerGeneratedText": llm_calls / generated_count if generated_count else None,
            "wallClockPerFinalIndividualSeconds": wall_clock / final_count if final_count else None,
        }


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def word_count(value: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", value, flags=re.UNICODE))


def render_template(template: str, variables: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = variables.get(key, match.group(0))
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    return re.sub(r"\{([A-Za-z0-9_]+)\}", replace, template)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON response must be an object.")
    return payload


def clean_generated_text(raw: str) -> str:
    text = re.sub(r"```[A-Za-z]*\n?", "", raw)
    text = text.replace("```", "")
    text = text.strip().strip("\"'`").strip()
    return text


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


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


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def cosine_label(vector: list[float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def call_lm_studio(
    config: dict[str, Any],
    stage_name: str,
    system_prompt: str,
    user_prompt: str,
    recorder: RunRecorder,
) -> LlmCallResult:
    lm_studio = config["lmStudio"]
    stage = config["stages"][stage_name]
    started_offset = time.perf_counter() - recorder.started
    started = time.perf_counter()
    client = LmStudioClient(
        lm_studio["baseUrl"],
        lm_studio["apiMode"],
        stage["model"],
        float(config["timeoutSeconds"]),
    )
    try:
        response = client.call(
            user_prompt,
            system_prompt=system_prompt,
            temperature=float(stage["temperature"]),
            top_p=float(stage["topP"]),
            top_k=int(stage["topK"]) if stage.get("topK") is not None else None,
            max_tokens=int(stage["maxTokens"]),
        )
    except LmStudioHttpError as error:
        elapsed = time.perf_counter() - started
        finished_offset = started_offset + elapsed
        recorder.add_llm_call(
            {
                "stage": stage_name,
                "status": "error",
                "model": stage["model"],
                "startedSeconds": started_offset,
                "finishedSeconds": finished_offset,
                "elapsedSeconds": elapsed,
                "promptChars": len(system_prompt) + len(user_prompt),
                "error": f"HTTP {error.status_code}: {error.response_text[:500]}",
            }
        )
        raise StrategyExecutionError(stage_name, f"LM Studio returned HTTP {error.status_code}.", {"response": error.response_text[:1000]}) from error
    except Exception as error:
        elapsed = time.perf_counter() - started
        finished_offset = started_offset + elapsed
        recorder.add_llm_call(
            {
                "stage": stage_name,
                "status": "error",
                "model": stage["model"],
                "startedSeconds": started_offset,
                "finishedSeconds": finished_offset,
                "elapsedSeconds": elapsed,
                "promptChars": len(system_prompt) + len(user_prompt),
                "error": str(error),
            }
        )
        raise StrategyExecutionError(stage_name, f"LM Studio call failed: {error}") from error

    elapsed = time.perf_counter() - started
    finished_offset = started_offset + elapsed
    text = str(response.get("text") or "")
    recorder.add_llm_call(
        {
            "stage": stage_name,
            "status": "ok",
            "model": stage["model"],
            "startedSeconds": started_offset,
            "finishedSeconds": finished_offset,
            "elapsedSeconds": elapsed,
            "promptChars": len(system_prompt) + len(user_prompt),
            "responseChars": len(text),
            "promptTokens": response["promptTokens"],
            "completionTokens": response["completionTokens"],
            "totalTokens": response["totalTokens"],
        }
    )
    return LlmCallResult(
        text,
        response.get("payload") if isinstance(response.get("payload"), dict) else {},
        elapsed,
        int(response["promptTokens"]),
        int(response["completionTokens"]),
        int(response["totalTokens"]),
    )


def choose_pool_sizes(n: int) -> dict[str, int]:
    base = max(1, math.ceil((1.25 * n) ** (1 / 3)))
    sizes = {
        "topics": base,
        "actions": math.ceil(1.6 * base),
        "roles": 2 * base,
    }
    order = ["roles", "actions", "topics"]
    cursor = 0
    while sizes["roles"] * sizes["topics"] * sizes["actions"] < 4 * n:
        sizes[order[cursor % len(order)]] += 1
        cursor += 1
    return sizes


def validate_string_list(items: Any, max_words: int) -> list[str]:
    if not isinstance(items, list):
        raise ValueError("Expected a JSON list.")
    clean: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        value = re.sub(r"\s+", " ", item.strip())
        key = normalize_text(value)
        if not value or "\n" in item or key in seen or key in GENERIC_VALUES:
            continue
        if word_count(value) > max_words:
            continue
        seen.add(key)
        clean.append(value)
    return clean


def validate_anchors(payload: dict[str, Any]) -> dict[str, list[str]]:
    expected = ("entities", "topics", "actions", "constraints")
    if set(payload.keys()) != set(expected):
        raise ValueError(f"Expected keys {', '.join(expected)}.")
    return {key: validate_string_list(payload[key], 8) for key in expected}


def extract_anchors(config: dict[str, Any], recorder: RunRecorder) -> dict[str, list[str]]:
    stage = config["stages"]["anchors"]
    user_prompt = render_template(stage["userPrompt"], {
        "reference_text": config["referenceText"],
        "domain": config["domain"],
    })
    result = call_lm_studio(config, "anchors", stage["systemPrompt"], user_prompt, recorder)
    try:
        anchors = validate_anchors(extract_json_object(result.text))
    except Exception as error:
        raise StrategyExecutionError("anchors", f"Invalid anchors JSON: {error}", {"rawOutput": result.text}) from error
    write_json(recorder.output_dir / "anchors.json", anchors)
    return anchors


def generate_pool(
    component: str,
    config: dict[str, Any],
    anchors: dict[str, list[str]],
    requested_count: int,
    recorder: RunRecorder,
) -> dict[str, Any]:
    stage_name = component
    stage = config["stages"][stage_name]
    user_prompt = render_template(stage["userPrompt"], {
        "reference_text": config["referenceText"],
        "domain": config["domain"],
        "component_type": component,
        "q_component": requested_count,
        "anchors": anchors,
    })
    result = call_lm_studio(config, stage_name, stage["systemPrompt"], user_prompt, recorder)
    try:
        payload = extract_json_object(result.text)
        raw_items = payload.get("items")
    except Exception as error:
        raise StrategyExecutionError(stage_name, f"Invalid pool JSON for {component}: {error}", {"rawOutput": result.text}) from error

    max_words = config["validation"][f"{component}MaxWords"]
    try:
        items = validate_string_list(raw_items, max_words)
    except Exception as error:
        raise StrategyExecutionError(stage_name, f"Invalid item list for {component}: {error}", {"rawOutput": result.text}) from error

    return {
        "component": component,
        "requested": requested_count,
        "rawItems": raw_items if isinstance(raw_items, list) else [],
        "items": items,
        "discarded": max(0, len(raw_items if isinstance(raw_items, list) else []) - len(items)),
    }


def merge_unique(existing: list[str], new_items: list[str]) -> list[str]:
    seen = {normalize_text(item) for item in existing}
    merged = list(existing)
    for item in new_items:
        key = normalize_text(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def expand_pool_once(
    pools: dict[str, list[str]],
    config: dict[str, Any],
    anchors: dict[str, list[str]],
    recorder: RunRecorder,
) -> dict[str, Any] | None:
    n = config["n"]
    product = len(pools["roles"]) * len(pools["topics"]) * len(pools["actions"])
    if product >= 3 * n:
        return None

    for component in ("roles", "actions", "topics"):
        other_components = [name for name in ("roles", "topics", "actions") if name != component]
        denominator = len(pools[other_components[0]]) * len(pools[other_components[1]])
        if denominator <= 0:
            continue
        required_total = math.ceil((3 * n) / denominator)
        q_extra = max(1, required_total - len(pools[component]))
        stage = config["stages"]["expansion"]
        user_prompt = render_template(stage["userPrompt"], {
            "reference_text": config["referenceText"],
            "domain": config["domain"],
            "component_type": component,
            "q_extra": q_extra,
            "existing_items": pools[component],
            "anchors": anchors,
        })
        result = call_lm_studio(config, "expansion", stage["systemPrompt"], user_prompt, recorder)
        try:
            payload = extract_json_object(result.text)
            raw_items = payload.get("items")
            max_words = config["validation"][f"{component}MaxWords"]
            new_items = validate_string_list(raw_items, max_words)
        except Exception as error:
            raise StrategyExecutionError("expansion", f"Invalid expansion JSON for {component}: {error}", {"rawOutput": result.text}) from error
        pools[component] = merge_unique(pools[component], new_items)
        return {
            "component": component,
            "requested": q_extra,
            "rawItems": raw_items if isinstance(raw_items, list) else [],
            "items": new_items,
        }

    raise StrategyExecutionError("expansion", "Cannot expand pools because at least two validated pools are empty.")


def build_prompt(candidate: dict[str, str], config: dict[str, Any]) -> str:
    component_list = f"role = {candidate['role']}; topic = {candidate['topic']}; action = {candidate['action']}"
    return render_template(config["promptTemplate"], {
        "role": candidate["role"],
        "topic": candidate["topic"],
        "action": candidate["action"],
        "component_list": component_list,
        "reference_text": config["referenceText"],
        "domain": config["domain"],
    })


def stratified_sample(pools: dict[str, list[str]], max_size: int, seed: int) -> list[dict[str, str]]:
    all_candidates = [
        {"role": role, "topic": topic, "action": action}
        for role in pools["roles"]
        for topic in pools["topics"]
        for action in pools["actions"]
    ]
    if len(all_candidates) <= max_size:
        return all_candidates

    rng = random.Random(seed)
    remaining = all_candidates[:]
    rng.shuffle(remaining)
    counts = {
        "roles": {item: 0 for item in pools["roles"]},
        "topics": {item: 0 for item in pools["topics"]},
        "actions": {item: 0 for item in pools["actions"]},
    }
    selected: list[dict[str, str]] = []
    while remaining and len(selected) < max_size:
        best_score = min(
            counts["roles"][candidate["role"]]
            + counts["topics"][candidate["topic"]]
            + counts["actions"][candidate["action"]]
            for candidate in remaining
        )
        best_indices = [
            index
            for index, candidate in enumerate(remaining)
            if counts["roles"][candidate["role"]]
            + counts["topics"][candidate["topic"]]
            + counts["actions"][candidate["action"]] == best_score
        ]
        chosen_index = rng.choice(best_indices)
        candidate = remaining.pop(chosen_index)
        selected.append(candidate)
        counts["roles"][candidate["role"]] += 1
        counts["topics"][candidate["topic"]] += 1
        counts["actions"][candidate["action"]] += 1
    return selected


def load_embedding_model(config: dict[str, Any], recorder: RunRecorder) -> Any:
    recorder.progress("embedding_model", f"Loading embedding model {config['embeddingModel']}.", 56, 7, 10)
    recorder.log("embeddings", f"Loading embedding model {config['embeddingModel']}.")
    model = shared_sbert_service().model_wrapper(config["embeddingModel"])
    shared_sbert_service().warmup(config["embeddingModel"])
    recorder.progress("embedding_model", f"Embedding model {config['embeddingModel']} ready.", 58, 7, 10)
    return model


def encode_texts(model: Any, texts: list[str], stage: str, recorder: RunRecorder) -> Any:
    started = time.perf_counter()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    elapsed = time.perf_counter() - started
    recorder.add_embedding_batch({
        "stage": stage,
        "model": getattr(model, "model_name", None),
        "sourceModel": getattr(model, "source_model", None),
        "textCount": len(texts),
        "elapsedSeconds": elapsed,
    })
    return embeddings


def select_diverse_candidates(candidates: list[dict[str, Any]], model: Any, n: int, recorder: RunRecorder) -> list[dict[str, Any]]:
    import numpy as np

    prompts = [candidate["prompt"] for candidate in candidates]
    recorder.progress("prompt_diversity", f"Encoding {len(prompts)} candidate prompts with SBERT.", 60, 7, 10)
    embeddings = encode_texts(model, prompts, "prompt_diversity", recorder)
    recorder.progress("prompt_diversity", "Computing prompt distance matrix.", 62, 7, 10)
    similarity = np.matmul(embeddings, embeddings.T)
    distances = 1.0 - similarity
    target = min(2 * n, len(candidates))
    if target <= 0:
        return []

    recorder.progress("prompt_diversity", f"Selecting {target} diverse prompts.", 63, 7, 10)
    average_distances = distances.mean(axis=1)
    selected_indices = [int(np.argmax(average_distances))]
    min_distances = distances[selected_indices[0]].copy()
    while len(selected_indices) < target:
        min_distances[selected_indices] = -1.0
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)
        min_distances = np.minimum(min_distances, distances[next_index])

    for index, candidate in enumerate(candidates):
        candidate["promptAverageDistance"] = float(average_distances[index])
    recorder.progress("prompt_diversity", f"Selected {len(selected_indices)} diverse prompts.", 64, 7, 10, len(selected_indices), target)
    return [{**candidates[index], "selectionRank": rank} for rank, index in enumerate(selected_indices, start=1)]


def render_generation_user_prompt(stage: dict[str, Any], prompt: str, candidate: dict[str, Any], config: dict[str, Any]) -> str:
    return render_template(stage["userPrompt"], {
        "prompt": prompt,
        "role": candidate["role"],
        "topic": candidate["topic"],
        "action": candidate["action"],
        "reference_text": config["referenceText"],
        "domain": config["domain"],
    })


def generate_texts(selected: list[dict[str, Any]], config: dict[str, Any], recorder: RunRecorder) -> list[dict[str, Any]]:
    stage = config["stages"]["generation"]
    parallelism = min(max(1, int(config["generationParallelism"])), max(1, len(selected)))
    results: list[dict[str, Any] | None] = [None] * len(selected)

    def persist_partial_results() -> None:
        write_json(recorder.output_dir / "generated_texts.json", [item for item in results if item is not None])

    def worker(index: int, candidate: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        user_prompt = render_generation_user_prompt(stage, candidate["prompt"], candidate, config)
        result = call_lm_studio(config, "generation", stage["systemPrompt"], user_prompt, recorder)
        generated_text = clean_generated_text(result.text)
        return index, {
            **candidate,
            "generationPrompt": user_prompt,
            "generatedText": generated_text,
            "rawGeneratedText": result.text,
            "generationElapsedSeconds": result.elapsed_seconds,
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="initial-generation") as executor:
        futures = {executor.submit(worker, index, candidate): index for index, candidate in enumerate(selected)}
        for future in as_completed(futures):
            index, item = future.result()
            results[index] = item
            completed += 1
            persist_partial_results()
            recorder.progress(
                "generation",
                f"Generated text {completed}/{len(selected)}.",
                65 + (20 * completed / max(1, len(selected))),
                7,
                10,
                completed,
                len(selected),
            )

    persist_partial_results()
    return [item for item in results if item is not None]


def validate_generated_texts(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    accepted: set[str] = set()
    reference_key = normalize_text(config["referenceText"])
    minimum = config["validation"]["generatedMinWords"]
    maximum = config["validation"]["generatedMaxWords"]
    for row in rows:
        text = row.get("generatedText") or ""
        key = normalize_text(text)
        words = word_count(text)
        if not text:
            row["status"] = "empty_generated_text"
        elif words < minimum or words > maximum:
            row["status"] = "invalid_length"
        elif key == reference_key:
            row["status"] = "literal_copy_reference"
        elif key in accepted:
            row["status"] = "duplicate_generated_text"
        else:
            row["status"] = "ok"
            accepted.add(key)
    return [row for row in rows if row.get("status") == "ok"]


def mark_non_dominated(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        vector = row.get("objectiveVector") or []
        if len(vector) < 2:
            row["nonDominated"] = False
            continue
        row["nonDominated"] = not any(
            other is not row
            and len(other.get("objectiveVector") or []) == len(vector)
            and all(a >= b for a, b in zip(other["objectiveVector"], vector))
            and any(a > b for a, b in zip(other["objectiveVector"], vector))
            for other in rows
        )


def pareto_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points), key=lambda point: (point[0], point[1]))
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
    return sorted(front, key=lambda point: point[0])


def calculate_hypervolume(points: list[tuple[float, float]]) -> float | None:
    front = pareto_points(points)
    if not front:
        return None
    collapsed: dict[float, float] = {}
    for x_value, y_value in front:
        collapsed[x_value] = max(collapsed.get(x_value, 0.0), y_value)
    hv = 0.0
    previous_x = 0.0
    for x_value, y_value in sorted(collapsed.items()):
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
    return sum(abs(distance - mean_distance) for distance in distances) / (len(distances) * mean_distance)


def evaluate_population(valid_rows: list[dict[str, Any]], config: dict[str, Any], model: Any, recorder: RunRecorder) -> list[dict[str, Any]]:
    import numpy as np

    texts = [config["referenceText"], *[row["generatedText"] for row in valid_rows]]
    recorder.progress("objectives", f"Encoding {len(texts)} texts for fidelity.", 89, 9, 10)
    embeddings = encode_texts(model, texts, "generated_fidelity", recorder)
    reference_embedding = embeddings[0]
    generated_embeddings = embeddings[1:]
    fidelities = np.matmul(generated_embeddings, reference_embedding)

    for row, fidelity in zip(valid_rows, fidelities):
        row["fidelity"] = float(fidelity)

    valid_rows.sort(key=lambda row: (row["fidelity"], row.get("promptAverageDistance") or 0.0), reverse=True)
    population = valid_rows[: config["n"]]
    final_texts = [row["generatedText"] for row in population]
    recorder.progress("objectives", f"Encoding {len(final_texts)} final texts for diversity.", 93, 9, 10)
    final_embeddings = encode_texts(model, final_texts, "final_diversity", recorder)
    if len(population) <= 1:
        diversities = [0.0] * len(population)
    else:
        similarity = np.matmul(final_embeddings, final_embeddings.T)
        diversities = []
        for index in range(len(population)):
            average_similarity = (float(similarity[index].sum()) - 1.0) / (len(population) - 1)
            diversities.append(clamp(1.0 - average_similarity, 0.0, 1.0))

    for rank, (row, diversity) in enumerate(zip(population, diversities), start=1):
        vector = [float(row["fidelity"]), float(diversity)]
        row["rank"] = rank
        row["diversity"] = float(diversity)
        row["objectiveVector"] = vector
        row["objectiveLabel"] = cosine_label(vector)
        row["objectiveNames"] = ["semantic_fidelity", "semantic_diversity"]

    mark_non_dominated(population)
    return population


def summarize_population(population: list[dict[str, Any]], config: dict[str, Any], recorder: RunRecorder, generated_count: int) -> dict[str, Any]:
    points = [
        (clamp((row["objectiveVector"][0] + 1.0) / 2.0, 0.0, 1.0), clamp(row["objectiveVector"][1], 0.0, 1.0))
        for row in population
        if row.get("nonDominated")
    ]
    hypervolume = calculate_hypervolume(points)
    spread = calculate_spread(points)
    best_fidelity = max(population, key=lambda row: row.get("fidelity", -2), default=None)
    best_diversity = max(population, key=lambda row: row.get("diversity", -1), default=None)
    metrics = {
        "totalRows": len(population),
        "completedRows": len([row for row in population if row.get("status") == "ok"]),
        "nonDominatedRows": sum(1 for row in population if row.get("nonDominated")),
        "bestObjectiveVector": best_fidelity.get("objectiveVector") if best_fidelity else [],
        "bestObjectiveLabel": best_fidelity.get("objectiveLabel") if best_fidelity else "--",
        "bestFidelity": best_fidelity.get("fidelity") if best_fidelity else None,
        "bestDiversity": best_diversity.get("diversity") if best_diversity else None,
        "hypervolume": hypervolume,
        "hypervolumeLabel": f"{hypervolume:.6f}" if hypervolume is not None else "No aplica",
        "spread": spread,
        "spreadLabel": f"{spread:.6f}" if spread is not None else "No aplica",
        "moConvention": "Maximization; HV uses normalized fidelity (f + 1) / 2 and diversity clamped to [0, 1] with reference [0, 0].",
    }
    return {
        "strategyId": config["strategyId"],
        "displayName": "Hybrid semantic initialization v7",
        "status": "completed",
        "rows": population,
        "metrics": metrics,
        "cost": recorder.cost_summary(len(population), generated_count),
        "outputDir": str(recorder.output_dir),
        "error": None,
    }


def run_strategy(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    recorder = RunRecorder(output_dir, config)
    recorder.progress("setup", "Starting hybrid semantic initialization.", 1, 1, 10)

    recorder.progress("anchors", "Extracting semantic anchors.", 6, 2, 10)
    anchors = extract_anchors(config, recorder)

    sizes = choose_pool_sizes(config["n"])
    write_json(output_dir / "pool_size_targets.json", sizes)
    pool_results: dict[str, dict[str, Any]] = {}
    pools: dict[str, list[str]] = {}
    for index, component in enumerate(("roles", "topics", "actions"), start=1):
        recorder.progress(component, f"Generating {component} pool ({index}/3).", 10 + index * 7, 3, 10, index, 3)
        result = generate_pool(component, config, anchors, sizes[component], recorder)
        pool_results[component] = result
        pools[component] = result["items"]

    write_json(output_dir / "pools.json", {"targets": sizes, "components": pool_results})
    product = len(pools["roles"]) * len(pools["topics"]) * len(pools["actions"])
    recorder.progress("validation", f"Validated pools produce {product} combinations.", 35, 4, 10)

    expansion_result = None
    if product < 3 * config["n"]:
        recorder.progress("expansion", "Expanding one pool because combinations are below 3N.", 40, 5, 10)
        expansion_result = expand_pool_once(pools, config, anchors, recorder)
        write_json(output_dir / "pool_expansion.json", expansion_result)
        product = len(pools["roles"]) * len(pools["topics"]) * len(pools["actions"])
        if product < 3 * config["n"]:
            raise StrategyExecutionError(
                "expansion",
                "Insufficient semantic component combinations after one expansion.",
                {
                    "required": 3 * config["n"],
                    "available": product,
                    "poolSizes": {key: len(value) for key, value in pools.items()},
                    "expansion": expansion_result,
                },
            )
    else:
        write_json(output_dir / "pool_expansion.json", None)

    recorder.progress("candidates", "Building stratified candidate set.", 47, 6, 10)
    candidates = stratified_sample(pools, 4 * config["n"], config["seed"])
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidateId"] = f"cand-{index:04d}"
        candidate["prompt"] = build_prompt(candidate, config)
    write_json(output_dir / "candidates.json", candidates)

    recorder.progress("prompt_diversity", "Selecting diverse prompts with SBERT.", 55, 7, 10)
    embedding_model = load_embedding_model(config, recorder)
    selected = select_diverse_candidates(candidates, embedding_model, config["n"], recorder)
    write_json(output_dir / "selected_prompts.json", selected)

    recorder.progress("generation", f"Generating {len(selected)} texts with LM Studio.", 65, 8, 10, 0, len(selected))
    generated_rows = generate_texts(selected, config, recorder)
    valid_rows = validate_generated_texts(generated_rows, config)
    write_json(output_dir / "generated_texts.json", generated_rows)
    if len(valid_rows) < config["n"]:
        raise StrategyExecutionError(
            "generated_text_validation",
            "Not enough valid generated texts.",
            {
                "required": config["n"],
                "valid": len(valid_rows),
                "generated": len(generated_rows),
                "statusCounts": status_counts(generated_rows),
            },
        )

    recorder.progress("objectives", "Computing fidelity, diversity and Pareto metrics.", 88, 9, 10)
    population = evaluate_population(valid_rows, config, embedding_model, recorder)
    write_json(output_dir / "population.json", population)

    result = summarize_population(population, config, recorder, len(generated_rows))
    recorder.progress("done", "Initial population run completed.", 100, 10, 10)
    write_json(output_dir / "result.json", result)
    return result


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def latest_logged_stage(output_dir: Path, fallback_stage: str) -> str:
    logs = read_json_or_default(output_dir / "logs.json", [])
    if not isinstance(logs, list):
        return fallback_stage
    for entry in reversed(logs):
        if not isinstance(entry, dict):
            continue
        stage = str(entry.get("stage") or "").strip()
        if stage:
            return stage
    return fallback_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an initial population strategy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    recorder: RunRecorder | None = None
    try:
        result = run_strategy(config, output_dir)
        return 0 if result.get("status") == "completed" else 2
    except StrategyExecutionError as error:
        recorder = RunRecorder(output_dir, config) if not (output_dir / "config.json").exists() else None
        failure = {
            "strategyId": config.get("strategyId", "hybrid-semantic-v7"),
            "displayName": "Hybrid semantic initialization v7",
            "status": "failed",
            "rows": [],
            "metrics": {},
            "cost": recorder.cost_summary(0, 0) if recorder else {},
            "outputDir": str(output_dir),
            "error": {
                "stage": error.stage,
                "message": str(error),
                "details": error.details,
            },
        }
        write_json(output_dir / "result.json", failure)
        print(f"[{error.stage}] ERROR: {error}", flush=True)
        return 2
    except Exception as error:
        stage = latest_logged_stage(output_dir, "unexpected")
        failure = {
            "strategyId": config.get("strategyId", "hybrid-semantic-v7"),
            "displayName": "Hybrid semantic initialization v7",
            "status": "failed",
            "rows": [],
            "metrics": {},
            "cost": {},
            "outputDir": str(output_dir),
            "error": {
                "stage": stage,
                "message": str(error),
                "details": {},
            },
        }
        write_json(output_dir / "result.json", failure)
        print(f"[{stage}] ERROR: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
