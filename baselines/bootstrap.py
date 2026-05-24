from __future__ import annotations

import importlib
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path
from typing import Any


NANOSECONDS_PER_SECOND = 1_000_000_000


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def response_value(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def nested_response_value(response: Any, *keys: str) -> Any:
    current = response
    for key in keys:
        current = response_value(current, key)
        if current is None:
            return None
    return current


def seconds_from_nanoseconds(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number / NANOSECONDS_PER_SECOND


class OllamaMetrics:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.started_at = utc_now()
        self.started_perf = time.perf_counter()
        self.calls: list[dict[str, Any]] = []
        self.write()

    def add_call(self, call: dict[str, Any]) -> None:
        with self.lock:
            call["index"] = len(self.calls) + 1
            self.calls.append(call)
            self.write_locked()

    def write(self) -> None:
        with self.lock:
            self.write_locked()

    def write_locked(self) -> None:
        summary = self.summary_locked()
        payload = {
            "schemaVersion": 1,
            "pid": os.getpid(),
            "startedAt": self.started_at,
            "updatedAt": utc_now(),
            "summary": summary,
            "calls": self.calls,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def summary_locked(self) -> dict[str, Any]:
        successful = [call for call in self.calls if call.get("status") == "ok"]
        failed = [call for call in self.calls if call.get("status") == "error"]
        prompt_tokens = sum_int(call.get("promptEvalCount") for call in self.calls)
        completion_tokens = sum_int(call.get("evalCount") for call in self.calls)
        return {
            "totalCalls": len(self.calls),
            "successfulCalls": len(successful),
            "failedCalls": len(failed),
            "clientWallClockSeconds": sum_float(call.get("clientWallClockSeconds") for call in self.calls),
            "ollamaTotalDurationSeconds": sum_float(call.get("ollamaTotalDurationSeconds") for call in self.calls),
            "ollamaLoadDurationSeconds": sum_float(call.get("ollamaLoadDurationSeconds") for call in self.calls),
            "ollamaPromptEvalDurationSeconds": sum_float(call.get("ollamaPromptEvalDurationSeconds") for call in self.calls),
            "ollamaEvalDurationSeconds": sum_float(call.get("ollamaEvalDurationSeconds") for call in self.calls),
            "promptEvalCount": prompt_tokens,
            "evalCount": completion_tokens,
            "totalTokens": prompt_tokens + completion_tokens,
        }


def sum_float(values: Any) -> float:
    total = 0.0
    for value in values:
        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def sum_int(values: Any) -> int:
    total = 0
    for value in values:
        try:
            total += int(value or 0)
        except (TypeError, ValueError):
            continue
    return total


def message_chars(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if isinstance(message, dict):
            total += len(str(message.get("content") or ""))
    return total


def response_chars(response: Any) -> int:
    content = nested_response_value(response, "message", "content")
    return len(str(content or ""))


def model_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    if "model" in kwargs:
        return str(kwargs["model"])
    if args:
        return str(args[0])
    return None


def messages_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "messages" in kwargs:
        return kwargs["messages"]
    if len(args) > 1:
        return args[1]
    return None


def build_call_record(
    method: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    elapsed: float,
    status: str,
    started_seconds: float,
    finished_seconds: float,
    response: Any = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "method": method,
        "status": status,
        "model": model_from_call(args, kwargs),
        "format": str(kwargs.get("format")) if kwargs.get("format") is not None else None,
        "clientWallClockSeconds": elapsed,
        "elapsedSeconds": elapsed,
        "startedSeconds": started_seconds,
        "finishedSeconds": finished_seconds,
        "promptChars": message_chars(messages_from_call(args, kwargs)),
        "responseChars": response_chars(response),
        "startedAt": utc_now(),
    }
    if error is not None:
        record["error"] = str(error)
    for source_key, output_key in (
        ("total_duration", "ollamaTotalDurationSeconds"),
        ("load_duration", "ollamaLoadDurationSeconds"),
        ("prompt_eval_duration", "ollamaPromptEvalDurationSeconds"),
        ("eval_duration", "ollamaEvalDurationSeconds"),
    ):
        seconds = seconds_from_nanoseconds(response_value(response, source_key))
        if seconds is not None:
            record[output_key] = seconds
    for source_key, output_key in (
        ("prompt_eval_count", "promptEvalCount"),
        ("eval_count", "evalCount"),
    ):
        value = response_value(response, source_key)
        if value is not None:
            record[output_key] = value
    return record


def install_ollama_metrics() -> None:
    metrics_path = os.environ.get("BASELINE_COST_METRICS_PATH")
    if not metrics_path:
        return

    import ollama

    metrics = OllamaMetrics(Path(metrics_path))

    for class_name in ("AsyncClient", "Client"):
        client_class = getattr(ollama, class_name, None)
        if client_class is None:
            continue
        for method_name in ("chat", "generate"):
            original = getattr(client_class, method_name, None)
            if original is None or getattr(original, "_baseline_metrics_wrapped", False):
                continue
            setattr(client_class, method_name, wrap_ollama_method(metrics, f"{class_name}.{method_name}", original))


def wrap_ollama_method(metrics: OllamaMetrics, method_label: str, original: Any) -> Any:
    if method_label.startswith("AsyncClient."):
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            started_seconds = started - metrics.started_perf
            try:
                response = await original(self, *args, **kwargs)
            except Exception as error:
                finished = time.perf_counter()
                metrics.add_call(build_call_record(method_label, args, kwargs, finished - started, "error", started_seconds, finished - metrics.started_perf, error=error))
                raise
            finished = time.perf_counter()
            metrics.add_call(build_call_record(method_label, args, kwargs, finished - started, "ok", started_seconds, finished - metrics.started_perf, response=response))
            return response

        async_wrapper._baseline_metrics_wrapped = True
        return async_wrapper

    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        started_seconds = started - metrics.started_perf
        try:
            response = original(self, *args, **kwargs)
        except Exception as error:
            finished = time.perf_counter()
            metrics.add_call(build_call_record(method_label, args, kwargs, finished - started, "error", started_seconds, finished - metrics.started_perf, error=error))
            raise
        finished = time.perf_counter()
        metrics.add_call(build_call_record(method_label, args, kwargs, finished - started, "ok", started_seconds, finished - metrics.started_perf, response=response))
        return response

    sync_wrapper._baseline_metrics_wrapped = True
    return sync_wrapper


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: bootstrap.py <script> [args...]")

    script_path = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script_path.parent))
    project_root = script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    preload_modules = [
        name.strip()
        for name in os.environ.get("BASELINE_PRELOAD_MODULES", "").split(",")
        if name.strip()
    ]

    for module_name in preload_modules:
        importlib.import_module(module_name)

    install_ollama_metrics()

    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
