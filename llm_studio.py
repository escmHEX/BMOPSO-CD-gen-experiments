from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class LmStudioError(Exception):
    pass


class LmStudioHttpError(LmStudioError):
    def __init__(self, status_code: int, response_text: str) -> None:
        super().__init__(f"LM Studio returned HTTP {status_code}.")
        self.status_code = status_code
        self.response_text = response_text


@dataclass(frozen=True)
class LmStudioResult:
    text: str
    payload: dict[str, Any]
    elapsed_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "payload": self.payload,
            "elapsedSeconds": self.elapsed_seconds,
            "promptTokens": self.prompt_tokens,
            "completionTokens": self.completion_tokens,
            "totalTokens": self.total_tokens,
        }


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def lm_studio_url(base_url: str, api_mode: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if api_mode not in {"native", "openai"}:
        raise ValueError("LM Studio apiMode debe ser native u openai.")
    prefix = "/api/v1" if api_mode == "native" else "/v1"
    return f"{base}{prefix}{endpoint}"


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item if isinstance(item, str) else item.get("text") or item.get("content") or "")
            for item in content
            if isinstance(item, (str, dict))
        )
    return ""


def extract_lm_studio_text(payload: dict[str, Any], api_mode: str) -> str:
    if api_mode == "native":
        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            return output_text.strip()

        output = payload.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    parts.append(content_to_text(item.get("content")))
            joined = "\n".join(part for part in parts if part).strip()
            if joined:
                return joined

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            return content_to_text(message.get("content")).strip()
        return str(first.get("text") or "").strip()
    return ""


def usage_tokens(payload: dict[str, Any]) -> tuple[int, int, int]:
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    prompt = int(finite_float(usage.get("prompt_tokens") or usage.get("promptTokens")))
    completion = int(finite_float(usage.get("completion_tokens") or usage.get("completionTokens")))
    total = int(finite_float(usage.get("total_tokens") or usage.get("totalTokens")))
    return prompt, completion, total or prompt + completion


def normalize_lm_studio_models(payload: dict[str, Any], api_mode: str) -> list[dict[str, Any]]:
    if api_mode == "native":
        models = payload.get("models") if isinstance(payload, dict) else []
        normalized = []
        for model in models if isinstance(models, list) else []:
            if not isinstance(model, dict) or model.get("type") != "llm":
                continue
            loaded_instances = model.get("loaded_instances")
            loaded_instance = loaded_instances[0] if isinstance(loaded_instances, list) and loaded_instances else None
            loaded_instance = loaded_instance if isinstance(loaded_instance, dict) else None
            model_id = loaded_instance.get("id") if loaded_instance else model.get("key")
            if not model_id:
                continue
            label = f"{model.get('display_name') or model.get('key')}{' (cargado)' if loaded_instance else ''}"
            normalized.append({"id": model_id, "label": label, "loaded": bool(loaded_instance)})
        return normalized

    data = payload.get("data") if isinstance(payload, dict) else []
    return [
        {"id": model["id"], "label": model["id"], "loaded": True}
        for model in data if isinstance(model, dict) and model.get("id")
    ]


class LmStudioClient:
    def __init__(self, base_url: str, api_mode: str, model: str = "", timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_mode = api_mode
        self.model = model
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request_json(lm_studio_url(self.base_url, self.api_mode, "/models"), None)
        return normalize_lm_studio_models(payload, self.api_mode)

    def call(
        self,
        user_prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.4,
        top_p: float = 0.95,
        top_k: int | None = None,
        max_tokens: int = 180,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.api_mode == "native":
            url = lm_studio_url(self.base_url, self.api_mode, "/chat")
            body: dict[str, Any] = {
                "model": self.model,
                "input": f"{system_prompt}\n\nUser instruction:\n{user_prompt}" if system_prompt.strip() else user_prompt,
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
                "stream": False,
                "store": False,
            }
        else:
            url = lm_studio_url(self.base_url, self.api_mode, "/chat/completions")
            body = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False,
            }
        if top_k is not None:
            body["top_k"] = top_k

        started = time.perf_counter()
        payload = self._request_json(url, body)
        elapsed = time.perf_counter() - started
        prompt_tokens, completion_tokens, total_tokens = usage_tokens(payload)
        return LmStudioResult(
            text=extract_lm_studio_text(payload, self.api_mode),
            payload=payload,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ).as_dict()

    def _request_json(self, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body_text = error.read().decode("utf-8", errors="replace")
            raise LmStudioHttpError(error.code, body_text) from error
