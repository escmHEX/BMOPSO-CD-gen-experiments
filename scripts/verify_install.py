#!/usr/bin/env bash
''':'
PYTHON_BIN="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_BIN=".venv/Scripts/python.exe"
fi
exec "$PYTHON_BIN" "$0" "$@"
':'''
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_defaults import DEFAULT_OLLAMA_MODEL, OLLAMA_OPENAI_API_MODE, OLLAMA_OPENAI_BASE_URL


REQUIRED_COMPARATOR_PROPOSALS = ("evolmd", "evolmd-mo", "mesap", "binary-mopso-cd")
REQUIRED_INITIAL_COMPARISON_STRATEGIES = ("hybrid-semantic-v7", "evolmd-initial", "evolmd-mo-initial")
REQUIRED_OLLAMA_MODELS = ("llama3", "llama3.1:8b", "gemma4:e4b")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raw_error = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {raw_error}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Could not reach {url}: {error.reason}") from error
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def check(name: str, action) -> CheckResult:
    try:
        detail = action()
        return CheckResult(name, True, str(detail or "ok"))
    except Exception as error:
        return CheckResult(name, False, str(error))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def model_aliases(name: str) -> set[str]:
    value = str(name or "").strip()
    if not value:
        return set()
    aliases = {value}
    if ":" not in value:
        aliases.add(f"{value}:latest")
    elif value.endswith(":latest"):
        aliases.add(value.removesuffix(":latest"))
    return aliases


def check_portal_health(base_url: str, timeout: float) -> str:
    payload = request_json(base_url, "/api/portal/health", timeout=timeout)
    require(payload.get("status") == "ok", f"unexpected health payload: {payload}")
    return f"pid={payload.get('pid')} port={payload.get('port')}"


def check_lm_models(base_url: str, timeout: float, lm_base_url: str, api_mode: str) -> str:
    payload = request_json(
        base_url,
        "/api/lm-studio/models",
        method="POST",
        payload={"baseUrl": lm_base_url, "apiMode": api_mode, "timeoutSeconds": int(timeout)},
        timeout=timeout + 5,
    )
    models = payload.get("models")
    require(isinstance(models, list), f"models response did not include a list: {payload}")
    available: set[str] = set()
    for model in models:
        if isinstance(model, dict):
            available.update(model_aliases(str(model.get("id") or model.get("label") or "")))
        else:
            available.update(model_aliases(str(model)))
    missing = [model for model in REQUIRED_OLLAMA_MODELS if not available.intersection(model_aliases(model))]
    require(not missing, f"missing Ollama model(s): {', '.join(missing)}")
    return f"{len(models)} model(s); required={', '.join(REQUIRED_OLLAMA_MODELS)}"


def check_lm_chat(base_url: str, timeout: float, lm_base_url: str, api_mode: str, model: str) -> str:
    payload = request_json(
        base_url,
        "/api/lm-studio/chat",
        method="POST",
        payload={
            "baseUrl": lm_base_url,
            "apiMode": api_mode,
            "model": model,
            "systemPrompt": "Reply with a short plain text answer.",
            "userPrompt": "Say ok.",
            "temperature": 0,
            "topP": 1,
            "maxTokens": 16,
            "timeoutSeconds": int(timeout),
        },
        timeout=timeout + 5,
    )
    text = str(payload.get("text") or "").strip()
    require(bool(text), f"chat response did not include text: {payload}")
    return text[:80]


def check_sbert_pair(base_url: str, timeout: float) -> str:
    payload = request_json(
        base_url,
        "/api/sbert/pair",
        method="POST",
        payload={
            "model": "all-MiniLM-L6-v2",
            "textA": "families need emergency shelter after flooding",
            "textB": "flooded families request urgent housing assistance",
        },
        timeout=timeout,
    )
    require("similarity" in payload, f"missing similarity: {payload}")
    return f"similarity={payload.get('similarity')}"


def check_ppdb_status(base_url: str, timeout: float) -> str:
    payload = request_json(
        base_url,
        "/api/turbulence-comparison/ppdb/status",
        method="POST",
        payload={},
        timeout=timeout,
    )
    require(payload.get("available") is True, payload.get("message") or f"PPDB unavailable: {payload}")
    return f"entries={payload.get('entries')}"


def check_binary_ppdb_files() -> str:
    source_path = ROOT / "data" / "external" / "ppdb" / "ppdb-2.0-s-all"
    sqlite_index_path = ROOT / "data" / "turbulence" / "ppdb_index.sqlite"
    require(
        source_path.exists() or sqlite_index_path.exists(),
        "Binary PPDB runtime files are missing. Rerun install without --skip-ppdb so "
        "data/external/ppdb/ppdb-2.0-s-all or data/turbulence/ppdb_index.sqlite exists.",
    )
    if sqlite_index_path.exists():
        return f"sqlite={sqlite_index_path.relative_to(ROOT)}"
    return f"source={source_path.relative_to(ROOT)}; sqlite will be built on first Binary run"


def check_initial_population(base_url: str, timeout: float) -> str:
    payload = request_json(base_url, "/api/initial-population/strategies", timeout=timeout)
    strategies = payload.get("strategies") if isinstance(payload.get("strategies"), list) else []
    require(any(item.get("strategyId") == "hybrid-semantic-v7" and item.get("available") for item in strategies), "hybrid strategy is not available")
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    lm = defaults.get("lmStudio") if isinstance(defaults.get("lmStudio"), dict) else {}
    require(lm.get("baseUrl") == OLLAMA_OPENAI_BASE_URL, f"unexpected initial-population baseUrl: {lm}")
    require(lm.get("apiMode") == OLLAMA_OPENAI_API_MODE, f"unexpected initial-population apiMode: {lm}")
    return "hybrid-semantic-v7 available"


def check_initial_population_comparison(base_url: str, timeout: float) -> str:
    payload = request_json(base_url, "/api/initial-population-comparison/strategies", timeout=timeout)
    strategies = payload.get("strategies") if isinstance(payload.get("strategies"), list) else []
    by_id = {str(item.get("strategyId")): item for item in strategies if isinstance(item, dict)}
    missing = [strategy_id for strategy_id in REQUIRED_INITIAL_COMPARISON_STRATEGIES if not by_id.get(strategy_id, {}).get("available")]
    require(not missing, f"initial comparison strategies unavailable: {', '.join(missing)}")
    return ", ".join(REQUIRED_INITIAL_COMPARISON_STRATEGIES)


def check_comparator_proposals(base_url: str, timeout: float) -> str:
    payload = request_json(base_url, "/api/comparator/proposals", timeout=timeout)
    proposals = payload.get("proposals") if isinstance(payload.get("proposals"), list) else []
    by_id = {str(item.get("proposalId")): item for item in proposals if isinstance(item, dict)}
    failures: list[str] = []
    for proposal_id in REQUIRED_COMPARATOR_PROPOSALS:
        proposal = by_id.get(proposal_id)
        if not proposal:
            failures.append(f"{proposal_id}: missing")
            continue
        if not proposal.get("available"):
            dependency_status = proposal.get("dependencyStatus") if isinstance(proposal.get("dependencyStatus"), dict) else {}
            missing = ", ".join(str(item) for item in dependency_status.get("missing") or [])
            detail = dependency_status.get("error") or (f"missing modules: {missing}" if missing else "not available")
            failures.append(f"{proposal_id}: {detail}")
    require(not failures, "; ".join(failures))
    return ", ".join(REQUIRED_COMPARATOR_PROPOSALS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that the portal installation is usable.")
    parser.add_argument("--base-url", default="http://127.0.0.1:4173")
    parser.add_argument("--lm-base-url", default=OLLAMA_OPENAI_BASE_URL)
    parser.add_argument("--api-mode", default=OLLAMA_OPENAI_API_MODE, choices=("native", "openai"))
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--skip-chat", action="store_true")
    args = parser.parse_args()

    checks = [
        check("portal health", lambda: check_portal_health(args.base_url, args.timeout)),
        check("llm models", lambda: check_lm_models(args.base_url, args.timeout, args.lm_base_url, args.api_mode)),
    ]
    if not args.skip_chat:
        checks.append(check("llm chat", lambda: check_lm_chat(args.base_url, args.timeout, args.lm_base_url, args.api_mode, args.model)))
    checks.extend(
        [
            check("sbert pair", lambda: check_sbert_pair(args.base_url, args.timeout)),
            check("ppdb status", lambda: check_ppdb_status(args.base_url, args.timeout)),
            check("binary ppdb", check_binary_ppdb_files),
            check("initial population", lambda: check_initial_population(args.base_url, args.timeout)),
            check("initial population comparison", lambda: check_initial_population_comparison(args.base_url, args.timeout)),
            check("proposal comparator", lambda: check_comparator_proposals(args.base_url, args.timeout)),
        ]
    )

    for result in checks:
        prefix = "OK" if result.ok else "FAIL"
        print(f"[{prefix}] {result.name}: {result.detail}")

    return 0 if all(result.ok for result in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
