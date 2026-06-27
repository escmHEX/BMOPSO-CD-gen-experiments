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
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_HEARTBEAT_SECONDS = 30.0


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str
    stdout: str = ""
    stderr: str = ""


def default_binary_python() -> Path:
    if platform.system().lower() == "windows":
        return ROOT / "baselines" / "venvs" / "binary-mopso-cd" / "Scripts" / "python.exe"
    return ROOT / "baselines" / "venvs" / "binary-mopso-cd" / "bin" / "python"


def signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def describe_return_code(return_code: int) -> str:
    if return_code < 0:
        signum = abs(return_code)
        return f"terminated by signal {signum} ({signal_name(signum)})"
    return f"exited with code {return_code}"


def cpu_flag_summary() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return "cpu flags unavailable on this platform"
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("flags"):
            _, _, flags = line.partition(":")
            flag_set = set(flags.split())
            interesting = ["sse4_2", "avx", "avx2", "avx512f", "fma"]
            present = [flag for flag in interesting if flag in flag_set]
            missing = [flag for flag in interesting if flag not in flag_set]
            return f"present={','.join(present) or 'none'} missing={','.join(missing) or 'none'}"
    return "cpu flags not found in /proc/cpuinfo"


def format_seconds(seconds: float) -> str:
    return f"{int(seconds)}s"


def portal_embedding_probe_code() -> str:
    return r'''
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(["diagnostic sentence one", "diagnostic sentence two"], convert_to_numpy=True)
print(f"shape={tuple(embeddings.shape)} dtype={embeddings.dtype}")
'''


def binary_embedding_probe_code() -> str:
    return r'''
from binary_mopso_cd.services.embedding import EmbeddingCache, EmbeddingService

service = EmbeddingService(
    model_name="all-MiniLM-L6-v2",
    resolved_model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=2,
    config_version="diagnostic",
    cache=EmbeddingCache(None),
)
embeddings = service.encode(["diagnostic sentence one", "diagnostic sentence two"], text_type="diagnostic")
print(f"shape={tuple(embeddings.shape)} dtype={embeddings.dtype}")
'''


def binary_prompt_reduction_probe_code() -> str:
    return r'''
import numpy as np

from binary_mopso_cd.initialization import greedy_max_min_indices
from binary_mopso_cd.services.embedding import EmbeddingCache, EmbeddingService

service = EmbeddingService(
    model_name="all-MiniLM-L6-v2",
    resolved_model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64,
    config_version="diagnostic",
    cache=EmbeddingCache(None),
)
prompts = [
    (
        "Generate a short social media message related to crises and emergencies "
        f"using semantic components role=person {index}, topic=quarantine, action=avoid phone."
    )
    for index in range(40)
]
embeddings = service.encode(prompts, text_type="prompt")
selected = greedy_max_min_indices(embeddings, 20)
scores = []
for index in selected:
    others = [other for other in selected if other != index]
    scores.append(float(np.min(1.0 - (embeddings[index] @ embeddings[others].T))))
print(
    f"shape={tuple(embeddings.shape)} dtype={embeddings.dtype} "
    f"selected={len(selected)} scores={len(scores)} min={min(scores):.6f}"
)
'''


def binary_run_prompt_reduction_probe_code(run_dir: Path) -> str:
    return f'''
from __future__ import annotations

import json
from pathlib import Path

from binary_mopso_cd.config import RuntimeConfig, load_yaml
from binary_mopso_cd.initialization import InitialPopulationBuilder
from binary_mopso_cd.router import ALG_PROMPT_RENDERER, SemanticRouter
from binary_mopso_cd.services.embedding import EmbeddingCache, EmbeddingService
from binary_mopso_cd.services.prompt_renderer import DeterministicPromptRenderer
from binary_mopso_cd.settings import ComponentSettings
from binary_mopso_cd.utils import rng_from_text

run_dir = Path({str(run_dir)!r})
config_path = run_dir / "config_effective.yaml"
diagnostics_path = run_dir / "initialization_pool_diagnostics.jsonl"
if not config_path.exists():
    raise FileNotFoundError(f"missing config_effective.yaml: {{config_path}}")
if not diagnostics_path.exists():
    raise FileNotFoundError(f"missing initialization_pool_diagnostics.jsonl: {{diagnostics_path}}")

config = RuntimeConfig(load_yaml(config_path))
components = ComponentSettings.from_config(config).order
pools = {{component: [] for component in components}}
rows = []
with diagnostics_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

for row in rows:
    task = str(row.get("task", ""))
    component = str(row.get("component", ""))
    if component not in pools or task not in {{"semantic_pool_generation", "semantic_pool_expansion"}}:
        continue
    if task == "semantic_pool_generation":
        current = [str(item) for item in row.get("existing_items", [])]
    else:
        current = [str(item) for item in row.get("existing_items") or pools.get(component, [])]
    current.extend(str(item) for item in row.get("valid_items", []))
    pools[component] = current

missing = [component for component in components if not pools.get(component)]
if missing:
    raise RuntimeError(f"run diagnostics do not contain all component pools: missing={{missing}}")


class ReplayExecutor:
    outdir = None

    def __init__(self, config: RuntimeConfig):
        self.config = config
        model_alias = str(config.get("models.sbert.default"))
        resolved_model_name = str(config.get(f"models.sbert.alternatives.{{model_alias}}", model_alias))
        self.embedding_service = EmbeddingService(
            model_name=model_alias,
            resolved_model_name=resolved_model_name,
            batch_size=int(config.get("models.sbert.batch_size", 64)),
            config_version=str(config.get("models.sbert.config_version", "sbert-v1")),
            cache=EmbeddingCache(None),
        )
        self.prompt_renderer = DeterministicPromptRenderer()

    def execute(self, task):
        if task.alg_name != ALG_PROMPT_RENDERER:
            raise ValueError(f"replay executor only supports prompt rendering, got {{task.alg_name}}")
        return self.prompt_renderer.render(
            dict(task.task_params["components"]),
            str(task.task_params.get("domain", self.config.get("experiment.domain"))),
        )


executor = ReplayExecutor(config)
builder = InitialPopulationBuilder(
    config,
    SemanticRouter(config),
    executor,  # type: ignore[arg-type]
    rng_from_text(config.seed, None),
)
domain = str(config.get("experiment.domain"))
pool_sizes = {{component: len(pools[component]) for component in components}}
candidates = builder._candidate_vectors(pools)
target = 2 * config.n
print(f"run_dir={{run_dir}}")
print(f"pool_sizes={{pool_sizes}} candidates={{len(candidates)}} target={{target}}")
reduced = builder._reduce_by_prompt_diversity(candidates, domain, target)
scores = [score for _, _, score in reduced]
score_min = min(scores) if scores else 0.0
score_max = max(scores) if scores else 0.0
print(f"reduced={{len(reduced)}} score_min={{score_min:.6f}} score_max={{score_max:.6f}}")
'''


def run_python_probe(
    name: str,
    python_executable: Path,
    code: str,
    timeout: float,
    *,
    heartbeat_seconds: float = PROBE_HEARTBEAT_SECONDS,
) -> ProbeResult:
    if not python_executable.exists():
        return ProbeResult(name, False, f"missing python executable: {python_executable}")
    print(f"[INFO] running {name}: python={python_executable} timeout={timeout:g}s", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        [str(python_executable), "-c", code],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    try:
        while True:
            elapsed = time.monotonic() - started
            remaining = timeout - elapsed
            if remaining <= 0:
                process.kill()
                stdout, stderr = process.communicate()
                return ProbeResult(name, False, f"timed out after {timeout:g}s", stdout.strip(), stderr.strip())
            wait_for = max(0.1, min(float(heartbeat_seconds), remaining))
            try:
                stdout, stderr = process.communicate(timeout=wait_for)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(f"[INFO] {name} still running after {format_seconds(elapsed)}", flush=True)
                continue
    except KeyboardInterrupt:
        process.kill()
        process.communicate()
        raise

    stdout = stdout.strip()
    stderr = stderr.strip()
    if process.returncode == 0:
        return ProbeResult(name, True, stdout or "ok", stdout, stderr)
    detail = describe_return_code(process.returncode)
    if process.returncode == -signal.SIGILL:
        detail += (
            "; SIGILL usually means a native dependency such as torch, numpy, scikit-learn, "
            "or sentence-transformers used CPU instructions unsupported by this node"
        )
    if stderr:
        detail += f"; stderr={stderr.splitlines()[-1]}"
    return ProbeResult(name, False, detail, stdout, stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose native ML runtime compatibility.")
    parser.add_argument("--binary-python", type=Path, default=default_binary_python())
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--skip-portal", action="store_true")
    parser.add_argument("--skip-binary", action="store_true")
    parser.add_argument("--binary-run-dir", type=Path)
    args = parser.parse_args()

    print(f"[INFO] python={sys.executable}")
    print(f"[INFO] platform={platform.platform()}")
    print(f"[INFO] machine={platform.machine()} processor={platform.processor() or 'unknown'}")
    print(f"[INFO] cpu_flags={cpu_flag_summary()}")

    results: list[ProbeResult] = []
    if not args.skip_portal:
        results.append(run_python_probe("portal sbert", Path(sys.executable), portal_embedding_probe_code(), args.timeout))
    if not args.skip_binary:
        results.append(run_python_probe("binary sbert", args.binary_python, binary_embedding_probe_code(), args.timeout))
        results.append(
            run_python_probe(
                "binary prompt reduction",
                args.binary_python,
                binary_prompt_reduction_probe_code(),
                args.timeout,
            )
        )
        if args.binary_run_dir is not None:
            results.append(
                run_python_probe(
                    "binary run prompt reduction",
                    args.binary_python,
                    binary_run_prompt_reduction_probe_code(args.binary_run_dir),
                    args.timeout,
                )
            )

    for result in results:
        prefix = "OK" if result.ok else "FAIL"
        print(f"[{prefix}] {result.name}: {result.detail}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
