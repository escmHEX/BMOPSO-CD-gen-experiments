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
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def run_python_probe(name: str, python_executable: Path, code: str, timeout: float) -> ProbeResult:
    if not python_executable.exists():
        return ProbeResult(name, False, f"missing python executable: {python_executable}")
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return ProbeResult(name, False, f"timed out after {timeout:g}s", error.stdout or "", error.stderr or "")

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode == 0:
        return ProbeResult(name, True, stdout or "ok", stdout, stderr)
    detail = describe_return_code(completed.returncode)
    if completed.returncode == -signal.SIGILL:
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

    for result in results:
        prefix = "OK" if result.ok else "FAIL"
        print(f"[{prefix}] {result.name}: {result.detail}")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
