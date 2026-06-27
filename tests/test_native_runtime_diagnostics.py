from __future__ import annotations

import importlib.util
import contextlib
import io
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "diagnose_native_runtime.py"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_native_runtime", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NativeRuntimeDiagnosticsTests(unittest.TestCase):
    def test_describe_return_code_reports_sigill(self):
        module = load_module()

        detail = module.describe_return_code(-4)

        self.assertEqual(detail, "terminated by signal 4 (SIGILL)")

    def test_binary_probe_code_uses_embedding_service(self):
        module = load_module()

        code = module.binary_embedding_probe_code()

        self.assertIn("EmbeddingService", code)
        self.assertIn("encode", code)

    def test_binary_prompt_reduction_probe_matches_initialization_shape(self):
        module = load_module()

        code = module.binary_prompt_reduction_probe_code()

        self.assertIn("EmbeddingService", code)
        self.assertIn("range(40)", code)
        self.assertIn("greedy_max_min_indices", code)
        self.assertIn("embeddings[index] @ embeddings[others].T", code)

    def test_binary_run_prompt_reduction_probe_replays_run_artifacts(self):
        module = load_module()

        code = module.binary_run_prompt_reduction_probe_code(Path("runs/comparator/example/binary/exec/run"))

        self.assertIn("initialization_pool_diagnostics.jsonl", code)
        self.assertIn("config_effective.yaml", code)
        self.assertIn("_candidate_vectors", code)
        self.assertIn("_reduce_by_prompt_diversity", code)

    def test_main_runs_binary_run_replay_when_run_dir_is_provided(self):
        module = load_module()
        calls: list[str] = []

        def fake_run_python_probe(name, python_executable, code, timeout, **kwargs):
            calls.append(name)
            return module.ProbeResult(name, True, "ok")

        argv = [
            "diagnose_native_runtime.py",
            "--skip-portal",
            "--binary-run-dir",
            "runs/comparator/example/binary/exec/run",
        ]
        with patch.object(sys, "argv", argv):
            with patch.object(module, "run_python_probe", side_effect=fake_run_python_probe):
                exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("binary run prompt reduction", calls)

    def test_run_python_probe_prints_start_message(self):
        module = load_module()

        class FakeProcess:
            returncode = 0

            def communicate(self, timeout=None):
                return "ok", ""

        output = io.StringIO()
        with patch.object(module.subprocess, "Popen", return_value=FakeProcess()):
            with contextlib.redirect_stdout(output):
                result = module.run_python_probe("binary sbert", Path(sys.executable), "print('ok')", 10)

        self.assertTrue(result.ok)
        self.assertIn("[INFO] running binary sbert:", output.getvalue())
        self.assertIn("timeout=10s", output.getvalue())

    def test_run_python_probe_prints_heartbeat_while_waiting(self):
        module = load_module()

        class FakeProcess:
            returncode = 0

            def __init__(self):
                self.calls = 0

            def communicate(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    raise module.subprocess.TimeoutExpired(["python"], timeout)
                return "ok", ""

        output = io.StringIO()
        with patch.object(module.subprocess, "Popen", return_value=FakeProcess()):
            with patch.object(module.time, "monotonic", side_effect=[0, 30, 31, 31]):
                with contextlib.redirect_stdout(output):
                    result = module.run_python_probe(
                        "binary sbert",
                        Path(sys.executable),
                        "print('ok')",
                        60,
                        heartbeat_seconds=30,
                    )

        self.assertTrue(result.ok)
        self.assertIn("[INFO] binary sbert still running after", output.getvalue())


if __name__ == "__main__":
    unittest.main()
