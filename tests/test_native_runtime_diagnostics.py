from __future__ import annotations

import importlib.util
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
