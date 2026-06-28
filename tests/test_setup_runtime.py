from __future__ import annotations

import ast
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RuntimeSetupTests(unittest.TestCase):
    def test_warmup_direct_runtime_imports_are_backend_requirements(self):
        import scripts.warmup_runtime as warmup_runtime

        module_path = Path(warmup_runtime.__file__)
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".", 1)[0])

        runtime_imports = {
            "sentence_transformers": "sentence-transformers",
            "transformers": "transformers",
            "nltk": "nltk",
            "spacy": "spacy",
        }
        requirements_text = Path("requirements.backend.txt").read_text(encoding="utf-8")
        declared_requirements = {
            line.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip().lower()
            for line in requirements_text.splitlines()
            if line.strip() and not line.startswith("#") and "://" not in line
        }

        missing = [
            requirement
            for module_name, requirement in runtime_imports.items()
            if module_name in imported_modules and requirement.lower() not in declared_requirements
        ]
        self.assertEqual(missing, [])

    def test_installers_reuse_existing_virtual_environments(self):
        ubuntu_installer = Path("scripts/install_ubuntu.sh").read_text(encoding="utf-8")
        windows_installer = Path("scripts/install_windows.ps1").read_text(encoding="utf-8")

        self.assertIn("Reusing existing virtual environment", ubuntu_installer)
        self.assertIn('elif [[ -x "$python_bin" ]]', ubuntu_installer)
        self.assertIn("UV_VENV_CLEAR=1", ubuntu_installer)
        self.assertIn("Reusing existing virtual environment", windows_installer)
        self.assertIn("elseif (Test-Path $python)", windows_installer)
        self.assertIn("UV_VENV_CLEAR=1", windows_installer)

    def test_installers_prepare_binary_ppdb_sqlite_index(self):
        ubuntu_installer = Path("scripts/install_ubuntu.sh").read_text(encoding="utf-8")
        windows_installer = Path("scripts/install_windows.ps1").read_text(encoding="utf-8")

        self.assertIn("data/turbulence/ppdb_index.sqlite", ubuntu_installer)
        self.assertIn("build_sqlite_index", ubuntu_installer)
        self.assertIn("data\\turbulence\\ppdb_index.sqlite", windows_installer)
        self.assertIn("build_sqlite_index", windows_installer)

    def test_verify_install_checks_binary_ppdb_runtime_files(self):
        verifier = Path("scripts/verify_install.py").read_text(encoding="utf-8")

        self.assertIn("binary ppdb", verifier)
        self.assertIn("ppdb_index.sqlite", verifier)
        self.assertIn("ppdb-2.0-s-all", verifier)

    def test_build_comparator_local_config_uses_ubuntu_venvs(self):
        from scripts.setup_runtime import build_comparator_local_config

        config = build_comparator_local_config(platform_name="linux")

        self.assertEqual(
            config["proposals"]["evolmd"]["pythonExecutable"],
            "baselines/venvs/evolmd/bin/python",
        )
        self.assertEqual(
            config["proposals"]["evolmd-mo"]["pythonExecutable"],
            "baselines/venvs/evolmd-mo/bin/python",
        )
        self.assertEqual(
            config["proposals"]["mesap"]["pythonExecutable"],
            "baselines/venvs/mesap/bin/python",
        )
        self.assertEqual(
            config["proposals"]["binary-mopso-cd"]["repositoryPath"],
            "baselines/external/binary-mopso-cd",
        )
        self.assertEqual(
            config["proposals"]["binary-mopso-cd"]["pythonExecutable"],
            "baselines/venvs/binary-mopso-cd/bin/python",
        )
        self.assertEqual(config["proposals"]["binary-mopso-cd"]["git"]["branch"], "dev")

    def test_build_comparator_local_config_uses_windows_venvs(self):
        from scripts.setup_runtime import build_comparator_local_config

        config = build_comparator_local_config(platform_name="win32")

        self.assertEqual(
            config["proposals"]["evolmd"]["pythonExecutable"],
            "baselines/venvs/evolmd/Scripts/python.exe",
        )
        self.assertEqual(
            config["proposals"]["binary-mopso-cd"]["pythonExecutable"],
            "baselines/venvs/binary-mopso-cd/Scripts/python.exe",
        )

    def test_base_comparator_config_does_not_hardcode_local_binary_windows_path(self):
        config = json.loads(Path("baselines/comparator_config.json").read_text(encoding="utf-8"))
        binary = config["proposals"]["binary-mopso-cd"]

        self.assertEqual(binary["repositoryPath"], "baselines/external/binary-mopso-cd")
        self.assertEqual(binary["pythonExecutable"], "")
        self.assertNotIn("C:\\", json.dumps(binary))

    def test_base_comparator_config_disables_repository_updates_by_default(self):
        config = json.loads(Path("baselines/comparator_config.json").read_text(encoding="utf-8"))

        self.assertIs(config["defaults"]["updateRepositoriesBeforeRun"], False)

    def test_setup_scripts_do_not_auto_upgrade_pip(self):
        scripts = (
            Path("scripts/setup_backend_env.ps1"),
            Path("scripts/install_windows.ps1"),
            Path("scripts/install_ubuntu.sh"),
        )

        for script_path in scripts:
            with self.subTest(script=str(script_path)):
                script = script_path.read_text(encoding="utf-8")
                self.assertNotIn("install --upgrade pip", script)
                self.assertNotIn('"--upgrade", "pip"', script)

    def test_windows_ollama_runtime_pins_gpu_runtime(self):
        script = Path("scripts/ollama_runtime.ps1").read_text(encoding="utf-8")

        self.assertIn('$OllamaRequiredVersion = "0.30.10"', script)
        self.assertIn('$OllamaPreferredLibrary = "cuda_v13"', script)
        self.assertIn('$OllamaFallbackLibrary = "cuda_v12"', script)
        self.assertIn('$env:CUDA_VISIBLE_DEVICES = "0"', script)
        self.assertIn("OllamaSetup.exe", script)
        self.assertIn("releases/download/v$OllamaRequiredVersion/OllamaSetup.exe", script)
        self.assertIn("sha256sum.txt", script)
        self.assertIn("Get-FileHash", script)
        self.assertIn("/api/generate", script)
        self.assertIn("/api/ps", script)
        self.assertIn("size_vram", script)
        self.assertIn("100% CPU", script)

    def test_windows_server_launchers_require_ollama_gpu_runtime(self):
        for script_path in (
            Path("scripts/start_server.ps1"),
            Path("scripts/start_server_daemon_windows.ps1"),
            Path("scripts/install_windows.ps1"),
        ):
            with self.subTest(script=str(script_path)):
                script = script_path.read_text(encoding="utf-8")
                self.assertIn('Join-Path $PSScriptRoot "ollama_runtime.ps1"', script)
                self.assertIn("Ensure-OllamaGpuRuntime", script)

    def test_windows_daemon_stop_cleans_existing_portal_server_processes(self):
        script = Path("scripts/start_server_daemon_windows.ps1").read_text(encoding="utf-8")

        self.assertIn("function Stop-PortalServerProcesses", script)
        self.assertIn("Get-CimInstance Win32_Process", script)
        self.assertIn("Stop-PortalServerProcesses", script)
        self.assertIn("Start-Process", script)
        self.assertIn("Wait-PortalBackendHealth", script)
        self.assertIn("--skip-port-release", script)

    def test_binary_repository_fallback_uses_relative_development_sibling(self):
        from baselines import comparator as comparator_module

        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            root = parent / "Experimentos"
            sibling_binary = parent / "Binary MOPSO-CD"
            root.mkdir()
            sibling_binary.mkdir()

            with (
                patch.object(comparator_module, "MODULE_ROOT", root),
                patch.object(
                    comparator_module,
                    "COMPARATOR_PROPOSAL_CONFIG",
                    {"binary-mopso-cd": {"repositoryPath": "baselines/external/binary-mopso-cd"}},
                ),
            ):
                self.assertEqual(
                    comparator_module.configured_binary_repository_path(),
                    "../Binary MOPSO-CD",
                )

    def test_comparator_config_path_prefers_local_config_when_env_is_missing(self):
        from baselines import comparator as comparator_module

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            default_config = config_dir / "comparator_config.json"
            local_config = config_dir / "comparator_config.local.json"
            default_config.write_text("{}", encoding="utf-8")
            local_config.write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    comparator_module.resolve_comparator_config_path(config_dir),
                    local_config,
                )

    def test_initial_population_baseline_uses_configured_proposal_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "baselines" / "external" / "evolmd").mkdir(parents=True)
            (root / "baselines" / "venvs" / "evolmd" / "Scripts").mkdir(parents=True)
            configured_python = root / "baselines" / "venvs" / "evolmd" / "Scripts" / "python.exe"
            configured_python.touch()
            config_path = root / "baselines" / "comparator_config.local.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "proposals": {
                            "evolmd": {
                                "repositoryPath": "baselines/external/evolmd",
                                "pythonExecutable": "baselines/venvs/evolmd/Scripts/python.exe",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"COMPARATOR_CONFIG_PATH": str(config_path)}):
                import baselines.comparator as comparator_module
                import initial_population.comparison as comparison_module

                importlib.reload(comparator_module)
                importlib.reload(comparison_module)
                service = comparison_module.InitialPopulationComparisonService(root)
                strategy = next(
                    strategy
                    for strategy in comparison_module.STRATEGIES
                    if strategy.strategy_id == "evolmd-initial"
                )
                strategy_dir = root / "runs" / "initial" / "evolmd"
                strategy_dir.mkdir(parents=True)
                execution_config = {
                    "referenceText": "reference",
                    "n": 1,
                    "seed": 42,
                    "baselines": {
                        "model": "llama3",
                        "bertModel": "bert-base-uncased",
                        "timeoutMinutes": 1,
                        "tempPrompts": 0.9,
                        "tempKeywords": 0.3,
                        "tempGeneration": 0.7,
                    },
                }
                with patch.object(
                    service,
                    "_run_process",
                    return_value={"returnCode": 0, "timedOut": False},
                ) as run_process:
                    service._run_baseline_process({}, strategy, strategy_dir, execution_config)

                command = run_process.call_args.args[2]
                self.assertEqual(command[0], str(configured_python))

            import baselines.comparator as comparator_module
            import initial_population.comparison as comparison_module

            importlib.reload(comparator_module)
            importlib.reload(comparison_module)


if __name__ == "__main__":
    unittest.main()
