from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class RuntimeSetupTests(unittest.TestCase):
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
