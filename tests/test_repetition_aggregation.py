from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from baselines import comparator as comparator_module
from baselines.bootstrap import install_evolmd_bertscore_guard
from baselines.comparator import ComparatorService, PROPOSALS, aggregate_proposal_repetitions
from baselines.comparator import aggregate_series
from baselines.comparator import embedding_front_rows_from_rows
from baselines.comparator import mark_non_dominated
from baselines.comparator import parse_simple_yaml_mapping
from baselines.comparator_metrics import build_charts_from_rows
from baselines.comparator_metrics import calculate_contribution
from baselines.comparator_metrics import calculate_extent
from baselines.comparator_metrics import calculate_unary_entropy
from initial_population.service import InitialPopulationService
from initial_population.comparison import InitialPopulationComparisonService
from turbulence_comparison.service import aggregate_turbulence_repetitions


class FakeSpacyToken:
    def __init__(self, lemma: str, pos: str):
        self.lemma_ = lemma
        self.pos_ = pos


class FakeSpacyModel:
    def __init__(self, docs):
        self._docs = docs

    def pipe(self, _texts):
        return iter(self._docs)


def command_set_values(command: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for index, item in enumerate(command):
        if item == "--set" and index + 1 < len(command):
            path, value = command[index + 1].split("=", 1)
            values[path] = value
        elif item.startswith("--set="):
            path, value = item[len("--set="):].split("=", 1)
            values[path] = value
    return values


def command_set_paths(command: list[str]) -> list[str]:
    paths: list[str] = []
    for index, item in enumerate(command):
        if item == "--set" and index + 1 < len(command):
            path, _value = command[index + 1].split("=", 1)
            paths.append(path)
        elif item.startswith("--set="):
            path, _value = item[len("--set="):].split("=", 1)
            paths.append(path)
    return paths


class RepetitionAggregationTests(unittest.TestCase):
    def test_initial_population_defaults_target_ollama_openai_compatible_endpoint(self):
        service = InitialPopulationService(Path("."))
        payload = service.default_config()

        self.assertEqual(payload["lmStudio"]["baseUrl"], "http://127.0.0.1:11434")
        self.assertEqual(payload["lmStudio"]["apiMode"], "openai")
        self.assertTrue(payload["stages"])
        self.assertTrue(all(stage["model"] == "llama3" for stage in payload["stages"].values()))

    def test_proposal_python_executable_detects_platform_repository_venv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "baseline"
            linux_python = repository / "venv" / "bin" / "python"
            windows_python = repository / "venv" / "Scripts" / "python.exe"
            linux_python.parent.mkdir(parents=True)
            windows_python.parent.mkdir(parents=True)
            linux_python.touch()
            windows_python.touch()

            proposal = replace(PROPOSALS[0], repository_path="baseline", python_executable="")

            with patch.object(comparator_module.sys, "platform", "linux"):
                self.assertEqual(
                    comparator_module.proposal_python_executable(root, repository, proposal),
                    str(linux_python),
                )
            with patch.object(comparator_module.sys, "platform", "win32"):
                self.assertEqual(
                    comparator_module.proposal_python_executable(root, repository, proposal),
                    str(windows_python),
                )

    def test_proposal_python_executable_detects_managed_baseline_venv_on_ubuntu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "baselines" / "external" / "evolmd"
            repository.mkdir(parents=True)
            managed_python = root / "baselines" / "venvs" / "evolmd" / "bin" / "python"
            managed_python.parent.mkdir(parents=True)
            managed_python.touch()

            proposal = replace(PROPOSALS[0], repository_path="baselines/external/evolmd", python_executable="")

            with patch.object(comparator_module.sys, "platform", "linux"):
                self.assertEqual(
                    comparator_module.proposal_python_executable(root, repository, proposal),
                    str(managed_python),
                )

    def test_configured_proposal_python_executable_overrides_local_venv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "baseline"
            (repository / ".venv" / "Scripts").mkdir(parents=True)
            configured_python = root / "custom" / "python.exe"
            configured_python.parent.mkdir(parents=True)
            configured_python.touch()

            proposal = replace(PROPOSALS[0], repository_path="baseline", python_executable=str(configured_python))

            self.assertEqual(
                comparator_module.proposal_python_executable(root, repository, proposal),
                str(configured_python),
            )

    def test_comparator_config_accepts_seed_and_repetitions(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "seed": 123,
                "repetitionsK": 4,
            }
        )
        self.assertEqual(parsed["seed"], 123)
        self.assertEqual(parsed["repetitionsK"], 4)
        self.assertIn("binary-mopso-cd", parsed["selectedProposalIds"])

    def test_comparator_config_timeout_defaults_to_10800_minutes(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config({"referenceText": "reference"})

        self.assertEqual(parsed["timeoutMinutes"], 10800)

    def test_comparator_config_rejects_timeout_below_2400_minutes(self):
        service = ComparatorService(Path("."))

        with self.assertRaises(ValueError):
            service._read_config({"referenceText": "reference", "timeoutMinutes": 120})

    def test_comparator_config_accepts_single_selected_proposal(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {"binary-mopso-cd": {"extraArgs": "--set selection.k=4"}},
            }
        )
        self.assertEqual(parsed["selectedProposalIds"], ["binary-mopso-cd"])
        self.assertEqual(parsed["proposalConfigs"]["binary-mopso-cd"]["extraArgs"], "--set selection.k=4")

    def test_comparator_config_accepts_multiple_instances_for_same_proposal(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "proposalInstances": [
                    {
                        "instanceId": "binary-a",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary A",
                        "proposalConfig": {"cliValues": {"selection.k": "4"}},
                    },
                    {
                        "instanceId": "binary-b",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary B",
                        "proposalConfig": {"cliValues": {"selection.k": "5"}},
                    },
                ],
            }
        )

        self.assertEqual(parsed["selectedProposalIds"], ["binary-mopso-cd"])
        self.assertEqual(len(parsed["proposalInstances"]), 2)
        instances = service._selected_instances(parsed)
        self.assertEqual([instance.instance_id for instance in instances], ["binary-a", "binary-b"])
        self.assertEqual(instances[0].proposal_config["cliValues"]["selection.k"], "4")
        self.assertEqual(instances[1].proposal_config["cliValues"]["selection.k"], "5")

    def test_comparator_config_accepts_same_initial_population_generator(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "proposalInstances": [
                    {
                        "instanceId": "binary-a",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary A",
                        "proposalConfig": {"cliValues": {"selection.k": "4"}},
                    },
                    {
                        "instanceId": "binary-b",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary B",
                        "proposalConfig": {"cliValues": {"selection.k": "5"}},
                    },
                ],
                "sameInitialPopulationForBmopso": {
                    "enabled": True,
                    "generatorInstanceId": "binary-a",
                },
            }
        )

        self.assertEqual(
            parsed["sameInitialPopulationForBmopso"],
            {"enabled": True, "generatorInstanceId": "binary-a", "scope": "per_repetition"},
        )

    def test_comparator_config_rejects_non_binary_same_initial_population_generator(self):
        service = ComparatorService(Path("."))
        with self.assertRaisesRegex(ValueError, "generatorInstanceId must reference a Binary MOPSO-CD instance"):
            service._read_config(
                {
                    "referenceText": "reference",
                    "proposalInstances": [
                        {
                            "instanceId": "binary-a",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary A",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        },
                        {
                            "instanceId": "mesap-a",
                            "proposalId": "mesap",
                            "displayName": "MESAP A",
                            "proposalConfig": {"cliValues": {}},
                        },
                    ],
                    "sameInitialPopulationForBmopso": {
                        "enabled": True,
                        "generatorInstanceId": "mesap-a",
                    },
                }
            )

    def test_comparator_config_rejects_duplicate_instances_for_same_proposal(self):
        service = ComparatorService(Path("."))
        with self.assertRaisesRegex(ValueError, "duplicateProposalInstances"):
            service._read_config(
                {
                    "referenceText": "reference",
                    "proposalInstances": [
                        {
                            "instanceId": "binary-a",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary A",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        },
                        {
                            "instanceId": "binary-b",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary B",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        },
                    ],
                }
            )

    def test_comparator_config_rejects_duplicate_mesap_instances(self):
        service = ComparatorService(Path("."))
        with self.assertRaisesRegex(ValueError, "duplicateProposalInstances"):
            service._read_config(
                {
                    "referenceText": "reference",
                    "proposalInstances": [
                        {
                            "instanceId": "mesap-a",
                            "proposalId": "mesap",
                            "displayName": "MESAP A",
                            "proposalConfig": {"cliValues": {"--k": "4"}},
                        },
                        {
                            "instanceId": "mesap-b",
                            "proposalId": "mesap",
                            "displayName": "MESAP B",
                            "proposalConfig": {"cliValues": {"--k": "4"}},
                        },
                    ],
                }
            )

    def test_comparator_config_allows_same_config_for_different_proposals(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "proposalInstances": [
                    {
                        "instanceId": "evolmd-mo-a",
                        "proposalId": "evolmd-mo",
                        "displayName": "EVOLMD-MO A",
                        "proposalConfig": {"cliValues": {}},
                    },
                    {
                        "instanceId": "binary-a",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary A",
                        "proposalConfig": {"cliValues": {}},
                    },
                ],
            }
        )

        self.assertEqual(parsed["selectedProposalIds"], ["evolmd-mo", "binary-mopso-cd"])

    def test_comparator_config_accepts_repository_update_settings(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "updateRepositoriesBeforeRun": True,
                "proposalGitConfigs": {
                    "binary-mopso-cd": {
                        "remote": "origin",
                        "branch": "dev",
                        "pullMode": "ff-only",
                        "expectedRemoteUrl": "https://github.com/escmHEX/BMOPSO-CD.git",
                    }
                },
            }
        )
        self.assertTrue(parsed["updateRepositoriesBeforeRun"])
        self.assertEqual(parsed["proposalGitConfigs"]["binary-mopso-cd"]["remote"], "origin")
        self.assertEqual(parsed["proposalGitConfigs"]["binary-mopso-cd"]["branch"], "dev")
        self.assertEqual(parsed["proposalGitConfigs"]["binary-mopso-cd"]["pullMode"], "ff-only")
        self.assertEqual(
            parsed["proposalGitConfigs"]["binary-mopso-cd"]["expectedRemoteUrl"],
            "https://github.com/escmHEX/BMOPSO-CD.git",
        )

    def test_comparator_tracks_completed_repetitions_in_proposal_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
            run = {
                "runId": "run-1",
                "runDir": str(root / "runs" / "comparator" / "run-1"),
                "status": "running",
                "startedAtEpoch": None,
                "config": {"repetitionsK": 3, "seed": 100, "generaciones": 2},
                "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal, 3)},
                "proposals": [],
                "progress": {},
                "costSummary": {},
                "logs": [],
            }

            def fake_execute_once(_run, _proposal, output_dir, seed):
                return {
                    "proposalId": proposal.proposal_id,
                    "displayName": proposal.display_name,
                    "status": "completed",
                    "outputDir": str(output_dir),
                    "rows": [],
                    "selectedRows": [],
                    "metrics": {},
                    "series": [],
                    "charts": {},
                    "cost": {},
                    "error": None,
                    "seed": seed,
                }

            with patch.object(service, "_execute_proposal_once", side_effect=fake_execute_once):
                result = service._execute_proposal(run, proposal)

            state = run["proposalStates"][proposal.proposal_id]
            self.assertEqual(result["completedRepetitions"], 3)
            self.assertEqual(result["repetitionsK"], 3)
            self.assertEqual(state["completedRepetitions"], 3)
            self.assertEqual(state["totalRepetitions"], 3)
            self.assertEqual(state["currentRepetitionIndex"], 3)

    def test_comparator_does_not_count_failed_single_repetition_as_completed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
            run = {
                "runId": "run-1",
                "runDir": str(root / "runs" / "comparator" / "run-1"),
                "status": "running",
                "startedAtEpoch": None,
                "config": {"repetitionsK": 1, "seed": 100, "generaciones": 2},
                "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal, 1)},
                "proposals": [],
                "progress": {},
                "costSummary": {},
                "logs": [],
            }

            def fake_execute_once(_run, _proposal, output_dir, _seed):
                return {
                    "proposalId": proposal.proposal_id,
                    "displayName": proposal.display_name,
                    "status": "failed",
                    "outputDir": str(output_dir),
                    "rows": [],
                    "metrics": {},
                    "cost": {},
                    "error": "httpx.ReadTimeout: timed out",
                }

            with patch.object(service, "_execute_proposal_once", side_effect=fake_execute_once):
                result = service._execute_proposal(run, proposal)

            state = run["proposalStates"][proposal.proposal_id]
            self.assertEqual(result["completedRepetitions"], 0)
            self.assertEqual(result["repetitionsK"], 1)
            self.assertEqual(state["completedRepetitions"], 0)
            self.assertEqual(state["totalRepetitions"], 1)
            self.assertEqual(state["currentRepetitionIndex"], 1)

    def test_comparator_counts_only_successful_repetitions_after_mixed_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
            run = {
                "runId": "run-1",
                "runDir": str(root / "runs" / "comparator" / "run-1"),
                "status": "running",
                "startedAtEpoch": None,
                "config": {"repetitionsK": 3, "seed": 100, "generaciones": 2},
                "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal, 3)},
                "proposals": [],
                "progress": {},
                "costSummary": {},
                "logs": [],
            }
            statuses = ["completed", "failed", "completed"]

            def fake_execute_once(_run, _proposal, output_dir, seed):
                status = statuses.pop(0)
                return {
                    "proposalId": proposal.proposal_id,
                    "displayName": proposal.display_name,
                    "status": status,
                    "outputDir": str(output_dir),
                    "rows": [],
                    "selectedRows": [],
                    "metrics": {},
                    "series": [],
                    "charts": {},
                    "cost": {},
                    "error": None if status == "completed" else "httpx.ReadTimeout: timed out",
                    "seed": seed,
                }

            with patch.object(service, "_execute_proposal_once", side_effect=fake_execute_once):
                result = service._execute_proposal(run, proposal)

            state = run["proposalStates"][proposal.proposal_id]
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completedRepetitions"], 2)
            self.assertEqual(result["repetitionsK"], 3)
            self.assertEqual(result["error"], "1 repeticion(es) fallaron.")
            self.assertEqual(state["completedRepetitions"], 2)
            self.assertEqual(state["totalRepetitions"], 3)
            self.assertEqual(state["currentRepetitionIndex"], 3)

    def test_record_proposal_result_preserves_zero_completed_repetitions(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        run = {
            "runId": "run-1",
            "runDir": "run-1",
            "status": "running",
            "startedAtEpoch": None,
            "config": {"repetitionsK": 1},
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal, 1)},
            "proposals": [],
            "progress": {},
            "costSummary": {},
            "logs": [],
            "repositoryUpdates": {},
        }
        result = {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "status": "failed",
            "completedRepetitions": 0,
            "repetitionsK": 1,
            "cost": {},
        }

        service._record_proposal_result_unlocked(run, result)

        state = run["proposalStates"][proposal.proposal_id]
        self.assertEqual(state["completedRepetitions"], 0)
        self.assertEqual(state["totalRepetitions"], 1)

    def test_comparator_executes_instance_in_isolated_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            config = service._read_config(
                {
                    "referenceText": "reference",
                    "proposalInstances": [
                        {
                            "instanceId": "binary-a",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary A",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        }
                    ],
                }
            )
            instance = service._selected_instances(config)[0]
            run = {
                "runId": "run-1",
                "runDir": str(root / "runs" / "comparator" / "run-1"),
                "status": "running",
                "startedAtEpoch": None,
                "config": config,
                "proposalStates": {instance.instance_id: service._initial_proposal_state(instance, 1)},
                "proposals": [],
                "progress": {},
                "costSummary": {},
                "logs": [],
            }
            observed_dirs = []

            def fake_execute_once(_run, _instance, output_dir, seed):
                observed_dirs.append(output_dir)
                return {
                    "status": "completed",
                    "outputDir": str(output_dir),
                    "rows": [],
                    "selectedRows": [],
                    "metrics": {},
                    "series": [],
                    "charts": {},
                    "cost": {},
                    "error": None,
                    "seed": seed,
                }

            with patch.object(service, "_execute_proposal_once", side_effect=fake_execute_once):
                result = service._execute_proposal(run, instance)

            self.assertEqual(observed_dirs[0], Path(run["runDir"]) / "binary-a")
            self.assertEqual(result["instanceId"], "binary-a")
            self.assertEqual(result["proposalId"], "binary-mopso-cd")

    def test_recontinue_run_preserves_completed_instances_and_cleans_partial_state(self):
        class InstantThread:
            def __init__(self, target, args=(), daemon=None):
                self._target = target
                self._args = args
                self.daemon = daemon

            def start(self):
                self._target(*self._args)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            config = service._read_config(
                {
                    "referenceText": "reference",
                    "selectedProposalIds": ["binary-mopso-cd"],
                    "repetitionsK": 2,
                    "seed": 7,
                    "proposalInstances": [
                        {
                            "instanceId": "binary-1",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary 1",
                            "proposalConfig": {"cliValues": {"selection.k": "1"}},
                        },
                        {
                            "instanceId": "binary-2",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary 2",
                            "proposalConfig": {"cliValues": {"selection.k": "2"}},
                        },
                        {
                            "instanceId": "binary-3",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary 3",
                            "proposalConfig": {"cliValues": {"selection.k": "3"}},
                        },
                        {
                            "instanceId": "binary-4",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary 4",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        },
                    ],
                }
            )
            instances = service._selected_instances(config)
            run_dir = root / "runs" / "comparator" / "recontinue-run"
            run_dir.mkdir(parents=True)
            for name in ("binary-1", "binary-2", "binary-3", "binary-4"):
                instance_dir = run_dir / name
                instance_dir.mkdir()
                (instance_dir / "marker.txt").write_text(name, encoding="utf-8")
            proposals = [
                {
                    "instanceId": "binary-1",
                    "proposalId": "binary-mopso-cd",
                    "displayName": "Binary 1",
                    "status": "completed",
                    "completedRepetitions": 2,
                    "repetitionsK": 2,
                    "cost": {},
                    "metrics": {},
                    "charts": {},
                    "rows": [],
                },
                {
                    "instanceId": "binary-2",
                    "proposalId": "binary-mopso-cd",
                    "displayName": "Binary 2",
                    "status": "completed",
                    "completedRepetitions": 2,
                    "repetitionsK": 2,
                    "cost": {},
                    "metrics": {},
                    "charts": {},
                    "rows": [],
                },
                {
                    "instanceId": "binary-3",
                    "proposalId": "binary-mopso-cd",
                    "displayName": "Binary 3",
                    "status": "failed",
                    "completedRepetitions": 1,
                    "repetitionsK": 2,
                    "cost": {},
                    "metrics": {},
                    "charts": {},
                    "rows": [],
                    "error": "partial crash",
                },
            ]
            states = {instance.instance_id: service._initial_proposal_state(instance, 2) for instance in instances}
            states["binary-1"].update({"status": "completed", "progress": 1.0, "completedRepetitions": 2})
            states["binary-2"].update({"status": "completed", "progress": 1.0, "completedRepetitions": 2})
            states["binary-3"].update({"status": "running", "progress": 0.45, "completedRepetitions": 1, "currentRepetitionIndex": 2})
            run = {
                "runId": "recontinue-run",
                "runDir": str(run_dir),
                "status": "running",
                "startedAtEpoch": None,
                "config": config,
                "proposalStates": states,
                "proposals": proposals,
                "progress": {},
                "costSummary": {},
                "logs": [
                    {"proposalId": "binary-1", "message": "completed log"},
                    {"proposalId": "binary-3", "message": "partial log"},
                    {"proposalId": "binary-4", "message": "queued partial log"},
                ],
                "repositoryUpdates": {},
                "cancelRequested": False,
                "activeProcesses": {},
            }
            service._runs["recontinue-run"] = run
            log_path = run_dir / "logs.jsonl"
            log_path.write_text(
                "\n".join(
                    json.dumps(entry, ensure_ascii=False)
                    for entry in [
                        {"proposalId": "binary-1", "message": "completed log"},
                        {"proposalId": "binary-3", "message": "partial log"},
                        {"proposalId": "binary-4", "message": "queued partial log"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            launched: list[str] = []

            def fake_run_proposals_sequential(run_payload, remaining_instances):
                launched.extend(instance.instance_id for instance in remaining_instances)
                for instance in remaining_instances:
                    service._record_proposal_result_unlocked(
                        run_payload,
                        {
                            "instanceId": instance.instance_id,
                            "proposalId": instance.proposal_id,
                            "displayName": instance.display_name,
                            "baseDisplayName": instance.base_display_name,
                            "proposalConfig": instance.proposal_config,
                            "status": "completed",
                            "completedRepetitions": 2,
                            "repetitionsK": 2,
                            "cost": {},
                            "metrics": {},
                            "charts": {},
                            "rows": [],
                            "error": None,
                        },
                    )

            with patch("baselines.comparator.threading.Thread", InstantThread):
                with patch.object(service, "_run_proposals_sequential", side_effect=fake_run_proposals_sequential):
                    result = service.recontinue_run("recontinue-run")

            self.assertEqual(launched, ["binary-3", "binary-4"])
            self.assertEqual(result["status"], "completed")
            self.assertTrue((run_dir / "binary-1" / "marker.txt").exists())
            self.assertTrue((run_dir / "binary-2" / "marker.txt").exists())
            self.assertFalse((run_dir / "binary-3" / "marker.txt").exists())
            self.assertFalse((run_dir / "binary-4" / "marker.txt").exists())
            self.assertNotIn("partial crash", json.dumps(result, ensure_ascii=False))
            self.assertEqual([proposal["instanceId"] for proposal in result["proposals"]], ["binary-1", "binary-2", "binary-3", "binary-4"])
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("completed log", log_text)
            self.assertNotIn("partial log", log_text)
            self.assertNotIn("queued partial log", log_text)
            self.assertNotIn("recontinu", log_text.lower())
            self.assertNotIn("binary-3", [entry["proposalId"] for entry in run["logs"]])
            self.assertNotIn("binary-4", [entry["proposalId"] for entry in run["logs"]])

    def test_recontinue_run_rejects_completed_run_without_touching_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "completed-run"
            run_dir.mkdir(parents=True)
            preserved = run_dir / "binary-1" / "marker.txt"
            preserved.parent.mkdir()
            preserved.write_text("keep", encoding="utf-8")
            run = {
                "runId": "completed-run",
                "runDir": str(run_dir),
                "status": "completed",
                "config": {"proposalInstances": []},
                "proposalStates": {},
                "proposals": [],
                "progress": {},
                "logs": [],
                "costSummary": {},
                "activeProcesses": {},
            }
            service._runs["completed-run"] = run

            with self.assertRaises(ValueError):
                service.recontinue_run("completed-run")

            self.assertTrue(preserved.exists())

    def test_recontinue_run_rejects_live_process_without_touching_files(self):
        class LiveProcess:
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "live-run"
            run_dir.mkdir(parents=True)
            preserved = run_dir / "binary-1" / "marker.txt"
            preserved.parent.mkdir()
            preserved.write_text("keep", encoding="utf-8")
            run = {
                "runId": "live-run",
                "runDir": str(run_dir),
                "status": "running",
                "config": {"proposalInstances": []},
                "proposalStates": {},
                "proposals": [],
                "progress": {},
                "logs": [],
                "costSummary": {},
                "activeProcesses": {"binary-1": LiveProcess()},
            }
            service._runs["live-run"] = run

            with self.assertRaises(RuntimeError):
                service.recontinue_run("live-run")

            self.assertTrue(preserved.exists())

    def test_recontinue_run_can_load_stale_running_summary_from_disk(self):
        class InstantThread:
            def __init__(self, target, args=(), daemon=None):
                self._target = target
                self._args = args
                self.daemon = daemon

            def start(self):
                self._target(*self._args)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            config = service._read_config(
                {
                    "referenceText": "reference",
                    "selectedProposalIds": ["binary-mopso-cd"],
                    "proposalInstances": [
                        {
                            "instanceId": "binary-1",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary 1",
                            "proposalConfig": {"cliValues": {"selection.k": "1"}},
                        },
                        {
                            "instanceId": "binary-2",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary 2",
                            "proposalConfig": {"cliValues": {"selection.k": "2"}},
                        },
                    ],
                }
            )
            instances = service._selected_instances(config)
            run_dir = root / "runs" / "comparator" / "disk-run"
            run_dir.mkdir(parents=True)
            (run_dir / "binary-2").mkdir()
            (run_dir / "binary-2" / "partial.txt").write_text("partial", encoding="utf-8")
            states = {instance.instance_id: service._initial_proposal_state(instance, 1) for instance in instances}
            states["binary-1"].update({"status": "completed", "progress": 1.0, "completedRepetitions": 1})
            states["binary-2"].update({"status": "running", "progress": 0.2})
            summary = {
                "runId": "disk-run",
                "runDir": str(run_dir),
                "status": "running",
                "config": config,
                "proposalStates": states,
                "proposals": [
                    {
                        "instanceId": "binary-1",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary 1",
                        "status": "completed",
                        "cost": {},
                        "metrics": {},
                        "charts": {},
                        "rows": [],
                    }
                ],
                "progress": {},
                "logs": [{"proposalId": "binary-2", "message": "partial"}],
                "costSummary": {},
                "cancelRequested": False,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            launched: list[str] = []

            def fake_run_proposals_sequential(run_payload, remaining_instances):
                launched.extend(instance.instance_id for instance in remaining_instances)
                for instance in remaining_instances:
                    service._record_proposal_result_unlocked(
                        run_payload,
                        {
                            "instanceId": instance.instance_id,
                            "proposalId": instance.proposal_id,
                            "displayName": instance.display_name,
                            "status": "completed",
                            "completedRepetitions": 1,
                            "repetitionsK": 1,
                            "cost": {},
                            "metrics": {},
                            "charts": {},
                            "rows": [],
                            "error": None,
                        },
                    )

            with patch("baselines.comparator.threading.Thread", InstantThread):
                with patch.object(service, "_run_proposals_sequential", side_effect=fake_run_proposals_sequential):
                    result = service.recontinue_run("disk-run")

            self.assertEqual(launched, ["binary-2"])
            self.assertEqual(result["status"], "completed")
            self.assertFalse((run_dir / "binary-2" / "partial.txt").exists())

    def test_comparator_reads_full_logs_in_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "run-logs"
            run_dir.mkdir(parents=True)
            run = {
                "runId": "run-logs",
                "runDir": str(run_dir),
                "status": "running",
                "config": {},
                "proposalStates": {},
                "progress": {},
                "logs": [],
            }
            service._runs["run-logs"] = run
            for index in range(7):
                service._append_log_unlocked(run, "system", f"line {index}")

            first = service.get_run_logs("run-logs", offset=0, limit=3)
            second = service.get_run_logs("run-logs", offset=first["nextOffset"], limit=3)
            third = service.get_run_logs("run-logs", offset=second["nextOffset"], limit=3)

            self.assertEqual([entry["message"] for entry in first["logs"]], ["line 0", "line 1", "line 2"])
            self.assertTrue(first["hasMore"])
            self.assertEqual([entry["message"] for entry in second["logs"]], ["line 3", "line 4", "line 5"])
            self.assertTrue(second["hasMore"])
            self.assertEqual([entry["message"] for entry in third["logs"]], ["line 6"])
            self.assertFalse(third["hasMore"])
            self.assertEqual(third["source"], "jsonl")

    def test_comparator_snapshot_zip_contains_complete_run_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "run-download"
            nested_dir = run_dir / "binary-mopso-cd" / "exec"
            nested_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text('{"runId":"run-download"}', encoding="utf-8")
            (nested_dir / "runtime.log").write_text("ok", encoding="utf-8")

            with tempfile.TemporaryDirectory() as snapshot_temp, tempfile.TemporaryDirectory() as zip_temp:
                snapshot_dir = service.snapshot_run_directory("run-download", Path(snapshot_temp))
                zip_path = Path(zip_temp) / "run-download.zip"
                service.write_run_snapshot_zip(snapshot_dir, zip_path)

                with zipfile.ZipFile(zip_path) as archive:
                    self.assertEqual(
                        sorted(archive.namelist()),
                        [
                            "run-download/binary-mopso-cd/exec/runtime.log",
                            "run-download/summary.json",
                        ],
                    )
                    self.assertEqual(archive.read("run-download/summary.json").decode("utf-8"), '{"runId":"run-download"}')

            self.assertEqual((run_dir / "summary.json").read_text(encoding="utf-8"), '{"runId":"run-download"}')

    def test_comparator_snapshot_rejects_run_id_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            service = ComparatorService(Path(temp_dir))

            with tempfile.TemporaryDirectory() as snapshot_temp:
                with self.assertRaises(ValueError):
                    service.snapshot_run_directory("../outside", Path(snapshot_temp))

    def test_comparator_snapshot_retries_file_that_changes_during_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "run-changing"
            run_dir.mkdir(parents=True)
            source_file = run_dir / "summary.json"
            source_file.write_text("first", encoding="utf-8")
            original_copy2 = comparator_module.shutil.copy2
            copy_attempts = 0

            def changing_copy(src, dst, *args, **kwargs):
                nonlocal copy_attempts
                copy_attempts += 1
                result = original_copy2(src, dst, *args, **kwargs)
                if copy_attempts == 1:
                    source_file.write_text("second", encoding="utf-8")
                return result

            with tempfile.TemporaryDirectory() as snapshot_temp:
                with patch.object(comparator_module.shutil, "copy2", side_effect=changing_copy):
                    snapshot_dir = service.snapshot_run_directory("run-changing", Path(snapshot_temp))

                self.assertGreaterEqual(copy_attempts, 2)
                self.assertEqual((snapshot_dir / "summary.json").read_text(encoding="utf-8"), "second")

    def test_comparator_snapshot_fails_when_file_never_stabilizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "run-unstable"
            run_dir.mkdir(parents=True)
            source_file = run_dir / "summary.json"
            source_file.write_text("0", encoding="utf-8")
            original_copy2 = comparator_module.shutil.copy2
            copy_attempts = 0

            def unstable_copy(src, dst, *args, **kwargs):
                nonlocal copy_attempts
                copy_attempts += 1
                result = original_copy2(src, dst, *args, **kwargs)
                source_file.write_text("x" * (copy_attempts + 1), encoding="utf-8")
                return result

            with tempfile.TemporaryDirectory() as snapshot_temp:
                with patch.object(comparator_module.shutil, "copy2", side_effect=unstable_copy):
                    with self.assertRaises(RuntimeError):
                        service.snapshot_run_directory("run-unstable", Path(snapshot_temp))

                self.assertEqual(copy_attempts, comparator_module.RUN_SNAPSHOT_COPY_ATTEMPTS)

    def test_fair_sequential_forces_effective_parallelism_to_one(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["evolmd-mo", "binary-mopso-cd"],
                "executionMode": "fair_sequential",
                "proposalParallelism": 3,
            }
        )
        self.assertEqual(parsed["proposalParallelism"], 3)
        self.assertEqual(parsed["effectiveProposalParallelism"], 1)
        self.assertTrue(parsed["costsComparable"])
        self.assertTrue(parsed["executionPolicy"]["costsComparable"])

    def test_exploratory_parallel_keeps_effective_parallelism_and_marks_costs_non_comparable(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["evolmd-mo", "binary-mopso-cd"],
                "executionMode": "exploratory_parallel",
                "proposalParallelism": 3,
            }
        )
        self.assertEqual(parsed["proposalParallelism"], 3)
        self.assertEqual(parsed["effectiveProposalParallelism"], 3)
        self.assertFalse(parsed["costsComparable"])
        self.assertFalse(parsed["executionPolicy"]["costsComparable"])

    def test_comparator_parses_timestamped_binary_stage_log(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        service._apply_log_progress_unlocked(
            run,
            proposal.proposal_id,
            "2026-06-03 00:05:01 | INFO | 3/6 Construyendo poblacion inicial",
        )
        state = run["proposalStates"][proposal.proposal_id]
        self.assertEqual(state["stageIndex"], 3)
        self.assertEqual(state["stageTotal"], 6)
        self.assertEqual(state["stageLabel"], "Construyendo poblacion inicial")

    def test_comparator_promotes_binary_initialization_detail_log(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        service._apply_log_progress_unlocked(
            run,
            proposal.proposal_id,
            "2026-06-03 00:05:01 | INFO | 3/6 Construyendo poblacion inicial",
        )
        progress_detail = (
            "initial population | text generation progress | completed=5/10 (50%) "
            "| failed=0 | max_concurrent=10 | elapsed=00:00:33"
        )
        service._apply_log_progress_unlocked(
            run,
            proposal.proposal_id,
            f"2026-06-03 00:05:45 | INFO | {progress_detail}",
        )

        state = run["proposalStates"][proposal.proposal_id]
        self.assertEqual(state["stageLabel"], progress_detail)
        self.assertGreater(state["progress"], 0.40)

    def test_comparator_parses_timestamped_binary_generation_log(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        service._apply_log_progress_unlocked(
            run,
            proposal.proposal_id,
            "2026-06-03 00:05:20 | INFO | run 1/1 | generation 29/30 | modified=5/20 | archive=10 | hv=0.301896 | elapsed=00:08:44",
        )
        state = run["proposalStates"][proposal.proposal_id]
        self.assertEqual(state["generationIndex"], 29)
        self.assertEqual(state["generationTotal"], 30)
        self.assertEqual(state["stageLabel"], "Generacion 29/30")

    def test_comparator_eta_uses_evolmd_generation_time_average(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "config": {"generaciones": 10, "repetitionsK": 1, "executionMode": "fair_sequential"},
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        service._apply_log_progress_unlocked(run, proposal.proposal_id, "Generacion 3/10")
        service._apply_log_progress_unlocked(run, proposal.proposal_id, "Tiempo Gen: 7.50s")

        timing = run["proposalStates"][proposal.proposal_id]["iterationTiming"]
        self.assertEqual(timing["completedIterations"], 3)
        self.assertEqual(timing["totalIterations"], 10)
        self.assertEqual(timing["remainingIterations"], 7)
        self.assertAlmostEqual(timing["averageIterationSeconds"], 7.5)
        self.assertAlmostEqual(run["progress"]["remainingSeconds"], 52.5)

    def test_comparator_eta_uses_binary_elapsed_delta(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "config": {"generaciones": 3, "repetitionsK": 1, "executionMode": "fair_sequential"},
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        service._apply_log_progress_unlocked(
            run,
            proposal.proposal_id,
            "2026-06-03 00:05:14 | INFO | run 1/1 | generation 1/3 started | elapsed=00:00:10",
        )
        service._apply_log_progress_unlocked(
            run,
            proposal.proposal_id,
            "2026-06-03 00:05:20 | INFO | run 1/1 | generation 1/3 | modified=2/20 | archive=4 | hv=0.1 | elapsed=00:00:16",
        )

        timing = run["proposalStates"][proposal.proposal_id]["iterationTiming"]
        self.assertEqual(timing["completedIterations"], 1)
        self.assertEqual(timing["remainingIterations"], 2)
        self.assertAlmostEqual(timing["averageIterationSeconds"], 6.0)
        self.assertAlmostEqual(run["progress"]["remainingSeconds"], 12.0)

    def test_comparator_eta_waits_for_first_completed_iteration(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "config": {"generaciones": 3, "repetitionsK": 1, "executionMode": "fair_sequential"},
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        service._apply_log_progress_unlocked(
            run,
            proposal.proposal_id,
            "2026-06-03 00:05:14 | INFO | run 1/1 | generation 1/3 started | elapsed=00:00:10",
        )

        timing = run["proposalStates"][proposal.proposal_id]["iterationTiming"]
        self.assertEqual(timing["completedIterations"], 0)
        self.assertIsNone(run["progress"]["remainingSeconds"])
        self.assertEqual(run["progress"]["remainingLabel"], "Esperando primera iteracion")

    def test_comparator_eta_ignores_duplicate_generation_completion_logs(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "config": {"generaciones": 3, "repetitionsK": 1, "executionMode": "fair_sequential"},
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        start_log = "2026-06-03 00:05:14 | INFO | run 1/1 | generation 1/3 started | elapsed=00:00:10"
        end_log = "2026-06-03 00:05:20 | INFO | run 1/1 | generation 1/3 | modified=2/20 | archive=4 | hv=0.1 | elapsed=00:00:16"
        service._apply_log_progress_unlocked(run, proposal.proposal_id, start_log)
        service._apply_log_progress_unlocked(run, proposal.proposal_id, end_log)
        service._apply_log_progress_unlocked(run, proposal.proposal_id, end_log)

        timing = run["proposalStates"][proposal.proposal_id]["iterationTiming"]
        self.assertEqual(timing["completedIterations"], 1)
        self.assertEqual(len(timing["durationSamples"]), 1)
        self.assertAlmostEqual(run["progress"]["remainingSeconds"], 12.0)

    def test_comparator_eta_total_iterations_include_repetitions(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd")
        run = {
            "status": "running",
            "startedAtEpoch": None,
            "cancelRequested": False,
            "config": {"generaciones": 2, "repetitionsK": 3, "executionMode": "fair_sequential"},
            "proposalStates": {proposal.proposal_id: service._initial_proposal_state(proposal)},
        }
        service._apply_log_progress_unlocked(run, proposal.proposal_id, "Generacion 1/2")
        service._apply_log_progress_unlocked(run, proposal.proposal_id, "Tiempo Gen: 5.00s")

        timing = run["proposalStates"][proposal.proposal_id]["iterationTiming"]
        self.assertEqual(timing["completedIterations"], 1)
        self.assertEqual(timing["totalIterations"], 6)
        self.assertEqual(timing["remainingIterations"], 5)
        self.assertAlmostEqual(run["progress"]["remainingSeconds"], 25.0)

    def test_comparator_config_rejects_unsafe_repository_update_settings(self):
        service = ComparatorService(Path("."))
        bad_configs = [
            {"remote": "origin;rm", "branch": "dev", "pullMode": "ff-only"},
            {"remote": "origin", "branch": "../main", "pullMode": "ff-only"},
            {"remote": "origin", "branch": "main", "pullMode": "merge"},
        ]
        for git_config in bad_configs:
            with self.subTest(git_config=git_config):
                with self.assertRaises(ValueError):
                    service._read_config(
                        {
                            "referenceText": "reference",
                            "selectedProposalIds": ["binary-mopso-cd"],
                            "proposalGitConfigs": {"binary-mopso-cd": git_config},
                        }
                    )

    def test_comparator_config_accepts_structured_cli_values(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "experiment.frozen_components": ["role", "topic", "action"],
                            "monitor.enabled": True,
                            "router.heuristics.word_replacement_candidates": False,
                            "router.task_models.synthetic_text_generation": "llama3.1:8b",
                            "semantic_components.order": ["topic", "role", "action"],
                            "models.sbert.default": "gte-small",
                        }
                    }
                },
            }
        )
        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertEqual(values["experiment.frozen_components"], ["role", "topic", "action"])
        self.assertTrue(values["monitor.enabled"])
        self.assertFalse(values["router.heuristics.word_replacement_candidates"])
        self.assertEqual(values["router.task_models.synthetic_text_generation"], "llama3.1:8b")
        self.assertEqual(values["semantic_components.order"], ["topic", "role", "action"])
        self.assertEqual(values["models.sbert.default"], "gte-small")

    def test_binary_cli_options_are_generated_from_default_yaml(self):
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        options_by_key = {
            str(option.get("key") or option.get("configPath") or option.get("flag")): option
            for option in proposal.cli_options
        }

        self.assertIn("--config", options_by_key)
        self.assertIn("mopso.archive_multiplier", options_by_key)
        self.assertIn("mopso.guided_trajectory_validation_enabled", options_by_key)
        self.assertIn("mopso.guided_trajectory_relative_margin", options_by_key)
        self.assertIn("router.task_models.synthetic_text_generation", options_by_key)
        self.assertIn("selection.lambda_mmr", options_by_key)
        self.assertEqual(options_by_key["mopso.archive_multiplier"]["flag"], "--set")
        self.assertEqual(options_by_key["mopso.archive_multiplier"]["type"], "float")
        self.assertEqual(options_by_key["mopso.alpha"]["type"], "float")
        guided_validation = options_by_key["mopso.guided_trajectory_validation_enabled"]
        guided_margin = options_by_key["mopso.guided_trajectory_relative_margin"]
        self.assertEqual(guided_validation["group"], "mopso")
        self.assertEqual(guided_validation["type"], "bool")
        self.assertTrue(guided_validation["allowFalse"])
        self.assertEqual(guided_validation["label"], "Validacion trayectoria guiada")
        self.assertIn("validacion angular", guided_validation["valueHelp"])
        self.assertEqual(guided_margin["group"], "mopso")
        self.assertEqual(guided_margin["type"], "float")
        self.assertEqual(guided_margin["label"], "Margen relativo trayectoria guiada")
        self.assertIn("no negativo", guided_margin["valueHelp"])
        self.assertEqual(options_by_key["router.heuristics.word_replacement_candidates"]["type"], "bool")
        self.assertTrue(options_by_key["router.heuristics.word_replacement_candidates"]["allowFalse"])
        self.assertEqual(options_by_key["experiment.n"]["source"], "managed")
        self.assertIn("auto: N", options_by_key["parallelism.particle_update_max_concurrent"]["valueHelp"])
        self.assertIn("auto: N", options_by_key["parallelism.initial_text_generation_max_concurrent"]["valueHelp"])
        self.assertEqual(options_by_key["experiment.frozen_components"]["type"], "component_multi_select")
        self.assertIn("poblacion inicial", options_by_key["experiment.frozen_components"]["valueHelp"])
        self.assertNotIn("no permite", options_by_key["experiment.frozen_components"]["valueHelp"])
        self.assertEqual(options_by_key["semantic_components.order"]["type"], "ordered_multi_select")
        self.assertEqual(options_by_key["semantic_components.expansion_order"]["type"], "ordered_multi_select")
        self.assertEqual(options_by_key["logging.level"]["ui"], "select")
        self.assertFalse(options_by_key["logging.level"]["allowCustom"])
        self.assertIn("llama3.1:8b", options_by_key["router.task_models.synthetic_text_generation"]["choices"])
        self.assertIn("qwen3:4b-instruct-2507-q4_K_M", options_by_key["router.task_models.synthetic_text_generation"]["choices"])
        self.assertIn("phi4-mini", options_by_key["router.task_models.synthetic_text_generation"]["choices"])
        self.assertIn("ministral-3:3b", options_by_key["router.task_models.synthetic_text_generation"]["choices"])
        self.assertTrue(options_by_key["router.task_models.synthetic_text_generation"]["allowCustom"])
        self.assertIn("Solo aplica a llamadas Ollama", options_by_key["ollama.timeout_seconds"]["valueHelp"])
        self.assertIn("PPDB", options_by_key["ollama.timeout_seconds"]["valueHelp"])

    def test_binary_proposal_cli_options_refresh_model_choices_from_default_yaml(self):
        service = ComparatorService(Path("."))
        config = comparator_module.load_binary_default_config(comparator_module.binary_default_config_path())
        config["ollama"]["model_options"] = list(config["ollama"]["model_options"]) + ["hot-model:7b"]

        with patch.object(comparator_module, "load_binary_default_config", return_value=config):
            proposals = service.list_proposals()
        binary = next(item for item in proposals if item["proposalId"] == "binary-mopso-cd")
        options_by_key = {
            str(option.get("key") or option.get("configPath") or option.get("flag")): option
            for option in binary["cliOptions"]
        }

        choices = options_by_key["router.task_models.synthetic_text_generation"]["choices"]
        self.assertIn("hot-model:7b", choices)

    def test_binary_cli_options_rebuild_from_current_default_yaml(self):
        config = {
            "ollama": {
                "alternative_model": "llama3",
                "model_options": ["llama3", "fresh-model:1b"],
                "model_capabilities": {
                    "fresh-model:1b": {
                        "thinking": True,
                    },
                },
            },
            "router": {
                "task_models": {
                    "synthetic_text_generation": "llama3",
                },
                "task_thinking": {
                    "synthetic_text_generation": None,
                },
            },
        }

        options, error = comparator_module.build_binary_cli_options(config)
        self.assertIsNone(error)
        options_by_key = {
            str(option.get("key") or option.get("configPath") or option.get("flag")): option
            for option in options
        }

        self.assertIn("fresh-model:1b", options_by_key["ollama.alternative_model"]["choices"])
        self.assertIn("fresh-model:1b", options_by_key["router.task_models.synthetic_text_generation"]["choices"])

    def test_comparator_public_defaults_use_binary_model_options_and_capabilities(self):
        service = ComparatorService(Path("."))

        defaults = service.public_defaults()

        self.assertIn("qwen3:4b-instruct-2507-q4_K_M", defaults["ollamaModelOptions"])
        self.assertIn("phi4-mini", defaults["ollamaModelOptions"])
        self.assertIn("ministral-3:3b", defaults["ollamaModelOptions"])
        self.assertTrue(defaults["ollamaModelCapabilities"]["qwen3.5:2b"]["thinking"])
        self.assertTrue(defaults["ollamaModelCapabilities"]["qwen3:4b-instruct-2507-q4_K_M"]["thinking"])
        self.assertFalse(defaults["ollamaModelCapabilities"]["llama3.1:8b"]["thinking"])
        self.assertEqual(
            defaults["ollamaModelCapabilities"]["qwen3.5:9b"]["validated_thinking_tasks"],
            ["semantic_anchor_extraction"],
        )
        self.assertEqual(
            defaults["ollamaModelCapabilities"]["lfm2.5:8b"]["validated_thinking_tasks"],
            ["semantic_anchor_extraction", "semantic_pool_generation"],
        )

    def test_binary_task_thinking_options_are_generated_for_structured_ui(self):
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        options_by_key = {
            str(option.get("key") or option.get("configPath") or option.get("flag")): option
            for option in proposal.cli_options
        }

        thinking = options_by_key["router.task_thinking.synthetic_text_generation"]

        self.assertEqual(thinking["flag"], "--set")
        self.assertEqual(thinking["type"], "thinking_mode")
        self.assertEqual(thinking["choices"], ["false", "low", "medium", "high"])
        self.assertTrue(thinking["allowFalse"])
        self.assertEqual(thinking["pairedModelPath"], "router.task_models.synthetic_text_generation")

    def test_simple_yaml_parser_supports_nested_sequences(self):
        parsed = parse_simple_yaml_mapping(
            """
            ollama:
              model_options:
                - "llama3.1:8b"
                - "qwen3.5:2b"
              model_capabilities:
                qwen3.5:2b:
                  thinking: true
                  validated_thinking_tasks:
                    - semantic_anchor_extraction
              stream: false
            """
        )

        self.assertEqual(parsed["ollama"]["model_options"], ["llama3.1:8b", "qwen3.5:2b"])
        self.assertTrue(parsed["ollama"]["model_capabilities"]["qwen3.5:2b"]["thinking"])
        self.assertEqual(
            parsed["ollama"]["model_capabilities"]["qwen3.5:2b"]["validated_thinking_tasks"],
            ["semantic_anchor_extraction"],
        )
        self.assertFalse(parsed["ollama"]["stream"])

    def test_binary_mopso_float_overrides_accept_decimal_values(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "mopso.alpha": "1.25",
                            "mopso.archive_multiplier": "0.5",
                            "mopso.guided_trajectory_relative_margin": "0.40",
                        }
                    }
                },
            }
        )
        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertEqual(values["mopso.alpha"], "1.25")
        self.assertEqual(values["mopso.archive_multiplier"], "0.5")
        self.assertEqual(values["mopso.guided_trajectory_relative_margin"], "0.40")

    def test_binary_mopso_guided_trajectory_bool_override_accepts_false(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "mopso.guided_trajectory_validation_enabled": False,
                        }
                    }
                },
            }
        )
        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertIs(values["mopso.guided_trajectory_validation_enabled"], False)

    def test_binary_task_thinking_allows_supported_model_even_without_task_validation(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "model": "qwen3.5:9b",
                "proposalInstances": [
                    {
                        "instanceId": "binary-mopso-cd-3",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary MOPSO-CD 3",
                        "proposalConfig": {
                            "cliValues": {
                                "router.task_models.semantic_pool_generation": "qwen3.5:9b",
                                "router.task_thinking.semantic_pool_generation": "low",
                            }
                        },
                    }
                ],
            }
        )

        values = parsed["proposalInstances"][0]["proposalConfig"]["cliValues"]
        self.assertEqual(values["router.task_thinking.semantic_pool_generation"], "low")

    def test_binary_task_thinking_rejects_model_without_thinking_support(self):
        service = ComparatorService(Path("."))
        with self.assertRaisesRegex(ValueError, "llama3.*semantic_pool_generation.*does not support thinking"):
            service._read_config(
                {
                    "referenceText": "reference",
                    "model": "llama3",
                    "selectedProposalIds": ["binary-mopso-cd"],
                    "proposalConfigs": {
                        "binary-mopso-cd": {
                            "cliValues": {
                                "router.task_models.semantic_pool_generation": "llama3",
                                "router.task_thinking.semantic_pool_generation": "low",
                            }
                        }
                    },
                }
            )

    def test_binary_task_thinking_allows_validated_model_task(self):
        service = ComparatorService(Path("."))

        parsed = service._read_config(
            {
                "referenceText": "reference",
                "model": "lfm2.5:8b",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "router.task_models.semantic_pool_generation": "lfm2.5:8b",
                            "router.task_thinking.semantic_pool_generation": "medium",
                        }
                    }
                },
            }
        )

        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertEqual(values["router.task_thinking.semantic_pool_generation"], "medium")

    def test_binary_task_thinking_accepts_legacy_true_for_supported_model(self):
        service = ComparatorService(Path("."))

        parsed = service._read_config(
            {
                "referenceText": "reference",
                "model": "qwen3.5:9b",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "router.task_models.semantic_anchor_extraction": "qwen3.5:9b",
                            "router.task_thinking.semantic_anchor_extraction": True,
                        }
                    }
                },
            }
        )

        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertIs(values["router.task_thinking.semantic_anchor_extraction"], True)

    def test_binary_cli_metadata_matches_real_config_casts_for_core_groups(self):
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        options_by_key = {
            str(option.get("key") or option.get("configPath") or option.get("flag")): option
            for option in proposal.cli_options
        }
        expected_types = {
            "mopso.alpha": "float",
            "mopso.archive_multiplier": "float",
            "mopso.c1": "float",
            "mopso.c2": "float",
            "mopso.dmax": "int",
            "mopso.k_retry": "int",
            "mopso.kcand": "int",
            "mopso.leader_tournament_size": "int",
            "mopso.guided_trajectory_validation_enabled": "bool",
            "mopso.guided_trajectory_relative_margin": "float",
            "mopso.omega_max": "float",
            "mopso.omega_min": "float",
            "mopso.p_anchor_enabled": "bool",
            "mopso.p_anchor_max": "float",
            "mopso.p_anchor_min": "float",
            "mopso.p_tur_max": "float",
            "mopso.p_tur_min": "float",
            "mopso.tau_dup": "float",
            "mopso.tau_tur_max": "float",
            "mopso.tau_tur_min": "float",
            "mopso.utility_weights.f1": "float",
            "mopso.utility_weights.f2": "float",
            "mopso.vmax": "float",
            "parallelism.enabled": "bool",
            "parallelism.initial_text_generation_max_concurrent": "int",
            "parallelism.particle_update_max_concurrent": "int",
            "selection.enabled": "bool",
            "selection.epsilon": "float",
            "selection.k": "int",
            "selection.lambda_mmr": "float",
            "selection.tau_max": "float",
            "selection.tau_min": "float",
        }

        for key, expected_type in expected_types.items():
            with self.subTest(key=key):
                self.assertIn(key, options_by_key)
                self.assertEqual(options_by_key[key]["type"], expected_type)

    def test_evolmd_ga_flags_are_proposal_specific_cli_values(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["evolmd"],
                "proposalConfigs": {
                    "evolmd": {
                        "cliValues": {
                            "--k": "5",
                            "--prob-crossover": "0.7",
                            "--prob-mutacion": "0.05",
                        }
                    }
                },
            }
        )
        values = parsed["proposalConfigs"]["evolmd"]["cliValues"]
        self.assertNotIn("k", parsed)
        self.assertEqual(values["--k"], "5")
        self.assertEqual(values["--prob-crossover"], "0.7")
        self.assertEqual(values["--prob-mutacion"], "0.05")

    def test_manual_proposal_cli_metadata_does_not_invent_ranges(self):
        flags_by_proposal = {
            "evolmd": ("--k", "--prob-crossover", "--prob-mutacion", "--num-elitismo"),
            "evolmd-mo": ("--k", "--prob-crossover", "--prob-mutacion", "--num-elitismo"),
            "mesap": ("--k", "--elite_size", "--prob_crossover", "--prob_mutation"),
        }

        for proposal_id, flags in flags_by_proposal.items():
            proposal = next(item for item in PROPOSALS if item.proposal_id == proposal_id)
            options_by_flag = {str(option.get("flag")): option for option in proposal.cli_options}
            for flag in flags:
                with self.subTest(proposal_id=proposal_id, flag=flag):
                    self.assertIn(flag, options_by_flag)
                    self.assertNotIn("min", options_by_flag[flag])
                    self.assertNotIn("max", options_by_flag[flag])

    def test_manual_proposal_cli_values_accept_argparse_compatible_ranges(self):
        service = ComparatorService(Path("."))
        cases = {
            "evolmd": {
                "--k": "0",
                "--prob-crossover": "1.2",
                "--prob-mutacion": "-0.1",
                "--num-elitismo": "-1",
            },
            "evolmd-mo": {
                "--k": "0",
                "--prob-crossover": "1.2",
                "--prob-mutacion": "-0.1",
                "--num-elitismo": "-1",
            },
            "mesap": {
                "--k": "0",
                "--elite_size": "-1",
                "--prob_crossover": "1.2",
                "--prob_mutation": "-0.1",
            },
        }

        for proposal_id, cli_values in cases.items():
            with self.subTest(proposal_id=proposal_id):
                parsed = service._read_config(
                    {
                        "referenceText": "reference",
                        "selectedProposalIds": [proposal_id],
                        "proposalConfigs": {proposal_id: {"cliValues": cli_values}},
                    }
                )
                self.assertEqual(parsed["proposalConfigs"][proposal_id]["cliValues"], cli_values)

    def test_cli_values_still_reject_wrong_types(self):
        service = ComparatorService(Path("."))
        cases = (
            ("binary-mopso-cd", {"mopso.alpha": "abc"}),
            ("evolmd", {"--k": "abc"}),
            ("evolmd-mo", {"--prob-crossover": "abc"}),
            ("mesap", {"--prob_mutation": "abc"}),
        )

        for proposal_id, cli_values in cases:
            with self.subTest(proposal_id=proposal_id):
                with self.assertRaises(ValueError):
                    service._read_config(
                        {
                            "referenceText": "reference",
                            "selectedProposalIds": [proposal_id],
                            "proposalConfigs": {proposal_id: {"cliValues": cli_values}},
                        }
                    )

    def test_evolmd_command_builds_proposal_specific_ga_flags(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["evolmd"],
                "proposalConfigs": {
                    "evolmd": {
                        "cliValues": {
                            "--k": "5",
                            "--prob-crossover": "0.7",
                            "--prob-mutacion": "0.05",
                        }
                    }
                },
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )
        self.assertEqual(command[command.index("--k") + 1], "5")
        self.assertEqual(command[command.index("--prob-crossover") + 1], "0.7")
        self.assertEqual(command[command.index("--prob-mutacion") + 1], "0.05")

    def test_evolmd_command_does_not_force_proposal_specific_defaults(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd")
        command = service._build_command(
            {"config": service._read_config({"referenceText": "reference", "selectedProposalIds": ["evolmd"]})},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )
        self.assertNotIn("--k", command)
        self.assertNotIn("--prob-crossover", command)
        self.assertNotIn("--prob-mutacion", command)

    def test_mesap_proposal_is_declared_from_fork(self):
        proposal = next(item for item in PROPOSALS if item.proposal_id == "mesap")

        self.assertEqual(proposal.display_name, "MESAP")
        self.assertEqual(proposal.repository_path, "baselines/external/mesap")
        self.assertEqual(proposal.result_file, "population_final.json")
        self.assertEqual(proposal.metrics_series_file, "metrics_log.csv")
        self.assertTrue(proposal.single_objective)
        self.assertEqual(
            proposal.git_expected_remote_url,
            "https://github.com/escmHEX/Modelo-Evolutivo-Semantico-Adaptativo-para-Prompts.git",
        )

    def test_mesap_command_builds_real_cli_flags(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "mesap")
        config = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["mesap"],
                "proposalConfigs": {
                    "mesap": {
                        "cliValues": {
                            "--k": "5",
                            "--elite_size": "3",
                            "--prob_crossover": "0.7",
                            "--prob_mutation": "0.05",
                            "--bert_model": "roberta-large",
                        }
                    }
                },
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )

        self.assertIn("--generations", command)
        self.assertIn("--outdir_base", command)
        self.assertIn("--reference_text", command)
        self.assertNotIn("--generaciones", command)
        self.assertNotIn("--outdir-base", command)
        self.assertNotIn("--texto-referencia", command)
        self.assertEqual(command[command.index("--generations") + 1], "3")
        self.assertEqual(command[command.index("--k") + 1], "5")
        self.assertEqual(command[command.index("--elite_size") + 1], "3")
        self.assertEqual(command[command.index("--prob_crossover") + 1], "0.7")
        self.assertEqual(command[command.index("--prob_mutation") + 1], "0.05")
        self.assertEqual(command[command.index("--bert_model") + 1], "roberta-large")

    def test_comparator_rejects_structured_managed_or_unknown_cli_values(self):
        service = ComparatorService(Path("."))
        for flag in ("experiment.seed", "--missing"):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    service._read_config(
                        {
                            "referenceText": "reference",
                            "selectedProposalIds": ["binary-mopso-cd"],
                            "proposalConfigs": {"binary-mopso-cd": {"cliValues": {flag: "1"}}},
                        }
                    )

    def test_comparator_rejects_unknown_selected_proposal(self):
        service = ComparatorService(Path("."))
        with self.assertRaises(ValueError):
            service._read_config({"referenceText": "reference", "selectedProposalIds": ["missing"]})

    def test_binary_extra_args_cannot_override_managed_isolation_flags(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        for flag in ("--outdir-base", "--seed"):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    service._validate_extra_args(proposal, [flag, "other"])

    def test_extra_args_cannot_override_common_comparison_flags(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd")
        for flag in ("--n", "--generaciones", "--model"):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    service._validate_extra_args(proposal, [flag, "other"])

    def test_extra_args_cannot_override_common_flags_with_equals_syntax(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        for flag in ("--n=99", "--iterations=99", "--model=other"):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    service._validate_extra_args(proposal, [flag])

    def test_binary_extra_args_cannot_override_managed_flags_with_equals_syntax(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        for flag in ("--outdir-base=other", "--reference-text=other", "--runs=3", "--seed=999"):
            with self.subTest(flag=flag):
                with self.assertRaises(ValueError):
                    service._validate_extra_args(proposal, [flag])

    def test_binary_extra_args_cannot_override_managed_yaml_paths(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        bad_args = [
            ["--set", "experiment.n=99"],
            ["--set=runtime.outdir_base=other"],
            ["--set", "router.task_models.synthetic_text_generation=other"],
            ["--set", "router.task_thinking.synthetic_text_generation=true"],
        ]
        for args in bad_args:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    service._validate_extra_args(proposal, args)

    def test_binary_command_uses_repetition_seed(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        command = service._build_command(
            {"config": service._read_config({"referenceText": "reference", "selectedProposalIds": ["binary-mopso-cd"]})},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )
        self.assertNotIn("--seed", command)
        self.assertEqual(command_set_values(command)["experiment.seed"], "777")

    def test_binary_command_defaults_parallelism_to_comparison_particle_count(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "n": 30,
                "selectedProposalIds": ["binary-mopso-cd"],
            }
        )

        command = service._build_command({"config": config}, proposal, Path("."), Path("out"), Path("reference.txt"), 777)
        values = command_set_values(command)

        self.assertEqual(values["parallelism.particle_update_max_concurrent"], "30")
        self.assertEqual(values["parallelism.initial_text_generation_max_concurrent"], "30")

    def test_binary_command_respects_partial_manual_parallelism_override(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "n": 30,
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "parallelism.particle_update_max_concurrent": "10",
                        }
                    }
                },
            }
        )

        command = service._build_command({"config": config}, proposal, Path("."), Path("out"), Path("reference.txt"), 777)
        values = command_set_values(command)

        self.assertEqual(values["parallelism.particle_update_max_concurrent"], "10")
        self.assertEqual(values["parallelism.initial_text_generation_max_concurrent"], "30")

    def test_binary_command_respects_manual_parallelism_extra_args(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "n": 30,
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "extraArgs": (
                            "--set parallelism.particle_update_max_concurrent=12 "
                            "--set=parallelism.initial_text_generation_max_concurrent=14"
                        ),
                    }
                },
            }
        )

        command = service._build_command({"config": config}, proposal, Path("."), Path("out"), Path("reference.txt"), 777)
        values = command_set_values(command)
        paths = command_set_paths(command)

        self.assertEqual(values["parallelism.particle_update_max_concurrent"], "12")
        self.assertEqual(values["parallelism.initial_text_generation_max_concurrent"], "14")
        self.assertEqual(paths.count("parallelism.particle_update_max_concurrent"), 1)
        self.assertEqual(paths.count("parallelism.initial_text_generation_max_concurrent"), 1)

    def test_binary_command_injects_portal_ppdb_paths_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ppdb_source = root / "data" / "external" / "ppdb" / "ppdb-2.0-s-all"
            ppdb_source.parent.mkdir(parents=True)
            ppdb_source.write_text("[X] ||| help ||| aid ||| features ||| Equivalence\n", encoding="utf-8")

            service = ComparatorService(root)
            proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
            config = service._read_config(
                {
                    "referenceText": "reference",
                    "selectedProposalIds": ["binary-mopso-cd"],
                }
            )

            command = service._build_command(
                {"config": config},
                proposal,
                root / "binary",
                root / "out",
                root / "reference.txt",
                777,
            )
            values = command_set_values(command)

            self.assertEqual(json.loads(values["models.ppdb.source_path"]), str(ppdb_source.resolve()))
            self.assertEqual(
                json.loads(values["models.ppdb.index_path"]),
                str((root / "data" / "turbulence" / "ppdb_index.sqlite").resolve()),
            )

    def test_binary_command_builds_structured_cli_values(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "--config": "configs/test.yaml",
                            "models.sbert.default": "gte-small",
                            "experiment.frozen_components": ["role", "topic", "action"],
                            "semantic_components.order": ["topic", "role", "action"],
                            "monitor.enabled": True,
                            "router.heuristics.semantic_anchor_extraction": False,
                            "router.heuristics.central_anchor_selection": False,
                            "router.heuristics.semantic_pool_generation": False,
                            "router.heuristics.semantic_pool_expansion": False,
                            "router.heuristics.semantic_component_influence_candidates": False,
                            "router.heuristics.word_replacement_candidates": False,
                            "router.task_models.semantic_pool_generation": "lfm2.5:8b",
                            "router.task_thinking.semantic_pool_generation": "low",
                        }
                    }
                },
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )
        self.assertEqual(command[:4], [command[0], "-m", "binary_mopso_cd", "--reference-text"])
        self.assertNotIn("--bert-model", command)
        self.assertNotIn("--freeze-components", command)
        self.assertEqual(command[command.index("--config") + 1], "configs/test.yaml")
        set_values = command_set_values(command)
        self.assertEqual(set_values["experiment.n"], "10")
        self.assertEqual(set_values["experiment.iterations"], "3")
        self.assertEqual(set_values["experiment.runs"], "1")
        self.assertEqual(set_values["experiment.seed"], "777")
        self.assertEqual(set_values["runtime.outdir_base"], json.dumps(str(Path("out").resolve()), ensure_ascii=False))
        self.assertEqual(set_values["ollama.default_model"], '"llama3"')
        self.assertEqual(set_values["models.sbert.default"], '"gte-small"')
        self.assertEqual(set_values["experiment.frozen_components"], '["role","topic","action"]')
        self.assertEqual(set_values["semantic_components.order"], '["topic","role","action"]')
        self.assertEqual(set_values["monitor.enabled"], "true")
        for key in (
            "router.heuristics.semantic_anchor_extraction",
            "router.heuristics.central_anchor_selection",
            "router.heuristics.semantic_pool_generation",
            "router.heuristics.semantic_pool_expansion",
            "router.heuristics.semantic_component_influence_candidates",
            "router.heuristics.word_replacement_candidates",
        ):
            self.assertEqual(set_values[key], "false")
        self.assertNotIn("router.task_models.semantic_anchor_extraction", set_values)
        self.assertNotIn("router.task_models.semantic_pool_expansion", set_values)
        self.assertNotIn("router.task_models.semantic_component_influence_candidates", set_values)
        self.assertNotIn("router.task_models.synthetic_text_generation", set_values)
        self.assertEqual(set_values["router.task_models.semantic_pool_generation"], '"lfm2.5:8b"')
        self.assertNotIn("router.task_models.central_anchor_selection", set_values)
        self.assertEqual(set_values["router.task_thinking.semantic_pool_generation"], '"low"')

    def test_binary_command_preserves_default_task_models_without_explicit_overrides(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "model": "llama3",
                "selectedProposalIds": ["binary-mopso-cd"],
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )

        set_values = command_set_values(command)
        self.assertEqual(set_values["ollama.default_model"], '"llama3"')
        self.assertNotIn("router.task_models.central_anchor_selection", set_values)
        self.assertFalse(any(path.startswith("router.task_models.") for path in command_set_paths(command)))

    def test_binary_command_maps_alternative_model_to_default_llama_task_models(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "ollama.alternative_model": "lfm2.5:8b",
                        }
                    }
                },
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )

        set_values = command_set_values(command)
        self.assertEqual(set_values["ollama.alternative_model"], '"lfm2.5:8b"')
        self.assertIn("router.task_models.semantic_anchor_extraction", set_values)
        self.assertIn("router.task_models.semantic_pool_generation", set_values)
        self.assertIn("router.task_models.semantic_pool_expansion", set_values)
        self.assertIn("router.task_models.semantic_component_influence_candidates", set_values)
        self.assertIn("router.task_models.synthetic_text_generation", set_values)
        self.assertEqual(set_values["router.task_models.semantic_anchor_extraction"], '"lfm2.5:8b"')
        self.assertEqual(set_values["router.task_models.semantic_pool_generation"], '"lfm2.5:8b"')
        self.assertEqual(set_values["router.task_models.semantic_pool_expansion"], '"lfm2.5:8b"')
        self.assertEqual(set_values["router.task_models.semantic_component_influence_candidates"], '"lfm2.5:8b"')
        self.assertEqual(set_values["router.task_models.synthetic_text_generation"], '"lfm2.5:8b"')
        self.assertNotIn("router.task_models.central_anchor_selection", set_values)

    def test_binary_command_keeps_explicit_task_model_over_alternative_model(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "ollama.alternative_model": "lfm2.5:8b",
                            "router.task_models.synthetic_text_generation": "qwen3.5:2b",
                        }
                    }
                },
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )

        set_values = command_set_values(command)
        self.assertEqual(set_values["router.task_models.synthetic_text_generation"], '"qwen3.5:2b"')
        self.assertIn("router.task_models.semantic_pool_generation", set_values)
        self.assertEqual(set_values["router.task_models.semantic_pool_generation"], '"lfm2.5:8b"')
        self.assertNotIn("router.task_models.central_anchor_selection", set_values)

    def test_binary_command_injects_server_safe_ollama_timeout_by_default(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )

        set_values = command_set_values(command)
        self.assertEqual(set_values["ollama.timeout_seconds"], "600")

    def test_binary_command_preserves_manual_ollama_timeout_override(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        config = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "ollama.timeout_seconds": 900,
                        }
                    }
                },
            }
        )
        command = service._build_command(
            {"config": config},
            proposal,
            Path("."),
            Path("out"),
            Path("reference.txt"),
            777,
        )

        set_values = command_set_values(command)
        paths = command_set_paths(command)
        self.assertEqual(set_values["ollama.timeout_seconds"], "900")
        self.assertEqual(paths.count("ollama.timeout_seconds"), 1)

    def test_binary_task_thinking_uses_default_task_model_when_model_not_overridden(self):
        service = ComparatorService(Path("."))

        parsed = service._read_config(
            {
                "referenceText": "reference",
                "model": "llama3",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {
                    "binary-mopso-cd": {
                        "cliValues": {
                            "router.task_thinking.central_anchor_selection": "high",
                        }
                    }
                },
            }
        )

        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertEqual(values["router.task_thinking.central_anchor_selection"], "high")

    def test_binary_task_thinking_rejects_default_task_model_without_thinking_support(self):
        service = ComparatorService(Path("."))

        with self.assertRaisesRegex(
            ValueError,
            "llama3\\.1:8b.*semantic_anchor_extraction.*does not support thinking",
        ):
            service._read_config(
                {
                    "referenceText": "reference",
                    "model": "qwen3.5:9b",
                    "selectedProposalIds": ["binary-mopso-cd"],
                    "proposalConfigs": {
                        "binary-mopso-cd": {
                            "cliValues": {
                                "router.task_thinking.semantic_anchor_extraction": "high",
                            }
                        }
                    },
                }
            )

    def test_binary_command_uses_instance_specific_cli_values(self):
        service = ComparatorService(Path("."))
        config = service._read_config(
            {
                "referenceText": "reference",
                "proposalInstances": [
                    {
                        "instanceId": "binary-a",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary A",
                        "proposalConfig": {
                            "cliValues": {
                                "selection.k": "4",
                                "mopso.guided_trajectory_validation_enabled": False,
                                "mopso.guided_trajectory_relative_margin": "0.40",
                            }
                        },
                    },
                    {
                        "instanceId": "binary-b",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary B",
                        "proposalConfig": {"cliValues": {"selection.k": "5"}},
                    },
                ],
            }
        )
        first, second = service._selected_instances(config)

        first_command = service._build_command({"config": config}, first, Path("."), Path("out-a"), Path("reference.txt"), 777)
        second_command = service._build_command({"config": config}, second, Path("."), Path("out-b"), Path("reference.txt"), 778)

        self.assertEqual(command_set_values(first_command)["selection.k"], "4")
        self.assertEqual(command_set_values(second_command)["selection.k"], "5")
        self.assertEqual(command_set_values(first_command)["mopso.guided_trajectory_validation_enabled"], "false")
        self.assertEqual(command_set_values(first_command)["mopso.guided_trajectory_relative_margin"], "0.40")
        self.assertNotIn("mopso.guided_trajectory_validation_enabled", command_set_values(second_command))
        self.assertNotIn("mopso.guided_trajectory_relative_margin", command_set_values(second_command))
        self.assertEqual(command_set_values(first_command)["experiment.seed"], "777")
        self.assertEqual(command_set_values(second_command)["experiment.seed"], "778")

    def test_binary_command_injects_same_initial_population_paths_for_dependent_instance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "run-1"
            population_path = run_dir / "binary-a" / "exec" / "native-run" / "data_initial_population.json"
            context_path = run_dir / "binary-a" / "exec" / "native-run" / "reference_context.json"
            population_path.parent.mkdir(parents=True)
            population_path.write_text("[]", encoding="utf-8")
            context_path.write_text("{}", encoding="utf-8")
            config = service._read_config(
                {
                    "referenceText": "reference",
                    "proposalInstances": [
                        {
                            "instanceId": "binary-a",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary A",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        },
                        {
                            "instanceId": "binary-b",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary B",
                            "proposalConfig": {"cliValues": {"selection.k": "5"}},
                        },
                    ],
                    "sameInitialPopulationForBmopso": {
                        "enabled": True,
                        "generatorInstanceId": "binary-a",
                    },
                }
            )
            first, second = service._selected_instances(config)
            run = {
                "config": config,
                "runDir": str(run_dir),
                "sameInitialPopulationArtifacts": {
                    "binary-a": {
                        "populationPath": str(population_path),
                        "referenceContextPath": str(context_path),
                        "repetitions": {},
                    }
                },
            }

            first_values = command_set_values(
                service._build_command(run, first, Path("."), run_dir / "binary-a" / "exec", Path("reference.txt"), 777)
            )
            second_values = command_set_values(
                service._build_command(run, second, Path("."), run_dir / "binary-b" / "exec", Path("reference.txt"), 778)
            )

            self.assertNotIn("initialization.population_input_path", first_values)
            self.assertEqual(json.loads(second_values["initialization.population_input_path"]), str(population_path.resolve()))
            self.assertEqual(json.loads(second_values["initialization.reference_context_input_path"]), str(context_path.resolve()))

    def test_binary_command_uses_repetition_specific_same_initial_population_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "run-1"
            rep2_population_path = run_dir / "binary-a" / "rep-002" / "exec" / "native-run" / "data_initial_population.json"
            rep2_context_path = run_dir / "binary-a" / "rep-002" / "exec" / "native-run" / "reference_context.json"
            rep2_population_path.parent.mkdir(parents=True)
            rep2_population_path.write_text("[]", encoding="utf-8")
            rep2_context_path.write_text("{}", encoding="utf-8")
            single_population_path = run_dir / "binary-a" / "exec" / "native-run" / "data_initial_population.json"
            single_context_path = run_dir / "binary-a" / "exec" / "native-run" / "reference_context.json"
            single_population_path.parent.mkdir(parents=True)
            single_population_path.write_text("[]", encoding="utf-8")
            single_context_path.write_text("{}", encoding="utf-8")
            config = service._read_config(
                {
                    "referenceText": "reference",
                    "repetitionsK": 2,
                    "proposalInstances": [
                        {
                            "instanceId": "binary-a",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary A",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        },
                        {
                            "instanceId": "binary-b",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary B",
                            "proposalConfig": {"cliValues": {"selection.k": "5"}},
                        },
                    ],
                    "sameInitialPopulationForBmopso": {
                        "enabled": True,
                        "generatorInstanceId": "binary-a",
                    },
                }
            )
            _first, second = service._selected_instances(config)
            run = {
                "config": config,
                "runDir": str(run_dir),
                "sameInitialPopulationArtifacts": {
                    "binary-a": {
                        "populationPath": str(run_dir / "binary-a" / "exec" / "native-run" / "data_initial_population.json"),
                        "referenceContextPath": str(run_dir / "binary-a" / "exec" / "native-run" / "reference_context.json"),
                        "repetitions": {
                            "2": {
                                "populationPath": str(rep2_population_path),
                                "referenceContextPath": str(rep2_context_path),
                            }
                        },
                    }
                },
            }

            values = command_set_values(
                service._build_command(
                    run,
                    second,
                    Path("."),
                    run_dir / "binary-b" / "rep-002" / "exec",
                    Path("reference.txt"),
                    778,
                )
            )

            self.assertEqual(json.loads(values["initialization.population_input_path"]), str(rep2_population_path.resolve()))
            self.assertEqual(json.loads(values["initialization.reference_context_input_path"]), str(rep2_context_path.resolve()))

    def test_parallel_same_initial_population_waits_for_generator_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_dir = root / "runs" / "comparator" / "run-1"
            run_dir.mkdir(parents=True)
            config = service._read_config(
                {
                    "referenceText": "reference",
                    "executionMode": "exploratory_parallel",
                    "proposalParallelism": 3,
                    "proposalInstances": [
                        {
                            "instanceId": "binary-a",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary A",
                            "proposalConfig": {"cliValues": {"selection.k": "4"}},
                        },
                        {
                            "instanceId": "binary-b",
                            "proposalId": "binary-mopso-cd",
                            "displayName": "Binary B",
                            "proposalConfig": {"cliValues": {"selection.k": "5"}},
                        },
                        {
                            "instanceId": "mesap-a",
                            "proposalId": "mesap",
                            "displayName": "MESAP A",
                            "proposalConfig": {"cliValues": {}},
                        },
                    ],
                    "sameInitialPopulationForBmopso": {
                        "enabled": True,
                        "generatorInstanceId": "binary-a",
                    },
                }
            )
            instances = service._selected_instances(config)
            run = {
                "runId": "run-1",
                "runDir": str(run_dir),
                "status": "running",
                "startedAtEpoch": time.time(),
                "config": config,
                "proposalStates": {
                    instance.instance_id: service._initial_proposal_state(instance, config.get("repetitionsK"))
                    for instance in instances
                },
                "proposals": [],
                "progress": {},
                "costSummary": {},
                "logs": [],
                "repositoryUpdates": {},
                "sameInitialPopulationArtifacts": {},
                "cancelRequested": False,
                "activeProcesses": {},
            }
            started: list[str] = []

            def completed_result(instance, output_dir):
                return {
                    "instanceId": instance.instance_id,
                    "proposalId": instance.proposal_id,
                    "displayName": instance.display_name,
                    "baseDisplayName": instance.base_display_name,
                    "proposalConfig": instance.proposal_config,
                    "status": "completed",
                    "outputDir": str(output_dir),
                    "rows": [],
                    "embeddingFrontRows": [],
                    "metrics": {},
                    "charts": {},
                    "cost": {},
                    "error": None,
                    "completedRepetitions": 1,
                    "repetitionsK": 1,
                }

            def fake_execute(run_payload, proposal):
                instance = service._coerce_instance(proposal, run_payload.get("config"))
                started.append(instance.instance_id)
                if instance.instance_id == "binary-a":
                    output_dir = run_dir / "binary-a" / "exec" / "native-run"
                    output_dir.mkdir(parents=True)
                    (output_dir / "data_initial_population.json").write_text("[]", encoding="utf-8")
                    (output_dir / "reference_context.json").write_text("{}", encoding="utf-8")
                    return completed_result(instance, output_dir)
                if instance.instance_id == "binary-b":
                    self.assertIn("binary-a", run_payload["sameInitialPopulationArtifacts"])
                    output_dir = run_dir / "binary-b" / "exec" / "native-run"
                    output_dir.mkdir(parents=True)
                    return completed_result(instance, output_dir)
                output_dir = run_dir / instance.instance_id
                output_dir.mkdir(parents=True)
                return completed_result(instance, output_dir)

            with patch.object(service, "_execute_proposal", side_effect=fake_execute):
                service._run_proposals_parallel(run, instances, 3)

            self.assertLess(started.index("binary-a"), started.index("binary-b"))
            self.assertEqual({proposal["instanceId"] for proposal in run["proposals"]}, {"binary-a", "binary-b", "mesap-a"})

    def test_binary_parallelism_auto_defaults_are_instance_specific(self):
        service = ComparatorService(Path("."))
        config = service._read_config(
            {
                "referenceText": "reference",
                "n": 30,
                "proposalInstances": [
                    {
                        "instanceId": "binary-auto",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary Auto",
                        "proposalConfig": {"cliValues": {"selection.k": "4"}},
                    },
                    {
                        "instanceId": "binary-manual",
                        "proposalId": "binary-mopso-cd",
                        "displayName": "Binary Manual",
                        "proposalConfig": {
                            "cliValues": {
                                "selection.k": "5",
                                "parallelism.particle_update_max_concurrent": "11",
                            }
                        },
                    },
                ],
            }
        )
        first, second = service._selected_instances(config)

        first_values = command_set_values(
            service._build_command({"config": config}, first, Path("."), Path("out-a"), Path("reference.txt"), 777)
        )
        second_values = command_set_values(
            service._build_command({"config": config}, second, Path("."), Path("out-b"), Path("reference.txt"), 778)
        )

        self.assertEqual(first_values["parallelism.particle_update_max_concurrent"], "30")
        self.assertEqual(first_values["parallelism.initial_text_generation_max_concurrent"], "30")
        self.assertEqual(second_values["parallelism.particle_update_max_concurrent"], "11")
        self.assertEqual(second_values["parallelism.initial_text_generation_max_concurrent"], "30")

    def test_repository_update_skips_dirty_repository(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd")
        snapshot = {
            "isGit": True,
            "dirty": True,
            "dirtyCount": 2,
            "branch": "main",
            "shortCommit": "abc123",
        }
        with patch.object(service, "_repository_git_snapshot", return_value=snapshot), patch.object(service, "_run_git") as run_git:
            result = service._update_repository_for_proposal(
                proposal,
                {"remote": "origin", "branch": "main", "pullMode": "ff-only"},
            )
        self.assertEqual(result["status"], "skipped_dirty")
        run_git.assert_not_called()

    def test_repository_update_skips_branch_mismatch(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")
        snapshot = {
            "isGit": True,
            "dirty": False,
            "dirtyCount": 0,
            "branch": "codex/work",
            "shortCommit": "abc123",
        }
        with patch.object(service, "_repository_git_snapshot", return_value=snapshot), patch.object(service, "_run_git") as run_git:
            result = service._update_repository_for_proposal(
                proposal,
                {"remote": "origin", "branch": "main", "pullMode": "ff-only"},
            )
        self.assertEqual(result["status"], "skipped_branch_mismatch")
        run_git.assert_not_called()

    def test_repository_update_skips_remote_mismatch(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        snapshot = {
            "isGit": True,
            "dirty": False,
            "dirtyCount": 0,
            "branch": "dev",
            "shortCommit": "abc123",
            "remoteUrl": "https://github.com/escmHEX/MOACO.git",
        }
        with patch.object(service, "_repository_git_snapshot", return_value=snapshot), patch.object(service, "_run_git") as run_git:
            result = service._update_repository_for_proposal(
                proposal,
                {
                    "remote": "origin",
                    "branch": "dev",
                    "pullMode": "ff-only",
                    "expectedRemoteUrl": "https://github.com/escmHEX/BMOPSO-CD.git",
                },
            )
        self.assertEqual(result["status"], "skipped_remote_mismatch")
        run_git.assert_not_called()

    def test_repository_update_fetches_and_pulls_fast_forward_only(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        before = {
            "isGit": True,
            "dirty": False,
            "dirtyCount": 0,
            "branch": "dev",
            "shortCommit": "abc123",
        }
        after = {
            "isGit": True,
            "dirty": False,
            "dirtyCount": 0,
            "branch": "dev",
            "shortCommit": "def456",
        }
        fetch = {"returnCode": 0, "stdout": "", "stderr": ""}
        pull = {"returnCode": 0, "stdout": "Already up to date.", "stderr": ""}
        with patch.object(service, "_repository_git_snapshot", side_effect=[before, after]), patch.object(
            service,
            "_run_git",
            side_effect=[fetch, pull],
        ) as run_git:
            result = service._update_repository_for_proposal(
                proposal,
                {"remote": "origin", "branch": "dev", "pullMode": "ff-only"},
            )
        self.assertEqual(result["status"], "up_to_date")
        self.assertEqual(run_git.call_args_list[0].args[1], ["fetch", "origin", "dev"])
        self.assertEqual(run_git.call_args_list[1].args[1], ["pull", "--ff-only", "origin", "dev"])

    def test_selected_proposals_validate_dependencies_before_starting(self):
        service = ComparatorService(Path("."))
        with patch.object(comparator_module, "proposal_entrypoint_exists", return_value=True), patch.object(
            comparator_module,
            "check_proposal_dependencies",
            return_value={
                "ok": False,
                "missing": ["torch"],
                "pythonExecutable": "python",
                "error": None,
            },
        ):
            with self.assertRaisesRegex(ValueError, "missing modules: torch"):
                service._validate_selected_proposals_available(
                    {
                        "selectedProposalIds": ["evolmd"],
                    }
                )

    def test_initial_population_config_accepts_repetitions(self):
        service = InitialPopulationComparisonService(Path("."))
        payload = service.default_config()
        payload["selectedStrategies"] = ["hybrid-semantic-v7"]
        payload["repetitionsK"] = 3
        parsed = service._read_config(payload)
        self.assertEqual(parsed["repetitionsK"], 3)

    def test_comparator_aggregates_metrics_across_repetitions(self):
        proposal = PROPOSALS[0]
        results = [
            {
                "status": "completed",
                "repetitionIndex": 1,
                "repetitionSeed": 10,
                "rows": [{"rank": 1, "status": "ok"}],
                "metrics": {
                    "totalRows": 10,
                    "completedRows": 10,
                    "objectiveNames": ["fitness"],
                    "bestObjectiveVector": [0.6],
                    "nonDominatedRows": 0,
                    "hypervolume": 0.2,
                    "extent": 0.4,
                    "unaryEntropy": 0.6,
                    "contribution": 0.25,
                    "postHocDiagnostic": True,
                    "bestDiagnosticObjectiveVector": [0.6, 0.5],
                    "postHocNonDominatedRows": 2,
                    "externalArchiveUpdateCount": 4,
                    "externalArchivePruneCount": 1,
                    "externalArchiveUpdateCountTotal": 4,
                    "externalArchivePruneCountTotal": 1,
                },
                "cost": {
                    "llmCalls": 2,
                    "llmEmptyContentCalls": 1,
                    "llmClientWallClockSeconds": 1.0,
                    "processWallClockSeconds": 4.0,
                    "promptEvalCount": 10,
                    "evalCount": 6,
                    "totalTokens": 20,
                },
            },
            {
                "status": "completed",
                "repetitionIndex": 2,
                "repetitionSeed": 11,
                "rows": [{"rank": 1, "status": "ok"}],
                "metrics": {
                    "totalRows": 8,
                    "completedRows": 8,
                    "objectiveNames": ["fitness"],
                    "bestObjectiveVector": [0.8],
                    "nonDominatedRows": 0,
                    "hypervolume": 0.4,
                    "extent": 0.2,
                    "unaryEntropy": 1.0,
                    "contribution": 0.75,
                    "postHocDiagnostic": True,
                    "bestDiagnosticObjectiveVector": [0.8, 0.7],
                    "postHocNonDominatedRows": 4,
                    "externalArchiveUpdateCount": 8,
                    "externalArchivePruneCount": 3,
                    "externalArchiveUpdateCountTotal": 8,
                    "externalArchivePruneCountTotal": 3,
                },
                "cost": {
                    "llmCalls": 3,
                    "llmEmptyContentCalls": 1,
                    "llmClientWallClockSeconds": 2.0,
                    "processWallClockSeconds": 8.0,
                    "promptEvalCount": 14,
                    "evalCount": 10,
                    "totalTokens": 30,
                },
            },
        ]
        aggregated = aggregate_proposal_repetitions(proposal, Path("out"), results, 2)
        self.assertEqual(aggregated["completedRepetitions"], 2)
        self.assertAlmostEqual(aggregated["metrics"]["completedRows"], 9.0)
        self.assertEqual(aggregated["metrics"]["bestObjectiveLabel"], "[0.700000]")
        self.assertEqual(aggregated["metrics"]["nonDominatedRows"], 3.0)
        self.assertEqual(aggregated["metrics"]["postHocNonDominatedRows"], 3.0)
        self.assertAlmostEqual(aggregated["metrics"]["nonDominatedRowsStdDev"], 2 ** 0.5)
        self.assertEqual(aggregated["metrics"]["nonDominatedRowsStdDevLabel"], "1.414214")
        self.assertAlmostEqual(aggregated["metrics"]["hypervolumeStdDev"], 0.1414213562373095)
        self.assertEqual(aggregated["metrics"]["hypervolumeStdDevLabel"], "0.141421")
        self.assertAlmostEqual(aggregated["metrics"]["extentStdDev"], 0.1414213562373095)
        self.assertAlmostEqual(aggregated["metrics"]["unaryEntropyStdDev"], 0.282842712474619)
        self.assertAlmostEqual(aggregated["metrics"]["contributionStdDev"], 0.3535533905932738)
        self.assertAlmostEqual(aggregated["metrics"]["externalArchiveUpdateCount"], 6.0)
        self.assertAlmostEqual(aggregated["metrics"]["externalArchivePruneCount"], 2.0)
        self.assertEqual(aggregated["metrics"]["externalArchiveUpdateCountTotal"], 12)
        self.assertEqual(aggregated["metrics"]["externalArchivePruneCountTotal"], 4)
        self.assertAlmostEqual(aggregated["cost"]["llmCalls"], 2.5)
        self.assertAlmostEqual(aggregated["cost"]["llmCallsStdDev"], 0.7071067811865476)
        self.assertEqual(aggregated["cost"]["llmCallsStdDevLabel"], "0.71")
        self.assertEqual(aggregated["cost"]["llmCallsTotal"], 5)
        self.assertAlmostEqual(aggregated["cost"]["llmEmptyContentCalls"], 1.0)
        self.assertEqual(aggregated["cost"]["llmEmptyContentCallsTotal"], 2)
        self.assertAlmostEqual(aggregated["cost"]["processWallClockSeconds"], 6.0)
        self.assertAlmostEqual(aggregated["cost"]["processWallClockSecondsStdDev"], 2.8284271247461903)
        self.assertEqual(aggregated["cost"]["processWallClockSecondsStdDevLabel"], "3s")
        self.assertAlmostEqual(aggregated["cost"]["llmClientWallClockSeconds"], 1.5)
        self.assertAlmostEqual(aggregated["cost"]["llmClientWallClockSecondsTotal"], 3.0)
        self.assertAlmostEqual(aggregated["cost"]["llmAverageCallSeconds"], 0.6)
        self.assertAlmostEqual(aggregated["cost"]["promptEvalCountStdDev"], 2.8284271247461903)
        self.assertEqual(aggregated["cost"]["promptEvalCountStdDevLabel"], "2.83")
        self.assertAlmostEqual(aggregated["cost"]["evalCountStdDev"], 2.8284271247461903)
        self.assertEqual(aggregated["cost"]["evalCountStdDevLabel"], "2.83")
        self.assertAlmostEqual(aggregated["cost"]["totalTokens"], 25.0)
        self.assertEqual(aggregated["cost"]["totalTokensTotal"], 50)
        self.assertEqual(aggregated["rows"][0]["repetitionSeed"], 10)

        summary = comparator_module.summarize_costs([{"cost": aggregated["cost"]}], run_elapsed_seconds=9.0)
        self.assertEqual(summary["llmCalls"], 5)
        self.assertEqual(summary["llmEmptyContentCalls"], 2)
        self.assertEqual(summary["totalTokens"], 50)
        self.assertAlmostEqual(summary["llmClientWallClockSeconds"], 3.0)

    def test_aggregate_series_averages_by_generation(self):
        series = aggregate_series([
            {
                "series": [
                    {
                        "generation": 1,
                        "hypervolume": 0.2,
                        "nonDominatedRows": 2,
                        "extent": 0.5,
                        "unaryEntropy": 0.8,
                        "contribution": 0.25,
                        "frontPoints": [[0.5, 0.7]],
                    }
                ]
            },
            {
                "series": [
                    {
                        "generation": 1,
                        "hypervolume": 0.4,
                        "nonDominatedRows": 4,
                        "extent": 0.3,
                        "unaryEntropy": 0.6,
                        "contribution": 0.75,
                        "frontPoints": [[0.7, 0.5]],
                    }
                ]
            },
        ])
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0]["hypervolume"], 0.3)
        self.assertAlmostEqual(series[0]["nonDominatedRows"], 3.0)
        self.assertAlmostEqual(series[0]["extent"], 0.4)
        self.assertAlmostEqual(series[0]["unaryEntropy"], 0.7)
        self.assertAlmostEqual(series[0]["contribution"], 0.5)
        self.assertEqual(series[0]["frontPoints"], [[0.5, 0.7], [0.7, 0.5]])

    def test_binary_rows_normalize_from_native_outputs(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": 0.2, "semanticDiversity": 0.6}],
        ):
            rows = service._normalize_rows(
                proposal,
                [
                    {
                        "solution_id": "s1",
                        "generated_text": "Generated",
                        "prompt": "Prompt",
                        "objectives": {"f1": 0.7, "f2": 0.4},
                        "components": {"role": "resident"},
                    }
                ],
                top_k=5,
                reference_text="reference",
            )
        self.assertEqual(rows[0]["proposalId"], "binary-mopso-cd")
        self.assertEqual(rows[0]["objectiveVector"], [0.7, 0.4])
        self.assertEqual(rows[0]["comparableObjectiveVector"], [0.6, 0.3])
        self.assertTrue(rows[0]["nonDominated"])

    def test_binary_summary_reads_external_archive_counts_from_metrics_csv(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "generation,archive_update_count,archive_prune_count",
                        "1,2,0",
                        "2,5,1",
                        "bad,9,9",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[{"semanticFidelity": 0.0, "semanticDiversity": 1.0}],
            ):
                rows = service._normalize_rows(
                    proposal,
                    [{"generated_text": "Generated", "objectives": {"f1": 0.0, "f2": 1.0}}],
                    top_k=5,
                    reference_text="reference",
                )
            metrics = service._summarize_rows(proposal, rows, output_dir)

        self.assertEqual(metrics["externalArchiveUpdateCount"], 5)
        self.assertEqual(metrics["externalArchivePruneCount"], 1)
        self.assertEqual(metrics["externalArchiveUpdateCountTotal"], 5)
        self.assertEqual(metrics["externalArchivePruneCountTotal"], 1)

    def test_binary_summary_falls_back_to_runtime_log_for_external_archive_counts(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "evolucion_metricas.csv").write_text(
                "generation,hypervolume\n1,0.25\n",
                encoding="utf-8",
            )
            (output_dir / "runtime.log").write_text(
                "\n".join(
                    [
                        "2026-06-03 00:05:20 | INFO | run 1/1 | generation 1/2 | archive_updates=3 | archive_prunes=1 | elapsed=00:00:16",
                        "2026-06-03 00:05:30 | INFO | run 1/1 finished | archive_updates=4 | archive_prunes=2 | elapsed=00:00:26",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[{"semanticFidelity": 0.0, "semanticDiversity": 1.0}],
            ):
                rows = service._normalize_rows(
                    proposal,
                    [{"generated_text": "Generated", "objectives": {"f1": 0.0, "f2": 1.0}}],
                    top_k=5,
                    reference_text="reference",
                )
            metrics = service._summarize_rows(proposal, rows, output_dir)

        self.assertEqual(metrics["externalArchiveUpdateCount"], 4)
        self.assertEqual(metrics["externalArchivePruneCount"], 2)

    def test_get_run_enriches_binary_internal_analysis_from_native_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "native-bmopso"
            run_dir = root / "runs" / "comparator" / run_id
            output_dir = run_dir / "binary-mopso-cd" / "exec" / "2026-06-26_03-19-22"
            output_dir.mkdir(parents=True)
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "generation,hypervolume,archive_size",
                        "0,0.05,1",
                        "1,0.11,2",
                        "2,0.22,3",
                    ]
                ),
                encoding="utf-8",
            )
            (output_dir / "archive_history.jsonl").write_text(
                json.dumps({"generation": 2, "hypervolume": 0.99}) + "\n",
                encoding="utf-8",
            )
            (output_dir / "pareto_front.json").write_text(
                json.dumps(
                    [
                        {"generated_text": "Generated A", "prompt": "Prompt A", "objectives": {"f1": 0.0, "f2": 1.0}},
                        {"generated_text": "Generated B", "prompt": "Prompt B", "objectives": {"f1": 0.4, "f2": 0.6}},
                        {"generated_text": "Generated C", "prompt": "Prompt C", "objectives": {"f1": -0.2, "f2": 0.2}},
                    ]
                ),
                encoding="utf-8",
            )
            (output_dir / "final_selection_hybrid.json").write_text(
                json.dumps(
                    [
                        {"generated_text": "Generated B", "prompt": "Prompt B", "objectives": {"f1": 0.4, "f2": 0.6}},
                    ]
                ),
                encoding="utf-8",
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "runId": run_id,
                        "status": "completed",
                        "runDir": str(run_dir),
                        "metricSchemaVersion": 4,
                        "metricCoordinateSpace": "comparable_normalized",
                        "proposals": [
                            {
                                "instanceId": "binary-a",
                                "proposalId": "binary-mopso-cd",
                                "displayName": "Binary A",
                                "baseDisplayName": "Binary MOPSO-CD",
                                "status": "completed",
                                "outputDir": str(output_dir),
                                "rows": [{"generatedText": "Generated A"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = ComparatorService(root)

            proposal = service.get_run(run_id)["proposals"][0]
            analysis = proposal["internalBmopsoAnalysis"]

        self.assertTrue(analysis["available"])
        self.assertEqual(analysis["instanceId"], "binary-a")
        self.assertEqual(analysis["coordinateSpace"], "binary_native_normalized")
        self.assertEqual(analysis["source"], "evolucion_metricas.csv")
        self.assertAlmostEqual(analysis["metrics"]["hypervolume"], 0.22)
        self.assertEqual([point["generation"] for point in analysis["series"]], [0, 1, 2])
        self.assertEqual([point["hypervolume"] for point in analysis["series"]], [0.05, 0.11, 0.22])
        self.assertEqual([point["archiveSize"] for point in analysis["series"]], [1, 2, 3])
        self.assertEqual(len(analysis["charts"]["pareto"]), 3)
        self.assertEqual(len(analysis["charts"]["nonDominated"]), 2)
        self.assertEqual(len(analysis["charts"]["selected"]), 1)
        self.assertEqual(analysis["charts"]["pareto"][0]["coordinateSpace"], "binary_native_normalized")

    def test_get_run_enriches_each_binary_instance_with_internal_analysis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "multiple-binary"
            run_dir = root / "runs" / "comparator" / run_id
            proposals = []
            for suffix, hv in (("a", "0.31"), ("b", "0.47")):
                output_dir = run_dir / f"binary-{suffix}" / "exec"
                output_dir.mkdir(parents=True)
                (output_dir / "evolucion_metricas.csv").write_text(
                    f"generation,hypervolume,archive_size\n1,{hv},4\n",
                    encoding="utf-8",
                )
                (output_dir / "pareto_front.json").write_text(
                    json.dumps(
                        [
                            {
                                "generated_text": f"Generated {suffix}",
                                "prompt": f"Prompt {suffix}",
                                "objectives": {"f1": 0.0, "f2": 1.0},
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                proposals.append(
                    {
                        "instanceId": f"binary-{suffix}",
                        "proposalId": "binary-mopso-cd",
                        "displayName": f"Binary {suffix.upper()}",
                        "status": "completed",
                        "outputDir": str(output_dir),
                    }
                )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "runId": run_id,
                        "status": "completed",
                        "runDir": str(run_dir),
                        "metricSchemaVersion": 4,
                        "metricCoordinateSpace": "comparable_normalized",
                        "proposals": proposals,
                    }
                ),
                encoding="utf-8",
            )
            service = ComparatorService(root)

            enriched = service.get_run(run_id)["proposals"]

        self.assertEqual(
            [proposal["internalBmopsoAnalysis"]["instanceId"] for proposal in enriched],
            ["binary-a", "binary-b"],
        )
        self.assertEqual(
            [proposal["internalBmopsoAnalysis"]["metrics"]["hypervolumeLabel"] for proposal in enriched],
            ["0.310000", "0.470000"],
        )

    def test_get_run_internal_analysis_uses_only_hybrid_selection_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "binary-no-hybrid-selection"
            run_dir = root / "runs" / "comparator" / run_id
            output_dir = run_dir / "binary" / "exec"
            output_dir.mkdir(parents=True)
            (output_dir / "evolucion_metricas.csv").write_text(
                "generation,hypervolume,archive_size\n1,0.33,2\n",
                encoding="utf-8",
            )
            front = [
                {"generated_text": "Generated A", "prompt": "Prompt A", "objectives": {"f1": 0.0, "f2": 1.0}},
                {"generated_text": "Generated B", "prompt": "Prompt B", "objectives": {"f1": 0.2, "f2": 0.8}},
            ]
            (output_dir / "pareto_front.json").write_text(json.dumps(front), encoding="utf-8")
            (output_dir / "pareto_ranked.json").write_text(json.dumps([front[0]]), encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "runId": run_id,
                        "status": "completed",
                        "runDir": str(run_dir),
                        "metricSchemaVersion": 4,
                        "metricCoordinateSpace": "comparable_normalized",
                        "proposals": [
                            {
                                "instanceId": "binary-a",
                                "proposalId": "binary-mopso-cd",
                                "displayName": "Binary A",
                                "status": "completed",
                                "outputDir": str(output_dir),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = ComparatorService(root)

            analysis = service.get_run(run_id)["proposals"][0]["internalBmopsoAnalysis"]

        self.assertEqual(len(analysis["charts"]["pareto"]), 2)
        self.assertEqual(analysis["charts"]["selected"], [])

    def test_get_run_skips_binary_internal_analysis_without_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "missing-native-artifacts"
            run_dir = root / "runs" / "comparator" / run_id
            output_dir = run_dir / "binary" / "exec"
            output_dir.mkdir(parents=True)
            (output_dir / "evolucion_metricas.csv").write_text(
                "generation,hypervolume\n1,0.25\n",
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "runId": run_id,
                        "status": "completed",
                        "runDir": str(run_dir),
                        "metricSchemaVersion": 4,
                        "metricCoordinateSpace": "comparable_normalized",
                        "proposals": [
                            {
                                "instanceId": "binary-a",
                                "proposalId": "binary-mopso-cd",
                                "displayName": "Binary A",
                                "status": "completed",
                                "outputDir": str(output_dir),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = ComparatorService(root)

            proposal = service.get_run(run_id)["proposals"][0]

        self.assertNotIn("internalBmopsoAnalysis", proposal)

    def test_get_run_does_not_add_internal_analysis_to_non_binary_proposals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "no-binary"
            run_dir = root / "runs" / "comparator" / run_id
            output_dir = run_dir / "evolmd-mo" / "exec"
            output_dir.mkdir(parents=True)
            (output_dir / "evolucion_metricas.csv").write_text(
                "generation,hypervolume\n1,0.25\n",
                encoding="utf-8",
            )
            (output_dir / "pareto_front.json").write_text("[]", encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "runId": run_id,
                        "status": "completed",
                        "runDir": str(run_dir),
                        "metricSchemaVersion": 4,
                        "metricCoordinateSpace": "comparable_normalized",
                        "proposals": [
                            {
                                "instanceId": "evolmd-mo",
                                "proposalId": "evolmd-mo",
                                "displayName": "EVOLMD-MO",
                                "status": "completed",
                                "outputDir": str(output_dir),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = ComparatorService(root)

            proposal = service.get_run(run_id)["proposals"][0]

        self.assertNotIn("internalBmopsoAnalysis", proposal)

    def test_mesap_rows_normalize_from_population_final_with_posthoc_metrics(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "mesap")

        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": 0.6, "semanticDiversity": 1.0}],
        ):
            rows = service._normalize_rows(
                proposal,
                [
                    {
                        "generated_data": "Generated",
                        "prompt": "Prompt",
                        "fitness": 0.73,
                        "role": "role",
                        "topic": "topic",
                    }
                ],
                top_k=5,
                reference_text="reference",
            )

        self.assertEqual(rows[0]["proposalId"], "mesap")
        self.assertEqual(rows[0]["objectiveVector"], [0.73])
        self.assertEqual(rows[0]["objectiveNames"], ["fitness"])
        self.assertEqual(rows[0]["diagnosticObjectiveVector"], [0.6, 1.0])
        self.assertEqual(rows[0]["comparableObjectiveVector"], [0.8, 0.5])
        self.assertTrue(rows[0]["postHocNonDominated"])

    def test_single_objective_summary_reports_posthoc_non_dominated_count(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd")
        metrics = service._summarize_rows(
            proposal,
            [
                {
                    "status": "ok",
                    "objectiveVector": [0.8],
                    "diagnosticObjectiveVector": [0.8, 0.2],
                    "postHocNonDominated": True,
                },
                {
                    "status": "ok",
                    "objectiveVector": [0.7],
                    "diagnosticObjectiveVector": [0.7, 0.5],
                    "postHocNonDominated": True,
                },
                {
                    "status": "ok",
                    "objectiveVector": [0.6],
                    "diagnosticObjectiveVector": [0.6, 0.1],
                    "postHocNonDominated": False,
                },
            ],
            Path("out"),
        )
        self.assertTrue(metrics["postHocDiagnostic"])
        self.assertEqual(metrics["postHocNonDominatedRows"], 2)
        self.assertEqual(metrics["nonDominatedRows"], 2)

    def test_charts_use_common_non_dominated_rows_for_evolmd(self):
        rows = [
            {
                "proposalId": "evolmd",
                "displayName": "EVOLMD",
                "status": "ok",
                "objectiveVector": [0.6],
                "diagnosticObjectiveVector": [0.6, 0.4],
                "comparableObjectiveVector": [0.8, 0.2],
                "generatedText": "single objective front",
                "postHocNonDominated": True,
                "nonDominated": True,
            },
            {
                "proposalId": "evolmd",
                "displayName": "EVOLMD",
                "status": "ok",
                "objectiveVector": [0.5],
                "diagnosticObjectiveVector": [0.5, 0.2],
                "comparableObjectiveVector": [0.75, 0.1],
                "generatedText": "dominated diagnostic row",
                "postHocNonDominated": False,
                "nonDominated": False,
            },
        ]
        charts = build_charts_from_rows(rows, [], [])
        self.assertEqual(len(charts["nonDominated"]), 1)
        self.assertEqual(charts["nonDominated"][0]["label"], "single objective front")
        self.assertEqual(charts["nonDominated"][0]["proposalId"], "evolmd")
        self.assertEqual(charts["nonDominated"][0]["x"], 0.8)
        self.assertEqual(charts["nonDominated"][0]["y"], 0.2)
        self.assertEqual(charts["nonDominated"][0]["nativeObjectiveVector"], [0.6, 0.4])

    def test_charts_use_native_non_dominated_rows_for_multiobjective_proposals(self):
        rows = [
            {
                "proposalId": "binary-mopso-cd",
                "displayName": "Binary MOPSO-CD",
                "status": "ok",
                "objectiveVector": [0.7, 0.5],
                "generatedText": "native front",
                "nonDominated": True,
                "postHocNonDominated": False,
            },
            {
                "proposalId": "evolmd-mo",
                "displayName": "EVOLMD-MO",
                "status": "ok",
                "objectiveVector": [0.6, 0.3],
                "generatedText": "native dominated",
                "nonDominated": False,
                "postHocNonDominated": True,
            },
        ]
        charts = build_charts_from_rows(rows, [], [])
        self.assertEqual(len(charts["nonDominated"]), 1)
        self.assertEqual(charts["nonDominated"][0]["label"], "native front")
        self.assertEqual(charts["nonDominated"][0]["proposalId"], "binary-mopso-cd")
        self.assertEqual(charts["nonDominated"][0]["x"], 0.85)
        self.assertEqual(charts["nonDominated"][0]["y"], 0.25)

    def test_embedding_front_rows_are_built_from_full_front_before_top_k(self):
        rows = [
            {
                "proposalId": "binary-mopso-cd",
                "instanceId": "binary-mopso-cd",
                "displayName": "Binary MOPSO-CD",
                "status": "ok",
                "generatedText": "front a",
                "rank": 1,
                "sourceIndex": 1,
                "nonDominated": True,
                "comparableObjectiveLabel": "[0.9, 0.4]",
            },
            {
                "proposalId": "binary-mopso-cd",
                "instanceId": "binary-mopso-cd",
                "displayName": "Binary MOPSO-CD",
                "status": "ok",
                "generatedText": "front b",
                "rank": 2,
                "sourceIndex": 2,
                "nonDominated": True,
                "comparableObjectiveLabel": "[0.8, 0.6]",
            },
            {
                "proposalId": "binary-mopso-cd",
                "instanceId": "binary-mopso-cd",
                "displayName": "Binary MOPSO-CD",
                "status": "ok",
                "generatedText": "dominated",
                "rank": 3,
                "sourceIndex": 3,
                "nonDominated": False,
            },
        ]
        selected_rows = [{"generatedText": "front b", "selectionRank": 1}]

        front_rows = embedding_front_rows_from_rows(rows, selected_rows)

        self.assertEqual([row["text"] for row in front_rows], ["front a", "front b"])
        self.assertFalse(front_rows[0]["selected"])
        self.assertTrue(front_rows[1]["selected"])
        self.assertEqual(front_rows[1]["selectionRank"], 1)

    def test_aggregate_repetitions_preserves_embedding_front_rows_beyond_visible_rows(self):
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        aggregated = aggregate_proposal_repetitions(
            proposal,
            Path("out"),
            [
                {
                    "status": "completed",
                    "repetitionIndex": 1,
                    "rows": [{"generatedText": "visible only"}],
                    "selectedRows": [],
                    "embeddingFrontRows": [
                        {"text": "front a", "proposalId": "binary-mopso-cd"},
                        {"text": "front b", "proposalId": "binary-mopso-cd"},
                    ],
                    "metrics": {},
                    "cost": {},
                    "series": [],
                }
            ],
            1,
        )

        self.assertEqual([row["text"] for row in aggregated["embeddingFrontRows"]], ["front a", "front b"])
        self.assertEqual(aggregated["embeddingFrontRows"][0]["repetitionIndex"], 1)

    def test_aggregate_repetitions_exposes_point_charts_per_repetition(self):
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        aggregated = aggregate_proposal_repetitions(
            proposal,
            Path("out"),
            [
                {
                    "status": "completed",
                    "repetitionIndex": 1,
                    "repetitionSeed": 42,
                    "rows": [{"generatedText": "visible rep 1"}],
                    "selectedRows": [],
                    "embeddingFrontRows": [
                        {"text": "rep 1 front a", "proposalId": "binary-mopso-cd"},
                        {"text": "rep 1 front b", "proposalId": "binary-mopso-cd"},
                    ],
                    "metrics": {"hypervolume": 0.11, "hypervolumeLabel": "0.110000"},
                    "charts": {
                        "pareto": [{"label": "rep 1 front a"}, {"label": "rep 1 front b"}],
                        "nonDominated": [{"label": "rep 1 front a"}, {"label": "rep 1 front b"}],
                        "selected": [],
                        "series": [],
                    },
                    "cost": {},
                    "series": [],
                },
                {
                    "status": "completed",
                    "repetitionIndex": 2,
                    "repetitionSeed": 43,
                    "rows": [{"generatedText": "visible rep 2"}],
                    "selectedRows": [],
                    "embeddingFrontRows": [
                        {"text": "rep 2 front a", "proposalId": "binary-mopso-cd"},
                        {"text": "rep 2 front b", "proposalId": "binary-mopso-cd"},
                        {"text": "rep 2 front c", "proposalId": "binary-mopso-cd"},
                    ],
                    "metrics": {"hypervolume": 0.22, "hypervolumeLabel": "0.220000"},
                    "charts": {
                        "pareto": [
                            {"label": "rep 2 front a"},
                            {"label": "rep 2 front b"},
                            {"label": "rep 2 front c"},
                        ],
                        "nonDominated": [
                            {"label": "rep 2 front a"},
                            {"label": "rep 2 front b"},
                            {"label": "rep 2 front c"},
                        ],
                        "selected": [],
                        "series": [],
                    },
                    "cost": {},
                    "series": [],
                },
            ],
            2,
        )

        repetitions = aggregated["pointChartRepetitions"]
        self.assertEqual([item["repetitionIndex"] for item in repetitions], [1, 2])
        self.assertEqual([item["repetitionSeed"] for item in repetitions], [42, 43])
        self.assertEqual([len(item["charts"]["pareto"]) for item in repetitions], [2, 3])
        self.assertEqual([len(item["embeddingFrontRows"]) for item in repetitions], [2, 3])
        self.assertEqual(repetitions[1]["metrics"]["hypervolumeLabel"], "0.220000")

    def test_binary_summary_uses_common_proxy_for_hypervolume(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": 0.0, "semanticDiversity": 1.5}],
        ):
            rows = service._normalize_rows(
                proposal,
                [
                    {
                        "solution_id": "s1",
                        "generated_text": "Generated",
                        "objectives": {"f1": 0.9, "f2": 0.1},
                    }
                ],
                top_k=5,
                reference_text="reference",
            )
        metrics = service._summarize_rows(proposal, rows, Path("out"))
        self.assertAlmostEqual(metrics["hypervolume"], 0.375)

    def test_evolmd_mo_and_binary_share_common_proxy_hv_normalization(self):
        service = ComparatorService(Path("."))
        evolmd_mo = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")
        binary = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")

        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": 0.0, "semanticDiversity": 1.5}],
        ):
            mo_rows = service._normalize_rows(
                evolmd_mo,
                [{"generated_data": "Generated", "objetivos": [0.9, 0.1]}],
                top_k=5,
                reference_text="reference",
            )
            binary_rows = service._normalize_rows(
                binary,
                [{"generated_text": "Generated", "objectives": {"f1": 0.2, "f2": 0.4}}],
                top_k=5,
                reference_text="reference",
            )

        self.assertEqual(mo_rows[0]["comparableObjectiveVector"], [0.5, 0.75])
        self.assertEqual(binary_rows[0]["comparableObjectiveVector"], [0.5, 0.75])
        self.assertEqual(mo_rows[0]["objectiveVector"], [0.9, 0.1])
        self.assertEqual(binary_rows[0]["objectiveVector"], [0.2, 0.4])
        self.assertAlmostEqual(service._summarize_rows(evolmd_mo, mo_rows, Path("out"))["hypervolume"], 0.375)
        self.assertAlmostEqual(service._summarize_rows(binary, binary_rows, Path("out"))["hypervolume"], 0.375)

    def test_evolmd_mo_native_objectives_do_not_drive_comparable_vector(self):
        service = ComparatorService(Path("."))
        evolmd_mo = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")

        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": -0.4, "semanticDiversity": 0.8}],
        ):
            rows = service._normalize_rows(
                evolmd_mo,
                [{"generated_data": "Generated", "objetivos": [0.957685112953186, 0.4842589497566223]}],
                top_k=5,
                reference_text="reference",
            )

        self.assertEqual(rows[0]["objectiveVector"], [0.957685112953186, 0.4842589497566223])
        self.assertAlmostEqual(rows[0]["comparableObjectiveVector"][0], 0.3)
        self.assertAlmostEqual(rows[0]["comparableObjectiveVector"][1], 0.4)

    def test_common_proxy_overrides_native_comparable_vectors_for_all_proposals(self):
        service = ComparatorService(Path("."))
        raw_by_proposal = {
            "evolmd": [{"generated_data": "Generated", "prompt": "Prompt", "fitness": 0.4}],
            "mesap": [{"generated_data": "Generated", "prompt": "Prompt", "fitness": 0.5}],
            "evolmd-mo": [{"generated_data": "Generated", "prompt": "Prompt", "objetivos": [-0.4, 1.6]}],
            "binary-mopso-cd": [{"generated_text": "Generated", "prompt": "Prompt", "objectives": {"f1": -0.4, "f2": 1.6}}],
        }

        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": 0.8, "semanticDiversity": 0.5}],
        ) as proxy:
            for proposal_id, raw_rows in raw_by_proposal.items():
                proposal = next(item for item in PROPOSALS if item.proposal_id == proposal_id)
                rows = service._normalize_rows(proposal, raw_rows, top_k=5, reference_text="reference")

                self.assertEqual(rows[0]["proxyObjectiveVector"], [0.8, 0.5])
                self.assertEqual(rows[0]["diagnosticObjectiveVector"], [0.8, 0.5])
                self.assertEqual(rows[0]["comparableObjectiveVector"], [0.9, 0.25])
                self.assertEqual(rows[0]["comparableObjectiveNames"], ["fidelity_sbert_proxy_normalized", "semantic_diversity_proxy_normalized"])

        self.assertEqual(proxy.call_count, 4)

    def test_common_proxy_skips_invalid_rows_and_metrics_use_proxy_front(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")

        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[
                {"semanticFidelity": 0.0, "semanticDiversity": 1.0},
                {"semanticFidelity": 0.6, "semanticDiversity": 0.2},
            ],
        ):
            rows = service._normalize_rows(
                proposal,
                [
                    {"generated_text": "front-a", "prompt": "Prompt A", "objectives": {"f1": -0.9, "f2": 0.1}},
                    {"generated_text": "", "prompt": "Invalid", "objectives": {"f1": 1.0, "f2": 2.0}},
                    {"generated_text": "front-b", "prompt": "Prompt B", "objectives": {"f1": -0.9, "f2": 0.1}},
                ],
                top_k=5,
                reference_text="reference",
            )

        invalid = next(row for row in rows if row["status"] != "ok")
        self.assertNotIn("proxyObjectiveVector", invalid)
        self.assertFalse(invalid["nonDominated"])

        metrics = service._summarize_rows(proposal, rows, Path("out"), include_artifact_metrics=False)
        self.assertEqual(metrics["nonDominatedRows"], 2)
        self.assertAlmostEqual(metrics["hypervolume"], 0.28)
        self.assertAlmostEqual(metrics["extent"], (0.3 + 0.4) ** 0.5)
        self.assertEqual(metrics["unaryEntropy"], 1.0)
        self.assertNotIn("spread", metrics)

    def test_front_metrics_follow_html_formulas(self):
        points = [(0.2, 0.8), (0.8, 0.2)]

        self.assertAlmostEqual(calculate_extent(points), (0.6 + 0.6) ** 0.5)
        self.assertAlmostEqual(calculate_unary_entropy(points, mu=5), 1.0)
        self.assertEqual(calculate_unary_entropy([(0.2, 0.8)], mu=5), 0.0)

        contributions = calculate_contribution(
            {
                "binary-mopso-cd": [(0.9, 0.4), (0.6, 0.8)],
                "evolmd-mo": [(0.9, 0.4), (0.3, 0.3)],
                "evolmd": [(0.2, 0.2)],
            }
        )
        self.assertAlmostEqual(contributions["binary-mopso-cd"], 0.75)
        self.assertAlmostEqual(contributions["evolmd-mo"], 0.25)
        self.assertAlmostEqual(contributions["evolmd"], 0.0)
        self.assertAlmostEqual(sum(contributions.values()), 1.0)

    def test_contribution_keeps_global_value_and_adds_repetition_std_dev(self):
        service = ComparatorService(Path("."))
        run = {
            "proposals": [
                {
                    "instanceId": "binary",
                    "proposalId": "binary-mopso-cd",
                    "status": "completed",
                    "metrics": {},
                    "charts": {"nonDominated": [{"x": 0.9, "y": 0.9}]},
                    "pointChartRepetitions": [
                        {
                            "repetitionIndex": 1,
                            "charts": {"nonDominated": [{"x": 0.9, "y": 0.9}]},
                        },
                        {
                            "repetitionIndex": 2,
                            "charts": {"nonDominated": [{"x": 0.2, "y": 0.2}]},
                        },
                    ],
                },
                {
                    "instanceId": "evolmd",
                    "proposalId": "evolmd",
                    "status": "completed",
                    "metrics": {},
                    "charts": {"nonDominated": [{"x": 0.8, "y": 0.8}]},
                    "pointChartRepetitions": [
                        {
                            "repetitionIndex": 1,
                            "charts": {"nonDominated": [{"x": 0.8, "y": 0.8}]},
                        },
                        {
                            "repetitionIndex": 2,
                            "charts": {"nonDominated": [{"x": 0.8, "y": 0.8}]},
                        },
                    ],
                },
            ],
        }

        service._apply_contribution_metrics_unlocked(run)

        binary_metrics = run["proposals"][0]["metrics"]
        evolmd_metrics = run["proposals"][1]["metrics"]
        self.assertAlmostEqual(binary_metrics["contribution"], 1.0)
        self.assertAlmostEqual(evolmd_metrics["contribution"], 0.0)
        self.assertAlmostEqual(binary_metrics["contributionStdDev"], 2 ** -0.5)
        self.assertEqual(binary_metrics["contributionStdDevLabel"], "0.707107")
        self.assertAlmostEqual(evolmd_metrics["contributionStdDev"], 2 ** -0.5)

    def test_evolmd_mo_legacy_series_reads_global_inertia_and_entropy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "Generacion,Max_Fidelidad,Max_Diversidad_Individual,Inercia_Global,Entropia_Global",
                        "1,0.8,0.7,0.3689031219,0.6886581024",
                        "2,0.9,0.8,0.3678999329,0.6905771496",
                    ]
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            evolmd_mo = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")

            series = service._read_legacy_metric_series(evolmd_mo, output_dir)

        self.assertEqual(len(series), 2)
        self.assertEqual(series[0]["generation"], 1)
        self.assertIsNone(series[0]["globalInertia"])
        self.assertIsNone(series[0]["globalEntropy"])
        self.assertIsNone(series[0]["hypervolume"])

    def test_evolmd_mo_history_series_uses_posthoc_diagnostics_instead_of_bugged_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "population_history.jsonl").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "population": [
                            {"generated_data": "Generated A", "objetivos": [0.0, 1.0]},
                            {"generated_data": "Generated B", "objetivos": [0.2, 0.8]},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "Generacion,Max_Fidelidad,Max_Diversidad_Individual,Inercia_Global,Entropia_Global",
                        "1,0.8,0.7,0.3689031219,0.6886581024",
                    ]
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            evolmd_mo = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")

            with patch.object(
                service,
                "_posthoc_population_diagnostics",
                return_value={"globalInertia": 0.125, "globalEntropy": 0.5},
            ), patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[
                    {"semanticFidelity": 0.0, "semanticDiversity": 1.0},
                    {"semanticFidelity": 0.2, "semanticDiversity": 0.8},
                ],
            ):
                series = service._build_metric_series(evolmd_mo, output_dir, [], "reference")

        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0]["hypervolume"], 0.29)
        self.assertAlmostEqual(series[0]["globalInertia"], 0.125)
        self.assertAlmostEqual(series[0]["globalEntropy"], 0.5)
        self.assertEqual(series[0]["source"], "population_history")

    def test_posthoc_population_diagnostics_uses_common_kmeans_definition(self):
        service = ComparatorService(Path("."))
        calls = []

        class FakeSbertService:
            def encode_texts(self, model_name, texts):
                calls.append((model_name, list(texts)))
                return [[float(index)] for index, _text in enumerate(texts)], {}

        class FakeKMeans:
            def __init__(self, n_clusters, n_init, random_state):
                self.n_clusters = n_clusters
                self.n_init = n_init
                self.random_state = random_state
                self.inertia_ = 12.0

            def fit(self, embeddings):
                self.embeddings = embeddings
                return self

        created_models = []

        def fake_kmeans(*args, **kwargs):
            model = FakeKMeans(*args, **kwargs)
            created_models.append(model)
            return model

        fake_sklearn = types.SimpleNamespace()
        fake_cluster = types.SimpleNamespace(KMeans=fake_kmeans)
        with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()), patch.dict(
            sys.modules,
            {"sklearn": fake_sklearn, "sklearn.cluster": fake_cluster},
        ), patch.object(service, "_posthoc_entity_entropy", return_value=0.25):
            metrics = service._posthoc_population_diagnostics(["a", "b", "c", "d", "e", "f"])

        self.assertEqual(calls, [(comparator_module.POSTHOC_EMBEDDING_MODEL, ["a", "b", "c", "d", "e", "f"])])
        self.assertEqual(created_models[0].n_clusters, 5)
        self.assertEqual(created_models[0].n_init, 10)
        self.assertEqual(created_models[0].random_state, 0)
        self.assertAlmostEqual(metrics["globalInertia"], 2.0)
        self.assertAlmostEqual(metrics["globalEntropy"], 0.25)

    def test_posthoc_entity_entropy_uses_pos_lemmas_and_log2_normalization(self):
        service = ComparatorService(Path("."))
        service._posthoc_spacy_model = FakeSpacyModel(
            [
                [
                    FakeSpacyToken("shelter", "NOUN"),
                    FakeSpacyToken("help", "VERB"),
                    FakeSpacyToken("quickly", "ADV"),
                ],
                [
                    FakeSpacyToken("urgent", "ADJ"),
                    FakeSpacyToken("shelter", "NOUN"),
                    FakeSpacyToken("Chile", "PROPN"),
                ],
            ]
        )

        score = service._posthoc_entity_entropy(["doc one", "doc two"])

        self.assertAlmostEqual(score, 1.5 / comparator_module.math.log2(6))

    def test_posthoc_entity_entropy_returns_none_when_spacy_model_is_missing(self):
        service = ComparatorService(Path("."))
        fake_spacy = types.SimpleNamespace(load=Mock(side_effect=OSError("missing model")))

        with patch.dict(sys.modules, {"spacy": fake_spacy}):
            score = service._posthoc_entity_entropy(["doc one", "doc two"])

        self.assertIsNone(score)

    def test_posthoc_population_diagnostics_preserves_inertia_when_entity_entropy_unavailable(self):
        service = ComparatorService(Path("."))

        class FakeSbertService:
            def encode_texts(self, _model_name, texts):
                return [[float(index)] for index, _text in enumerate(texts)], {}

        class FakeKMeans:
            inertia_ = 12.0

            def __init__(self, *_args, **_kwargs):
                pass

            def fit(self, _embeddings):
                return self

        fake_sklearn = types.SimpleNamespace()
        fake_cluster = types.SimpleNamespace(KMeans=FakeKMeans)
        with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()), patch.dict(
            sys.modules,
            {"sklearn": fake_sklearn, "sklearn.cluster": fake_cluster},
        ), patch.object(service, "_posthoc_entity_entropy", return_value=None):
            metrics = service._posthoc_population_diagnostics(["a", "b", "c", "d", "e", "f"])

        self.assertAlmostEqual(metrics["globalInertia"], 2.0)
        self.assertIsNone(metrics["globalEntropy"])

    def test_front_point_diagnostics_returns_embeddings_and_cached_entity_terms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_id = "front-diagnostics-run"
            run_dir = root / "runs" / "comparator" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"runId": run_id, "status": "completed", "proposals": []}),
                encoding="utf-8",
            )
            service = ComparatorService(root)
            service._posthoc_spacy_model = FakeSpacyModel(
                [
                    [FakeSpacyToken("shelter", "NOUN"), FakeSpacyToken("quickly", "ADV")],
                    [FakeSpacyToken("help", "VERB"), FakeSpacyToken("urgent", "ADJ")],
                ]
            )
            calls: list[tuple[str, list[str]]] = []

            class FakeSbertService:
                def encode_texts(self, model_name, texts):
                    calls.append((model_name, list(texts)))
                    return [[float(index), float(index + 1)] for index, _text in enumerate(texts)], {
                        "embeddingModel": model_name,
                        "sourceModel": model_name,
                        "embeddingTexts": len(texts),
                        "embeddingWallClockSeconds": 0.01,
                    }

            payload = {
                "points": [
                    {"key": "p1", "text": "Need urgent shelter"},
                    {"key": "p2", "text": "Help families"},
                ]
            }

            with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()):
                first = service.get_run_front_point_diagnostics(run_id, payload)
                second = service.get_run_front_point_diagnostics(run_id, payload)

        self.assertEqual(calls, [(comparator_module.POSTHOC_EMBEDDING_MODEL, ["Need urgent shelter", "Help families"])])
        self.assertEqual(first["points"], second["points"])
        self.assertEqual(first["points"][0]["key"], "p1")
        self.assertEqual(first["points"][0]["embedding"], [0.0, 1.0])
        self.assertEqual(first["points"][0]["entityTerms"], ["shelter"])
        self.assertEqual(first["points"][0]["entityTokenCount"], 2)
        self.assertEqual(first["points"][1]["entityTerms"], ["help", "urgent"])

    def test_front_point_diagnostics_returns_none_for_missing_run(self):
        service = ComparatorService(Path("."))

        self.assertIsNone(service.get_run_front_point_diagnostics("missing", {"points": []}))

    def test_binary_monitor_series_reads_inertia_and_entropy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "monitor_metrics.csv").write_text(
                "\n".join(
                    [
                        "generation,kmeans_inertia,entity_entropy,monitor_overhead_seconds",
                        "1,0.125,0.693,0.01",
                        "2,0.250,0.810,0.02",
                    ]
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))

            series = service._read_binary_monitor_metric_series(output_dir)

        self.assertEqual(len(series), 2)
        self.assertEqual(series[0]["generation"], 1)
        self.assertAlmostEqual(series[0]["globalInertia"], 0.125)
        self.assertAlmostEqual(series[0]["globalEntropy"], 0.693)
        self.assertEqual(series[0]["source"], "binary_monitor")

    def test_binary_archive_series_merges_monitor_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "archive_history.jsonl").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "archive": [
                            {
                                "generated_text": "Generated A",
                                "prompt": "Prompt A",
                                "objectives": {"f1": 0.0, "f2": 1.0},
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "monitor_metrics.csv").write_text(
                "generation,kmeans_inertia,entity_entropy\n1,0.125,0.693\n",
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")

            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[{"semanticFidelity": 0.0, "semanticDiversity": 1.0}],
            ):
                series = service._build_metric_series(proposal, output_dir, [], "reference")

        self.assertEqual(len(series), 1)
        self.assertIsNotNone(series[0]["hypervolume"])
        self.assertAlmostEqual(series[0]["globalInertia"], 0.125)
        self.assertAlmostEqual(series[0]["globalEntropy"], 0.693)
        self.assertEqual(series[0]["source"], "archive_history")

    def test_binary_native_series_adds_final_proxy_point_without_archive_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "generation,hypervolume,archive_size",
                        "1,0.11,2",
                        "2,0.22,3",
                    ]
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")

            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[
                    {"semanticFidelity": 0.0, "semanticDiversity": 1.0},
                    {"semanticFidelity": 0.4, "semanticDiversity": 0.6},
                ],
            ):
                final_rows = service._normalize_rows(
                    proposal,
                    [
                        {"generated_text": "Generated A", "prompt": "Prompt A", "objectives": {"f1": 0.9, "f2": 0.1}},
                        {"generated_text": "Generated B", "prompt": "Prompt B", "objectives": {"f1": 0.2, "f2": 0.8}},
                    ],
                    top_k=5,
                    reference_text="reference",
                )
                series = service._build_metric_series(proposal, output_dir, final_rows, "reference")

        self.assertEqual([point["generation"] for point in series], [1, 2])
        self.assertEqual(series[0]["source"], "native")

    def test_binary_metric_series_adds_generation_zero_from_initial_population(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "data_initial_population.json").write_text(
                json.dumps(
                    [
                        {"generated_text": "Initial A", "prompt": "Prompt A", "objectives": {"f1": 0.9, "f2": 0.1}},
                        {"generated_text": "Initial B", "prompt": "Prompt B", "objectives": {"f1": 0.2, "f2": 0.8}},
                    ]
                ),
                encoding="utf-8",
            )
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "generation,hypervolume,archive_size",
                        "1,0.11,2",
                        "2,0.22,3",
                    ]
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")

            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[
                    {"semanticFidelity": 0.0, "semanticDiversity": 1.0},
                    {"semanticFidelity": 0.4, "semanticDiversity": 0.6},
                ],
            ), patch.object(
                service,
                "_posthoc_population_diagnostics",
                return_value={"globalInertia": 0.2, "globalEntropy": 0.3},
            ):
                series = service._build_metric_series(proposal, output_dir, [], "reference")

        self.assertEqual([point["generation"] for point in series], [0, 1, 2])
        self.assertEqual(series[0]["source"], "initial_population")
        self.assertIsNotNone(series[0]["hypervolume"])
        self.assertIsNotNone(series[0]["extent"])
        self.assertIsNotNone(series[0]["unaryEntropy"])
        self.assertAlmostEqual(series[0]["globalInertia"], 0.2)
        self.assertAlmostEqual(series[0]["globalEntropy"], 0.3)
        self.assertEqual(series[0]["frontPoints"], [[0.5, 0.5], [0.7, 0.3]])
        self.assertEqual(series[1]["source"], "native")

    def test_legacy_series_adds_final_proxy_point_without_population_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "Generacion,Max_Fidelidad,Max_Diversidad_Individual,Inercia_Global,Entropia_Global",
                        "1,0.8,0.7,0.3689031219,0.6886581024",
                        "2,0.9,0.8,0.3678999329,0.6905771496",
                    ]
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")

            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[
                    {"semanticFidelity": 0.0, "semanticDiversity": 1.0},
                    {"semanticFidelity": 0.4, "semanticDiversity": 0.6},
                ],
            ):
                final_rows = service._normalize_rows(
                    proposal,
                    [
                        {"generated_data": "Generated A", "prompt": "Prompt A", "objetivos": [0.9, 0.1]},
                        {"generated_data": "Generated B", "prompt": "Prompt B", "objetivos": [0.2, 0.8]},
                    ],
                    top_k=5,
                    reference_text="reference",
                )
                series = service._build_metric_series(proposal, output_dir, final_rows, "reference")

        self.assertEqual([point["generation"] for point in series], [1, 2])
        self.assertEqual(series[0]["source"], "legacy_csv_without_front")
        self.assertIsNone(series[0]["globalInertia"])
        self.assertIsNone(series[0]["globalEntropy"])
        self.assertEqual(series[1]["source"], "legacy_csv_without_front")
        self.assertIsNone(series[1]["globalInertia"])
        self.assertIsNone(series[1]["globalEntropy"])

    def test_series_contribution_uses_final_only_generation(self):
        service = ComparatorService(Path("."))
        proposals = [
            {
                "proposalId": "binary-mopso-cd",
                "instanceId": "binary-mopso-cd",
                "series": [
                    {
                        "generation": 0,
                        "frontPoints": [[0.9, 0.4], [0.6, 0.8]],
                    }
                ],
            },
            {
                "proposalId": "evolmd-mo",
                "instanceId": "evolmd-mo",
                "series": [
                    {
                        "generation": 0,
                        "frontPoints": [[0.9, 0.4], [0.3, 0.3]],
                    }
                ],
            },
        ]

        service._apply_series_contribution_metrics(proposals)

        self.assertAlmostEqual(proposals[0]["series"][0]["contribution"], 0.75)
        self.assertAlmostEqual(proposals[1]["series"][0]["contribution"], 0.25)

    def test_history_series_uses_posthoc_diagnostics_when_native_diagnostics_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "population_history.jsonl").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "population": [
                            {
                                "generated_data": "Generated A",
                                "prompt": "Prompt A",
                                "fitness": 0.7,
                            },
                            {
                                "generated_data": "Generated B",
                                "prompt": "Prompt B",
                                "fitness": 0.6,
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "metrics_log.csv").write_text(
                "generation,best_fitness\n1,0.7\n",
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            proposal = next(item for item in PROPOSALS if item.proposal_id == "mesap")

            with patch.object(
                service,
                "_posthoc_population_diagnostics",
                return_value={"globalInertia": 0.33, "globalEntropy": 0.44},
            ), patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[
                    {"semanticFidelity": 0.6, "semanticDiversity": 1.0},
                    {"semanticFidelity": 0.5, "semanticDiversity": 1.2},
                ],
            ):
                series = service._build_metric_series(proposal, output_dir, [], "reference")

        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0]["globalInertia"], 0.33)
        self.assertAlmostEqual(series[0]["globalEntropy"], 0.44)
        self.assertEqual(series[0]["source"], "population_history")

    def test_mesap_history_series_uses_population_history_for_posthoc_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "population_history.jsonl").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "population": [
                            {
                                "generated_data": "Generated",
                                "prompt": "Prompt",
                                "fitness": 0.73,
                                "role": "role",
                                "topic": "topic",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            proposal = next(item for item in PROPOSALS if item.proposal_id == "mesap")

            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[{"semanticFidelity": 0.6, "semanticDiversity": 1.0}],
            ):
                series = service._build_metric_series(proposal, output_dir, [], "reference")

        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["generation"], 1)
        self.assertAlmostEqual(series[0]["hypervolume"], 0.4)
        self.assertEqual(series[0]["nonDominatedRows"], 1)
        self.assertEqual(series[0]["extent"], 0.0)
        self.assertEqual(series[0]["unaryEntropy"], 0.0)
        self.assertIsNone(series[0]["contribution"])
        self.assertEqual(series[0]["source"], "population_history")

    def test_legacy_series_without_global_diagnostics_keeps_missing_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "evolucion_metricas.csv").write_text(
                "\n".join(
                    [
                        "Generacion,Max_Fidelidad,Max_Diversidad_Individual",
                        "1,0.8,0.7",
                    ]
                ),
                encoding="utf-8",
            )
            service = ComparatorService(Path("."))
            evolmd_mo = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")

            series = service._read_legacy_metric_series(evolmd_mo, output_dir)

        self.assertEqual(len(series), 1)
        self.assertIsNone(series[0]["globalInertia"])
        self.assertIsNone(series[0]["globalEntropy"])

    def test_aggregate_series_preserves_global_diagnostics(self):
        series = aggregate_series(
            [
                {
                    "series": [
                        {"generation": 1, "globalInertia": 0.2, "globalEntropy": 0.6},
                        {"generation": 2, "globalInertia": 0.4, "globalEntropy": 0.8},
                    ]
                },
                {
                    "series": [
                        {"generation": 1, "globalInertia": 0.4, "globalEntropy": 0.8},
                        {"generation": 2},
                    ]
                },
            ]
        )

        self.assertAlmostEqual(series[0]["globalInertia"], 0.3)
        self.assertAlmostEqual(series[0]["globalEntropy"], 0.7)
        self.assertAlmostEqual(series[1]["globalInertia"], 0.4)
        self.assertAlmostEqual(series[1]["globalEntropy"], 0.8)

    def test_aggregate_proposal_repetitions_exposes_terminal_global_diagnostics_as_metrics(self):
        proposal = next(item for item in PROPOSALS if item.proposal_id == "evolmd-mo")
        aggregated = aggregate_proposal_repetitions(
            proposal,
            Path("."),
            [
                {
                    "status": "completed",
                    "rows": [],
                    "selectedRows": [],
                    "metrics": {"totalRows": 0, "completedRows": 0},
                    "series": [
                        {"generation": 1, "globalInertia": 0.2, "globalEntropy": 0.6},
                        {"generation": 2, "globalInertia": 0.4, "globalEntropy": 0.8},
                    ],
                    "cost": {},
                }
            ],
            repetitions_k=1,
        )

        self.assertIn("globalInertia", aggregated["metrics"])
        self.assertAlmostEqual(aggregated["metrics"]["globalInertia"], 0.4)
        self.assertEqual(aggregated["metrics"]["globalInertiaLabel"], "0.400000")
        self.assertIn("globalEntropy", aggregated["metrics"])
        self.assertAlmostEqual(aggregated["metrics"]["globalEntropy"], 0.8)
        self.assertEqual(aggregated["metrics"]["globalEntropyLabel"], "0.800000")

    def test_legacy_summary_without_comparable_points_is_recompute_recommended(self):
        service = ComparatorService(Path("."))
        run = {
            "status": "completed",
            "proposals": [
                {
                    "proposalId": "evolmd-mo",
                    "rows": [{"objectiveVector": [0.9, 0.5]}],
                    "charts": {"pareto": [{"x": 0.9, "y": 0.5}]},
                }
            ],
        }

        status = service._with_metric_recompute_status(run)["metricRecomputeStatus"]

        self.assertTrue(status["available"])
        self.assertTrue(status["recommended"])
        self.assertTrue(status["legacyChartPointsDetected"])

    def test_recompute_run_metrics_rebuilds_legacy_summary_from_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_id = "legacy-run"
            run_dir = root / "runs" / "comparator" / run_id
            output_dir = run_dir / "evolmd-mo" / "exec" / "out"
            output_dir.mkdir(parents=True)
            (output_dir / "pareto_front.json").write_text(
                json.dumps(
                    [
                        {
                            "generated_data": "Generated",
                            "prompt": "Prompt",
                            "objetivos": [0.0, 1.5],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            summary = {
                "runId": run_id,
                "status": "completed",
                "runDir": str(run_dir),
                "config": {"referenceText": "reference", "topK": 10, "repetitionsK": 1},
                "logs": [],
                "proposals": [
                    {
                        "proposalId": "evolmd-mo",
                        "displayName": "EVOLMD-MO",
                        "status": "completed",
                        "outputDir": str(output_dir),
                        "rows": [
                            {
                                "proposalId": "evolmd-mo",
                                "status": "ok",
                                "objectiveVector": [0.0, 1.5],
                                "generatedText": "Generated",
                            }
                        ],
                        "metrics": {"hypervolume": 0.75},
                        "charts": {"pareto": [{"x": 0.0, "y": 1.5}]},
                        "cost": {},
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            with patch.object(
                comparator_module,
                "calculate_posthoc_semantic_scores",
                return_value=[{"semanticFidelity": 0.0, "semanticDiversity": 1.5}],
            ), patch.object(
                service,
                "_build_metric_series",
                return_value=[{"generation": 1, "globalInertia": 0.3, "globalEntropy": 0.7}],
            ):
                recomputed = service.recompute_run_metrics(run_id)
            proposal = recomputed["proposals"][0]

        self.assertEqual(recomputed["metricSchemaVersion"], 4)
        self.assertEqual(recomputed["metricCoordinateSpace"], "comparable_normalized")
        self.assertFalse(recomputed["metricRecomputeStatus"]["recommended"])
        self.assertAlmostEqual(proposal["metrics"]["hypervolume"], 0.375)
        self.assertAlmostEqual(proposal["metrics"]["globalInertia"], 0.3)
        self.assertAlmostEqual(proposal["metrics"]["globalEntropy"], 0.7)
        self.assertEqual(proposal["rows"][0]["comparableObjectiveVector"], [0.5, 0.75])
        self.assertEqual(proposal["charts"]["pareto"][0]["coordinateSpace"], "comparable_normalized")

    def test_embedding_projection_payload_uses_front_rows_without_raw_embeddings(self):
        class FakeSbertService:
            def encode_texts(self, model_name, texts):
                return (
                    [
                        [1.0, 0.0, 0.0],
                        [0.8, 0.2, 0.0],
                        [0.2, 0.8, 0.0],
                    ],
                    {
                        "embeddingModel": model_name,
                        "sourceModel": "sentence-transformers/all-MiniLM-L6-v2",
                        "embeddingTexts": len(texts),
                        "embeddingWallClockSeconds": 0.01,
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_id = "projection-run"
            run_dir = root / "runs" / "comparator" / run_id
            run_dir.mkdir(parents=True)
            summary = {
                "runId": run_id,
                "status": "completed",
                "runDir": str(run_dir),
                "config": {"referenceText": "reference"},
                "proposals": [
                    {
                        "proposalId": "binary-mopso-cd",
                        "instanceId": "binary-mopso-cd",
                        "displayName": "Binary MOPSO-CD",
                        "status": "completed",
                        "embeddingFrontRows": [
                            {"text": "front a", "selected": True, "selectionRank": 1, "objectiveLabel": "[0.9, 0.4]"},
                            {"text": "front b", "selected": False, "objectiveLabel": "[0.8, 0.6]"},
                        ],
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()):
                payload = service.get_run_embedding_projection(run_id, "pca")

        self.assertEqual(payload["runId"], run_id)
        self.assertEqual(payload["method"], "pca")
        self.assertEqual(payload["effectiveMethod"], "pca")
        self.assertEqual(payload["embeddingTexts"], 3)
        self.assertIn("x", payload["reference"])
        self.assertEqual(len(payload["proposals"][0]["points"]), 2)
        self.assertNotIn("embeddings", payload)
        self.assertNotIn("embedding", payload["proposals"][0]["points"][0])

    def test_embedding_projection_filters_front_rows_by_repetition(self):
        class FakeSbertService:
            def encode_texts(self, model_name, texts):
                embeddings = [[1.0, 0.0, 0.0]]
                for index, _text in enumerate(texts[1:], start=1):
                    embeddings.append([0.0, float(index), 1.0 - (index * 0.1)])
                return (
                    embeddings,
                    {
                        "embeddingModel": model_name,
                        "sourceModel": "sentence-transformers/all-MiniLM-L6-v2",
                        "embeddingTexts": len(texts),
                        "embeddingWallClockSeconds": 0.01,
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_id = "projection-repetition-run"
            run_dir = root / "runs" / "comparator" / run_id
            run_dir.mkdir(parents=True)
            summary = {
                "runId": run_id,
                "status": "completed",
                "runDir": str(run_dir),
                "config": {"referenceText": "reference"},
                "proposals": [
                    {
                        "proposalId": "binary-mopso-cd",
                        "instanceId": "binary-mopso-cd",
                        "displayName": "Binary MOPSO-CD",
                        "status": "completed",
                        "pointChartRepetitions": [
                            {
                                "repetitionIndex": 1,
                                "charts": {
                                    "pareto": [
                                        {"label": "rep 1 front", "proposalId": "binary-mopso-cd"},
                                    ],
                                    "nonDominated": [],
                                    "selected": [],
                                },
                                "embeddingFrontRows": [
                                    {"text": "rep 1 front", "proposalId": "binary-mopso-cd"},
                                ],
                            },
                            {
                                "repetitionIndex": 2,
                                "charts": {
                                    "pareto": [
                                        {"label": "rep 2 front a", "proposalId": "binary-mopso-cd"},
                                        {"label": "rep 2 front b", "proposalId": "binary-mopso-cd"},
                                    ],
                                    "nonDominated": [],
                                    "selected": [],
                                },
                                "embeddingFrontRows": [
                                    {"text": "rep 2 front a", "proposalId": "binary-mopso-cd"},
                                    {"text": "rep 2 front b", "proposalId": "binary-mopso-cd"},
                                ],
                            },
                        ],
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()):
                payload = service.get_run_embedding_projection(run_id, "pca", repetition=2)

        self.assertEqual(payload["repetitionIndex"], 2)
        self.assertEqual(payload["embeddingTexts"], 3)
        self.assertEqual(
            [point["text"] for point in payload["proposals"][0]["points"]],
            ["rep 2 front a", "rep 2 front b"],
        )

    def test_embedding_projection_uses_binary_visible_pareto_for_selected_repetition(self):
        class FakeSbertService:
            def encode_texts(self, model_name, texts):
                embeddings = [[1.0, 0.0, 0.0]]
                for index, _text in enumerate(texts[1:], start=1):
                    embeddings.append([0.0, float(index), 1.0 - (index * 0.01)])
                return (
                    embeddings,
                    {
                        "embeddingModel": model_name,
                        "sourceModel": "sentence-transformers/all-MiniLM-L6-v2",
                        "embeddingTexts": len(texts),
                        "embeddingWallClockSeconds": 0.01,
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_id = "binary-visible-projection-run"
            run_dir = root / "runs" / "comparator" / run_id
            run_dir.mkdir(parents=True)
            summary = {
                "runId": run_id,
                "status": "completed",
                "runDir": str(run_dir),
                "config": {"referenceText": "reference"},
                "proposals": [
                    {
                        "proposalId": "binary-mopso-cd",
                        "instanceId": "binary-mopso-cd-2",
                        "displayName": "Binary MOPSO-CD - c1>c2",
                        "status": "completed",
                        "embeddingFrontRows": [
                            {"text": f"aggregated {index}", "proposalId": "binary-mopso-cd"}
                            for index in range(45)
                        ],
                        "pointChartRepetitions": [
                            {
                                "repetitionIndex": 1,
                                "charts": {
                                    "pareto": [{"label": "rep 1 native", "proposalId": "binary-mopso-cd"}],
                                    "nonDominated": [{"label": "rep 1 comparable", "proposalId": "binary-mopso-cd"}],
                                    "selected": [],
                                },
                                "embeddingFrontRows": [{"text": "rep 1 comparable", "proposalId": "binary-mopso-cd"}],
                            },
                            {
                                "repetitionIndex": 2,
                                "charts": {
                                    "pareto": [
                                        {"label": f"rep 2 native {index}", "proposalId": "binary-mopso-cd"}
                                        for index in range(37)
                                    ],
                                    "nonDominated": [
                                        {"label": f"rep 2 comparable {index}", "proposalId": "binary-mopso-cd"}
                                        for index in range(29)
                                    ],
                                    "selected": [],
                                },
                                "embeddingFrontRows": [
                                    {"text": f"rep 2 comparable {index}", "proposalId": "binary-mopso-cd"}
                                    for index in range(29)
                                ],
                            },
                        ],
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()):
                payload = service.get_run_embedding_projection(run_id, "pca", repetition=2)

        self.assertEqual(payload["repetitionIndex"], 2)
        self.assertEqual(payload["embeddingTexts"], 38)
        self.assertEqual(len(payload["proposals"][0]["points"]), 37)
        self.assertEqual(payload["proposals"][0]["points"][0]["text"], "rep 2 native 0")
        self.assertEqual(payload["proposals"][0]["points"][-1]["text"], "rep 2 native 36")

    def test_embedding_projection_uses_non_binary_visible_non_dominated_for_selected_repetition(self):
        class FakeSbertService:
            def encode_texts(self, model_name, texts):
                embeddings = [[1.0, 0.0, 0.0]]
                for index, _text in enumerate(texts[1:], start=1):
                    embeddings.append([0.0, float(index), 1.0 - (index * 0.1)])
                return (
                    embeddings,
                    {
                        "embeddingModel": model_name,
                        "sourceModel": "sentence-transformers/all-MiniLM-L6-v2",
                        "embeddingTexts": len(texts),
                        "embeddingWallClockSeconds": 0.01,
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_id = "non-binary-visible-projection-run"
            run_dir = root / "runs" / "comparator" / run_id
            run_dir.mkdir(parents=True)
            summary = {
                "runId": run_id,
                "status": "completed",
                "runDir": str(run_dir),
                "config": {"referenceText": "reference"},
                "proposals": [
                    {
                        "proposalId": "evolmd-mo",
                        "instanceId": "evolmd-mo-1",
                        "displayName": "EVOLMD-MO",
                        "status": "completed",
                        "pointChartRepetitions": [
                            {
                                "repetitionIndex": 2,
                                "charts": {
                                    "pareto": [
                                        {"label": "dominated visible candidate", "proposalId": "evolmd-mo"},
                                        {"label": "front a", "proposalId": "evolmd-mo"},
                                        {"label": "front b", "proposalId": "evolmd-mo"},
                                    ],
                                    "nonDominated": [
                                        {"label": "front a", "proposalId": "evolmd-mo"},
                                        {"label": "front b", "proposalId": "evolmd-mo"},
                                    ],
                                    "selected": [],
                                },
                                "embeddingFrontRows": [
                                    {"text": "legacy front a", "proposalId": "evolmd-mo"},
                                    {"text": "legacy front b", "proposalId": "evolmd-mo"},
                                    {"text": "legacy extra", "proposalId": "evolmd-mo"},
                                ],
                            },
                        ],
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()):
                payload = service.get_run_embedding_projection(run_id, "pca", repetition=2)

        self.assertEqual(payload["embeddingTexts"], 3)
        self.assertEqual(
            [point["text"] for point in payload["proposals"][0]["points"]],
            ["front a", "front b"],
        )

    def test_embedding_projection_uses_repetition_charts_when_point_chart_repetitions_are_absent(self):
        class FakeSbertService:
            def encode_texts(self, model_name, texts):
                embeddings = [[1.0, 0.0, 0.0]]
                for index, _text in enumerate(texts[1:], start=1):
                    embeddings.append([0.0, float(index), 1.0 - (index * 0.1)])
                return (
                    embeddings,
                    {
                        "embeddingModel": model_name,
                        "sourceModel": "sentence-transformers/all-MiniLM-L6-v2",
                        "embeddingTexts": len(texts),
                        "embeddingWallClockSeconds": 0.01,
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = ComparatorService(root)
            run_id = "legacy-repetition-chart-projection-run"
            run_dir = root / "runs" / "comparator" / run_id
            run_dir.mkdir(parents=True)
            summary = {
                "runId": run_id,
                "status": "completed",
                "runDir": str(run_dir),
                "config": {"referenceText": "reference"},
                "proposals": [
                    {
                        "proposalId": "binary-mopso-cd",
                        "instanceId": "binary-mopso-cd-legacy",
                        "displayName": "Binary MOPSO-CD",
                        "status": "completed",
                        "embeddingFrontRows": [
                            {"text": f"aggregated {index}", "proposalId": "binary-mopso-cd"}
                            for index in range(6)
                        ],
                        "repetitions": [
                            {
                                "status": "completed",
                                "repetitionIndex": 1,
                                "charts": {
                                    "pareto": [{"label": "rep 1 visible", "proposalId": "binary-mopso-cd"}],
                                    "nonDominated": [],
                                    "selected": [],
                                },
                                "embeddingFrontRows": [{"text": "rep 1 legacy", "proposalId": "binary-mopso-cd"}],
                            },
                            {
                                "status": "completed",
                                "repetitionIndex": 2,
                                "charts": {
                                    "pareto": [
                                        {"label": "rep 2 visible a", "proposalId": "binary-mopso-cd"},
                                        {"label": "rep 2 visible b", "proposalId": "binary-mopso-cd"},
                                    ],
                                    "nonDominated": [],
                                    "selected": [],
                                },
                                "embeddingFrontRows": [{"text": "rep 2 legacy", "proposalId": "binary-mopso-cd"}],
                            },
                        ],
                    }
                ],
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            with patch.object(comparator_module, "shared_sbert_service", return_value=FakeSbertService()):
                payload = service.get_run_embedding_projection(run_id, "pca", repetition=2)

        self.assertEqual(payload["embeddingTexts"], 3)
        self.assertEqual(
            [point["text"] for point in payload["proposals"][0]["points"]],
            ["rep 2 visible a", "rep 2 visible b"],
        )

    def test_binary_internal_bmopso_analysis_is_attached_per_point_repetition(self):
        service = ComparatorService(Path("."))

        def fake_internal_analysis(result, _identity_source):
            repetition_index = result.get("repetitionIndex")
            return {
                "available": True,
                "repetitionIndex": repetition_index,
                "repetitionSeed": result.get("repetitionSeed"),
                "proposalId": "binary-mopso-cd",
                "instanceId": "binary-mopso-cd",
                "displayName": "Binary MOPSO-CD",
                "metrics": {
                    "hypervolume": repetition_index / 10,
                    "hypervolumeLabel": f"{repetition_index / 10:.6f}",
                },
                "series": [{"generation": 1, "hypervolume": repetition_index / 10}],
                "charts": {
                    "pareto": [{"x": repetition_index / 10, "y": 0.5}],
                    "selected": [],
                    "nonDominated": [{"x": repetition_index / 10, "y": 0.5}],
                },
            }

        run = {
            "proposals": [
                {
                    "proposalId": "binary-mopso-cd",
                    "instanceId": "binary-mopso-cd",
                    "displayName": "Binary MOPSO-CD",
                    "status": "completed",
                    "pointChartRepetitions": [
                        {"repetitionIndex": 1, "charts": {"pareto": []}},
                        {"repetitionIndex": 2, "charts": {"pareto": []}},
                    ],
                    "repetitions": [
                        {"status": "completed", "repetitionIndex": 1, "repetitionSeed": 42},
                        {"status": "completed", "repetitionIndex": 2, "repetitionSeed": 43},
                    ],
                }
            ],
        }

        with patch.object(service, "_build_single_internal_bmopso_analysis", side_effect=fake_internal_analysis):
            public = service._with_internal_bmopso_analysis(run)

        proposal = public["proposals"][0]
        self.assertIn("internalBmopsoAnalysis", proposal)
        repetitions = proposal["pointChartRepetitions"]
        self.assertEqual(
            [item["internalBmopsoAnalysis"]["repetitionIndex"] for item in repetitions],
            [1, 2],
        )
        self.assertEqual(
            [item["internalBmopsoAnalysis"]["metrics"]["hypervolumeLabel"] for item in repetitions],
            ["0.100000", "0.200000"],
        )

    def test_embedding_projection_returns_none_for_missing_run(self):
        service = ComparatorService(Path("."))
        self.assertIsNone(service.get_run_embedding_projection("missing-run", "pca"))

    def test_embedding_projection_rejects_invalid_method(self):
        service = ComparatorService(Path("."))
        with self.assertRaises(ValueError):
            service.get_run_embedding_projection("missing-run", "mds")

    def test_evolmd_bertscore_guard_assigns_zero_to_empty_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_dir = Path(temp_dir) / "metrics"
            metrics_dir.mkdir()
            (metrics_dir / "__init__.py").write_text("", encoding="utf-8")
            (metrics_dir / "bert.py").write_text(
                "\n".join(
                    [
                        "def bertscore_individuos(individuos, ref_text, model_type, lang='en'):",
                        "    if any(not item.get('generated_data') for item in individuos):",
                        "        raise RuntimeError('empty candidate reached BERTScore')",
                        "    for item in individuos:",
                        "        item['fitness'] = 0.7",
                        "    return individuos",
                    ]
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, temp_dir)
            previous_metrics = sys.modules.pop("metrics", None)
            previous_bert = sys.modules.pop("metrics.bert", None)
            try:
                install_evolmd_bertscore_guard()
                import metrics.bert as bert_module

                rows = [{"generated_data": ""}, {"generated_data": "valid text"}]
                result = bert_module.bertscore_individuos(
                    rows,
                    "reference text",
                    "bert-base-uncased",
                )
                self.assertIs(result, rows)
                self.assertEqual(rows[0]["fitness"], 0.0)
                self.assertEqual(rows[0]["fitness_status"], "empty_generated_data")
                self.assertEqual(rows[1]["fitness"], 0.7)
                self.assertEqual(rows[1]["fitness_status"], "ok")
            finally:
                sys.path.remove(temp_dir)
                sys.modules.pop("metrics", None)
                sys.modules.pop("metrics.bert", None)
                if previous_metrics is not None:
                    sys.modules["metrics"] = previous_metrics
                if previous_bert is not None:
                    sys.modules["metrics.bert"] = previous_bert

    def test_non_dominated_excludes_invalid_rows(self):
        rows = [
            {"status": "empty_generated_data", "objectiveVector": [1.0, 1.0]},
            {"status": "ok", "objectiveVector": [0.8, 0.8]},
            {"status": "ok", "objectiveVector": [0.6, 0.6]},
        ]
        mark_non_dominated(rows)
        self.assertFalse(rows[0]["nonDominated"])
        self.assertTrue(rows[1]["nonDominated"])
        self.assertFalse(rows[2]["nonDominated"])

    def test_evolmd_posthoc_diagnostics_excludes_invalid_rows(self):
        service = ComparatorService(Path("."))
        rows = [
            {
                "status": "ok",
                "objectiveVector": [0.8],
                "generatedText": "valid text",
            },
            {
                "status": "empty_generated_data",
                "objectiveVector": [0.0],
                "generatedText": "",
            },
        ]

        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": 0.4, "semanticDiversity": 1.2}],
        ) as mocked:
            service._attach_evolmd_posthoc_diagnostics(rows, "reference text")

        mocked.assert_called_once_with(["valid text"], "reference text")
        self.assertEqual(rows[0]["objectiveVector"], [0.8])
        self.assertEqual(rows[0]["diagnosticObjectiveVector"], [0.4, 1.2])
        self.assertEqual(rows[0]["comparableObjectiveVector"], [0.7, 0.6])
        self.assertTrue(rows[0]["postHocNonDominated"])
        self.assertNotIn("diagnosticObjectiveVector", rows[1])
        self.assertFalse(rows[1]["postHocNonDominated"])

    def test_mo_metrics_exclude_invalid_rows_with_high_objectives(self):
        service = ComparatorService(Path("."))
        proposal = PROPOSALS[1]
        with patch.object(
            comparator_module,
            "calculate_posthoc_semantic_scores",
            return_value=[{"semanticFidelity": 0.5, "semanticDiversity": 0.5}],
        ):
            rows = service._normalize_rows(
                proposal,
                [
                    {"generated_data": "", "objetivos": [1.0, 1.0]},
                    {"generated_data": "valid text", "objetivos": [0.5, 0.5]},
                ],
                top_k=10,
                reference_text="reference",
            )
        invalid = next(row for row in rows if row["status"] != "ok")
        valid = next(row for row in rows if row["status"] == "ok")
        self.assertFalse(invalid["nonDominated"])
        self.assertTrue(valid["nonDominated"])

        metrics = service._summarize_rows(proposal, rows, Path("out"))
        self.assertEqual(metrics["completedRows"], 1)
        self.assertAlmostEqual(metrics["hypervolume"], 0.1875)

    def test_binary_costs_distinguish_unreported_tokens_from_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "llm_calls.jsonl").write_text(
                '{"elapsed_seconds": 1.5, "status": "ok"}\n',
                encoding="utf-8",
            )

            cost = comparator_module.build_binary_cost_metrics(
                {"processWallClockSeconds": 2.0, "returnCode": 0},
                output_dir,
                False,
            )

        self.assertEqual(cost["totalTokens"], 0)
        self.assertFalse(cost["hasTokenReport"])
        self.assertFalse(cost["hasOllamaDurationReport"])

    def test_binary_costs_sum_reported_ollama_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "llm_calls.jsonl").write_text(
                "\n".join(
                    [
                        '{"elapsed_seconds": 1, "promptEvalCount": 3, "evalCount": 4, "ollamaTotalDurationSeconds": 0.5}',
                        '{"elapsed_seconds": 2, "prompt_eval_count": 5, "eval_count": 6, "total_duration": 700000000}',
                    ]
                ),
                encoding="utf-8",
            )

            cost = comparator_module.build_binary_cost_metrics(
                {"processWallClockSeconds": 4.0, "returnCode": 0},
                output_dir,
                False,
            )

        self.assertEqual(cost["promptEvalCount"], 8)
        self.assertEqual(cost["evalCount"], 10)
        self.assertEqual(cost["totalTokens"], 18)
        self.assertAlmostEqual(cost["ollamaTotalDurationSeconds"], 1.2)
        self.assertTrue(cost["hasTokenReport"])
        self.assertTrue(cost["hasOllamaDurationReport"])

    def test_binary_costs_report_empty_content_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "llm_calls.jsonl").write_text(
                "\n".join(
                    [
                        '{"semantic_task": "semantic_pool_generation", "elapsed_seconds": 1, "content_chars": 0, "thinking_chars": 14416, "empty_content": true}',
                        '{"semantic_task": "semantic_anchor_extraction", "elapsed_seconds": 2, "content_chars": 25}',
                    ]
                ),
                encoding="utf-8",
            )

            cost = comparator_module.build_binary_cost_metrics(
                {"processWallClockSeconds": 4.0, "returnCode": 1},
                output_dir,
                False,
            )

        self.assertEqual(cost["llmCalls"], 2)
        self.assertEqual(cost["llmEmptyContentCalls"], 1)
        self.assertEqual(cost["llmTaskBreakdown"]["semantic_pool_generation"]["emptyContentCalls"], 1)
        self.assertEqual(cost["llmTaskBreakdown"]["semantic_anchor_extraction"]["emptyContentCalls"], 0)

    def test_process_failure_message_uses_python_exception_log_detail(self):
        service = ComparatorService(Path("."))
        run = {
            "config": {"timeoutMinutes": 60},
            "logs": [
                {"proposalId": "binary-mopso-cd", "message": "2026-06-11 01:19:42 | ERROR | run 1/1 failed"},
                {"proposalId": "binary-mopso-cd", "message": "Traceback (most recent call last):"},
                {
                    "proposalId": "binary-mopso-cd",
                    "message": "RuntimeError: Initial semantic pools are insufficient: product=0, required=180",
                },
            ],
        }

        message = service._process_failure_message(run, "binary-mopso-cd", 1, False)

        self.assertEqual(
            message,
            "Process exited with code 1. RuntimeError: Initial semantic pools are insufficient: product=0, required=180",
        )

    def test_process_failure_message_uses_read_timeout_exception_log_detail(self):
        service = ComparatorService(Path("."))
        run = {
            "config": {"timeoutMinutes": 60},
            "logs": [
                {"proposalId": "binary-mopso-cd", "message": "Traceback (most recent call last):"},
                {
                    "proposalId": "binary-mopso-cd",
                    "message": (
                        'File "C:\\Users\\Admin\\Desktop\\Implementacion\\Binary MOPSO-CD\\.venv\\Lib\\'
                        'site-packages\\httpx\\_transports\\default.py", line 118, in map_httpcore_exceptions'
                    ),
                },
                {"proposalId": "binary-mopso-cd", "message": "httpx.ReadTimeout: timed out"},
            ],
        }

        message = service._process_failure_message(run, "binary-mopso-cd", 1, False)

        self.assertEqual(message, "Process exited with code 1. httpx.ReadTimeout: timed out")

    def test_process_failure_message_names_negative_signal_exit(self):
        service = ComparatorService(Path("."))
        run = {
            "config": {"timeoutMinutes": 60},
            "logs": [],
        }

        message = service._process_failure_message(run, "binary-mopso-cd", -4, False)

        self.assertIn("Process terminated by signal 4 (SIGILL).", message)
        self.assertIn("native Python dependency", message)
        self.assertIn("scripts/diagnose_native_runtime.py", message)

    def test_binary_failed_costs_can_read_existing_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_base = Path(temp_dir) / "exec"
            output_dir = output_base / "2026-06-11_01-17-48"
            output_dir.mkdir(parents=True)
            (output_dir / "llm_calls.jsonl").write_text(
                '{"elapsed_seconds": 1.5, "promptEvalCount": 2, "evalCount": 3, "total_duration": 4000000000}\n',
                encoding="utf-8",
            )

            cost = comparator_module.build_binary_cost_metrics(
                {"processWallClockSeconds": 2.0, "returnCode": 1},
                output_dir,
                False,
            )

        self.assertEqual(cost["returnCode"], 1)
        self.assertEqual(cost["llmCalls"], 1)
        self.assertEqual(cost["totalTokens"], 5)
        self.assertAlmostEqual(cost["ollamaTotalDurationSeconds"], 4.0)

    def test_turbulence_aggregates_rates_and_final_deltas(self):
        config = {
            "strategies": ["llm"],
            "individualCount": 2,
            "seed": 7,
            "repetitionsK": 2,
            "finalFidelityMin": 0.0,
            "finalFidelityMax": 1.0,
            "kCandidates": 5,
            "operatorParallelism": 1,
            "generationParallelism": 1,
            "turbulenceMinSimilarity": 0.65,
            "turbulenceMaxSimilarity": 0.9,
            "embeddingModel": "fake",
            "distilbertModel": "fake",
        }
        repetitions = [
            {
                "movementCount": 2,
                "configSummary": {"repetitionIndex": 1, "seed": 7},
                "baseline": {"completedRows": 2, "averageFidelity": 0.5, "averageDiversity": 0.3, "rows": []},
                "strategies": [
                    {
                        "strategyId": "llm",
                        "operatorMetrics": {
                            "moves": 2,
                            "successes": 1,
                            "coverageCount": 2,
                            "similarityCount": 1,
                            "similaritySum": 0.7,
                            "operatorCumulativeSeconds": 1.0,
                            "operatorWallClockSeconds": 1.0,
                            "operatorParallelism": 1,
                            "cost": {"llmCalls": 2},
                        },
                        "finalMetrics": {"completedRows": 2, "averageFidelity": 0.6, "averageDiversity": 0.4, "rows": []},
                        "generationCost": {},
                        "finalEmbeddingCost": {},
                        "movementRows": [{"success": True}],
                        "finalRows": [],
                    }
                ],
            },
            {
                "movementCount": 2,
                "configSummary": {"repetitionIndex": 2, "seed": 8},
                "baseline": {"completedRows": 2, "averageFidelity": 0.7, "averageDiversity": 0.5, "rows": []},
                "strategies": [
                    {
                        "strategyId": "llm",
                        "operatorMetrics": {
                            "moves": 2,
                            "successes": 2,
                            "coverageCount": 2,
                            "similarityCount": 2,
                            "similaritySum": 1.5,
                            "operatorCumulativeSeconds": 1.0,
                            "operatorWallClockSeconds": 1.0,
                            "operatorParallelism": 1,
                            "cost": {"llmCalls": 2},
                        },
                        "finalMetrics": {"completedRows": 2, "averageFidelity": 0.8, "averageDiversity": 0.6, "rows": []},
                        "generationCost": {},
                        "finalEmbeddingCost": {},
                        "movementRows": [{"success": True}],
                        "finalRows": [],
                    }
                ],
            },
        ]
        aggregated = aggregate_turbulence_repetitions(repetitions, config)
        strategy = aggregated["strategies"][0]
        self.assertEqual(aggregated["completedRepetitions"], 2)
        self.assertAlmostEqual(aggregated["baseline"]["averageFidelity"], 0.6)
        self.assertAlmostEqual(strategy["operatorMetrics"]["successRate"], 0.75)
        self.assertAlmostEqual(strategy["operatorMetrics"]["averageSimilarity"], 2.2 / 3)
        self.assertAlmostEqual(strategy["finalMetrics"]["averageFidelity"], 0.7)
        self.assertAlmostEqual(strategy["deltas"]["averageFidelityDelta"], 0.1)
        self.assertEqual(strategy["repetitions"][0]["repetitionIndex"], 1)
        self.assertEqual(strategy["repetitions"][1]["repetitionSeed"], 8)


if __name__ == "__main__":
    unittest.main()
