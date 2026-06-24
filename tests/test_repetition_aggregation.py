from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
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
    def test_proposal_python_executable_detects_venv_for_any_proposal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "baseline"
            venv_python = repository / "venv" / "Scripts" / "python.exe"
            venv_python.parent.mkdir(parents=True)
            venv_python.touch()

            proposal = replace(PROPOSALS[0], repository_path="baseline", python_executable="")

            self.assertEqual(
                comparator_module.proposal_python_executable(root, repository, proposal),
                str(venv_python),
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
        self.assertIn("router.task_models.synthetic_text_generation", options_by_key)
        self.assertIn("selection.lambda_mmr", options_by_key)
        self.assertEqual(options_by_key["mopso.archive_multiplier"]["flag"], "--set")
        self.assertEqual(options_by_key["mopso.archive_multiplier"]["type"], "float")
        self.assertEqual(options_by_key["mopso.alpha"]["type"], "float")
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
                        }
                    }
                },
            }
        )
        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertEqual(values["mopso.alpha"], "1.25")
        self.assertEqual(values["mopso.archive_multiplier"], "0.5")

    def test_binary_task_thinking_rejects_unvalidated_model_task_before_run(self):
        payload = {
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

        service = ComparatorService(Path("."))
        with self.assertRaisesRegex(ValueError, "qwen3\\.5:9b.*semantic_pool_generation.*not validated"):
            service._read_config(payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_service = ComparatorService(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "semantic_pool_generation"):
                temp_service.start_run(payload)
            self.assertFalse(temp_service.runs_root.exists())

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

    def test_binary_task_thinking_accepts_legacy_true_but_validates_task(self):
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
        self.assertEqual(set_values["router.task_models.semantic_anchor_extraction"], '"llama3"')
        self.assertEqual(set_values["router.task_models.semantic_pool_generation"], '"lfm2.5:8b"')
        self.assertEqual(set_values["router.task_thinking.semantic_pool_generation"], '"low"')

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
        first, second = service._selected_instances(config)

        first_command = service._build_command({"config": config}, first, Path("."), Path("out-a"), Path("reference.txt"), 777)
        second_command = service._build_command({"config": config}, second, Path("."), Path("out-b"), Path("reference.txt"), 778)

        self.assertEqual(command_set_values(first_command)["selection.k"], "4")
        self.assertEqual(command_set_values(second_command)["selection.k"], "5")
        self.assertEqual(command_set_values(first_command)["experiment.seed"], "777")
        self.assertEqual(command_set_values(second_command)["experiment.seed"], "778")

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
        self.assertAlmostEqual(aggregated["metrics"]["externalArchiveUpdateCount"], 6.0)
        self.assertAlmostEqual(aggregated["metrics"]["externalArchivePruneCount"], 2.0)
        self.assertEqual(aggregated["metrics"]["externalArchiveUpdateCountTotal"], 12)
        self.assertEqual(aggregated["metrics"]["externalArchivePruneCountTotal"], 4)
        self.assertAlmostEqual(aggregated["cost"]["llmCalls"], 2.5)
        self.assertEqual(aggregated["cost"]["llmCallsTotal"], 5)
        self.assertAlmostEqual(aggregated["cost"]["llmEmptyContentCalls"], 1.0)
        self.assertEqual(aggregated["cost"]["llmEmptyContentCallsTotal"], 2)
        self.assertAlmostEqual(aggregated["cost"]["llmClientWallClockSeconds"], 1.5)
        self.assertAlmostEqual(aggregated["cost"]["llmClientWallClockSecondsTotal"], 3.0)
        self.assertAlmostEqual(aggregated["cost"]["llmAverageCallSeconds"], 0.6)
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

        self.assertEqual([point["generation"] for point in series], [0, 1, 2])
        self.assertEqual(series[0]["source"], "final_only")
        self.assertIsNotNone(series[0]["hypervolume"])
        self.assertIsNotNone(series[0]["extent"])
        self.assertIsNotNone(series[0]["unaryEntropy"])
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

        self.assertEqual([point["generation"] for point in series], [0, 1, 2])
        self.assertEqual(series[0]["source"], "final_only")
        self.assertIsNotNone(series[0]["hypervolume"])
        self.assertIsNotNone(series[0]["extent"])
        self.assertIsNotNone(series[0]["unaryEntropy"])
        self.assertEqual(series[0]["frontPoints"], [[0.5, 0.5], [0.7, 0.3]])
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
