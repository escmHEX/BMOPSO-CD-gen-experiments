from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from baselines import comparator as comparator_module
from baselines.bootstrap import install_evolmd_bertscore_guard
from baselines.comparator import ComparatorService, PROPOSALS, aggregate_proposal_repetitions
from baselines.comparator import aggregate_series
from baselines.comparator import mark_non_dominated
from baselines.comparator_metrics import build_charts_from_rows
from initial_population.comparison import InitialPopulationComparisonService
from turbulence_comparison.service import aggregate_turbulence_repetitions


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

    def test_comparator_config_accepts_single_selected_proposal(self):
        service = ComparatorService(Path("."))
        parsed = service._read_config(
            {
                "referenceText": "reference",
                "selectedProposalIds": ["binary-mopso-cd"],
                "proposalConfigs": {"binary-mopso-cd": {"extraArgs": "--freeze-components role"}},
            }
        )
        self.assertEqual(parsed["selectedProposalIds"], ["binary-mopso-cd"])
        self.assertEqual(parsed["proposalConfigs"]["binary-mopso-cd"]["extraArgs"], "--freeze-components role")

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
            "2026-06-03 00:05:20 | INFO | run 1/1 | generation 29/30 | modified=5/20 | archive=10 | hv=0.301896 | spread=0.258775 | elapsed=00:08:44",
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
            "2026-06-03 00:05:20 | INFO | run 1/1 | generation 1/3 | modified=2/20 | archive=4 | hv=0.1 | spread=0.2 | elapsed=00:00:16",
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
        end_log = "2026-06-03 00:05:20 | INFO | run 1/1 | generation 1/3 | modified=2/20 | archive=4 | hv=0.1 | spread=0.2 | elapsed=00:00:16"
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
                            "--freeze-components": ["role", "topic"],
                            "--enable-monitor": True,
                            "--router-heuristic": {"word_replacement_candidates": False},
                            "--task-model": {"synthetic_text_generation": "llama3.1:8b"},
                        }
                    }
                },
            }
        )
        values = parsed["proposalConfigs"]["binary-mopso-cd"]["cliValues"]
        self.assertEqual(values["--freeze-components"], ["role", "topic"])
        self.assertTrue(values["--enable-monitor"])
        self.assertFalse(values["--router-heuristic"]["word_replacement_candidates"])
        self.assertEqual(values["--task-model"]["synthetic_text_generation"], "llama3.1:8b")

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

    def test_comparator_rejects_structured_managed_or_unknown_cli_values(self):
        service = ComparatorService(Path("."))
        for flag in ("--seed", "--missing"):
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
        self.assertIn("--seed", command)
        self.assertEqual(command[command.index("--seed") + 1], "777")

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
                            "--bert-model": "gte-small",
                            "--freeze-components": ["role", "topic"],
                            "--enable-monitor": True,
                            "--router-heuristic": {"word_replacement_candidates": False},
                            "--task-model": {"synthetic_text_generation": "llama3.1:8b"},
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
        self.assertEqual(command[command.index("--bert-model") + 1], "gte-small")
        self.assertEqual(command.count("--bert-model"), 1)
        self.assertEqual(command[command.index("--freeze-components") + 1], "role,topic")
        self.assertIn("--enable-monitor", command)
        self.assertIn("word_replacement_candidates=false", command)
        self.assertIn("synthetic_text_generation=llama3.1:8b", command)

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
                    "spread": 0.4,
                    "postHocDiagnostic": True,
                    "bestDiagnosticObjectiveVector": [0.6, 0.5],
                    "postHocNonDominatedRows": 2,
                },
                "cost": {"llmCalls": 2, "llmClientWallClockSeconds": 1.0, "totalTokens": 20},
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
                    "spread": 0.2,
                    "postHocDiagnostic": True,
                    "bestDiagnosticObjectiveVector": [0.8, 0.7],
                    "postHocNonDominatedRows": 4,
                },
                "cost": {"llmCalls": 3, "llmClientWallClockSeconds": 2.0, "totalTokens": 30},
            },
        ]
        aggregated = aggregate_proposal_repetitions(proposal, Path("out"), results, 2)
        self.assertEqual(aggregated["completedRepetitions"], 2)
        self.assertAlmostEqual(aggregated["metrics"]["completedRows"], 9.0)
        self.assertEqual(aggregated["metrics"]["bestObjectiveLabel"], "[0.700000]")
        self.assertEqual(aggregated["metrics"]["nonDominatedRows"], 3.0)
        self.assertEqual(aggregated["metrics"]["postHocNonDominatedRows"], 3.0)
        self.assertEqual(aggregated["cost"]["llmCalls"], 5)
        self.assertEqual(aggregated["rows"][0]["repetitionSeed"], 10)

    def test_aggregate_series_averages_by_generation(self):
        series = aggregate_series([
            {"series": [{"generation": 1, "hypervolume": 0.2, "nonDominatedRows": 2, "spread": 0.5}]},
            {"series": [{"generation": 1, "hypervolume": 0.4, "nonDominatedRows": 4, "spread": 0.3}]},
        ])
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(series[0]["hypervolume"], 0.3)
        self.assertAlmostEqual(series[0]["nonDominatedRows"], 3.0)
        self.assertAlmostEqual(series[0]["spread"], 0.4)

    def test_binary_rows_normalize_from_native_outputs(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
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
        )
        self.assertEqual(rows[0]["proposalId"], "binary-mopso-cd")
        self.assertEqual(rows[0]["objectiveVector"], [0.7, 0.4])
        self.assertTrue(rows[0]["nonDominated"])

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

    def test_charts_use_posthoc_non_dominated_rows_for_evolmd(self):
        rows = [
            {
                "proposalId": "evolmd",
                "displayName": "EVOLMD",
                "status": "ok",
                "objectiveVector": [0.6],
                "diagnosticObjectiveVector": [0.6, 0.4],
                "generatedText": "single objective front",
                "postHocNonDominated": True,
                "nonDominated": False,
            },
            {
                "proposalId": "evolmd",
                "displayName": "EVOLMD",
                "status": "ok",
                "objectiveVector": [0.5],
                "diagnosticObjectiveVector": [0.5, 0.2],
                "generatedText": "dominated diagnostic row",
                "postHocNonDominated": False,
                "nonDominated": True,
            },
        ]
        charts = build_charts_from_rows(rows, [], [])
        self.assertEqual(len(charts["nonDominated"]), 1)
        self.assertEqual(charts["nonDominated"][0]["label"], "single objective front")
        self.assertEqual(charts["nonDominated"][0]["proposalId"], "evolmd")

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

    def test_binary_summary_uses_native_diversity_scale_for_hypervolume(self):
        service = ComparatorService(Path("."))
        proposal = next(item for item in PROPOSALS if item.proposal_id == "binary-mopso-cd")
        rows = service._normalize_rows(
            proposal,
            [
                {
                    "solution_id": "s1",
                    "generated_text": "Generated",
                    "objectives": {"f1": 0.0, "f2": 1.5},
                }
            ],
            top_k=5,
        )
        metrics = service._summarize_rows(proposal, rows, Path("out"))
        self.assertAlmostEqual(metrics["hypervolume"], 0.375)

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

        with patch.object(comparator_module, "calculate_posthoc_semantic_diversity", return_value=[0.25]) as mocked:
            service._attach_evolmd_posthoc_diagnostics(rows)

        mocked.assert_called_once_with(["valid text"])
        self.assertEqual(rows[0]["diagnosticObjectiveVector"], [0.8, 0.25])
        self.assertTrue(rows[0]["postHocNonDominated"])
        self.assertNotIn("diagnosticObjectiveVector", rows[1])
        self.assertFalse(rows[1]["postHocNonDominated"])

    def test_mo_metrics_exclude_invalid_rows_with_high_objectives(self):
        service = ComparatorService(Path("."))
        proposal = PROPOSALS[1]
        rows = service._normalize_rows(
            proposal,
            [
                {"generated_data": "", "objetivos": [1.0, 1.0]},
                {"generated_data": "valid text", "objetivos": [0.5, 0.5]},
            ],
            top_k=10,
        )
        invalid = next(row for row in rows if row["status"] != "ok")
        valid = next(row for row in rows if row["status"] == "ok")
        self.assertFalse(invalid["nonDominated"])
        self.assertTrue(valid["nonDominated"])

        metrics = service._summarize_rows(proposal, rows, Path("out"))
        self.assertEqual(metrics["completedRows"], 1)
        self.assertAlmostEqual(metrics["hypervolume"], 0.375)

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
