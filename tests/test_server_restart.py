from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import server
from baselines import comparator as comparator_module


class FakeTcpServer:
    allow_reuse_address = False

    def __init__(self, address, handler):
        self.address = address
        self.handler = handler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def serve_forever(self):
        raise RuntimeError("stop test server")


class ToolPortalRestartTests(unittest.TestCase):
    def test_server_defaults_target_ollama_openai_compatible_endpoint(self):
        self.assertEqual(server.DEFAULT_PORT, 4173)
        self.assertEqual(server.DEFAULT_LM_STUDIO, "http://127.0.0.1:11434")

    def test_active_work_summary_detects_running_runs(self):
        services = {
            "comparator": SimpleNamespace(
                _runs={
                    "active-run": {"status": "running"},
                    "done-run": {"status": "completed"},
                }
            ),
            "initialPopulation": SimpleNamespace(_runs={}),
        }

        active = server.portal_active_work_summary(services)

        self.assertEqual(
            active,
            [
                {
                    "service": "comparator",
                    "runId": "active-run",
                    "status": "running",
                }
            ],
        )

    def test_active_work_summary_ignores_terminal_cancel_requested_runs(self):
        services = {
            "initialPopulation": SimpleNamespace(
                _runs={
                    "cancelling-run": {
                        "status": "completed",
                        "cancelRequested": True,
                    }
                }
            )
        }

        active = server.portal_active_work_summary(services)

        self.assertEqual(active, [])

    def test_restart_command_preserves_server_arguments(self):
        root = Path("C:/portal")

        command = server.build_portal_restart_command(
            root=root,
            host="127.0.0.1",
            port=4173,
            lm_studio_base="http://127.0.0.1:1234",
        )

        self.assertEqual(command[0], server.windowless_python_executable(server.sys.executable))
        self.assertEqual(command[1], str(root / "server.py"))
        self.assertIn("--host", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--port", command)
        self.assertIn("4173", command)
        self.assertIn("--lm-studio", command)
        self.assertIn("http://127.0.0.1:1234", command)
        self.assertIn(server.PORTAL_SKIP_PORT_RELEASE_FLAG, command)

    def test_restart_command_prefers_windowless_server_on_windows(self):
        with patch.object(server.sys, "platform", "win32"), patch.object(
            server.sys,
            "executable",
            r"C:\portal\.venv\Scripts\python.exe",
        ), patch.object(server.Path, "exists", return_value=True):
            command = server.build_portal_restart_command(
                root=Path("C:/portal"),
                host="127.0.0.1",
                port=4173,
                lm_studio_base="http://127.0.0.1:1234",
            )

        self.assertEqual(command[0], r"C:\portal\.venv\Scripts\pythonw.exe")

    def test_restart_command_prefers_project_venv_when_current_python_is_global(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripts_dir = root / ".venv" / "Scripts"
            scripts_dir.mkdir(parents=True)
            (scripts_dir / "python.exe").write_text("", encoding="utf-8")
            (scripts_dir / "pythonw.exe").write_text("", encoding="utf-8")

            with patch.object(server.sys, "platform", "win32"), patch.object(
                server.sys,
                "executable",
                r"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe",
            ):
                command = server.build_portal_restart_command(
                    root=root,
                    host="127.0.0.1",
                    port=4173,
                    lm_studio_base="http://127.0.0.1:1234",
                )

        self.assertEqual(command[0], str(scripts_dir / "pythonw.exe"))

    def test_comparator_embedding_projection_get_returns_json_500_on_unexpected_error(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            path="/api/comparator/runs/run-1/embedding-projection?method=pca",
            comparator_path_parts=lambda: ["runs", "run-1", "embedding-projection"],
            comparator_service=SimpleNamespace(
                get_run_embedding_projection=Mock(side_effect=RuntimeError("No module named numpy"))
            ),
            send_json=lambda status, payload: sent.append((status, payload)),
        )

        server.ToolPortalHandler.handle_comparator_get(fake_handler)

        self.assertEqual(sent, [(500, {"error": "No module named numpy"})])

    def test_comparator_charting_get_and_post_use_effective_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "baselines" / "comparator_config.local.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps(
                    {
                        "defaults": {"model": "llama3"},
                        "proposals": {},
                        "charting": {
                            "library": "Apache ECharts",
                            "labels": {
                                "hypervolume": {
                                    "title": "Hipervolumen guardado",
                                    "xAxis": "Generacion",
                                    "yAxis": "Hipervolumen",
                                }
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            service = comparator_module.ComparatorService(root, config_path=config_path)
            sent: list[tuple[int, dict]] = []

            get_handler = SimpleNamespace(
                path="/api/comparator/charting",
                comparator_path_parts=lambda: ["charting"],
                comparator_service=service,
                send_json=lambda status, payload: sent.append((status, payload)),
            )
            server.ToolPortalHandler.handle_comparator_get(get_handler)

            self.assertEqual(sent[0][0], 200)
            self.assertEqual(sent[0][1]["charting"]["labels"]["hypervolume"]["title"], "Hipervolumen guardado")

            post_handler = SimpleNamespace(
                path="/api/comparator/charting",
                comparator_path_parts=lambda: ["charting"],
                comparator_service=service,
                read_json_body=lambda: {
                    "labels": {
                        "pareto": {
                            "title": "Frente tesis",
                            "xAxis": "Fidelidad normalizada",
                            "yAxis": "Diversidad normalizada",
                        }
                    }
                },
                send_json=lambda status, payload: sent.append((status, payload)),
            )
            server.ToolPortalHandler.handle_comparator_post(post_handler)

            self.assertEqual(sent[1][0], 200)
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["charting"]["labels"]["pareto"]["title"], "Frente tesis")
            self.assertEqual(saved["charting"]["library"], "Apache ECharts")

    def test_comparator_charting_post_rejects_blank_label_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "baselines" / "comparator_config.local.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text('{"charting": {}}', encoding="utf-8")
            sent: list[tuple[int, dict]] = []
            fake_handler = SimpleNamespace(
                path="/api/comparator/charting",
                comparator_path_parts=lambda: ["charting"],
                comparator_service=comparator_module.ComparatorService(root, config_path=config_path),
                read_json_body=lambda: {"labels": {"hypervolume": {"title": ""}}},
                send_json=lambda status, payload: sent.append((status, payload)),
            )

            server.ToolPortalHandler.handle_comparator_post(fake_handler)

            self.assertEqual(sent[0][0], 400)
            self.assertIn("title", sent[0][1]["error"])

    def test_comparator_run_download_get_returns_zip_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "runs" / "comparator" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "summary.json").write_text("{}", encoding="utf-8")

            responses: list[int] = []
            headers: list[tuple[str, str]] = []
            body = bytearray()
            fake_handler = SimpleNamespace(
                path="/api/comparator/runs/run-1/download",
                comparator_path_parts=lambda: ["runs", "run-1", "download"],
                comparator_service=comparator_module.ComparatorService(root),
                send_response=lambda status: responses.append(status),
                send_header=lambda key, value: headers.append((key, value)),
                end_headers=lambda: None,
                wfile=SimpleNamespace(write=lambda chunk: body.extend(chunk)),
            )
            fake_handler.send_binary_file = server.ToolPortalHandler.send_binary_file.__get__(fake_handler)
            fake_handler.send_comparator_run_download = server.ToolPortalHandler.send_comparator_run_download.__get__(
                fake_handler
            )

            server.ToolPortalHandler.handle_comparator_get(fake_handler)

            self.assertEqual(responses, [200])
            self.assertIn(("Content-Type", "application/zip"), headers)
            self.assertIn(
                ("Content-Disposition", 'attachment; filename="comparator-run-run-1.zip"'),
                headers,
            )
            zip_path = Path(temp_dir) / "response.zip"
            zip_path.write_bytes(body)
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["run-1/summary.json"])

    def test_comparator_run_download_get_returns_404_when_run_is_missing(self):
        sent: list[tuple[int, dict]] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_handler = SimpleNamespace(
                path="/api/comparator/runs/missing/download",
                comparator_path_parts=lambda: ["runs", "missing", "download"],
                comparator_service=comparator_module.ComparatorService(Path(temp_dir)),
                send_json=lambda status, payload: sent.append((status, payload)),
            )
            fake_handler.send_comparator_run_download = server.ToolPortalHandler.send_comparator_run_download.__get__(
                fake_handler
            )

            server.ToolPortalHandler.handle_comparator_get(fake_handler)

        self.assertEqual(sent, [(404, {"error": "Run not found."})])

    def test_restart_helper_prefers_windowless_python_on_windows(self):
        with patch.object(server.sys, "platform", "win32"), patch.object(
            server.sys,
            "executable",
            r"C:\portal\.venv\Scripts\python.exe",
        ), patch.object(server.Path, "exists", return_value=True):
            helper_command = server.build_portal_restart_helper_command(
                command=[r"C:\portal\.venv\Scripts\python.exe", "server.py"],
                cwd=Path("C:/portal"),
                parent_pid=1234,
                delay_seconds=0,
            )

        self.assertEqual(helper_command[0], r"C:\portal\.venv\Scripts\pythonw.exe")
        self.assertEqual(json.loads(helper_command[4]), [r"C:\portal\.venv\Scripts\python.exe", "server.py"])
        self.assertEqual(helper_command[-2:], ["127.0.0.1", "4173"])

    def test_current_systemd_user_service_matches_current_main_pid(self):
        completed = Mock(returncode=0, stdout="ActiveState=active\nMainPID=4321\n", stderr="")

        with patch.object(server.sys, "platform", "linux"), patch.object(
            server.shutil,
            "which",
            return_value="/usr/bin/systemctl",
        ), patch.dict(
            server.os.environ,
            {server.PORTAL_SYSTEMD_SERVICE_ENV: "custom-portal"},
            clear=False,
        ):
            service_name = server.current_systemd_user_service(current_pid=4321, runner=Mock(return_value=completed))

        self.assertEqual(service_name, "custom-portal.service")

    def test_current_systemd_user_service_ignores_other_main_pid(self):
        completed = Mock(returncode=0, stdout="ActiveState=active\nMainPID=9999\n", stderr="")

        with patch.object(server.sys, "platform", "linux"), patch.object(
            server.shutil,
            "which",
            return_value="/usr/bin/systemctl",
        ):
            service_name = server.current_systemd_user_service(current_pid=4321, runner=Mock(return_value=completed))

        self.assertIsNone(service_name)

    def test_systemd_restart_helper_uses_no_block_user_restart(self):
        command = server.build_portal_systemd_restart_helper_command("bmopso-cd-experiments.service", 0.2)

        self.assertEqual(command[1], "-c")
        self.assertIn("--no-block", command[2])
        self.assertEqual(command[-2:], ["0.2", "bmopso-cd-experiments.service"])

    def test_hidden_subprocess_kwargs_avoid_windows_console_creation(self):
        with patch.object(server.sys, "platform", "win32"):
            kwargs = server.hidden_subprocess_kwargs()

        self.assertIn("creationflags", kwargs)
        self.assertTrue(kwargs["creationflags"] & getattr(server.subprocess, "CREATE_NO_WINDOW", 0))
        self.assertIn("startupinfo", kwargs)

    def test_tool_portal_html_marker_is_accepted(self):
        self.assertTrue(server.is_tool_portal_html("<title>Portal de herramientas LLM</title><strong>Tesis LLM</strong>"))
        self.assertFalse(server.is_tool_portal_html("<title>Other app</title>"))

    def test_schedule_restart_launches_helper_and_exits_current_process(self):
        calls = []
        exits = []

        def fake_launcher(command, **kwargs):
            calls.append((command, kwargs))

        thread = server.schedule_portal_restart(
            command=["python", "server.py"],
            cwd=Path("C:/portal"),
            delay_seconds=0,
            launcher=fake_launcher,
            terminator=lambda code: exits.append(code),
            current_pid=9876,
        )
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls), 1)
        helper_command = calls[0][0]
        self.assertIn("-c", helper_command)
        self.assertIn("9876", helper_command)
        self.assertIn('["python", "server.py"]', helper_command)
        self.assertIn("C:\\portal", helper_command)
        self.assertEqual(calls[0][1]["cwd"], "C:\\portal")
        self.assertEqual(exits, [0])

    def test_restart_endpoint_uses_systemd_when_current_process_is_user_service(self):
        sent: list[tuple[int, dict]] = []
        fake_handler = SimpleNamespace(
            portal_path_parts=lambda: ["restart"],
            read_request_body=lambda: b"{}",
            headers={server.PORTAL_RESTART_HEADER: "1"},
            portal_services=lambda: {},
            send_json=lambda status, payload: sent.append((status, payload)),
            portal_root=Path("C:/portal"),
            portal_host="0.0.0.0",
            portal_port=4173,
            lm_studio_base="http://127.0.0.1:11434",
            portal_started_at="2026-06-27T03:09:25+00:00",
            wfile=SimpleNamespace(flush=Mock()),
        )

        with patch.object(
            server,
            "current_systemd_user_service",
            return_value="bmopso-cd-experiments.service",
        ), patch.object(server, "schedule_portal_systemd_restart") as schedule_systemd_restart, patch.object(
            server,
            "schedule_portal_restart",
        ) as schedule_restart:
            server.ToolPortalHandler.handle_portal_post(fake_handler)

        self.assertEqual(sent[0][0], 202)
        self.assertEqual(sent[0][1]["restartMode"], "systemd-user")
        schedule_systemd_restart.assert_called_once_with("bmopso-cd-experiments.service", Path("C:/portal"))
        schedule_restart.assert_not_called()

    def test_main_skips_port_release_for_internal_restart(self):
        service = SimpleNamespace()
        argv = [
            "server.py",
            "--host",
            "127.0.0.1",
            "--port",
            "4173",
            "--lm-studio",
            "http://127.0.0.1:1234",
            server.PORTAL_SKIP_PORT_RELEASE_FLAG,
        ]
        with patch.object(server.sys, "argv", argv), patch.object(
            server.socketserver,
            "ThreadingTCPServer",
            FakeTcpServer,
        ), patch.object(server, "ComparatorService", return_value=service), patch.object(
            server,
            "InitialPopulationService",
            return_value=service,
        ), patch.object(
            server,
            "InitialPopulationComparisonService",
            return_value=service,
        ), patch.object(
            server,
            "ReferenceTextStore",
            return_value=service,
        ), patch.object(
            server,
            "TurbulenceComparisonService",
            return_value=service,
        ), patch.object(
            server,
            "SbertSimilarityService",
            return_value=service,
        ), patch.object(
            server,
            "release_existing_server_port",
        ) as release_port, patch(
            "builtins.print",
        ):
            with self.assertRaisesRegex(RuntimeError, "stop test server"):
                server.main()

        release_port.assert_not_called()

    def test_release_port_accepts_verified_portal_when_command_line_is_unavailable(self):
        with patch.object(server.sys, "platform", "win32"), patch.object(
            server,
            "find_windows_listener_pid",
            return_value=1234,
        ), patch.object(server, "windows_process_command_line", return_value=""), patch.object(
            server,
            "is_tool_portal_listener",
            return_value=True,
        ), patch.object(server, "windows_process_parent_id", return_value=None), patch.object(
            server,
            "wait_for_port_release",
        ) as wait_for_port_release, patch.object(server.subprocess, "run", return_value=Mock(returncode=0)) as run:
            server.release_existing_server_port("127.0.0.1", 4173)

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][:3], ["taskkill", "/PID", "1234"])
        wait_for_port_release.assert_called_once_with("127.0.0.1", 4173)

    def test_terminate_windows_process_tree_falls_back_when_taskkill_times_out(self):
        with patch.object(
            server.subprocess,
            "run",
            side_effect=server.subprocess.TimeoutExpired(["taskkill"], 15),
        ), patch.object(server, "stop_windows_process") as stop_windows_process:
            server.terminate_windows_process_tree(1234)

        stop_windows_process.assert_called_once_with(1234)

    def test_terminate_windows_process_tree_uses_direct_fallback_when_stop_process_times_out(self):
        with patch.object(
            server.subprocess,
            "run",
            side_effect=server.subprocess.TimeoutExpired(["taskkill"], 15),
        ), patch.object(
            server,
            "stop_windows_process",
            side_effect=server.subprocess.TimeoutExpired(["Stop-Process"], 10),
        ), patch.object(server, "terminate_windows_process_direct") as terminate_direct:
            server.terminate_windows_process_tree(1234)

        terminate_direct.assert_called_once_with(1234)

    def test_comparator_subprocess_kwargs_avoid_windows_console_creation(self):
        with patch.object(comparator_module.sys, "platform", "win32"):
            kwargs = comparator_module.hidden_subprocess_kwargs()

        self.assertIn("creationflags", kwargs)
        self.assertTrue(kwargs["creationflags"] & getattr(comparator_module.subprocess, "CREATE_NO_WINDOW", 0))
        self.assertIn("startupinfo", kwargs)

    def test_comparator_dependency_check_uses_hidden_subprocess_kwargs(self):
        proposal = comparator_module.ProposalDefinition(
            proposal_id="test",
            display_name="Test",
            repository_path=".",
            description="",
            objective_names=("fitness",),
            result_file="result.json",
            single_objective=True,
            required_modules=("json",),
        )
        completed = Mock(returncode=0, stdout='{"missing":[]}', stderr="")
        with patch.object(comparator_module.sys, "platform", "win32"), patch.object(
            comparator_module,
            "proposal_python_executable",
            return_value="python",
        ), patch.object(
            comparator_module.subprocess,
            "run",
            return_value=completed,
        ) as run:
            status = comparator_module.check_proposal_dependencies(Path("."), proposal)

        self.assertTrue(status["ok"])
        self.assertIn("creationflags", run.call_args.kwargs)
        self.assertTrue(run.call_args.kwargs["creationflags"] & getattr(comparator_module.subprocess, "CREATE_NO_WINDOW", 0))


if __name__ == "__main__":
    unittest.main()
