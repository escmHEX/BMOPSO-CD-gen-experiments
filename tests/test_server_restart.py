from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import server


class ToolPortalRestartTests(unittest.TestCase):
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

        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(root / "server.py"))
        self.assertIn("--host", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--port", command)
        self.assertIn("4173", command)
        self.assertIn("--lm-studio", command)
        self.assertIn("http://127.0.0.1:1234", command)

    def test_tool_portal_html_marker_is_accepted(self):
        self.assertTrue(server.is_tool_portal_html("<title>Portal de herramientas LLM</title><strong>Tesis LLM</strong>"))
        self.assertFalse(server.is_tool_portal_html("<title>Other app</title>"))

    def test_schedule_restart_invokes_launcher_without_real_process(self):
        calls = []

        def fake_launcher(command, **kwargs):
            calls.append((command, kwargs))

        thread = server.schedule_portal_restart(
            command=["python", "server.py"],
            cwd=Path("C:/portal"),
            delay_seconds=0,
            launcher=fake_launcher,
        )
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(calls[0][0], ["python", "server.py"])
        self.assertEqual(calls[0][1]["cwd"], "C:\\portal")

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


if __name__ == "__main__":
    unittest.main()
