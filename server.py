from __future__ import annotations

import argparse
import ctypes
import http.server
import json
import mimetypes
import os
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from baselines.comparator import ComparatorService
from initial_population.comparison import InitialPopulationComparisonService
from initial_population.service import InitialPopulationService
from llm_studio import LmStudioClient, LmStudioHttpError
from reference_text_store import ReferenceTextStore
from sbert_service import SbertSimilarityService
from turbulence_comparison.service import TurbulenceComparisonService


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4173
DEFAULT_LM_STUDIO = "http://127.0.0.1:1234"
PROXY_PREFIX = "/lmstudio"
COMPARATOR_PREFIX = "/api/comparator"
MIMETYPES = {".mjs": "application/javascript"}

for extension, content_type in MIMETYPES.items():
    mimetypes.add_type(content_type, extension)
INITIAL_POPULATION_PREFIX = "/api/initial-population"
INITIAL_POPULATION_COMPARISON_PREFIX = "/api/initial-population-comparison"
REFERENCE_TEXTS_PREFIX = "/api/reference-texts"
TURBULENCE_COMPARISON_PREFIX = "/api/turbulence-comparison"
SBERT_PREFIX = "/api/sbert"
LM_STUDIO_API_PREFIX = "/api/lm-studio"
PORTAL_API_PREFIX = "/api/portal"
PORTAL_RESTART_HEADER = "X-Tool-Portal-Restart"
PORTAL_ACTIVE_STATUSES = {"queued", "running"}
PORTAL_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
PORTAL_STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")
PORTAL_RESTART_HELPER_TIMEOUT_SECONDS = 12
PORTAL_SKIP_PORT_RELEASE_FLAG = "--skip-port-release"


def release_existing_server_port(host: str, port: int) -> None:
    if sys.platform != "win32":
        return

    process_id = find_windows_listener_pid(host, port)
    if process_id is None or process_id == os.getpid():
        return

    command_line = windows_process_command_line(process_id)
    if "server.py" not in command_line and not is_tool_portal_listener(host, port):
        raise RuntimeError(
            f"Port {port} is already in use by a different process: PID {process_id}, {command_line}"
        )

    target_process_id = process_id
    parent_process_id = windows_process_parent_id(process_id)
    if parent_process_id and parent_process_id != os.getpid():
        parent_command_line = windows_process_command_line(parent_process_id).lower()
        if "server.py" in parent_command_line and ("py.exe" in parent_command_line or "\\py " in parent_command_line):
            target_process_id = parent_process_id

    terminate_windows_process_tree(target_process_id)
    wait_for_port_release(host, port)


def find_windows_listener_pid(host: str, port: int) -> int | None:
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        **hidden_subprocess_kwargs(),
    )
    candidate_hosts = {host, "0.0.0.0", "::", "::1", "127.0.0.1"}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING":
            continue
        local_address = parts[1]
        if not local_address.endswith(f":{port}"):
            continue
        address = local_address.rsplit(":", 1)[0].strip("[]")
        if address not in candidate_hosts:
            continue
        try:
            return int(parts[4])
        except ValueError:
            return None
    return None


def windows_process_command_line(process_id: int) -> str:
    script = f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {process_id}\").CommandLine"
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout.strip()


def windows_process_parent_id(process_id: int) -> int | None:
    script = f"(Get-CimInstance Win32_Process -Filter \"ProcessId = {process_id}\").ParentProcessId"
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def is_tool_portal_listener(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    static_url = f"http://{probe_host}:{port}/LLM/"
    try:
        with urllib.request.urlopen(static_url, timeout=3) as response:
            if response.status == 200 and is_tool_portal_html(response.read().decode("utf-8", errors="ignore")):
                return True
    except Exception:
        pass

    url = f"http://{probe_host}:{port}{COMPARATOR_PREFIX}/proposals"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("proposals"), list) and isinstance(payload.get("defaults"), dict)


def is_tool_portal_html(text: str) -> bool:
    return "Portal de herramientas LLM" in text and "Tesis LLM" in text


def terminate_windows_process_tree(process_id: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(process_id), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        try:
            stop_windows_process(process_id)
        except subprocess.TimeoutExpired:
            terminate_windows_process_direct(process_id)


def stop_windows_process(process_id: int) -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {process_id} -Force"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        **hidden_subprocess_kwargs(),
    )


def terminate_windows_process_direct(process_id: int) -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    process_terminate = 0x0001
    handle = kernel32.OpenProcess(process_terminate, False, int(process_id))
    if not handle:
        return
    try:
        kernel32.TerminateProcess(handle, 1)
    finally:
        kernel32.CloseHandle(handle)


def wait_for_port_release(host: str, port: int) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if find_windows_listener_pid(host, port) is None:
            return
        time.sleep(0.2)
    process_id = find_windows_listener_pid(host, port)
    raise RuntimeError(f"Port {port} is still in use after terminating previous server process {process_id}.")


def first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    return values[0]


def portal_health_payload(host: str, port: int, lm_studio_base: str, started_at: str) -> dict:
    return {
        "status": "ok",
        "pid": os.getpid(),
        "startedAt": started_at,
        "host": host,
        "port": port,
        "lmStudio": lm_studio_base,
    }


def portal_active_work_summary(services: dict[str, object]) -> list[dict]:
    active: list[dict] = []
    for service_name, service in services.items():
        runs = getattr(service, "_runs", {})
        lock = getattr(service, "_lock", None)
        if lock:
            with lock:
                snapshot = dict(runs)
        else:
            snapshot = dict(runs)
        for fallback_run_id, run in snapshot.items():
            if not isinstance(run, dict):
                continue
            status = str(run.get("status") or "")
            cancel_pending = bool(run.get("cancelRequested")) and status not in PORTAL_TERMINAL_STATUSES
            if status not in PORTAL_ACTIVE_STATUSES and not cancel_pending:
                continue
            active.append(
                {
                    "service": service_name,
                    "runId": str(run.get("runId") or fallback_run_id),
                    "status": status,
                }
            )
    return active


def windowless_python_executable(python_executable: str | None = None) -> str:
    executable = Path(python_executable or sys.executable)
    if sys.platform == "win32" and executable.name.lower() == "python.exe":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.exists():
            return str(pythonw)
    return str(executable)


def build_portal_restart_command(root: Path, host: str, port: int, lm_studio_base: str) -> list[str]:
    return [
        windowless_python_executable(sys.executable),
        str(root / "server.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--lm-studio",
        lm_studio_base,
        PORTAL_SKIP_PORT_RELEASE_FLAG,
    ]


def portal_restart_endpoint_from_command(command: list[str]) -> tuple[str, int]:
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    for index, value in enumerate(command):
        if value == "--host" and index + 1 < len(command):
            host = command[index + 1]
        elif value == "--port" and index + 1 < len(command):
            try:
                port = int(command[index + 1])
            except ValueError:
                port = DEFAULT_PORT
    return host, port


def hidden_subprocess_kwargs(detached: bool = False) -> dict:
    if sys.platform != "win32":
        return {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if detached:
        creationflags |= subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    kwargs: dict = {}
    if creationflags:
        kwargs["creationflags"] = creationflags
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return kwargs


def portal_restart_subprocess_kwargs(cwd: Path) -> dict:
    return {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        **hidden_subprocess_kwargs(detached=True),
    }


def build_portal_restart_helper_command(
    command: list[str],
    cwd: Path,
    parent_pid: int,
    delay_seconds: float,
    timeout_seconds: float = PORTAL_RESTART_HELPER_TIMEOUT_SECONDS,
) -> list[str]:
    host, port = portal_restart_endpoint_from_command(command)
    helper_code = r"""
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time

parent_pid = int(sys.argv[1])
command = json.loads(sys.argv[2])
cwd = sys.argv[3]
delay_seconds = float(sys.argv[4])
timeout_seconds = float(sys.argv[5])
host = sys.argv[6]
port = int(sys.argv[7])

def parent_alive(pid):
    if pid <= 0:
        return False
    if sys.platform == "win32":
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def terminate_parent(pid):
    if pid <= 0 or not parent_alive(pid):
        return
    if sys.platform == "win32":
        kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "check": False,
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0
            kwargs["startupinfo"] = startupinfo
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            **kwargs,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

def probe_host(value):
    if value in ("", "0.0.0.0", "::"):
        return "127.0.0.1"
    return value

def port_is_listening(host_value, port_value):
    target_host = probe_host(host_value)
    family = socket.AF_INET6 if ":" in target_host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            return sock.connect_ex((target_host, port_value)) == 0
    except OSError:
        return False

deadline = time.monotonic() + timeout_seconds
while parent_alive(parent_pid) and time.monotonic() < deadline:
    time.sleep(0.1)

if parent_alive(parent_pid):
    terminate_parent(parent_pid)
    deadline = time.monotonic() + 4
    while parent_alive(parent_pid) and time.monotonic() < deadline:
        time.sleep(0.1)

if delay_seconds > 0:
    time.sleep(delay_seconds)

deadline = time.monotonic() + timeout_seconds
while port_is_listening(host, port) and time.monotonic() < deadline:
    time.sleep(0.1)

kwargs = {
    "cwd": cwd,
    "stdin": subprocess.DEVNULL,
    "stdout": subprocess.DEVNULL,
    "stderr": subprocess.DEVNULL,
    "close_fds": True,
}
if sys.platform == "win32":
    creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
    kwargs["creationflags"] = creationflags
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    kwargs["startupinfo"] = startupinfo

subprocess.Popen(command, **kwargs)
"""
    return [
        windowless_python_executable(),
        "-c",
        helper_code,
        str(parent_pid),
        json.dumps(command),
        str(cwd),
        str(delay_seconds),
        str(timeout_seconds),
        host,
        str(port),
    ]


def schedule_portal_restart(
    command: list[str],
    cwd: Path,
    delay_seconds: float = 0.35,
    launcher=None,
    terminator=None,
    current_pid: int | None = None,
) -> threading.Thread:
    process_launcher = launcher or subprocess.Popen
    process_terminator = terminator or os._exit
    parent_pid = current_pid if current_pid is not None else os.getpid()

    def restart() -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        helper_command = build_portal_restart_helper_command(command, cwd, parent_pid, 0)
        process_launcher(helper_command, **portal_restart_subprocess_kwargs(cwd))
        process_terminator(0)

    thread = threading.Thread(target=restart, name="portal-backend-restart", daemon=True)
    thread.start()
    return thread


class ToolPortalHandler(http.server.SimpleHTTPRequestHandler):
    lm_studio_base = DEFAULT_LM_STUDIO
    portal_root = Path(__file__).resolve().parent
    portal_host = DEFAULT_HOST
    portal_port = DEFAULT_PORT
    portal_started_at = PORTAL_STARTED_AT
    comparator_service: ComparatorService
    initial_population_service: InitialPopulationService
    initial_population_comparison_service: InitialPopulationComparisonService
    reference_text_store: ReferenceTextStore
    turbulence_comparison_service: TurbulenceComparisonService
    sbert_service: SbertSimilarityService

    def do_OPTIONS(self) -> None:
        if self.is_api_or_proxy_path():
            self.send_response(204)
            self.send_cors_headers()
            self.end_headers()
            return
        super().do_OPTIONS()

    def do_GET(self) -> None:
        if self.path.startswith(PORTAL_API_PREFIX):
            self.handle_portal_get()
            return
        if self.path.startswith(REFERENCE_TEXTS_PREFIX):
            self.handle_reference_texts_get()
            return
        if self.path.startswith(TURBULENCE_COMPARISON_PREFIX):
            self.handle_turbulence_comparison_get()
            return
        if self.path.startswith(INITIAL_POPULATION_COMPARISON_PREFIX):
            self.handle_initial_population_comparison_get()
            return
        if self.path.startswith(INITIAL_POPULATION_PREFIX):
            self.handle_initial_population_get()
            return
        if self.path.startswith(COMPARATOR_PREFIX):
            self.handle_comparator_get()
            return
        if self.path.startswith(PROXY_PREFIX):
            self.proxy_to_lm_studio()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith(PORTAL_API_PREFIX):
            self.handle_portal_post()
            return
        if self.path.startswith(REFERENCE_TEXTS_PREFIX):
            self.handle_reference_texts_post()
            return
        if self.path.startswith(TURBULENCE_COMPARISON_PREFIX):
            self.handle_turbulence_comparison_post()
            return
        if self.path.startswith(SBERT_PREFIX):
            self.handle_sbert_post()
            return
        if self.path.startswith(LM_STUDIO_API_PREFIX):
            self.handle_lm_studio_api_post()
            return
        if self.path.startswith(INITIAL_POPULATION_COMPARISON_PREFIX):
            self.handle_initial_population_comparison_post()
            return
        if self.path.startswith(INITIAL_POPULATION_PREFIX):
            self.handle_initial_population_post()
            return
        if self.path.startswith(COMPARATOR_PREFIX):
            self.handle_comparator_post()
            return
        if self.path.startswith(PROXY_PREFIX):
            self.proxy_to_lm_studio()
            return
        self.send_error(404, "Not found")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        if self.is_api_or_proxy_path():
            self.send_cors_headers()
        super().end_headers()

    def is_api_or_proxy_path(self) -> bool:
        return (
            self.path.startswith(PROXY_PREFIX)
            or self.path.startswith(COMPARATOR_PREFIX)
            or self.path.startswith(INITIAL_POPULATION_PREFIX)
            or self.path.startswith(INITIAL_POPULATION_COMPARISON_PREFIX)
            or self.path.startswith(REFERENCE_TEXTS_PREFIX)
            or self.path.startswith(TURBULENCE_COMPARISON_PREFIX)
            or self.path.startswith(SBERT_PREFIX)
            or self.path.startswith(LM_STUDIO_API_PREFIX)
            or self.path.startswith(PORTAL_API_PREFIX)
        )

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def handle_portal_get(self) -> None:
        path_parts = self.portal_path_parts()
        if path_parts == ["health"]:
            self.send_json(
                200,
                portal_health_payload(
                    self.portal_host,
                    self.portal_port,
                    self.lm_studio_base,
                    self.portal_started_at,
                ),
            )
            return
        self.send_json(404, {"error": "Not found."})

    def handle_portal_post(self) -> None:
        path_parts = self.portal_path_parts()
        if path_parts != ["restart"]:
            self.send_json(404, {"error": "Not found."})
            return
        self.read_request_body()
        if self.headers.get(PORTAL_RESTART_HEADER) != "1":
            self.send_json(403, {"error": "Missing portal restart header."})
            return
        active_work = portal_active_work_summary(self.portal_services())
        if active_work:
            self.send_json(
                409,
                {
                    "error": "No se puede reiniciar el backend mientras hay corridas activas.",
                    "activeRuns": active_work,
                },
            )
            return
        command = build_portal_restart_command(
            self.portal_root,
            self.portal_host,
            self.portal_port,
            self.lm_studio_base,
        )
        self.send_json(
            202,
            {
                "status": "restarting",
                "pid": os.getpid(),
                "startedAt": self.portal_started_at,
            },
        )
        try:
            self.wfile.flush()
        except OSError:
            pass
        schedule_portal_restart(command, self.portal_root)

    def portal_services(self) -> dict[str, object]:
        return {
            "comparator": self.comparator_service,
            "initialPopulation": self.initial_population_service,
            "initialPopulationComparison": self.initial_population_comparison_service,
            "turbulenceComparison": self.turbulence_comparison_service,
        }

    def proxy_to_lm_studio(self) -> None:
        target_url = self.build_target_url()
        body = self.read_request_body()
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        request = urllib.request.Request(target_url, data=body, headers=headers, method=self.command)

        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = response.read()
                self.send_response(response.status)
                self.copy_response_headers(response.headers)
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.copy_response_headers(error.headers)
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:
            payload = json.dumps({"error": str(error)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def build_target_url(self) -> str:
        parsed = urllib.parse.urlsplit(self.path)
        stripped_path = parsed.path.removeprefix(PROXY_PREFIX)
        if not stripped_path.startswith("/"):
            stripped_path = f"/{stripped_path}"
        target = urllib.parse.urljoin(f"{self.lm_studio_base}/", stripped_path.lstrip("/"))
        if parsed.query:
            target = f"{target}?{parsed.query}"
        return target

    def read_request_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return None
        return self.rfile.read(length)

    def copy_response_headers(self, headers) -> None:
        ignored = {"connection", "content-encoding", "transfer-encoding", "access-control-allow-origin"}
        for key, value in headers.items():
            if key.lower() not in ignored:
                self.send_header(key, value)

    def handle_comparator_get(self) -> None:
        path_parts = self.comparator_path_parts()

        if path_parts == ["proposals"]:
            self.send_json(
                200,
                {
                    "proposals": self.comparator_service.list_proposals(),
                    "defaults": self.comparator_service.public_defaults(),
                },
            )
            return

        if len(path_parts) == 2 and path_parts[0] == "runs":
            run = self.comparator_service.get_run(path_parts[1])
            if not run:
                self.send_json(404, {"error": "Run not found."})
                return
            self.send_json(200, run)
            return

        if len(path_parts) == 3 and path_parts[0] == "runs" and path_parts[2] == "embedding-projection":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                payload = self.comparator_service.get_run_embedding_projection(
                    path_parts[1],
                    method=first_query_value(query, "method") or "pca",
                )
            except ValueError as error:
                self.send_json(400, {"error": str(error)})
                return
            if not payload:
                self.send_json(404, {"error": "Run not found."})
                return
            self.send_json(200, payload)
            return

        if len(path_parts) == 3 and path_parts[0] == "runs" and path_parts[2] == "logs":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            try:
                offset = int((query.get("offset") or ["0"])[0] or 0)
                limit = int((query.get("limit") or ["5000"])[0] or 5000)
            except ValueError:
                self.send_json(400, {"error": "offset and limit must be integers."})
                return
            payload = self.comparator_service.get_run_logs(path_parts[1], offset=offset, limit=limit)
            if not payload:
                self.send_json(404, {"error": "Run not found."})
                return
            self.send_json(200, payload)
            return

        self.send_json(404, {"error": "Not found."})

    def handle_comparator_post(self) -> None:
        path_parts = self.comparator_path_parts()

        try:
            if path_parts == ["runs"]:
                run = self.comparator_service.start_run(self.read_json_body())
                self.send_json(202, run)
                return

            if len(path_parts) == 3 and path_parts[0] == "runs" and path_parts[2] == "cancel":
                self.read_request_body()
                run = self.comparator_service.cancel_run(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            if len(path_parts) == 3 and path_parts[0] == "runs" and path_parts[2] == "recompute-metrics":
                self.read_request_body()
                run = self.comparator_service.recompute_run_metrics(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            message = str(error)
            payload = {"error": message}
            if message.startswith("duplicateProposalInstances:"):
                payload["code"] = "duplicateProposalInstances"
                payload["error"] = message.split(":", 1)[1].strip()
            self.send_json(400, payload)
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_initial_population_get(self) -> None:
        path_parts = self.initial_population_path_parts()
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if path_parts == ["strategies"]:
                self.send_json(200, {
                    "strategies": self.initial_population_service.list_strategies(),
                    "defaults": self.initial_population_service.default_config(),
                })
                return

            if path_parts == ["lm-studio", "models"]:
                base_url = first_query_value(query, "baseUrl")
                api_mode = first_query_value(query, "apiMode")
                self.send_json(200, self.initial_population_service.list_lm_studio_models(base_url, api_mode))
                return

            if len(path_parts) == 2 and path_parts[0] == "runs":
                run = self.initial_population_service.get_run(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_initial_population_post(self) -> None:
        path_parts = self.initial_population_path_parts()

        try:
            if path_parts == ["runs"]:
                run = self.initial_population_service.start_run(self.read_json_body())
                self.send_json(202, run)
                return

            if len(path_parts) == 3 and path_parts[0] == "runs" and path_parts[2] == "cancel":
                run = self.initial_population_service.cancel_run(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_initial_population_comparison_get(self) -> None:
        path_parts = self.initial_population_comparison_path_parts()

        try:
            if path_parts == ["strategies"]:
                self.send_json(200, {
                    "strategies": self.initial_population_comparison_service.list_strategies(),
                    "defaults": self.initial_population_comparison_service.default_config(),
                })
                return

            if len(path_parts) == 2 and path_parts[0] == "runs":
                run = self.initial_population_comparison_service.get_run(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_initial_population_comparison_post(self) -> None:
        path_parts = self.initial_population_comparison_path_parts()

        try:
            if path_parts == ["runs"]:
                run = self.initial_population_comparison_service.start_run(self.read_json_body())
                self.send_json(202, run)
                return

            if len(path_parts) == 3 and path_parts[0] == "runs" and path_parts[2] == "cancel":
                run = self.initial_population_comparison_service.cancel_run(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_reference_texts_get(self) -> None:
        path_parts = self.reference_texts_path_parts()
        try:
            if not path_parts:
                self.send_json(200, {"items": self.reference_text_store.list_texts()})
                return
            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_reference_texts_post(self) -> None:
        path_parts = self.reference_texts_path_parts()
        try:
            if not path_parts:
                item = self.reference_text_store.save_text(self.read_json_body())
                self.send_json(201, {"item": item, "items": self.reference_text_store.list_texts()})
                return
            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_turbulence_comparison_get(self) -> None:
        path_parts = self.turbulence_comparison_path_parts()
        try:
            if len(path_parts) == 2 and path_parts[0] == "runs":
                run = self.turbulence_comparison_service.get_run(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_turbulence_comparison_post(self) -> None:
        path_parts = self.turbulence_comparison_path_parts()
        try:
            if path_parts == ["ppdb", "status"]:
                self.send_json(200, self.turbulence_comparison_service.ppdb_status(self.read_json_body()))
                return

            if path_parts == ["ppdb", "prepare"]:
                self.send_json(200, self.turbulence_comparison_service.prepare_ppdb(self.read_json_body()))
                return

            if path_parts == ["runs"]:
                run = self.turbulence_comparison_service.start_run(self.read_json_body())
                self.send_json(202, run)
                return

            if len(path_parts) == 3 and path_parts[0] == "runs" and path_parts[2] == "cancel":
                run = self.turbulence_comparison_service.cancel_run(path_parts[1])
                if not run:
                    self.send_json(404, {"error": "Run not found."})
                    return
                self.send_json(200, run)
                return

            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_sbert_post(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        try:
            if parsed.path == f"{SBERT_PREFIX}/pair":
                self.send_json(200, self.sbert_service.calculate_pair(self.read_json_body()))
                return
            if parsed.path == f"{SBERT_PREFIX}/embeddings":
                self.send_json(200, self.sbert_service.calculate_embeddings(self.read_json_body()))
                return
            self.send_json(404, {"error": "Not found."})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def handle_lm_studio_api_post(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        try:
            payload = self.read_json_body()
            client = self.build_lm_studio_client(payload)
            if parsed.path == f"{LM_STUDIO_API_PREFIX}/models":
                self.send_json(200, {"models": client.list_models()})
                return
            if parsed.path == f"{LM_STUDIO_API_PREFIX}/chat":
                result = client.call(
                    str(payload.get("userPrompt") or payload.get("prompt") or ""),
                    system_prompt=str(payload.get("systemPrompt") or ""),
                    temperature=float(payload.get("temperature", 0.4)),
                    top_p=float(payload.get("topP", 0.95)),
                    top_k=int(payload["topK"]) if payload.get("topK") is not None else None,
                    max_tokens=int(payload.get("maxTokens", 180)),
                )
                self.send_json(200, result)
                return
            self.send_json(404, {"error": "Not found."})
        except LmStudioHttpError as error:
            self.send_json(502, {"error": str(error), "statusCode": error.status_code, "response": error.response_text[:1000]})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": str(error)})

    def build_lm_studio_client(self, payload: dict) -> LmStudioClient:
        base_url = str(payload.get("baseUrl") or self.lm_studio_base).strip().rstrip("/")
        api_mode = str(payload.get("apiMode") or "native").strip()
        model = str(payload.get("model") or "").strip()
        timeout_seconds = float(payload.get("timeoutSeconds") or 120)
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("baseUrl de LM Studio debe ser http(s).")
        return LmStudioClient(base_url, api_mode, model, timeout_seconds)

    def comparator_path_parts(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        api_path = parsed.path.removeprefix(COMPARATOR_PREFIX).strip("/")
        if not api_path:
            return []
        return [urllib.parse.unquote(part) for part in api_path.split("/") if part]

    def portal_path_parts(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        api_path = parsed.path.removeprefix(PORTAL_API_PREFIX).strip("/")
        if not api_path:
            return []
        return [urllib.parse.unquote(part) for part in api_path.split("/") if part]

    def initial_population_path_parts(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        api_path = parsed.path.removeprefix(INITIAL_POPULATION_PREFIX).strip("/")
        if not api_path:
            return []
        return [urllib.parse.unquote(part) for part in api_path.split("/") if part]

    def initial_population_comparison_path_parts(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        api_path = parsed.path.removeprefix(INITIAL_POPULATION_COMPARISON_PREFIX).strip("/")
        if not api_path:
            return []
        return [urllib.parse.unquote(part) for part in api_path.split("/") if part]

    def reference_texts_path_parts(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        api_path = parsed.path.removeprefix(REFERENCE_TEXTS_PREFIX).strip("/")
        if not api_path:
            return []
        return [urllib.parse.unquote(part) for part in api_path.split("/") if part]

    def turbulence_comparison_path_parts(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        api_path = parsed.path.removeprefix(TURBULENCE_COMPARISON_PREFIX).strip("/")
        if not api_path:
            return []
        return [urllib.parse.unquote(part) for part in api_path.split("/") if part]

    def read_json_body(self) -> dict:
        body = self.read_request_body()
        if not body:
            return {}
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON body: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def send_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the thesis LLM tool portal with an LM Studio proxy.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--lm-studio", default=DEFAULT_LM_STUDIO)
    parser.add_argument(PORTAL_SKIP_PORT_RELEASE_FLAG, action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    handler = lambda *handler_args, **handler_kwargs: ToolPortalHandler(
        *handler_args,
        directory=str(root),
        **handler_kwargs,
    )
    ToolPortalHandler.lm_studio_base = args.lm_studio.rstrip("/")
    ToolPortalHandler.portal_root = root
    ToolPortalHandler.portal_host = args.host
    ToolPortalHandler.portal_port = args.port
    ToolPortalHandler.portal_started_at = PORTAL_STARTED_AT
    ToolPortalHandler.comparator_service = ComparatorService(root)
    ToolPortalHandler.initial_population_service = InitialPopulationService(root)
    ToolPortalHandler.initial_population_comparison_service = InitialPopulationComparisonService(root)
    ToolPortalHandler.reference_text_store = ReferenceTextStore(root)
    ToolPortalHandler.turbulence_comparison_service = TurbulenceComparisonService(root)
    ToolPortalHandler.sbert_service = SbertSimilarityService()

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    if not args.skip_port_release:
        release_existing_server_port(args.host, args.port)
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving portal at http://{args.host}:{args.port}/LLM/")
        print(f"Proxying {PROXY_PREFIX}/* to {ToolPortalHandler.lm_studio_base}/*")
        print(f"Serving comparator API at http://{args.host}:{args.port}{COMPARATOR_PREFIX}/")
        print(f"Serving initial population API at http://{args.host}:{args.port}{INITIAL_POPULATION_PREFIX}/")
        print(f"Serving initial population comparison API at http://{args.host}:{args.port}{INITIAL_POPULATION_COMPARISON_PREFIX}/")
        print(f"Serving reference texts API at http://{args.host}:{args.port}{REFERENCE_TEXTS_PREFIX}/")
        print(f"Serving turbulence comparison API at http://{args.host}:{args.port}{TURBULENCE_COMPARISON_PREFIX}/")
        print(f"Serving SBERT API at http://{args.host}:{args.port}{SBERT_PREFIX}/")
        print(f"Serving LM Studio API at http://{args.host}:{args.port}{LM_STUDIO_API_PREFIX}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
