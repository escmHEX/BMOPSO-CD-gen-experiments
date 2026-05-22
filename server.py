from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from baselines.comparator import ComparatorService


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4173
DEFAULT_LM_STUDIO = "http://127.0.0.1:1234"
PROXY_PREFIX = "/lmstudio"
COMPARATOR_PREFIX = "/api/comparator"


def release_existing_server_port(host: str, port: int) -> None:
    if sys.platform != "win32":
        return

    script = f"""
$conn = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue |
  Where-Object {{ $_.LocalAddress -eq '{host}' -or $_.LocalAddress -eq '0.0.0.0' -or $_.LocalAddress -eq '::' -or $_.LocalAddress -eq '::1' -or $_.LocalAddress -eq '127.0.0.1' }} |
  Select-Object -First 1
if ($conn) {{
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)"
  [pscustomobject]@{{ pid = $conn.OwningProcess; commandLine = $proc.CommandLine }} | ConvertTo-Json -Compress
}}
"""

    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    payload = result.stdout.strip()
    if not payload:
        return

    try:
        process_info = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Could not inspect the process listening on port {port}: {payload}") from error

    process_id = int(process_info["pid"])
    command_line = str(process_info.get("commandLine") or "")
    if process_id == os.getpid():
        return
    if "server.py" not in command_line:
        raise RuntimeError(
            f"Port {port} is already in use by a different process: PID {process_id}, {command_line}"
        )

    subprocess.run(
        ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {process_id} -Force"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    time.sleep(1)


class ToolPortalHandler(http.server.SimpleHTTPRequestHandler):
    lm_studio_base = DEFAULT_LM_STUDIO
    comparator_service: ComparatorService

    def do_OPTIONS(self) -> None:
        if self.path.startswith(PROXY_PREFIX) or self.path.startswith(COMPARATOR_PREFIX):
            self.send_response(204)
            self.send_cors_headers()
            self.end_headers()
            return
        super().do_OPTIONS()

    def do_GET(self) -> None:
        if self.path.startswith(COMPARATOR_PREFIX):
            self.handle_comparator_get()
            return
        if self.path.startswith(PROXY_PREFIX):
            self.proxy_to_lm_studio()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith(COMPARATOR_PREFIX):
            self.handle_comparator_post()
            return
        if self.path.startswith(PROXY_PREFIX):
            self.proxy_to_lm_studio()
            return
        self.send_error(404, "Not found")

    def end_headers(self) -> None:
        if self.path.startswith(PROXY_PREFIX) or self.path.startswith(COMPARATOR_PREFIX):
            self.send_cors_headers()
        super().end_headers()

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

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
            self.send_json(200, {"proposals": self.comparator_service.list_proposals()})
            return

        if len(path_parts) == 2 and path_parts[0] == "runs":
            run = self.comparator_service.get_run(path_parts[1])
            if not run:
                self.send_json(404, {"error": "Run not found."})
                return
            self.send_json(200, run)
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
                run = self.comparator_service.cancel_run(path_parts[1])
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

    def comparator_path_parts(self) -> list[str]:
        parsed = urllib.parse.urlsplit(self.path)
        api_path = parsed.path.removeprefix(COMPARATOR_PREFIX).strip("/")
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
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    handler = lambda *handler_args, **handler_kwargs: ToolPortalHandler(
        *handler_args,
        directory=str(root),
        **handler_kwargs,
    )
    ToolPortalHandler.lm_studio_base = args.lm_studio.rstrip("/")
    ToolPortalHandler.comparator_service = ComparatorService(root)

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    release_existing_server_port(args.host, args.port)
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving portal at http://{args.host}:{args.port}/LLM/")
        print(f"Proxying {PROXY_PREFIX}/* to {ToolPortalHandler.lm_studio_base}/*")
        print(f"Serving comparator API at http://{args.host}:{args.port}{COMPARATOR_PREFIX}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
