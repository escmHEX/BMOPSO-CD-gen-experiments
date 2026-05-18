from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4173
DEFAULT_LM_STUDIO = "http://127.0.0.1:1234"
PROXY_PREFIX = "/lmstudio"


class ToolPortalHandler(http.server.SimpleHTTPRequestHandler):
    lm_studio_base = DEFAULT_LM_STUDIO

    def do_OPTIONS(self) -> None:
        if self.path.startswith(PROXY_PREFIX):
            self.send_response(204)
            self.send_cors_headers()
            self.end_headers()
            return
        super().do_OPTIONS()

    def do_GET(self) -> None:
        if self.path.startswith(PROXY_PREFIX):
            self.proxy_to_lm_studio()
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith(PROXY_PREFIX):
            self.proxy_to_lm_studio()
            return
        self.send_error(404, "Not found")

    def end_headers(self) -> None:
        if self.path.startswith(PROXY_PREFIX):
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

    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
        print(f"Serving portal at http://{args.host}:{args.port}/LLM/")
        print(f"Proxying {PROXY_PREFIX}/* to {ToolPortalHandler.lm_studio_base}/*")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
