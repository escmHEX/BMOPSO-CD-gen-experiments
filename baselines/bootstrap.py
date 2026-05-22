from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: bootstrap.py <script> [args...]")

    script_path = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(script_path.parent))
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    preload_modules = [
        name.strip()
        for name in os.environ.get("BASELINE_PRELOAD_MODULES", "").split(",")
        if name.strip()
    ]

    for module_name in preload_modules:
        importlib.import_module(module_name)

    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
