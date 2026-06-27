from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


BINARY_REPOSITORY_URL = "https://github.com/escmHEX/BMOPSO-CD.git"
BINARY_REPOSITORY_BRANCH = "dev"
BINARY_LOCAL_REPOSITORY = "baselines/external/binary-mopso-cd"
COMPARATOR_LOCAL_CONFIG = "baselines/comparator_config.local.json"
PROPOSAL_VENV_NAMES = {
    "evolmd": "evolmd",
    "evolmd-mo": "evolmd-mo",
    "mesap": "mesap",
    "binary-mopso-cd": "binary-mopso-cd",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slash_path(*parts: str) -> str:
    return "/".join(part.strip("/\\") for part in parts if part)


def venv_python_path(proposal_id: str, platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    venv_name = PROPOSAL_VENV_NAMES[proposal_id]
    if platform_name.startswith("win"):
        return _slash_path("baselines", "venvs", venv_name, "Scripts", "python.exe")
    return _slash_path("baselines", "venvs", venv_name, "bin", "python")


def load_base_comparator_config(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    config_path = root / "baselines" / "comparator_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Comparator config must be a JSON object: {config_path}")
    return payload


def build_comparator_local_config(
    root: Path | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(load_base_comparator_config(root))
    proposals = config.setdefault("proposals", {})
    if not isinstance(proposals, dict):
        raise ValueError("Comparator proposals config must be an object.")

    for proposal_id in ("evolmd", "evolmd-mo", "mesap"):
        proposal = proposals.setdefault(proposal_id, {})
        if not isinstance(proposal, dict):
            raise ValueError(f"Comparator proposal config must be an object: {proposal_id}")
        proposal["repositoryPath"] = _slash_path("baselines", "external", proposal_id)
        proposal["pythonExecutable"] = venv_python_path(proposal_id, platform_name)
        proposal.setdefault("pythonPathEntries", [])

    binary = proposals.setdefault("binary-mopso-cd", {})
    if not isinstance(binary, dict):
        raise ValueError("Comparator Binary proposal config must be an object.")
    binary["repositoryPath"] = BINARY_LOCAL_REPOSITORY
    binary["pythonExecutable"] = venv_python_path("binary-mopso-cd", platform_name)
    binary.setdefault("pythonPathEntries", [])
    git_config = binary.setdefault("git", {})
    if not isinstance(git_config, dict):
        raise ValueError("Comparator Binary git config must be an object.")
    git_config["remote"] = "origin"
    git_config["branch"] = BINARY_REPOSITORY_BRANCH
    git_config["pullMode"] = "ff-only"
    git_config["expectedRemoteUrl"] = BINARY_REPOSITORY_URL

    return config


def write_comparator_local_config(
    root: Path | None = None,
    platform_name: str | None = None,
    output: str | Path = COMPARATOR_LOCAL_CONFIG,
) -> Path:
    root = root or project_root()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    config = build_comparator_local_config(root, platform_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local runtime setup files for this portal.")
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--platform", default=sys.platform, choices=("linux", "win32"))
    parser.add_argument("--output", default=COMPARATOR_LOCAL_CONFIG)
    parser.add_argument("--write-comparator-config", action="store_true")
    args = parser.parse_args()

    if args.write_comparator_config:
        path = write_comparator_local_config(args.root, args.platform, args.output)
        print(path)
        return

    payload = build_comparator_local_config(args.root, args.platform)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
