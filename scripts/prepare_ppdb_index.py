from __future__ import annotations

"""Prepare a compact PPDB index for the turbulence comparison backend.

Usage example:
    py scripts/prepare_ppdb_index.py --source C:/datasets/ppdb-2.0-s-all.gz

The official PPDB files are distributed separately from the repository. This
script keeps only paraphrases whose left or right side appears in
LLM/data/pso-individuals.json, so the generated JSON remains small enough to
version if needed.
"""

import argparse
import gzip
import json
import re
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile


TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def component_terms(root: Path) -> set[str]:
    path = root / "LLM" / "data" / "pso-individuals.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for individual in payload.get("individuals", []):
        for vector_key in ("rol", "topico", "accion"):
            value = str(individual.get(vector_key) or "")
            add_terms(value, keys)
        for owner in ("pbest", "lider"):
            vector = individual.get(owner) or {}
            for vector_key in ("rol", "topico", "accion"):
                value = str(vector.get(vector_key) or "")
                add_terms(value, keys)
    return keys


def add_terms(value: str, keys: set[str]) -> None:
    normalized = normalize_text(value)
    if normalized:
        keys.add(normalized)
    tokens = TOKEN_RE.findall(value)
    for token in tokens:
        keys.add(normalize_text(token))
    for index in range(len(tokens) - 1):
        keys.add(normalize_text(" ".join(tokens[index:index + 2])))
    for index in range(len(tokens) - 2):
        keys.add(normalize_text(" ".join(tokens[index:index + 3])))


def source_path(source: str) -> Path:
    if source.startswith(("http://", "https://")):
        with NamedTemporaryFile(delete=False, suffix=Path(source).suffix or ".ppdb") as file:
            with urllib.request.urlopen(source, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            return Path(file.name)
    return Path(source)


def resolve_project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_ppdb_line(line: str) -> tuple[str, str] | None:
    parts = [part.strip() for part in line.split("|||")]
    if len(parts) >= 3:
        return parts[1], parts[2]
    tab_parts = [part.strip() for part in line.split("\t")]
    if len(tab_parts) >= 2:
        return tab_parts[0], tab_parts[1]
    return None


def build_index(source: Path, keys: set[str], limit_per_key: int) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}

    def add_mapping(key: str, value: str, normalized_value: str) -> None:
        if key not in keys or not normalized_value or key == normalized_value:
            return
        bucket = index.setdefault(key, [])
        if value not in bucket and len(bucket) < limit_per_key:
            bucket.append(value)

    with open_text(source) as file:
        for line in file:
            parsed = parse_ppdb_line(line)
            if not parsed:
                continue
            left, right = parsed
            left_key = normalize_text(left)
            right_key = normalize_text(right)
            if not left_key or not right_key:
                continue
            add_mapping(left_key, right, right_key)
            add_mapping(right_key, left, left_key)
    return {key: values for key, values in sorted(index.items()) if values}


def prepare_ppdb_index(
    root: Path,
    source: str | Path,
    output: str | Path = "data/turbulence/ppdb_index.json",
    limit_per_key: int = 30,
) -> dict[str, object]:
    source_text = str(source)
    source_candidate = source_path(source_text)
    source_file = source_candidate if source_candidate.is_absolute() else root / source_candidate
    if not source_file.exists():
        raise FileNotFoundError(f"PPDB source not found: {source_file}")

    output_path = resolve_project_path(root, output)
    keys = component_terms(root)
    entries = build_index(source_file, keys, max(1, int(limit_per_key)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "source": source_text,
        "sourcePath": str(source_file),
        "keyCount": len(keys),
        "entryCount": sum(len(values) for values in entries.values()),
        "entries": entries,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "outputPath": str(output_path),
        "sourcePath": str(source_file),
        "keyCount": len(keys),
        "indexedKeys": len(entries),
        "entryCount": payload["entryCount"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact PPDB index for the turbulence comparison tool.")
    parser.add_argument("--source", required=True, help="Local PPDB file path or direct URL. .gz files are supported.")
    parser.add_argument("--output", default="data/turbulence/ppdb_index.json")
    parser.add_argument("--limit-per-key", type=int, default=30)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    summary = prepare_ppdb_index(root, args.source, args.output, args.limit_per_key)
    print(f"PPDB index written to {summary['outputPath']} with {summary['indexedKeys']} keyed entries.")


if __name__ == "__main__":
    main()
