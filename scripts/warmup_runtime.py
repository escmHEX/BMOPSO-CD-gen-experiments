from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_OLLAMA_MODELS = ("llama3", "llama3.1:8b", "gemma4:e4b")
DEFAULT_SBERT_MODELS = ("sentence-transformers/all-MiniLM-L6-v2", "thenlper/gte-small")
DEFAULT_TRANSFORMER_MODELS = ("bert-base-uncased", "distilbert/distilbert-base-uncased")
DEFAULT_NLTK_PACKAGES = (
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger_eng",
    "punkt",
    "punkt_tab",
    "stopwords",
)


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, check=True)


def normalize_ollama_name(name: str) -> set[str]:
    value = name.strip()
    if not value:
        return set()
    values = {value}
    if ":" not in value:
        values.add(f"{value}:latest")
    return values


def installed_ollama_models() -> set[str]:
    completed = subprocess.run(
        ["ollama", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "No se pudo consultar ollama list.")
    names: set[str] = set()
    for index, line in enumerate(completed.stdout.splitlines()):
        if index == 0 and line.lower().startswith("name"):
            continue
        first = line.split(maxsplit=1)[0] if line.split() else ""
        if first:
            names.add(first)
    return names


def warmup_ollama(models: tuple[str, ...]) -> None:
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama no esta instalado o no esta en PATH.")
    installed = installed_ollama_models()
    for model in models:
        if installed.intersection(normalize_ollama_name(model)):
            print(f"Ollama model already available: {model}")
            continue
        run_command(["ollama", "pull", model])
        installed = installed_ollama_models()


def warmup_python_models(
    sbert_models: tuple[str, ...],
    transformer_models: tuple[str, ...],
    nltk_packages: tuple[str, ...],
) -> None:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
    import nltk
    import spacy

    for package in nltk_packages:
        print(f"Downloading NLTK package: {package}", flush=True)
        nltk.download(package, quiet=False)

    print("Loading spaCy model: en_core_web_sm", flush=True)
    spacy.load("en_core_web_sm")

    for model in sbert_models:
        print(f"Loading SBERT model: {model}", flush=True)
        SentenceTransformer(model)

    for model in transformer_models:
        print(f"Loading transformer model: {model}", flush=True)
        AutoTokenizer.from_pretrained(model)
        if model == "distilbert/distilbert-base-uncased":
            AutoModelForMaskedLM.from_pretrained(model)
        else:
            AutoModel.from_pretrained(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache the runtime models used by the portal.")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-python-models", action="store_true")
    parser.add_argument("--ollama-model", action="append", dest="ollama_models")
    args = parser.parse_args()

    if not args.skip_ollama:
        warmup_ollama(tuple(args.ollama_models or DEFAULT_OLLAMA_MODELS))
    if not args.skip_python_models:
        warmup_python_models(DEFAULT_SBERT_MODELS, DEFAULT_TRANSFORMER_MODELS, DEFAULT_NLTK_PACKAGES)

    print("Runtime warmup complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
