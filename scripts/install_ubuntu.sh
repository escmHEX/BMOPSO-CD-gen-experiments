#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

FORCE_PPDB=0
SKIP_PPDB=0
SKIP_OLLAMA_INSTALL=0
SKIP_MODEL_PULL=0
SKIP_WARMUP=0
NO_SUDO=0
OLLAMA_LOCAL_DIR="${OLLAMA_LOCAL_DIR:-$ROOT/.local/ollama}"
OLLAMA_LOCAL_MODELS_DIR="${OLLAMA_MODELS:-$ROOT/.local/ollama-models}"
SPACY_MODEL_WHEEL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

usage() {
  cat <<'USAGE'
Usage: bash scripts/install_ubuntu.sh [options]

Options:
  --force-ppdb          Redownload PPDB and rebuild the compact index.
  --skip-ppdb           Skip Kaggle PPDB download and index generation.
  --skip-ollama-install Do not install Ollama if it is missing.
  --skip-model-pull     Skip Ollama model pulls during warmup.
  --skip-warmup         Skip all model warmup steps.
  --no-sudo             Never use sudo. Install Ollama under .local/ollama.
  -h, --help            Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-ppdb) FORCE_PPDB=1 ;;
    --skip-ppdb) SKIP_PPDB=1 ;;
    --skip-ollama-install) SKIP_OLLAMA_INSTALL=1 ;;
    --skip-model-pull) SKIP_MODEL_PULL=1 ;;
    --skip-warmup) SKIP_WARMUP=1 ;;
    --no-sudo) NO_SUDO=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing command '$1'. Install it and rerun this script."
}

has_sudo_access() {
  [[ "$NO_SUDO" -eq 0 ]] || return 1
  command -v sudo >/dev/null 2>&1 || return 1
  if sudo -n true >/dev/null 2>&1; then
    return 0
  fi
  [[ -t 0 ]] || return 1
  sudo -v >/dev/null 2>&1
}

maybe_apt_install() {
  local package="$1"
  if command -v apt-get >/dev/null 2>&1 && has_sudo_access; then
    sudo apt-get update
    sudo apt-get install -y "$package"
  fi
}

ensure_command() {
  local command_name="$1"
  local package_name="${2:-$1}"
  if command -v "$command_name" >/dev/null 2>&1; then
    return
  fi
  maybe_apt_install "$package_name"
  if command -v "$command_name" >/dev/null 2>&1; then
    return
  fi
  if [[ "$NO_SUDO" -eq 1 ]] || ! has_sudo_access; then
    fail "Missing command '$command_name'. Without sudo, install/load package '$package_name' first and rerun."
  fi
  fail "Missing command '$command_name'. Install package '$package_name' and rerun."
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  ensure_command curl curl
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv installation did not add uv to PATH. Add ~/.local/bin to PATH and rerun."
}

use_local_ollama_env() {
  export PATH="$OLLAMA_LOCAL_DIR/bin:$PATH"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-$OLLAMA_LOCAL_MODELS_DIR}"
}

ollama_archive_name() {
  case "$(uname -m)" in
    x86_64|amd64) echo "ollama-linux-amd64.tar.zst" ;;
    aarch64|arm64) echo "ollama-linux-arm64.tar.zst" ;;
    *) fail "Unsupported CPU architecture for local Ollama install: $(uname -m)" ;;
  esac
}

extract_tar_zst() {
  local archive="$1"
  local destination="$2"
  mkdir -p "$destination"
  local helper_python="${PORTAL_PYTHON:-}"
  [[ -n "$helper_python" && -x "$helper_python" ]] || fail "Cannot extract $archive without a prepared Python venv."
  uv pip install --python "$helper_python" zstandard
  "$helper_python" - "$archive" "$destination" <<'PY'
from pathlib import Path
import sys
import tarfile

import zstandard

archive = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()

with archive.open("rb") as raw:
    reader = zstandard.ZstdDecompressor().stream_reader(raw)
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Refusing to extract outside destination: {member.name}")
            tar.extract(member, path=destination, filter="data")
PY
}

install_ollama_local() {
  ensure_command curl curl
  local archive_name
  archive_name="$(ollama_archive_name)"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local resolved_root
  local resolved_install_dir
  resolved_root="$(cd "$ROOT" && pwd -P)"
  mkdir -p "$(dirname "$OLLAMA_LOCAL_DIR")"
  resolved_install_dir="$(cd "$(dirname "$OLLAMA_LOCAL_DIR")" && pwd -P)/$(basename "$OLLAMA_LOCAL_DIR")"
  [[ "$resolved_install_dir" == "$resolved_root/"* ]] || fail "OLLAMA_LOCAL_DIR must be inside the repository when the installer manages it: $OLLAMA_LOCAL_DIR"
  echo "Installing Ollama locally under $OLLAMA_LOCAL_DIR."
  curl -fL "https://ollama.com/download/$archive_name" -o "$tmp_dir/$archive_name" || {
    rm -rf "$tmp_dir"
    return 1
  }
  rm -rf "$OLLAMA_LOCAL_DIR"
  extract_tar_zst "$tmp_dir/$archive_name" "$OLLAMA_LOCAL_DIR" || {
    rm -rf "$tmp_dir"
    return 1
  }
  rm -rf "$tmp_dir"
  use_local_ollama_env
  command -v ollama >/dev/null 2>&1 || fail "Local Ollama install finished but $OLLAMA_LOCAL_DIR/bin/ollama is not usable."
}

ensure_ollama() {
  if [[ -x "$OLLAMA_LOCAL_DIR/bin/ollama" ]]; then
    use_local_ollama_env
  fi
  if command -v ollama >/dev/null 2>&1; then
    return
  fi
  if [[ "$SKIP_OLLAMA_INSTALL" -eq 1 ]]; then
    fail "Ollama is missing and --skip-ollama-install was passed."
  fi
  if [[ "$NO_SUDO" -eq 1 ]] || ! has_sudo_access; then
    install_ollama_local
    return
  fi
  ensure_command curl curl
  if curl -fsSL https://ollama.com/install.sh | sh; then
    command -v ollama >/dev/null 2>&1 && return
  fi
  echo "System Ollama install did not finish cleanly; falling back to local user install." >&2
  install_ollama_local
}

ensure_ollama_service() {
  ensure_ollama
  if [[ "$NO_SUDO" -eq 0 ]] && command -v systemctl >/dev/null 2>&1 && has_sudo_access; then
    sudo systemctl enable --now ollama >/dev/null 2>&1 || true
  fi
  if ! ollama list >/dev/null 2>&1; then
    mkdir -p runs
    nohup ollama serve > runs/ollama-serve.log 2>&1 &
    sleep 3
  fi
  ollama list >/dev/null 2>&1 || fail "Ollama is installed but not responding. Start it with 'ollama serve' and rerun."
}

python_in_venv() {
  local venv_path="$1"
  echo "$venv_path/bin/python"
}

ensure_venv() {
  local venv_path="$1"
  local python_bin
  python_bin="$(python_in_venv "$venv_path")"
  if [[ "${UV_VENV_CLEAR:-}" == "1" || "${UV_VENV_CLEAR:-}" == "true" ]]; then
    uv venv --clear --python 3.13 "$venv_path"
  elif [[ -x "$python_bin" ]]; then
    echo "Reusing existing virtual environment at: $venv_path"
  elif [[ -e "$venv_path" ]]; then
    fail "$venv_path exists but $python_bin is missing. Remove that directory or rerun with UV_VENV_CLEAR=1."
  else
    uv venv --python 3.13 "$venv_path"
  fi
  uv pip install --python "$python_bin" --upgrade pip
}

install_requirements() {
  local python_bin="$1"
  local requirements="$2"
  [[ -f "$requirements" ]] || fail "Requirements file not found: $requirements"
  uv pip install --python "$python_bin" -r "$requirements"
}

install_spacy_model() {
  local python_bin="$1"
  uv pip install --python "$python_bin" "$SPACY_MODEL_WHEEL"
}

sync_submodules() {
  require_command git
  git submodule sync --recursive
  git submodule update --init --recursive
}

sync_binary_repository() {
  local binary_dir="baselines/external/binary-mopso-cd"
  if [[ -d "$binary_dir/.git" ]]; then
    git -C "$binary_dir" fetch origin
    git -C "$binary_dir" checkout dev
    git -C "$binary_dir" pull --ff-only origin dev
    return
  fi
  if [[ -e "$binary_dir" ]]; then
    fail "$binary_dir exists but is not a git repository. Move it or remove it before rerunning."
  fi
  git clone --branch dev https://github.com/escmHEX/BMOPSO-CD.git "$binary_dir"
}

has_kaggle_credentials() {
  [[ -n "${KAGGLE_API_TOKEN:-}" ]] && return 0
  [[ -f "$HOME/.kaggle/access_token" ]] && return 0
  [[ -f "$HOME/.kaggle/kaggle.json" ]] && return 0
  [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]] && return 0
  return 1
}

prepare_ppdb() {
  local portal_python="$1"
  local ppdb_dir="data/external/ppdb"
  local ppdb_source="$ppdb_dir/ppdb-2.0-s-all"
  local ppdb_index="data/turbulence/ppdb_index.json"
  if [[ "$SKIP_PPDB" -eq 1 ]]; then
    echo "Skipping PPDB download and index generation."
    return
  fi
  if [[ "$FORCE_PPDB" -eq 1 || ! -f "$ppdb_source" ]]; then
    has_kaggle_credentials || fail "Kaggle credentials are missing. Configure KAGGLE_API_TOKEN, ~/.kaggle/access_token, ~/.kaggle/kaggle.json, or KAGGLE_USERNAME/KAGGLE_KEY for cudawarrior/ppdb-2-0-s-all."
    mkdir -p "$ppdb_dir"
    "$portal_python" -m kaggle datasets download -d cudawarrior/ppdb-2-0-s-all -p "$ppdb_dir" --unzip
    if [[ ! -f "$ppdb_source" ]]; then
      local candidate
      candidate="$(find "$ppdb_dir" -maxdepth 2 -type f \( -name 'ppdb-2.0-s-all' -o -name 'ppdb-2.0-s-all.gz' \) | head -n 1 || true)"
      [[ -n "$candidate" ]] || fail "Kaggle download completed but ppdb-2.0-s-all was not found under $ppdb_dir."
      if [[ "$candidate" == *.gz ]]; then
        ensure_command gzip gzip
        gzip -dc "$candidate" > "$ppdb_source"
      else
        cp "$candidate" "$ppdb_source"
      fi
    fi
  fi
  if [[ "$FORCE_PPDB" -eq 1 || ! -f "$ppdb_index" ]]; then
    "$portal_python" scripts/prepare_ppdb_index.py --source "$ppdb_source" --output "$ppdb_index"
  fi
}

prepare_binary_ppdb_index() {
  local binary_python="$1"
  local ppdb_source="data/external/ppdb/ppdb-2.0-s-all"
  local ppdb_index="data/turbulence/ppdb_index.sqlite"
  if [[ "$SKIP_PPDB" -eq 1 ]]; then
    echo "Skipping Binary PPDB SQLite index generation."
    return
  fi
  [[ -f "$ppdb_source" ]] || fail "PPDB source is missing at $ppdb_source; rerun without --skip-ppdb after configuring Kaggle credentials."
  if [[ "$FORCE_PPDB" -eq 1 || ! -f "$ppdb_index" ]]; then
    "$binary_python" - "$ppdb_source" "$ppdb_index" <<'PY'
from pathlib import Path
import sys

from binary_mopso_cd.services.ppdb import build_sqlite_index

build_sqlite_index(Path(sys.argv[1]), Path(sys.argv[2]))
PY
  fi
}

if [[ -x "$OLLAMA_LOCAL_DIR/bin/ollama" ]]; then
  use_local_ollama_env
fi

ensure_command git git
ensure_command curl curl
ensure_uv
uv python install 3.13

sync_submodules
sync_binary_repository

ensure_venv ".venv"
PORTAL_PYTHON="$(python_in_venv ".venv")"
install_requirements "$PORTAL_PYTHON" "requirements.backend.txt"
uv pip install --python "$PORTAL_PYTHON" kaggle

for proposal in evolmd evolmd-mo mesap; do
  ensure_venv "baselines/venvs/$proposal"
  PROPOSAL_PYTHON="$(python_in_venv "baselines/venvs/$proposal")"
  install_requirements "$PROPOSAL_PYTHON" "baselines/external/$proposal/requirements.txt"
  if [[ "$proposal" == "evolmd-mo" ]]; then
    install_spacy_model "$PROPOSAL_PYTHON"
  fi
done

ensure_venv "baselines/venvs/binary-mopso-cd"
BINARY_PYTHON="$(python_in_venv "baselines/venvs/binary-mopso-cd")"
uv pip install --python "$BINARY_PYTHON" -e "baselines/external/binary-mopso-cd[models]"
install_spacy_model "$BINARY_PYTHON"
"$PORTAL_PYTHON" scripts/diagnose_native_runtime.py --skip-portal --binary-python "$BINARY_PYTHON"

"$PORTAL_PYTHON" scripts/setup_runtime.py --write-comparator-config --platform linux
prepare_ppdb "$PORTAL_PYTHON"
prepare_binary_ppdb_index "$BINARY_PYTHON"

ensure_ollama_service
if [[ "$SKIP_WARMUP" -eq 0 ]]; then
  WARMUP_ARGS=()
  if [[ "$SKIP_MODEL_PULL" -eq 1 ]]; then
    WARMUP_ARGS+=(--skip-ollama)
  fi
  "$PORTAL_PYTHON" scripts/warmup_runtime.py "${WARMUP_ARGS[@]}"
fi

cat <<EOF
Setup complete.
Use:
  bash scripts/start_server_daemon_ubuntu.sh start --host 0.0.0.0
  bash scripts/verify_install.py --base-url http://127.0.0.1:4173
EOF
