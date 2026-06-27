#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

FORCE_PPDB=0
SKIP_PPDB=0
SKIP_OLLAMA_INSTALL=0
SKIP_MODEL_PULL=0
SKIP_WARMUP=0
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

maybe_apt_install() {
  local package="$1"
  if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
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
  command -v "$command_name" >/dev/null 2>&1 || fail "Missing command '$command_name'. Install package '$package_name' and rerun."
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

ensure_ollama() {
  if command -v ollama >/dev/null 2>&1; then
    return
  fi
  if [[ "$SKIP_OLLAMA_INSTALL" -eq 1 ]]; then
    fail "Ollama is missing and --skip-ollama-install was passed."
  fi
  ensure_command curl curl
  curl -fsSL https://ollama.com/install.sh | sh
  command -v ollama >/dev/null 2>&1 || fail "Ollama installation finished but ollama is not in PATH."
}

ensure_ollama_service() {
  ensure_ollama
  if command -v systemctl >/dev/null 2>&1; then
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
  uv venv --python 3.13 "$venv_path"
  local python_bin
  python_bin="$(python_in_venv "$venv_path")"
  "$python_bin" -m pip install --upgrade pip
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
    has_kaggle_credentials || fail "Kaggle credentials are missing. Configure ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY for cudawarrior/ppdb-2-0-s-all."
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

ensure_command git git
ensure_command curl curl
ensure_command unzip unzip
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

"$PORTAL_PYTHON" scripts/setup_runtime.py --write-comparator-config --platform linux
prepare_ppdb "$PORTAL_PYTHON"

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
