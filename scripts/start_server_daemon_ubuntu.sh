#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SERVICE_NAME="bmopso-cd-experiments"
ACTION="${1:-start}"
HOST="127.0.0.1"
PORT="4173"
LM_STUDIO="http://127.0.0.1:11434"
USE_NOHUP=0

if [[ $# -gt 0 ]]; then
  shift
fi

usage() {
  cat <<'USAGE'
Usage: bash scripts/start_server_daemon_ubuntu.sh <start|stop|restart|status> [options]

Options:
  --host HOST          Bind host. Use 0.0.0.0 on a server if remote access is needed.
  --port PORT          Server port. Defaults to 4173.
  --lm-studio URL      Ollama/LM Studio base URL. Defaults to http://127.0.0.1:11434.
  --nohup             Force nohup fallback instead of systemd --user.
  -h, --help          Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:?missing host}"; shift ;;
    --port) PORT="${2:?missing port}"; shift ;;
    --lm-studio) LM_STUDIO="${2:?missing URL}"; shift ;;
    --nohup) USE_NOHUP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

PYTHON="$ROOT/.venv/bin/python"
SERVER="$ROOT/server.py"
CONFIG="$ROOT/baselines/comparator_config.local.json"
DAEMON_DIR="$ROOT/runs/server-daemon"
PID_FILE="$DAEMON_DIR/server.pid"
LOG_FILE="$DAEMON_DIR/server.log"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_runtime() {
  [[ -x "$PYTHON" ]] || fail "Missing $PYTHON. Run bash scripts/install_ubuntu.sh first."
  [[ -f "$SERVER" ]] || fail "Missing $SERVER."
  [[ -f "$CONFIG" ]] || fail "Missing $CONFIG. Run bash scripts/install_ubuntu.sh first."
}

systemd_available() {
  [[ "$USE_NOHUP" -eq 0 ]] || return 1
  command -v systemctl >/dev/null 2>&1 || return 1
  systemctl --user show-environment >/dev/null 2>&1 || return 1
}

write_systemd_unit() {
  local user_dir="$HOME/.config/systemd/user"
  local unit="$user_dir/$SERVICE_NAME.service"
  mkdir -p "$user_dir"
  cat > "$unit" <<EOF
[Unit]
Description=BMOPSO-CD experiments portal
After=network.target

[Service]
Type=simple
WorkingDirectory=$ROOT
Environment=COMPARATOR_CONFIG_PATH=$CONFIG
ExecStart=$PYTHON $SERVER --host $HOST --port $PORT --lm-studio $LM_STUDIO
Restart=on-failure
RestartSec=5
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE

[Install]
WantedBy=default.target
EOF
  echo "$unit"
}

enable_linger_if_possible() {
  command -v loginctl >/dev/null 2>&1 || return 0
  loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes' && return 0
  if command -v sudo >/dev/null 2>&1; then
    sudo loginctl enable-linger "$USER" >/dev/null 2>&1 || {
      echo "Warning: could not enable linger. If the service stops after SSH logout, run: sudo loginctl enable-linger $USER" >&2
    }
  else
    echo "Warning: sudo is unavailable. To survive SSH logout, run as admin: loginctl enable-linger $USER" >&2
  fi
}

nohup_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

nohup_start() {
  require_runtime
  mkdir -p "$DAEMON_DIR"
  if nohup_running; then
    echo "Server already running with PID $(cat "$PID_FILE")."
    return
  fi
  COMPARATOR_CONFIG_PATH="$CONFIG" nohup "$PYTHON" "$SERVER" --host "$HOST" --port "$PORT" --lm-studio "$LM_STUDIO" >> "$LOG_FILE" 2>&1 &
  echo "$!" > "$PID_FILE"
  echo "Server started with PID $(cat "$PID_FILE"). Logs: $LOG_FILE"
}

nohup_stop() {
  if ! nohup_running; then
    echo "Server is not running under nohup."
    rm -f "$PID_FILE"
    return
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" || true
  rm -f "$PID_FILE"
  echo "Stopped PID $pid."
}

systemd_start() {
  require_runtime
  mkdir -p "$DAEMON_DIR"
  write_systemd_unit >/dev/null
  enable_linger_if_possible
  systemctl --user daemon-reload
  systemctl --user enable --now "$SERVICE_NAME.service"
  systemctl --user status "$SERVICE_NAME.service" --no-pager
}

systemd_stop() {
  systemctl --user stop "$SERVICE_NAME.service" || true
}

systemd_status() {
  systemctl --user status "$SERVICE_NAME.service" --no-pager
}

if systemd_available; then
  case "$ACTION" in
    start) systemd_start ;;
    stop) systemd_stop ;;
    restart) systemd_stop; systemd_start ;;
    status) systemd_status ;;
    *) usage; exit 2 ;;
  esac
else
  case "$ACTION" in
    start) nohup_start ;;
    stop) nohup_stop ;;
    restart) nohup_stop; nohup_start ;;
    status)
      if nohup_running; then
        echo "Server running with PID $(cat "$PID_FILE"). Logs: $LOG_FILE"
      else
        echo "Server is not running under nohup."
        exit 3
      fi
      ;;
    *) usage; exit 2 ;;
  esac
fi
