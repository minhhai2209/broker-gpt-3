#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV="venv"
[[ -d .venv ]] && VENV=".venv"
REQ_FILE="$ROOT_DIR/requirements.txt"
PY_BIN=""

ensure_venv() {
  if [[ -x "$VENV/bin/python" ]]; then
    PY_BIN="$VENV/bin/python"
  else
    echo "[setup] Creating virtualenv at ./$VENV"
    python3 -m venv "$VENV"
    PY_BIN="$VENV/bin/python"
  fi

  local stamp="$VENV/.pip-stamp"
  if [[ ! -f "$stamp" || "$REQ_FILE" -nt "$stamp" ]]; then
    echo "[setup] Installing requirements"
    "$PY_BIN" -m pip install -r "$REQ_FILE" >/dev/null
    touch "$stamp"
  fi
}

run_engine() {
  ensure_venv
  echo "[engine] Using: $PY_BIN"
  "$PY_BIN" -m scripts.engine.data_engine --config "${1:-config/data_engine.yaml}"
}

run_tests() {
  ensure_venv
  echo "[tests] Using: $PY_BIN"
  "$PY_BIN" -m unittest discover -s tests -p "test_*.py" -v
}

run_server() {
  ensure_venv
  echo "[server] Using: $PY_BIN"
  export DATA_ENGINE_CONFIG="${DATA_ENGINE_CONFIG:-config/data_engine.yaml}"
  PORT="${PORT:-8787}" "$PY_BIN" -m scripts.api.server
}

main() {
  local task="${1:-engine}"
  shift || true
  case "$task" in
    engine)
      run_engine "$@"
      ;;
    tests)
      run_tests "$@"
      ;;
    server)
      run_server "$@"
      ;;
    *)
      echo "Usage: $0 [engine|tests|server]" >&2
      exit 2
      ;;
  esac
}

main "$@"
