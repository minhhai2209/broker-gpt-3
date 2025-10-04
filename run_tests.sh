#!/usr/bin/env bash
set -euo pipefail

# Helper to run the Python test suite with a local venv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV="venv"
[[ -d .venv ]] && VENV=".venv"

PY_BIN=""

ensure_venv() {
  if [[ -x "$VENV/bin/python" ]]; then
    PY_BIN="$VENV/bin/python"
    return
  fi
  echo "[setup] Creating virtualenv at ./$VENV"
  python3 -m venv "$VENV"
  PY_BIN="$VENV/bin/python"
  echo "[setup] Installing requirements"
  "$VENV/bin/pip" install -r requirements.txt >/dev/null
}

main() {
  ensure_venv
  echo "[run] Using Python: $PY_BIN"
  echo "[run] Running tests..."
  "$PY_BIN" -m unittest discover -s tests -p "test_*.py" -v
}

main "$@"

