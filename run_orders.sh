#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the end-to-end order generator.
# - Creates/uses a Python venv at ./venv (or ./.venv if present)
# - Ensures config/policy_overrides.json exists (copies sample if available)
# - Runs scripts/generate_orders.py and prints output paths

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
  echo "[run] Generating orders..."
  "$PY_BIN" scripts/generate_orders.py
  echo
  echo "[done] Outputs (if generated):"
  [[ -f out/orders/orders_final.csv ]] && echo " - out/orders/orders_final.csv"
  [[ -f out/orders/orders_print.txt ]] && echo " - out/orders/orders_print.txt"
  [[ -f out/orders/orders_reasoning.csv ]] && echo " - out/orders/orders_reasoning.csv"
  [[ -f out/orders/orders_analysis.txt ]] && echo " - out/orders/orders_analysis.txt"
}

main "$@"

