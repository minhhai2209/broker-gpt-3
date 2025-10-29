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
  local has_config=0
  for arg in "$@"; do
    case "$arg" in
      --config|--config=*)
        has_config=1
        break
        ;;
    esac
  done
  if [[ $has_config -eq 0 ]]; then
    set -- --config config/data_engine.yaml "$@"
  fi
  echo "[engine] Ensuring universe CSV"
  "$PY_BIN" -m scripts.tools.build_universe --industry-map "$ROOT_DIR/data/industry_map.csv" --output "$ROOT_DIR/data/universe/vn100.csv"
  echo "[engine] Using: $PY_BIN"
  "$PY_BIN" -m scripts.engine.data_engine "$@"
}

run_tests() {
  ensure_venv
  echo "[tests] Using: $PY_BIN"
  "$PY_BIN" -m unittest discover -s tests -p "test_*.py" -v
}

run_server() {
  echo "[server] Deprecated. Use TCBS scraper instead." >&2
  exit 2
}

run_prompts() {
  ensure_venv
  echo "[prompts] Using: $PY_BIN"
  "$PY_BIN" -m scripts.tools.gen_prompts "$@"
}

main() {
  local task="${1:-all}"
  shift || true
  case "$task" in
    all)
      # Run TCBS (headful by default) then the data engine.
      # Reuse this script to leverage existing logic and flags.
      echo "[all] Step 1/2: Running TCBS scraper (headful)"
      "$0" tcbs --headful "$@"
      echo "[all] Step 2/2: Running data engine"
      "$0" engine
      ;;
    engine)
      run_engine "$@"
      ;;
    tests)
      run_tests "$@"
      ;;
    server)
      run_server "$@" # kept for compatibility; now deprecated
      ;;
    prompts)
      run_prompts "$@"
      ;;
    tcbs)
      ensure_venv
      echo "[tcbs] Using: $PY_BIN"
      # Ensure Chromium is installed for Playwright (best-effort)
      "$PY_BIN" -m playwright install chromium >/dev/null 2>&1 || true
      # Helper to read TCBS_PROFILE without sourcing the entire .env (password may contain $)
      env_profile="${TCBS_PROFILE:-}"
      if [[ -z "$env_profile" && -f .env ]]; then
        env_profile="$(grep -E '^TCBS_PROFILE=' .env | tail -n 1 | cut -d= -f2- | tr -d '\r' | sed 's/^\"//; s/\"$//')"
      fi
      # If first arg looks like a flag, do not treat as profile; fall back to env or 'tcbs'
      if [[ "${1:-}" =~ ^- || -z "${1:-}" ]]; then
        profile="${env_profile:-tcbs}"
      else
        profile="${1}"
        shift || true
      fi
      "$PY_BIN" -m scripts.scrapers.tcbs --profile "$profile" "$@"
      ;;
    *)
      echo "Usage: $0 [all|engine|tests|server|tcbs]" >&2
      exit 2
      ;;
  esac
}

main "$@"
