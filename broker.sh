#!/usr/bin/env bash
set -euo pipefail

# Unified helper
# Usage: ./broker.sh [orders|tests|policy|server|fundamentals] [extra args]
# - No arg defaults to: orders
# - Only one task allowed at a time

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

cleanup_out_dir() {
  local target="$ROOT_DIR/out"
  if [[ -d "$target" ]]; then
    echo "[orders] Removing stale out/ directory"
    rm -rf "$target"
  fi
}

run_orders() {
  cleanup_out_dir
  ensure_venv
  echo "[orders] Using: $PY_BIN"
  "$PY_BIN" -u scripts/generate_orders.py
  echo
  echo "[done] Outputs (if generated):"
  [[ -f out/orders/orders_final.csv ]] && echo " - out/orders/orders_final.csv"
  [[ -f out/orders/orders_print.txt ]] && echo " - out/orders/orders_print.txt"
  [[ -f out/orders/orders_reasoning.csv ]] && echo " - out/orders/orders_reasoning.csv"
  [[ -f out/orders/orders_analysis.txt ]] && echo " - out/orders/orders_analysis.txt"
  [[ -f out/orders/trade_suggestions.txt ]] && echo " - out/orders/trade_suggestions.txt"
}

run_tests() {
  ensure_venv
  echo "[tests] Using: $PY_BIN"
  if [[ "${BROKER_COVERAGE:-0}" != "0" ]]; then
    echo "[tests] Coverage enabled (BROKER_COVERAGE=1)"
    mkdir -p out/coverage || true
    "$PY_BIN" -m coverage run -m unittest discover -s tests -p "test_*.py" -v
    "$PY_BIN" -m coverage report -m --skip-empty
    "$PY_BIN" -m coverage xml -o out/coverage/coverage.xml
    "$PY_BIN" -m coverage html -d out/coverage/html
    echo "[coverage] HTML: out/coverage/html/index.html"
    echo "[coverage] XML:  out/coverage/coverage.xml"
  else
    "$PY_BIN" -m unittest discover -s tests -p "test_*.py" -v
  fi
}

ensure_playwright_browser() {
  if ! "$PY_BIN" -m playwright --version >/dev/null 2>&1; then
    return
  fi
  local marker="$VENV/.playwright-chromium"
  if [[ -f "$marker" ]]; then
    return
  fi
  echo "[setup] Downloading Playwright Chromium (first run may take a while)"
  "$PY_BIN" -m playwright install chromium >/dev/null
  touch "$marker"
}

gh_switch_account() {
  local user="minhhai2209"
  if ! command -v gh >/dev/null 2>&1; then
    echo "[warn] GitHub CLI (gh) not found; pushing with existing git auth"
    return 0
  fi
  echo "[gh] Switching auth to account: $user"
  # Some gh versions support 'auth switch'; ignore failure if unavailable
  if gh auth switch -u "$user" >/dev/null 2>&1; then
    :
  else
    echo "[warn] 'gh auth switch' not available or failed; proceeding with current gh session"
  fi
  gh auth status -h github.com || echo "[warn] gh auth status check failed"
}

commit_and_push_policy() {
  local file="config/policy_overrides.json"
  if [[ ! -f "$file" ]]; then
    echo "[error] Missing $file; nothing to commit"
    exit 1
  fi
  # Only stage that single file
  git add -N "$file" >/dev/null 2>&1 || true
  # Check if there are changes in that file
  if git diff --quiet -- "$file"; then
    echo "[policy] No changes to commit in $file"
    return 0
  fi
  # Ensure we are on a branch (not detached) for pull/rebase safety
  local branch
  branch=$(git rev-parse --abbrev-ref HEAD)
  if [[ "$branch" == "HEAD" ]]; then
    echo "[error] Detached HEAD; checkout a branch before pushing"
    exit 2
  fi
  echo "[git] Fetching and rebasing on origin/$branch"
  git fetch origin "$branch"
  # Rebase to incorporate any remote updates before committing/pushing
  if ! git rebase "origin/$branch"; then
    echo "[error] Rebase failed; resolve conflicts then re-run"
    exit 3
  fi
  git add "$file"
  git commit -m "chore(policy): update policy_overrides via Codex CLI"
  gh_switch_account
  # Final safety: small retry window in case of concurrent pushes
  for i in 1 2 3; do
    if git push origin "$branch"; then
      echo "[policy] Committed and pushed $file"
      return 0
    fi
    echo "[warn] Push failed, retry $i/3 after rebase"
    git fetch origin "$branch"
    git rebase "origin/$branch" || true
    sleep 2
  done
  echo "[error] Failed to push after retries"
  exit 4
}

run_policy() {
  ensure_venv
  echo "[policy] Using: $PY_BIN"
  echo "[git] Pulling latest changes"
  git pull
  # Unbuffered Python for streaming logs into server
  "$PY_BIN" -u scripts/ai/generate_policy_overrides.py
  commit_and_push_policy
}

run_fundamentals() {
  ensure_venv
  ensure_playwright_browser
  echo "[fundamentals] Using: $PY_BIN"
  "$PY_BIN" scripts/collect_vietstock_fundamentals.py "$@"
}

main() {
  local task="${1:-orders}"
  shift || true
  case "$task" in
    orders)
      run_orders "$@"
      ;;
    tests)
      run_tests "$@"
      ;;
    coverage)
      BROKER_COVERAGE=1 run_tests "$@"
      ;;
    policy)
      run_policy "$@"
      ;;
    server)
      ensure_venv
      echo "[server] Using: $PY_BIN"
      PORT="${PORT:-8787}" "$PY_BIN" scripts/api/server.py
      ;;
    fundamentals)
      run_fundamentals "$@"
      ;;
    *)
      echo "Usage: $0 [orders|tests|policy|server|fundamentals] [extra args]" >&2
      exit 2
      ;;
  esac
}

main "$@"
