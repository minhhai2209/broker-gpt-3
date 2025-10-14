#!/usr/bin/env bash
set -euo pipefail

# Minimal replay stub for 3 days bull/side/bear (placeholder)
# Usage: scripts/replay.sh T=3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

: "${T:=3}"
echo "[replay] Running smoke replay for $T days (stub)"
./broker.sh orders >/dev/null || true
echo "[replay] DONE (stub)"

