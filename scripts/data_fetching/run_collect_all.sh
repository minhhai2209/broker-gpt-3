#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -d venv ]]; then
  source venv/bin/activate
fi

python scripts/data_fetching/run_data_jobs.py --group nightly "$@"
python scripts/data_fetching/run_data_jobs.py --group real_time "$@"

echo "[done] Nightly data jobs completed"
