from __future__ import annotations

import json
from pathlib import Path

from scripts.engine.config_io import ensure_policy_override_file, suggest_tuning


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = REPO_ROOT / "config"


def test_curated_bias_overlay_merges_and_takes_precedence(tmp_path):
    curated_path = CFG / "policy_curated_overrides.json"
    curated_path.parent.mkdir(parents=True, exist_ok=True)
    # Backup existing curated overlay (if any)
    backup = None
    if curated_path.exists():
        backup = curated_path.read_text(encoding="utf-8")
    try:
        curated = {"ticker_bias": {"AAA": 0.03}}
        curated_path.write_text(json.dumps(curated, ensure_ascii=False, indent=2), encoding="utf-8")
        out_path = ensure_policy_override_file()
        assert out_path.exists()
        merged = suggest_tuning(None, None)
        tb = dict(merged.get("ticker_bias", {}) or {})
        assert tb.get("AAA") == 0.03
    finally:
        if backup is None:
            curated_path.unlink(missing_ok=True)
        else:
            curated_path.write_text(backup, encoding="utf-8")

