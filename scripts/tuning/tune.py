from __future__ import annotations

"""
Unified tuning entrypoint.

Steps
- Clean out/ to avoid stale artifacts
- Build merged runtime policy (baseline + optional single overlay) and set daily band
- Build pipeline artifacts
- Run all calibrators (writing to out/orders/policy_overrides.json)
- Run AI overrides calibrator (Codex) to update sector/ticker biases
- Persist final overlay to config/policy_overrides.json (single source)
- Remove legacy overlay files if present
"""

from pathlib import Path
import json
import re
import shutil


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"
CFG_DIR = BASE_DIR / "config"


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def main() -> int:
    # Clean out/
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    # Merge baseline + overlay and set band for presets builder
    from scripts.engine.config_io import ensure_policy_override_file
    from scripts.build_presets_all import set_daily_band_pct
    pol_path = ensure_policy_override_file()
    raw = _strip_json_comments(pol_path.read_text(encoding="utf-8"))
    obj = json.loads(raw)
    try:
        band = float((((obj.get('market') or {}).get('microstructure') or {}).get('daily_band_pct')))
    except Exception as exc:
        raise SystemExit(f"daily_band_pct missing/invalid in merged policy: {exc}") from exc
    set_daily_band_pct(band)

    # Build artifacts
    from scripts.engine.pipeline import ensure_pipeline_artifacts
    ensure_pipeline_artifacts()

    # Run calibrators (write runtime overrides)
    from scripts.tuning.calibrators import (
        calibrate_regime_components,
        calibrate_regime,
        calibrate_market_filter,
        calibrate_breadth_floor,
        calibrate_leader_gates,
        calibrate_risk_limits,
        calibrate_sizing,
        calibrate_thresholds,
        calibrate_softmax_tau,
        calibrate_near_ceiling,
        calibrate_liquidity,
        calibrate_dynamic_caps,
        calibrate_ttl_minutes,
        calibrate_fill_prob,
        calibrate_watchlist,
    )

    calibrate_regime_components.calibrate(write=True)
    calibrate_regime.calibrate(horizon=63, write=True)
    calibrate_market_filter.calibrate(write=True)
    calibrate_breadth_floor.calibrate(write=True)
    calibrate_leader_gates.calibrate(write=True)
    calibrate_risk_limits.calibrate(write=True)
    calibrate_sizing.calibrate(write=True)
    calibrate_thresholds.calibrate(write=True)
    calibrate_softmax_tau.calibrate(write=True)
    calibrate_near_ceiling.calibrate(write=True)
    calibrate_liquidity.calibrate(write=True)
    calibrate_dynamic_caps.calibrate(write=True)
    calibrate_ttl_minutes.calibrate(write=True)
    calibrate_fill_prob.calibrate(write=True)
    calibrate_watchlist.calibrate(write=True)

    # Run AI overrides calibrator (Codex)
    try:
        from scripts.tuning.calibrators import calibrate_ai_overrides
        calibrate_ai_overrides.calibrate(write=True)
    except SystemExit as exc:
        # Surface clear error; unified tune requires Codex when enabled
        raise
    except Exception as exc:
        raise SystemExit(f"AI overrides calibrator failed: {exc}") from exc

    # Persist overlay
    src = OUT_DIR / 'orders' / 'policy_overrides.json'
    if not src.exists():
        raise SystemExit('Missing out/orders/policy_overrides.json after tuning')
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    dst = CFG_DIR / 'policy_overrides.json'
    dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
    # Remove legacy overlays if present
    for old in (CFG_DIR / 'policy_nightly_overrides.json', CFG_DIR / 'policy_ai_overrides.json'):
        try:
            old.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
    print(f"[tune] Wrote unified overlay -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

