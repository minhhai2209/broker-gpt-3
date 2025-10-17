from __future__ import annotations

"""Unified tuning entrypoint.

Steps
- Clean out/ to avoid stale artifacts
- Run Codex-driven AI overrides (pre-phase) to publish config/policy_ai_overrides.json
- Build merged runtime policy (baseline + overlays) and set daily band
- Build pipeline artifacts
- Run numeric calibrators (writing to out/orders/policy_overrides.json)
- Leave unified overlay in out/orders/policy_overrides.json for CI to persist if desired
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
    print('[tune] START unified tuning pipeline')
    # Clean out/
    if OUT_DIR.exists():
        print('[tune] Removing previous out/ directory')
        shutil.rmtree(OUT_DIR)

    # Run AI pre-phase before merging overlays
    print('[tune] Running AI overrides calibrator (Codex pre-phase)')
    # Require Codex by default; users can export BROKER_REQUIRE_CODEX=0 to skip
    import os as _os
    _os.environ.setdefault('BROKER_REQUIRE_CODEX', '1')
    from scripts.tuning.calibrators import calibrate_ai_overrides

    calibrate_ai_overrides.calibrate(write=True)

    # Merge baseline + overlays and set band for presets builder
    print('[tune] Merging baseline + overlays via ensure_policy_override_file()')
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
    print('[tune] Building pipeline artifacts')
    ensure_pipeline_artifacts()

    # Run calibrators (write runtime overrides)
    from scripts.tuning.calibrators import (
        calibrate_regime_components,
        calibrate_regime,
        calibrate_market_filter,
        calibrate_ticker_stops,
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
    # New deterministic calibrators (generic / reusable)
    from scripts.tuning.calibrators import calibrate_budget_topk, calibrate_execution_fill
    from scripts.tuning.calibrators.policy_write import (
        expect_calibrations,
        verify_calibrations,
        reset_tracking,
    )

    print('[tune] Running calibrators')
    reset_tracking()
    expect_calibrations([
        calibrate_regime_components.__name__,
        calibrate_regime.__name__,
        calibrate_market_filter.__name__,
        calibrate_ticker_stops.__name__,
        calibrate_breadth_floor.__name__,
        calibrate_leader_gates.__name__,
        calibrate_risk_limits.__name__,
        calibrate_sizing.__name__,
        calibrate_budget_topk.__name__,
        calibrate_execution_fill.__name__,
        calibrate_thresholds.__name__,
        calibrate_softmax_tau.__name__,
        calibrate_near_ceiling.__name__,
        calibrate_liquidity.__name__,
        calibrate_dynamic_caps.__name__,
        calibrate_ttl_minutes.__name__,
        calibrate_fill_prob.__name__,
        calibrate_watchlist.__name__,
    ])
    calibrate_regime_components.calibrate(write=True)
    calibrate_regime.calibrate(horizon=63, write=True)
    calibrate_market_filter.calibrate(write=True)
    calibrate_ticker_stops.calibrate(write=True)
    calibrate_breadth_floor.calibrate(write=True)
    calibrate_leader_gates.calibrate(write=True)
    calibrate_risk_limits.calibrate(write=True)
    calibrate_sizing.calibrate(write=True)
    # Generic breadth/budget + execution acceptance before thresholds top‑K gating
    calibrate_budget_topk.calibrate(write=True)
    calibrate_execution_fill.calibrate(write=True)
    calibrate_thresholds.calibrate(write=True)
    calibrate_softmax_tau.calibrate(write=True)
    calibrate_near_ceiling.calibrate(write=True)
    calibrate_liquidity.calibrate(write=True)
    calibrate_dynamic_caps.calibrate(write=True)
    calibrate_ttl_minutes.calibrate(write=True)
    calibrate_fill_prob.calibrate(write=True)
    calibrate_watchlist.calibrate(write=True)

    missing_writes = verify_calibrations()
    if missing_writes:
        missing_list = ', '.join(sorted(missing_writes))
        raise SystemExit(f"[tune] Missing policy writes from calibrators: {missing_list}")

    # Surface final overlay path for downstream automation (GitHub Action persists when appropriate)
    src = OUT_DIR / 'orders' / 'policy_overrides.json'
    if not src.exists():
        raise SystemExit('Missing out/orders/policy_overrides.json after tuning')
    # Always publish to repo config for audit/CI (carry over _meta)
    dst = CFG_DIR / 'policy_overrides.json'
    shutil.copyfile(src, dst)
    print(f"[tune] Unified overlay ready at {src} and published to {dst}")
    print('[tune] DONE')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
