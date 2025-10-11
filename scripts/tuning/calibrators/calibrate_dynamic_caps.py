from __future__ import annotations

"""
Calibrate sizing.dynamic_caps bounds from ENP target and sector concentration target.

Orthodox basis
- Position size bounds relate inversely to desired breadth (ENP). A simple,
  transparent mapping sets pos_max ≈ c1 / ENP_total and sector_max ≈ k × pos_max.
- Bounds are then pos_min < pos_max and sector_min < sector_max, with modest
  ratios (e.g., 0.7× and 0.8×) to provide a corridor for market_score scaling.

Inputs
- config/policy_overrides.json
  - add_max, new_max (fallback to compute ENP if enp_total_target not provided)
  - sizing.dynamic_caps object must exist (schema requires)
  - calibration_targets.dynamic_caps.{enp_total_target?, sector_limit_target?}

Outputs
- Writes sizing.dynamic_caps.{pos_min,pos_max,sector_min,sector_max}.

Fail-fast: missing keys or invalid values.
"""

from pathlib import Path
import json
import re

from scripts.tuning.calibrators.policy_write import write_policy


BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / 'out'
ORDERS_PATH = OUT_DIR / 'orders' / 'policy_overrides.json'
CONFIG_PATH = BASE_DIR / 'config' / 'policy_overrides.json'


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load() -> dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    return json.loads(_strip(src.read_text(encoding='utf-8')))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def calibrate(write: bool = False) -> tuple[float, float, float, float]:
    obj = _load()
    try:
        dc = dict((obj.get('sizing', {}) or {}).get('dynamic_caps', {}) or {})
    except Exception as exc:
        raise SystemExit(f'Missing sizing.dynamic_caps: {exc}') from exc
    if not dc:
        raise SystemExit('sizing.dynamic_caps object missing in config')

    tgt = (obj.get('calibration_targets', {}) or {}).get('dynamic_caps', {})
    enp_total = tgt.get('enp_total_target', None)
    sector_target = tgt.get('sector_limit_target', None)
    if enp_total is None:
        # Fallback to add_max+new_max as breadth proxy
        try:
            enp_total = float(int(obj['add_max']) + int(obj['new_max']))
        except Exception as exc:
            raise SystemExit(f'Provide calibration_targets.dynamic_caps.enp_total_target or valid add_max/new_max: {exc}') from exc
    try:
        enp_total = float(enp_total)
    except Exception as exc:
        raise SystemExit(f'invalid enp_total_target: {exc}') from exc
    if enp_total <= 0:
        raise SystemExit('enp_total_target must be > 0')

    # Map breadth to caps using simple, transparent constants
    c1 = 1.2  # slight allowance above 1/ENP for focus when conviction high
    pos_max_base = _clamp(c1 / enp_total, 0.05, 0.20)
    pos_min_base = _clamp(0.7 * pos_max_base, 0.02, pos_max_base)

    if sector_target is None:
        sector_max_base = _clamp(4.0 * pos_max_base, 0.25, 0.40)
    else:
        try:
            sector_max_base = _clamp(float(sector_target), 0.10, 0.60)
        except Exception as exc:
            raise SystemExit(f'invalid sector_limit_target: {exc}') from exc
    sector_min_base = _clamp(0.8 * sector_max_base, 0.10, sector_max_base)

    if write:
        sizing = dict(obj.get('sizing', {}) or {})
        dyn = dict(sizing.get('dynamic_caps', {}) or {})
        dyn['pos_min'] = float(pos_min_base)
        dyn['pos_max'] = float(pos_max_base)
        dyn['sector_min'] = float(sector_min_base)
        dyn['sector_max'] = float(sector_max_base)
        sizing['dynamic_caps'] = dyn
        obj['sizing'] = sizing
        write_policy(
            calibrator=__name__,
            policy=obj,
            orders_path=ORDERS_PATH,
            config_path=CONFIG_PATH,
        )
    return float(pos_min_base), float(pos_max_base), float(sector_min_base), float(sector_max_base)


def main():
    # Always write tuned dynamic caps to runtime overrides
    pmin, pmax, smin, smax = calibrate(write=True)
    print(f"[calibrate.dyncaps] pos_min={pmin:.3f}, pos_max={pmax:.3f}, sector_min={smin:.3f}, sector_max={smax:.3f}")


if __name__ == '__main__':
    main()
