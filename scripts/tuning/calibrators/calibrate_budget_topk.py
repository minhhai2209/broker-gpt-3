from __future__ import annotations

"""
Deterministic calibration for BUY breadth and budget mapping.

Principles
- Keep K (new_max/add_max) adaptive to regime (risk_on_probability) and
  index ATR percentile while respecting guardrails and auditability.
- Do not look into realized fills; only use current cross‑section diagnostics
  and portfolio size. Calibrate mapping buy_budget_by_regime to generic values
  that the engine uses at runtime (no per‑run ad‑hoc knobs).
- Provide a small, data-driven relaxation margin for the market breadth guard so
  risk-on sessions with orderly volatility do not fail the breadth filter by a
  few basis points. The guard still tightens automatically in stressed tapes.

Inputs
- out/orders/policy_overrides.json (or config/policy_overrides.json fallback)
- out/portfolio_clean.csv
- out/session_summary.csv, out/sector_strength.csv

Outputs
- Writes: buy_budget_by_regime, add_max, new_max into the runtime policy file.

Fail‑fast
- Missing required files/columns -> SystemExit with a clear message.
"""

from pathlib import Path
from typing import Dict
import json
import re
import math
import pandas as pd

from scripts.tuning.calibrators.policy_write import write_policy

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / "out"
DATA_DIR = BASE_DIR / "data"
ORDERS_PATH = OUT_DIR / "orders" / "policy_overrides.json"
CONFIG_PATH = BASE_DIR / "config" / "policy_overrides.json"


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.M)
    text = re.sub(r"(^|\s)#.*$", "", text, flags=re.M)
    return text


def _load_policy() -> Dict:
    src = ORDERS_PATH if ORDERS_PATH.exists() else CONFIG_PATH
    if not src.exists():
        raise SystemExit(f"Missing policy file: {src}")
    return json.loads(_strip(src.read_text(encoding="utf-8")))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if df is None or df.empty:
        raise SystemExit(f"{path} is empty")
    return df


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def calibrate(*, write: bool = True) -> Dict[str, float]:
    pol = _load_policy()
    # Inputs for regime
    session = _load_csv(OUT_DIR / "session_summary.csv")
    sectors = _load_csv(OUT_DIR / "sector_strength.csv")

    # Portfolio size and breadth target
    portfolio = _load_csv(OUT_DIR / "portfolio_clean.csv")
    if "Ticker" not in portfolio.columns:
        raise SystemExit("portfolio_clean.csv missing Ticker column")
    holdings = set(str(t).strip().upper() for t in portfolio["Ticker"].astype(str))
    n_hold = len(holdings)
    industry = _load_csv(DATA_DIR / "industry_map.csv")
    universe = [str(t).strip().upper() for t in industry.get("Ticker", pd.Series(dtype=str)).astype(str)]
    universe = [t for t in universe if t and t not in {"VNINDEX", "VN30", "VN100"}]
    universe_size = len(set(universe))
    clip_hi = max(0, min(universe_size if universe_size > 0 else 50, 50))  # schema: add_max/new_max ∈ [0,50]

    try:
        from scripts.orders.order_engine import get_market_regime
    except Exception as exc:
        raise SystemExit(f"Unable to import engine: {exc}") from exc

    regime = get_market_regime(session, sectors, pol)
    risk_on_prob = float(getattr(regime, "risk_on_probability", 0.0) or 0.0)
    atr_pctile = float(getattr(regime, "index_atr_percentile", 0.5) or 0.5)

    sizing = dict(pol.get("sizing", {}) or {})
    min_names_target = int(float(sizing.get("min_names_target", 0) or 0))
    gap_to_min = max(0, min_names_target - n_hold)

    # Volatility penalty using configured soft/hard ATR percentiles when available
    mf = dict(pol.get("market_filter", {}) or {})
    soft = float(mf.get("index_atr_soft_pct", 0.90) or 0.90)
    hard = float(mf.get("index_atr_hard_pct", 0.97) or 0.97)
    if hard <= soft:
        hard = max(soft + 0.01, 0.96)
    # Map atr_pctile in [soft..hard] -> penalty in [0..1]
    vol_penalty = 1.0 - _clip((atr_pctile - soft) / (hard - soft), 0.0, 1.0)

    # Risk factor from logistic probability
    risk_factor = _clip((risk_on_prob - 0.45) / 0.25, 0.0, 1.0)

    # Breadth guard relaxation — enabled only when risk-on odds and volatility allow.
    if risk_factor > 0.0:
        base_relax = 0.005 + 0.015 * risk_factor  # 0.5%..2.0%
        breadth_relax = _clip(base_relax * vol_penalty, 0.0, 0.03)
    else:
        breadth_relax = 0.0

    # NEW breadth target — adaptive with regime floors, clipped to [0..10]
    if getattr(regime, "risk_on", False):
        k_new = clip_hi
        k_add = clip_hi
    else:
        base_floor = 3 if getattr(regime, "is_neutral", False) else 0
        k_new = base_floor + int(math.ceil(gap_to_min * (0.4 + 0.6 * risk_factor) * vol_penalty))
        k_new = int(_clip(k_new, 0, clip_hi))
        k_add = int(math.ceil(n_hold * (0.10 + 0.20 * risk_factor) * vol_penalty))
        k_add = max(k_add, k_new)
        k_add = int(_clip(k_add, 0, clip_hi))

    # Regime budget mapping — generic, audit‑friendly (engine picks at runtime)
    # Keep within 0.02..0.30 and modest vs baseline guidance
    budget_map = {
        "risk_off": 0.02,
        "neutral": 0.10,
        "risk_on": 0.20,
    }

    out = {
        "new_max": k_new,
        "add_max": k_add,
        "breadth_relax_margin": breadth_relax,
        **{f"budget_{k}": v for k, v in budget_map.items()},
    }

    if write:
        obj = pol
        obj["new_max"] = int(k_new)
        obj["add_max"] = int(k_add)
        obj["buy_budget_by_regime"] = dict(budget_map)
        mf_conf = dict(obj.get("market_filter", {}) or {})
        mf_conf["breadth_relax_margin"] = float(breadth_relax)
        obj["market_filter"] = mf_conf
        write_policy(
            calibrator=__name__,
            policy=obj,
            orders_path=ORDERS_PATH,
            config_path=CONFIG_PATH,
        )
    return out


if __name__ == "__main__":
    vals = calibrate(write=True)
    print("[calibrate.budget_topk] " + ", ".join(f"{k}={v}" for k, v in vals.items()))
