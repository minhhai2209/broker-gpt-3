from __future__ import annotations

"""
Emit a session-scoped patch_tune.json from long-term curated signals.

Source of truth (long-term, user-maintained): data/curated_signals.json
Schema (per-ticker, prices in 'nghìn đồng/cp' i.e. thousands VND per share):
{
  "meta": {"updated": "2025-10-16"},
  "tickers": [
    {
      "ticker": "STB",
      "tier": "A",
      "pullback_low_k": 35.0,
      "pullback_high_k": 36.0,
      "breakout_k": 40.0,
      "stop_k": 32.0,
      "note": "Hậu tái cơ cấu..."
    }
  ]
}

Behaviour:
- Reads out/snapshot.csv (Price column) then selects tickers where:
  * pullback_low_k <= Price <= pullback_high_k -> apply +pb_bias (default 0.06)
  * Price >= breakout_k -> apply +brk_bias (default 0.08)
  * else optional baseline bias by tier (A: 0.02, B: 0.01)
- Writes out/orders/patch_tune.json with meta.ttl= end of day (VN timezone).

Guardrails:
- Bias values are later clamped to [-0.20, 0.20] by aggregator.
- This module never edits config/policy_overrides.json; it only emits runtime patch.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out"
ORDERS_DIR = OUT_DIR / "orders"

VN_TZ = timezone(timedelta(hours=7))


@dataclass
class CuratedItem:
    ticker: str
    tier: str
    pullback_low_k: Optional[float]
    pullback_high_k: Optional[float]
    breakout_k: Optional[float]
    stop_k: Optional[float]


def _read_curated(path: Path) -> List[CuratedItem]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for row in (raw.get("tickers") or []):
        try:
            items.append(
                CuratedItem(
                    ticker=str(row.get("ticker", "")).strip().upper(),
                    tier=str(row.get("tier", "")).strip().upper(),
                    pullback_low_k=(float(row.get("pullback_low_k")) if row.get("pullback_low_k") is not None else None),
                    pullback_high_k=(float(row.get("pullback_high_k")) if row.get("pullback_high_k") is not None else None),
                    breakout_k=(float(row.get("breakout_k")) if row.get("breakout_k") is not None else None),
                    stop_k=(float(row.get("stop_k")) if row.get("stop_k") is not None else None),
                )
            )
        except Exception:
            continue
    return [it for it in items if it.ticker]


def _bias_for(item: CuratedItem, price_k: Optional[float]) -> Optional[float]:
    if price_k is None or not (price_k == price_k):  # NaN guard
        return None
    # Default baselines by tier (very light) so as not to affect EXIT downgrade
    base_by_tier = {"A": 0.02, "B": 0.01}
    pb_bias = 0.06
    brk_bias = 0.08
    # Pullback check
    if item.pullback_low_k is not None and item.pullback_high_k is not None:
        if item.pullback_low_k <= price_k <= item.pullback_high_k:
            return pb_bias
    # Breakout add check
    if item.breakout_k is not None and price_k >= item.breakout_k:
        return brk_bias
    # Baseline tilt by tier
    return base_by_tier.get(item.tier, 0.0)


def emit_curated_patch(curated_path: Path | None = None, snapshot_path: Path | None = None) -> Optional[Path]:
    curated_path = curated_path or (DATA_DIR / "curated_signals.json")
    snapshot_path = snapshot_path or (OUT_DIR / "snapshot.csv")
    items = _read_curated(curated_path)
    if not items:
        return None
    if not snapshot_path.exists():
        # No snapshot -> cannot evaluate conditions; emit only baseline tier tilts
        price_map: Dict[str, Optional[float]] = {}
    else:
        snap = pd.read_csv(snapshot_path)
        if "Ticker" not in snap.columns:
            return None
        snap = snap.set_index("Ticker")
        price_map = {}
        for it in items:
            price = None
            if it.ticker in snap.index:
                s = snap.loc[it.ticker]
                for key in ("Price", "P"):
                    if key in s.index:
                        try:
                            price = float(s.get(key))
                            break
                        except Exception:
                            price = None
            price_map[it.ticker] = price
    bias_entries: Dict[str, float] = {}
    for it in items:
        bias = _bias_for(it, price_map.get(it.ticker))
        if bias is None:
            continue
        if abs(bias) < 1e-6:
            continue
        bias_entries[f"ticker_bias.{it.ticker}"] = float(bias)
    if not bias_entries:
        return None
    # TTL end-of-day VN
    now_vn = datetime.now(VN_TZ)
    eod = now_vn.replace(hour=23, minute=59, second=0, microsecond=0)
    payload = {
        "meta": {
            "source": "curated_signals",
            "ttl": eod.isoformat(),
        },
        "bias": bias_entries,
    }
    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ORDERS_DIR / "patch_tune.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    try:
        p = emit_curated_patch()
        print(f"curated patch: {p}")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

