from __future__ import annotations

"""
Execution Ladder — Scale‑In/Scale‑Out helper for VN100 engine.

This module generates per‑order ladder levels (price, qty, TTL, reprice rule)
based on the runtime policy (execution.ladder), snapshot, and metrics. It is
invoked by the order engine after action & sizing but before writing outputs.

Design goals (per spec):
- Respect HOSE tick/lot sizes and daily band; never cross market when no_cross.
- Split a logical order into 2–5 levels by regime profile with offsets/weights.
- Enforce per‑level ADTV cap and drop levels with hopeless fill probability.
- Emit expanded orders_final.csv (BundleId/Level rows) and per‑level quality.
- Fail fast on missing configuration; fallback to tick‑based offsets if ATR% is
  unavailable for a ticker (while logging diagnostics for auditability).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import math
import json
import pandas as pd

from scripts.utils import hose_tick_size, round_to_tick, clip_to_band, to_float


@dataclass
class Level:
    bundle_id: str
    level: int
    side: str
    ticker: str
    qty: int
    limit_price: float
    ttl_min: int
    reprice_rule: int
    notes: str = ""
    # Diagnostics
    price_ref: float = 0.0
    atr_pct: float = 0.0  # decimal (e.g., 0.025 for 2.5%)
    delta_ticks: float = 0.0


def _profile_from_regime(regime: object) -> str:
    try:
        if bool(getattr(regime, 'risk_on', False)):
            return 'risk_on'
        # Prefer explicit neutral flag if provided
        if bool(getattr(regime, 'is_neutral', False)):
            return 'neutral'
    except Exception:
        pass
    return 'risk_off'


def _tick_for_price(price_k: float) -> float:
    return hose_tick_size(price_k)


def _mk_price(side: str, pref: float, delta: float, atr_pct: float) -> float:
    sgn = -1.0 if side == 'BUY' else +1.0
    raw = pref * (1.0 + sgn * delta * atr_pct)
    tick = _tick_for_price(pref if pref > 0 else raw)
    return round_to_tick(raw, tick)


def _round_to_lot(qty: float, lot: int) -> int:
    if lot <= 0:
        return int(max(0, round(qty)))
    return int(max(0, int(round(qty / lot)) * lot))


def _adtv_shares(metrics_row: Optional[pd.Series], price_ref: float) -> float:
    if metrics_row is None:
        return 0.0
    adtv_k = to_float(metrics_row.get('AvgTurnover20D_k'))
    if adtv_k is None or adtv_k <= 0 or price_ref <= 0:
        return 0.0
    return float(adtv_k) * 1000.0 / float(price_ref)


def _delta_ticks(side: str, price_ref: float, level_price: float) -> float:
    tick = _tick_for_price(price_ref if price_ref > 0 else level_price)
    if tick <= 0:
        return 0.0
    if side == 'BUY':
        return max(0.0, (price_ref - level_price) / tick)
    return max(0.0, (level_price - price_ref) / tick)


def _detect_limit_lock(side: str, snap_row: Optional[pd.Series], price_ref: Optional[float]) -> bool:
    if snap_row is None or price_ref is None:
        return False
    tick = _tick_for_price(price_ref)
    tol = max(tick, 1e-6)
    try:
        if side == 'BUY':
            ceil_val = None
            for col in ('Ceiling', 'CeilingPrice', 'Ceiling_Tick', 'CeilingTick'):
                v = to_float(snap_row.get(col))
                if v is not None:
                    ceil_val = v; break
            if ceil_val is None:
                return False
            high_px = to_float(snap_row.get('High')) or to_float(snap_row.get('DayHigh')) or to_float(snap_row.get('HighPrice'))
            close_px = to_float(snap_row.get('Close')) or to_float(snap_row.get('Price')) or to_float(snap_row.get('Last'))
            if high_px is not None and abs(high_px - ceil_val) <= tol and price_ref >= ceil_val - tol:
                return True
            if close_px is not None and abs(close_px - ceil_val) <= tol and price_ref >= ceil_val - tol:
                return True
            return False
        # SELL -> floor lock
        floor_val = None
        for col in ('Floor', 'FloorPrice', 'Floor_Tick', 'FloorTick'):
            v = to_float(snap_row.get(col))
            if v is not None:
                floor_val = v; break
        if floor_val is None:
            return False
        low_px = to_float(snap_row.get('Low')) or to_float(snap_row.get('DayLow')) or to_float(snap_row.get('LowPrice'))
        close_px = to_float(snap_row.get('Close')) or to_float(snap_row.get('Price')) or to_float(snap_row.get('Last'))
        if low_px is not None and abs(low_px - floor_val) <= tol and price_ref <= floor_val + tol:
            return True
        if close_px is not None and abs(close_px - floor_val) <= tol and price_ref <= floor_val + tol:
            return True
        return False
    except Exception:
        return False


def estimate_fill_prob(delta_ticks: float, size_ratio: float, regime_score: float) -> float:
    # Simple logistic heuristic; coefficients subject to calibration
    a, b, c, d = 0.2, -0.12, -1.8, 0.3
    z = a + b * abs(float(delta_ticks)) + c * max(0.0, float(size_ratio)) + d * max(0.0, float(regime_score))
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0


def generate_ladder_levels(
    orders: List[object],
    *,
    regime: object,
    snapshot: pd.DataFrame,
    metrics: pd.DataFrame,
    presets: pd.DataFrame,
) -> Tuple[List[Level], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Expand logical orders into per‑level ladder instructions.

    Returns (levels, dropped, debug_records).
    dropped: list of dicts with reason and minimal fields for watchlist.
    debug_records: list of per‑bundle diagnostics for ladder.log
    """
    if orders is None:
        return [], [], []
    exec_conf = dict(getattr(regime, 'execution', {}) or {})
    ladder_conf = dict(exec_conf.get('ladder', {}) or {})
    if not ladder_conf.get('enabled', False):
        return [], [], []
    # Required keys with explicit fail‑fast
    required = [
        'max_levels','lot_size','method','buy_offsets_atr','sell_offsets_atr','weights',
        'ttl_min','reprice_ticks','min_fill_prob','adtv_cap_frac','atc_guard_minutes','skip_if_limit_lock','no_cross'
    ]
    for k in required:
        if k not in ladder_conf:
            raise SystemExit(f"Missing execution.ladder.{k} in runtime policy")

    lot = int(ladder_conf.get('lot_size', 100) or 100)
    max_levels = int(ladder_conf.get('max_levels', 3) or 3)
    method = str(ladder_conf.get('method'))
    min_fill_prob = float(ladder_conf.get('min_fill_prob', 0.20) or 0.20)
    adtv_cap_frac = float(ladder_conf.get('adtv_cap_frac', 0.03) or 0.03)
    skip_if_limit_lock = bool(ladder_conf.get('skip_if_limit_lock', True))
    no_cross = bool(ladder_conf.get('no_cross', True))
    profile = _profile_from_regime(regime)
    reprice_map = dict(ladder_conf.get('reprice_ticks', {}) or {})
    reprice_base = int(reprice_map.get(profile, 0) or 0)
    ttl_list = list(ladder_conf.get('ttl_min', []) or [])
    if not ttl_list:
        ttl_list = [5, 15, 30]

    snap = snapshot.set_index('Ticker') if not snapshot.empty else pd.DataFrame()
    met = metrics.set_index('Ticker') if not metrics.empty else pd.DataFrame()
    pre = presets.set_index('Ticker') if not presets.empty else pd.DataFrame()

    def _as_row(df: pd.DataFrame, t: str) -> Optional[pd.Series]:
        if df is None or df.empty or t not in df.index:
            return None
        row = df.loc[t]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    regime_score = float(getattr(regime, 'market_score', 0.5) or 0.5)
    levels: List[Level] = []
    dropped: List[Dict[str, Any]] = []
    debug_recs: List[Dict[str, Any]] = []

    bundle_seq = 0
    for o in orders:
        t = str(getattr(o, 'ticker'))
        side = str(getattr(o, 'side')).upper()
        Q = int(getattr(o, 'quantity'))
        base_note = str(getattr(o, 'note', '') or '')
        snap_row = _as_row(snap, t)
        met_row = _as_row(met, t)
        pre_row = _as_row(pre, t)
        price_ref = None
        if snap_row is not None:
            price_ref = to_float(snap_row.get('Price')) or to_float(snap_row.get('P'))
        if price_ref is None:
            price_ref = float(getattr(o, 'limit_price', 0.0) or 0.0)
        if price_ref is None or price_ref <= 0:
            # If even base is missing, skip ladder gracefully (cannot price levels)
            continue

        atr_pct = 0.0
        if met_row is not None:
            ap = to_float(met_row.get('ATR14_Pct'))
            if ap is not None and ap > 0:
                atr_pct = float(ap) / 100.0
        adtv_sh = _adtv_shares(met_row, price_ref)
        adtv_lot = adtv_sh / max(1, lot)

        # Determine offsets
        deltas: List[float] = []
        if method == 'atr_pct' and atr_pct > 0:
            offs_map = ladder_conf['buy_offsets_atr'] if side == 'BUY' else ladder_conf['sell_offsets_atr']
            prof = offs_map.get(profile)
            if not isinstance(prof, list) or len(prof) == 0:
                raise SystemExit(f"execution.ladder offsets missing for profile '{profile}'")
            deltas = [float(x) for x in prof][:max_levels]
        else:
            # fallback in ticks (approximate via steps array)
            k = [0, 2, 5, 8, 12][:max_levels]
            deltas = [float(x) for x in k]

        weights_map = ladder_conf['weights']
        weights = [float(x) for x in weights_map.get(profile, [])][:max_levels]
        if not weights:
            raise SystemExit(f"execution.ladder.weights missing for profile '{profile}'")
        # Normalize weights defensively
        s = sum(max(0.0, w) for w in weights)
        if s <= 0:
            weights = [1.0 / max(1, len(deltas))] * len(deltas)
        else:
            weights = [max(0.0, w) / s for w in weights]

        L = min(max_levels, max(1, len(deltas)))
        # Liquidity-aware reduction: if ADTV in lots is tiny, collapse to ≤2 levels
        if adtv_lot > 0 and adtv_lot < 20 and L > 2:
            L = 2

        # Compute raw quantities and cap per-level by ADTV
        q_raw = [max(0.0, w * Q) for w in weights[:L]]
        q = [_round_to_lot(x, lot) for x in q_raw]
        cap_shares = max(0.0, adtv_sh * adtv_cap_frac)
        cap_level = _round_to_lot(cap_shares, lot) if cap_shares > 0 else None
        for i in range(len(q)):
            if cap_level is not None and q[i] > cap_level:
                q[i] = int(cap_level)
        # Redistribute remainder + lot rounding (prefer near‑price first)
        used = sum(q)
        remain = max(0, Q - used)
        spins = 0
        while remain >= lot and spins < 64:
            for j in range(len(q)):
                room_ok = True
                if cap_level is not None and (q[j] + lot) > cap_level:
                    room_ok = False
                if room_ok:
                    q[j] += lot
                    remain -= lot
                    if remain < lot:
                        break
            spins += 1

        # Price levels, guards, and π filter
        bundle_seq += 1
        bundle_id = f"B{bundle_seq:03d}-{t}-{side}"
        dropped_reasons: List[str] = []
        lvl_records: List[Dict[str, Any]] = []
        idx_price = to_float(snap_row.get('Price')) if snap_row is not None else None
        floor_tick = to_float(pre_row.get('BandFloor_Tick')) if pre_row is not None else None
        ceil_tick = to_float(pre_row.get('BandCeiling_Tick')) if pre_row is not None else None
        lock = _detect_limit_lock(side, snap_row, idx_price)
        if skip_if_limit_lock and lock:
            dropped.append({
                'Ticker': t, 'Side': side, 'Quantity': int(Q), 'LimitPrice': float(price_ref),
                'Reason': 'limit_lock', 'BundleId': bundle_id,
            })
            debug_recs.append({
                'bundle_id': bundle_id, 'ticker': t, 'side': side,
                'price_ref': price_ref, 'atr_pct': atr_pct, 'levels': [],
                'dropped_reasons': ['limit_lock'],
            })
            continue

        for i in range(L):
            if q[i] <= 0:
                continue
            if method == 'atr_pct' and atr_pct > 0:
                p_i = _mk_price(side, price_ref, deltas[i], atr_pct)
            else:
                step = _tick_for_price(price_ref)
                sgn = -1.0 if side == 'BUY' else +1.0
                p_i = round_to_tick(price_ref + sgn * deltas[i] * step, step)
            # Clip to daily band
            p_i = clip_to_band(p_i, floor_tick, ceil_tick)
            # Guard: no_cross (do not cross market price)
            if no_cross and idx_price is not None:
                if side == 'BUY' and p_i > idx_price:
                    p_i = round_to_tick(idx_price, _tick_for_price(idx_price))
                if side == 'SELL' and p_i < idx_price:
                    p_i = round_to_tick(idx_price, _tick_for_price(idx_price))
            # TTL/reprice per slot
            ttl = int(ttl_list[i] if i < len(ttl_list) else ttl_list[-1])
            reprice = int(reprice_base if side == 'BUY' else -reprice_base)
            dticks = _delta_ticks(side, price_ref, p_i)
            size_ratio = (q[i] / max(1.0, adtv_sh)) if adtv_sh > 0 else 0.0
            regime_score = float(getattr(regime, 'market_score', 0.5) or 0.5)
            pi = estimate_fill_prob(dticks, size_ratio, regime_score)
            # π filter
            if pi < min_fill_prob:
                dropped.append({
                    'Ticker': t, 'Side': side, 'Quantity': int(q[i]), 'LimitPrice': float(p_i),
                    'Reason': 'low_fill_prob', 'BundleId': bundle_id, 'Level': i + 1,
                })
                continue
            note_bits = ["LADDER", f"P_ref={price_ref:.2f}", f"ATR%={(atr_pct*100.0):.2f}", f"Δ={dticks:.1f}t", f"Regime={profile}"]
            if base_note:
                note_bits.append(base_note)
            lvl = Level(
                bundle_id=bundle_id,
                level=i + 1,
                side=side,
                ticker=t,
                qty=int(q[i]),
                limit_price=float(f"{p_i:.2f}"),
                ttl_min=ttl,
                reprice_rule=reprice,
                notes="; ".join(note_bits),
                price_ref=float(price_ref),
                atr_pct=float(atr_pct),
                delta_ticks=float(dticks),
            )
            levels.append(lvl)
            lvl_records.append({
                'level': i + 1,
                'qty': int(q[i]),
                'price': float(p_i),
                'ttl': ttl,
                'reprice': reprice,
                'delta_ticks': float(dticks),
                'pi': float(pi),
            })
        debug_recs.append({
            'bundle_id': bundle_id,
            'ticker': t,
            'side': side,
            'regime_profile': profile,
            'price_ref': price_ref,
            'atr_pct': atr_pct,
            'adtv_shares': adtv_sh,
            'levels': lvl_records,
            'dropped_reasons': dropped_reasons,
        })

    return levels, dropped, debug_recs


def write_ladder_final_csv(levels: List[Level], path: Path) -> None:
    """Write minimal 4-column CSV for direct broker entry.

    Per operator preference, keep orders_final.csv schema as:
    Ticker, Side, Quantity, LimitPrice

    We still emit one row per ladder level, but only the essential fields.
    All diagnostics and per-level metadata are written to orders_quality.csv
    and out/debug/ladder.log.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    # Stable sort: BUY first, then by bundle, then level ascending
    sorted_levels = sorted(levels, key=lambda lv: (lv.side != 'BUY', lv.ticker, lv.bundle_id, lv.level))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Ticker','Side','Quantity','LimitPrice'])
        for lv in sorted_levels:
            w.writerow([lv.ticker, lv.side, int(lv.qty), f"{lv.limit_price:.2f}"])


def build_ladder_quality_rows(
    levels: List[Level],
    *,
    snapshot: pd.DataFrame,
    presets: pd.DataFrame,
    metrics: pd.DataFrame,
    regime: object,
    feats_all: Dict[str, Dict[str, float]] | None = None,
) -> List[List[Any]]:
    # Reuse existing execution metrics heuristic to provide FillRate/Slip estimates
    from scripts.orders.orders_io import _compute_execution_metrics  # type: ignore
    snap = snapshot.set_index('Ticker') if not snapshot.empty else pd.DataFrame()
    pre = presets.set_index('Ticker') if not presets.empty else pd.DataFrame()
    met = metrics.set_index('Ticker') if not metrics.empty else pd.DataFrame()
    pricing_conf = dict(getattr(regime, 'pricing', {}) or {})
    th = dict(getattr(regime, 'thresholds', {}) or {})
    rows: List[List[Any]] = []
    for lv in levels:
        t = lv.ticker
        snap_row = snap.loc[t] if (t in snap.index) else None
        pre_row = pre.loc[t] if (t in pre.index) else None
        market = None
        if snap_row is not None:
            market = to_float(snap_row.get('Price')) or to_float(snap_row.get('P'))
        tick = hose_tick_size(market if market is not None else lv.limit_price)
        ceil_tick = to_float(pre_row.get('BandCeiling_Tick')) if pre_row is not None else None
        floor_tick = to_float(pre_row.get('BandFloor_Tick')) if pre_row is not None else None
        feats = (feats_all or {}).get(t, {}) if feats_all else {}
        # ADTV in lots for size ratio
        adtv_lot = 0.0
        if t in met.index:
            mrow = met.loc[t]
            adtv_k = to_float(mrow.get('AvgTurnover20D_k'))
            if adtv_k is not None and market is not None and market > 0:
                adtv_sh = float(adtv_k) * 1000.0 / float(market)
                adtv_lot = adtv_sh / 100.0
        metrics = _compute_execution_metrics(
            ticker=t,
            side=lv.side,
            qty=int(lv.qty),
            limit=float(lv.limit_price),
            market=market,
            tick=tick,
            ceil_tick=ceil_tick,
            floor_tick=floor_tick,
            feats=feats,
            pricing_conf=pricing_conf,
            thresholds=th,
            snapshot_row=snap_row,
            ticker_overrides=dict(getattr(regime, 'ticker_overrides', {}) or {}),
        )
        fill_rate = float(metrics.get('fill_rate', 1.0) or 1.0)
        slip_bps = float(metrics.get('slip_bps', 0.0) or 0.0)
        alpha_exp = 0.0
        try:
            if lv.side == 'BUY':
                alpha_exp = max(0.0, (lv.price_ref - lv.limit_price) / max(lv.price_ref, 1e-9))
            else:
                alpha_exp = max(0.0, (lv.limit_price - lv.price_ref) / max(lv.price_ref, 1e-9))
        except Exception:
            alpha_exp = 0.0
        rows.append([
            lv.bundle_id, lv.level, lv.side, lv.ticker, int(lv.qty), f"{lv.limit_price:.2f}",
            (f"{market:.2f}" if market is not None else ''),
            f"{estimate_fill_prob(lv.delta_ticks, (lv.qty / max(1.0, adtv_lot)), float(getattr(regime, 'market_score', 0.5) or 0.5)):.3f}",
            f"{fill_rate:.3f}",
            f"{alpha_exp:.3f}",
            f"{slip_bps:.1f}",
            f"{lv.price_ref:.2f}",
            f"{lv.delta_ticks:.1f}",
            f"{lv.atr_pct*100.0:.2f}",
            lv.notes,
        ])
    return rows


def write_ladder_quality_csv(level_rows: List[List[Any]], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'BundleId','Level','Side','Ticker','Qty','LimitPrice','MarketPrice',
            'FillProbExp','FillRateExp','AlphaExp','SlipBpsExp','PriceRef','DeltaTicks','ATRPct','Notes'
        ])
        for r in level_rows:
            w.writerow(r)


def write_ladder_log(debug_records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for rec in debug_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
