from __future__ import annotations

import csv
import math
import os
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Dict, List, Iterator

import pandas as pd

from scripts.utils import to_float, hose_tick_size, detect_session_phase_now_vn, round_to_tick

LOT_SIZE = 100


def _pricing_in_session_now() -> bool:
    """Local-only notion of session for pricing adjustments.

    Treats morning, lunch, afternoon, and ATC as "in session" for the purpose
    of aligning limit prices to the current market. This function is private to
    orders IO and does NOT alter other modules' session semantics.
    """
    try:
        phase = str(detect_session_phase_now_vn()).strip().lower()
    except Exception:
        # If phase detection fails, stay conservative and do not adjust.
        return False
    return phase in {"morning", "lunch", "afternoon", "atc"}


def _adjust_limit_to_market(side: str, limit: float, market: float | None) -> float:
    """Clamp limit price to market during pricing session.

    - BUY: if limit > market, use market
    - SELL: if limit < market, use market
    Only applies when _pricing_in_session_now() is True and market is available.
    """
    if market is None or not _pricing_in_session_now():
        return float(limit)
    if side == 'BUY':
        return float(market) if float(limit) > float(market) else float(limit)
    # SELL
    return float(market) if float(limit) < float(market) else float(limit)


@contextmanager
def _open_for_write(path: Path, mode: str = 'w', *, newline: str | None = None) -> Iterator[StringIO | object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, newline=newline, encoding='utf-8') as handle:
        yield handle


def _as_series(df: pd.DataFrame, ticker: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if ticker not in df.index:
        return None
    row = df.loc[ticker]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _snap_value(row: pd.Series | None, columns: List[str]) -> float | None:
    if row is None:
        return None
    for col in columns:
        if col in row.index:
            val = to_float(row.get(col))
            if val is not None:
                return float(val)
    return None


def _detect_limit_lock(side: str, snap_row: pd.Series | None, market: float | None, *, ceil_tick: float | None, floor_tick: float | None, tick: float) -> bool:
    if snap_row is None or market is None:
        return False
    tol = max(tick, 1e-6)
    if side == 'BUY':
        ceil_val = _snap_value(snap_row, ['Ceiling', 'CeilingPrice', 'Ceiling_Tick', 'CeilingTick'])
        if ceil_val is None:
            ceil_val = ceil_tick
        if ceil_val is None:
            return False
        high_px = _snap_value(snap_row, ['High', 'DayHigh', 'HighPrice'])
        close_px = _snap_value(snap_row, ['Close', 'Price', 'Last'])
        if high_px is not None and abs(high_px - ceil_val) <= tol and market >= ceil_val - tol:
            return True
        if close_px is not None and abs(close_px - ceil_val) <= tol and market >= ceil_val - tol:
            return True
        return False
    floor_val = _snap_value(snap_row, ['Floor', 'FloorPrice', 'Floor_Tick', 'FloorTick'])
    if floor_val is None:
        floor_val = floor_tick
    if floor_val is None:
        return False
    low_px = _snap_value(snap_row, ['Low', 'DayLow', 'LowPrice'])
    close_px = _snap_value(snap_row, ['Close', 'Price', 'Last'])
    if low_px is not None and abs(low_px - floor_val) <= tol and market <= floor_val + tol:
        return True
    if close_px is not None and abs(close_px - floor_val) <= tol and market <= floor_val + tol:
        return True
    return False


def _compute_execution_metrics(
    *,
    ticker: str,
    side: str,
    qty: int,
    limit: float,
    market: float | None,
    tick: float,
    ceil_tick: float | None,
    floor_tick: float | None,
    feats: Dict,
    pricing_conf: Dict,
    thresholds: Dict,
    snapshot_row: pd.Series | None,
    ticker_overrides: Dict[str, Dict[str, object]] | None = None,
) -> Dict[str, object]:
    fill_conf = dict(pricing_conf.get('fill_prob', {}) or {})
    for req in ('base', 'cross', 'near_ceiling', 'min', 'decay_scale_min_ticks'):
        if req not in fill_conf or fill_conf[req] is None:
            raise SystemExit(f"Missing pricing.fill_prob.{req} in policy")
    fill = float(fill_conf['base'])
    notes: List[str] = []
    atr_pct = float(feats.get('atr_pct', 0.0) or 0.0)
    price_ref = market if market is not None else limit
    dist_ticks = 0.0
    if market is not None:
        if not thresholds or thresholds.get('near_ceiling_pct') is None:
            raise SystemExit("Missing thresholds.near_ceiling_pct in policy")
        near_thr = float(thresholds.get('near_ceiling_pct'))
        if ceil_tick is not None and market >= (near_thr * ceil_tick):
            fill = float(fill_conf['near_ceiling'])
            notes.append('near_ceiling')
        else:
            dist_ticks = abs(market - limit) / max(tick, 1e-6)
            if side == 'BUY':
                if limit >= market:
                    fill = float(fill_conf['cross'])
                else:
                    atr_ticks = (atr_pct * market) / max(tick, 1e-6) if market > 0 else 0.0
                    scale = max(float(fill_conf['decay_scale_min_ticks']), atr_ticks, 1e-6)
                    fill = max(float(fill_conf['min']), math.exp(-dist_ticks / scale))
            else:
                if limit <= market:
                    fill = float(fill_conf['cross'])
                else:
                    atr_ticks = (atr_pct * market) / max(tick, 1e-6) if market > 0 else 0.0
                    scale = max(float(fill_conf['decay_scale_min_ticks']), atr_ticks, 1e-6)
                    fill = max(float(fill_conf['min']), math.exp(-dist_ticks / scale))
    limit_lock = _detect_limit_lock(side, snapshot_row, market, ceil_tick=ceil_tick, floor_tick=floor_tick, tick=tick)
    if limit_lock:
        fill = 0.0
        notes.append('limit_lock')
    adtv_k = float(feats.get('adtv20_k', 0.0) or 0.0)
    avg_volume = 0.0
    if adtv_k > 0.0 and price_ref and price_ref > 0.0:
        avg_volume = adtv_k / price_ref
    kappa = float(fill_conf.get('partial_fill_kappa', 0.65) or 0.65)
    kappa = max(0.0, min(1.0, kappa))
    min_notional = float(fill_conf.get('min_fill_notional_vnd', 0.0) or 0.0)
    min_shares = float(LOT_SIZE)
    if price_ref and price_ref > 0.0 and min_notional > 0.0:
        min_shares = max(min_shares, min_notional / (price_ref * 1000.0))
    denom = max(min_shares, float(qty)) if qty else min_shares
    if avg_volume > 0.0 and denom > 0.0:
        fill_rate = max(0.0, min(1.0, kappa * (avg_volume / denom)))
    else:
        fill_rate = 1.0
    slip_conf = dict(pricing_conf.get('slippage_model', {}) or {})
    alpha = float(slip_conf.get('alpha_bps', 0.0) or 0.0)
    beta_dist = float(slip_conf.get('beta_dist_per_tick', 0.0) or 0.0)
    beta_size = float(slip_conf.get('beta_size', 0.0) or 0.0)
    beta_vol = float(slip_conf.get('beta_vol', 0.0) or 0.0)
    size_ratio = float(qty) / avg_volume if avg_volume > 0.0 else 0.0
    slip_bps = alpha + beta_dist * dist_ticks + beta_size * size_ratio + beta_vol * (atr_pct * 100.0)
    mae_floor = float(slip_conf.get('mae_bps', 0.0) or 0.0)
    if ticker_overrides and ticker in ticker_overrides:
        ov = ticker_overrides.get(ticker) or {}
        if isinstance(ov, dict) and ov.get('mae_bps') is not None:
            try:
                mae_floor = max(mae_floor, float(ov.get('mae_bps')))
            except Exception:
                mae_floor = mae_floor
    slip_bps = float(max(mae_floor, slip_bps, 0.0))
    return {
        'fill_prob_price': max(0.0, min(1.0, float(fill))),
        'fill_rate': max(0.0, min(1.0, float(fill_rate))),
        'slip_bps': slip_bps,
        'limit_lock': bool(limit_lock),
        'notes': notes,
        'distance_ticks': float(dist_ticks),
        'size_ratio': float(size_ratio),
    }


def print_orders(orders, snapshot: pd.DataFrame) -> str:
    snap = snapshot.set_index("Ticker")
    lines = []
    total_buy = 0.0; total_sell = 0.0
    for o in orders:
        market_price = None
        if o.ticker in snap.index:
            market_price = to_float(snap.loc[o.ticker].get("Price")) or to_float(snap.loc[o.ticker].get("P"))
        # Use effective limit for display when pricing session rules apply
        limit_eff = _adjust_limit_to_market(o.side, float(o.limit_price), market_price)
        mp_str = f"{market_price:.2f}" if market_price is not None else "NA"
        line = f"{o.ticker} — LO — {o.quantity} — Giá đặt: {limit_eff:.2f} (nghìn) — Giá thị trường: {mp_str} (nghìn) — {o.note}"
        lines.append(line)
        if o.side == "BUY": total_buy += o.quantity * limit_eff
        else: total_sell += o.quantity * limit_eff
    lines += ["", "BẢNG TỔNG HỢP TIỀN:", f"• Tổng mua: {total_buy:.0f} (nghìn)", f"• Tổng bán: {total_sell:.0f} (nghìn)", f"• Net (mua − bán): {total_buy - total_sell:.0f} (nghìn)", "• Gợi ý: nếu cần xem bằng VND, nhân 1.000 ngoài phạm vi prompt."]
    return "\n".join(lines)


def write_orders_csv(
    orders,
    path: Path,
    *,
    snapshot: pd.DataFrame | None = None,
    presets: pd.DataFrame | None = None,
    regime: dict | object | None = None,
    feats_all: dict | None = None,
    scores: dict | None = None,
) -> None:
    """Write orders to CSV.

    Backward compatible: if no market context is provided, write 4 columns
    (Ticker,Side,Quantity,LimitPrice). If context is provided, append useful
    execution/profitability fields to help operator prioritize input.
    """
    enrich = snapshot is not None and presets is not None and feats_all is not None and scores is not None
    snap = snapshot.set_index("Ticker") if (enrich and not snapshot.empty) else pd.DataFrame()
    pre = presets.set_index("Ticker") if (enrich and not presets.empty) else pd.DataFrame()
    th = {}
    if enrich and regime is not None:
        try:
            th = dict(getattr(regime, 'thresholds', {}) or {})
        except Exception:
            th = {}
    with _open_for_write(path, "w", newline="") as f:
        w = csv.writer(f)
        # New behavior: always write a minimal, stable schema to avoid mis-entry.
        # Columns: Ticker, Side, Quantity, LimitPrice
        # Sorting is still determined internally by Priority (not shown here).
        w.writerow(["Ticker", "Side", "Quantity", "LimitPrice"]) 
        # If we have context, compute internal priority for sorting only.
        rows = []
        # Transaction cost (round-trip) fraction from policy (pricing.tc_roundtrip_frac)
        pricing_conf = dict(getattr(regime, 'pricing', {}) or {})
        if 'tc_roundtrip_frac' not in pricing_conf:
            raise SystemExit("Missing pricing.tc_roundtrip_frac in policy")
        try:
            tc = float(pricing_conf['tc_roundtrip_frac'])
        except Exception as exc:
            raise SystemExit(f"Invalid pricing.tc_roundtrip_frac: {exc}") from exc
        ticker_overrides = dict(getattr(regime, 'ticker_overrides', {}) or {})
        for o in orders:
            t = o.ticker; side = o.side; qty = o.quantity; limit = float(o.limit_price)
            market = None
            snap_row = _as_series(snap, t) if enrich else None
            pre_row = _as_series(pre, t) if enrich else None
            if snap_row is not None:
                market = to_float(snap_row.get("Price")) or to_float(snap_row.get("P"))
            # Adjust limit to market during pricing session without affecting other modules
            limit_eff = _adjust_limit_to_market(side, limit, market)
            prio_raw = 0.0
            prio_net = 0.0
            if enrich:
                tick = hose_tick_size(market if market is not None else limit)
                ceil_tick = to_float(pre_row.get('BandCeiling_Tick')) or to_float(pre_row.get('BandCeilingRaw')) if pre_row is not None else None
                floor_tick = to_float(pre_row.get('BandFloor_Tick')) or to_float(pre_row.get('BandFloorRaw')) if pre_row is not None else None
                feats = feats_all.get(t, {}) if feats_all else {}
                metrics = _compute_execution_metrics(
                    ticker=t,
                    side=side,
                    qty=qty,
                    limit=limit_eff,
                    market=market,
                    tick=tick,
                    ceil_tick=ceil_tick,
                    floor_tick=floor_tick,
                    feats=feats,
                    pricing_conf=pricing_conf,
                    thresholds=th,
                    snapshot_row=snap_row,
                    ticker_overrides=ticker_overrides,
                )
                fill_prob_price = float(metrics.get('fill_prob_price', 0.0))
                fill_rate = float(metrics.get('fill_rate', 1.0))
                slip_bps = float(metrics.get('slip_bps', 0.0))
                raw_score = max(0.0, float(scores.get(t, 0.0) if scores else 0.0))
                fill_combined = max(0.0, fill_prob_price * fill_rate)
                prio_raw = raw_score * fill_combined
                slip_frac = max(0.0, slip_bps / 10000.0)
                prio_net = prio_raw * max(0.0, 1.0 - tc)
                if slip_frac > 0.0:
                    prio_net = prio_net / (1.0 + slip_frac)
            rows.append([t, side, qty, f"{limit_eff:.2f}", prio_net, prio_raw])
        # Sort BUY first, then by net priority (desc). Column not written.
        rows.sort(
            key=lambda r: (
                r[1] != 'BUY',
                -float(r[4]) if isinstance(r[4], (int, float, str)) else 0.0,
                -float(r[5]) if isinstance(r[5], (int, float, str)) else 0.0,
            )
        )
        for t, side, qty, limit_str, _prio_net, _prio_raw in rows:
            w.writerow([t, side, qty, limit_str])


def write_orders_reasoning(actions: dict, scores: dict, feats_all: dict, path: Path) -> None:
    with _open_for_write(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ticker", "Action", "Score", "above_ma20", "above_ma50", "rsi", "macdh_pos", "liq_norm", "atr_pct", "pnl_pct"])
        for t, a in actions.items():
            feats = feats_all.get(t, {})
            w.writerow([
                t, a,
                f"{scores.get(t, 0.0):.3f}",
                feats.get("above_ma20", 0.0),
                feats.get("above_ma50", 0.0),
                f"{feats.get('rsi', 0.0):.1f}",
                feats.get("macdh_pos", 0.0),
                f"{feats.get('liq_norm', 0.0):.2f}",
                f"{feats.get('atr_pct', 0.0):.2f}",
                f"{feats.get('pnl_pct', 0.0):.3f}",
            ])


def write_orders_quality(orders, snapshot: pd.DataFrame, presets: pd.DataFrame, regime: dict | object, feats_all: dict, scores: dict, path: Path) -> None:
    """Write per-order execution/profitability heuristics to CSV to help prioritization."""
    snap = snapshot.set_index("Ticker") if not snapshot.empty else pd.DataFrame()
    pre = presets.set_index("Ticker") if not presets.empty else pd.DataFrame()
    ttl_override_map = dict(getattr(regime, 'ttl_overrides', {}) or {})
    try:
        idx_atr_pctile = float(getattr(regime, 'index_atr_percentile', 0.5) or 0.5)
    except Exception:
        idx_atr_pctile = 0.5
    try:
        th = dict(getattr(regime, 'thresholds', {}) or {})
    except Exception:
        th = {}
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    pricing_conf = dict(getattr(regime, 'pricing', {}) or {})
    if 'tc_roundtrip_frac' not in pricing_conf:
        raise SystemExit("Missing pricing.tc_roundtrip_frac in policy")
    try:
        tc = float(pricing_conf['tc_roundtrip_frac'])
    except Exception as exc:
        raise SystemExit(f"Invalid pricing.tc_roundtrip_frac: {exc}") from exc
    ticker_overrides = dict(getattr(regime, 'ticker_overrides', {}) or {})
    for o in orders:
        t = o.ticker
        side = o.side
        qty = int(o.quantity)
        limit = float(o.limit_price)
        snap_row = _as_series(snap, t)
        pre_row = _as_series(pre, t)
        market = None
        if snap_row is not None:
            market = to_float(snap_row.get('Price')) or to_float(snap_row.get('P'))
        tick = hose_tick_size(market if market is not None else limit)
        ceil_tick = to_float(pre_row.get('BandCeiling_Tick')) or to_float(pre_row.get('BandCeilingRaw')) if pre_row is not None else None
        floor_tick = to_float(pre_row.get('BandFloor_Tick')) or to_float(pre_row.get('BandFloorRaw')) if pre_row is not None else None
        feats = feats_all.get(t, {}) or {}
        exec_metrics = _compute_execution_metrics(
            ticker=t,
            side=side,
            qty=qty,
            limit=limit,
            market=market,
            tick=tick,
            ceil_tick=ceil_tick,
            floor_tick=floor_tick,
            feats=feats,
            pricing_conf=pricing_conf,
            thresholds=th,
            snapshot_row=snap_row,
            ticker_overrides=ticker_overrides,
        )
        fill_prob_price = float(exec_metrics.get('fill_prob_price', 0.0))
        fill_rate = float(exec_metrics.get('fill_rate', 1.0))
        slip_bps = float(exec_metrics.get('slip_bps', 0.0))
        limit_lock = bool(exec_metrics.get('limit_lock', False))
        notes = list(exec_metrics.get('notes', []) or [])
        # Expected R (TP/SL ratio) from thresholds & ATR
        exp_r = 0.0
        exp_r_net = 0.0
        try:
            tp_pct = th.get('tp_pct'); sl_pct = th.get('sl_pct')
            tp_floor = float(th.get('tp_floor_pct') or 0.0)
            sl_floor = float(th.get('sl_floor_pct') or 0.0)
            tp_atr_mult = th.get('tp_atr_mult'); sl_atr_mult = th.get('sl_atr_mult')
            atr_pct = float(feats.get('atr_pct', 0.0) or 0.0)
            dyn_tp = (float(tp_atr_mult) * atr_pct) if (tp_atr_mult is not None and atr_pct > 0) else None
            dyn_sl = (float(sl_atr_mult) * atr_pct) if (sl_atr_mult is not None and atr_pct > 0) else None
            vals_tp = [v for v in [tp_pct, dyn_tp] if v is not None and float(v) > 0]
            vals_sl = [v for v in [sl_pct, dyn_sl] if v is not None and float(v) > 0]
            tp_eff = min(vals_tp) if vals_tp else (tp_floor if tp_floor > 0 else None)
            sl_eff = min(vals_sl) if vals_sl else (sl_floor if sl_floor > 0 else None)
            if tp_eff and sl_eff and sl_eff > 0:
                tp_net = max(0.0, float(tp_eff) - 2.0 * tc)
                sl_net = max(1e-9, float(sl_eff) + 2.0 * tc)
                exp_r = float(tp_net) / float(sl_net)
                slip_frac = max(0.0, slip_bps / 10000.0)
                tp_net_adj = max(0.0, float(tp_eff) - 2.0 * tc - slip_frac)
                sl_net_adj = max(1e-9, float(sl_eff) + 2.0 * tc + slip_frac)
                if sl_net_adj > 0:
                    exp_r_net = max(0.0, tp_net_adj / sl_net_adj)
        except Exception:
            exp_r = 0.0
            exp_r_net = 0.0
        raw_score = max(0.0, float(scores.get(t, 0.0) or 0.0))
        fill_combined = max(0.0, fill_prob_price * fill_rate)
        priority_raw = raw_score * fill_combined
        slip_frac = max(0.0, slip_bps / 10000.0)
        priority_net = priority_raw * max(0.0, 1.0 - tc)
        if slip_frac > 0.0:
            priority_net = priority_net / (1.0 + slip_frac)
        # TTL suggestion based on index volatility percentile
        try:
            mf_conf = getattr(regime, 'market_filter', {}) or {}
            if 'index_atr_soft_pct' not in mf_conf or 'index_atr_hard_pct' not in mf_conf:
                raise SystemExit("Missing market_filter.index_atr_soft_pct/hard_pct in policy")
            soft = float(mf_conf['index_atr_soft_pct']); hard = float(mf_conf['index_atr_hard_pct'])
            ou_conf = dict(getattr(regime, 'orders_ui', {}) or {})
            ttl_conf = dict(ou_conf.get('ttl_minutes', {}) or {})
            bucket_conf = ou_conf.get('ttl_bucket_minutes', {}) or {}
            bucket_state = ou_conf.get('ttl_bucket_state', {}) or {}
            bucket_label = bucket_state.get('current') if isinstance(bucket_state, dict) else None
            if bucket_label and isinstance(bucket_conf, dict) and bucket_label in bucket_conf:
                bucket_vals = bucket_conf.get(bucket_label) or {}
                for key in ('base', 'soft', 'hard'):
                    if key in bucket_vals:
                        ttl_conf[key] = bucket_vals[key]
            if not {'base', 'soft', 'hard'}.issubset(ttl_conf):
                raise SystemExit("Missing orders_ui.ttl_minutes.{base,soft,hard} in policy")
            ttl_base = int(ttl_conf['base']); ttl_soft = int(ttl_conf['soft']); ttl_hard = int(ttl_conf['hard'])
            ttl = ttl_hard if idx_atr_pctile >= hard else (ttl_soft if idx_atr_pctile >= soft else ttl_base)
            if t in ttl_override_map:
                try:
                    ttl = int(ttl_override_map[t])
                except Exception:
                    ttl = ttl
        except Exception as exc:
            raise SystemExit(f"Failed TTL computation: {exc}") from exc
        # Derived signal from note
        sig = ''
        try:
            note = getattr(o, 'note', '') or ''
            s = str(note)
            if 'Chốt lời' in s or 'Take Profit' in s:
                sig = 'TP'
            elif 'Bán toàn bộ' in s or 'Exit' in s:
                sig = 'EXIT'
            elif 'Bán bớt' in s or 'Giảm' in s:
                sig = 'TRIM'
            elif 'Mua mới' in s:
                sig = 'NEW'
            elif 'Mua gia tăng' in s or 'bổ sung' in s:
                sig = 'ADD'
        except Exception:
            sig = ''
        if limit_lock and 'limit_lock' not in notes:
            notes.append('limit_lock')
        rows.append({
            'Ticker': t,
            'Side': side,
            'Quantity': qty,
            'LimitPrice': f"{limit:.2f}",
            'MarketPrice': (f"{market:.2f}" if market is not None else ''),
            'FillProb': round(fill_prob_price, 3),
            'FillRateExp': round(fill_rate, 3),
            'ExpR': round(float(exp_r), 2),
            'exp_r_net': round(float(exp_r_net), 2),
            'Priority': round(priority_raw, 3),
            'priority_net': round(priority_net, 3),
            'TTL_Min': int(ttl),
            'SlipBps': round(slip_bps, 1),
            'SlipPct': round(slip_bps / 10000.0, 4),
            'Signal': sig,
            'LimitLock': 'Y' if limit_lock else '',
            'Notes': ';'.join(sorted(set(notes))),
        })
    df = pd.DataFrame(rows)
    try:
        df['__side_key'] = (df['Side'] != 'BUY').astype(int)
        sort_cols = ['__side_key', 'priority_net', 'Priority'] if 'priority_net' in df.columns else ['__side_key', 'Priority']
        df.sort_values(by=sort_cols, ascending=[True, False, False][:len(sort_cols)], inplace=True)
        df.drop(columns=['__side_key'], inplace=True)
    except Exception:
        pass
    if _test_mode_enabled():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)





def write_orders_csv_enriched(
    orders,
    path: Path,
    *,
    snapshot: pd.DataFrame,
    presets: pd.DataFrame,
    regime: dict | object,
    feats_all: dict,
    scores: dict,
) -> None:
    """Write enriched orders CSV with execution/profitability fields."""
    if snapshot is None or presets is None or feats_all is None or scores is None:
        raise SystemExit("write_orders_csv_enriched requires market context (snapshot/presets/feats/scores)")
    snap = snapshot.set_index("Ticker") if not snapshot.empty else pd.DataFrame()
    pre = presets.set_index("Ticker") if not presets.empty else pd.DataFrame()
    try:
        th = dict(getattr(regime, 'thresholds', {}) or {})
    except Exception:
        th = {}
    pricing_conf = dict(getattr(regime, 'pricing', {}) or {})
    if 'tc_roundtrip_frac' not in pricing_conf:
        raise SystemExit("Missing pricing.tc_roundtrip_frac in policy")
    try:
        tc = float(pricing_conf['tc_roundtrip_frac'])
    except Exception as exc:
        raise SystemExit(f"Invalid pricing.tc_roundtrip_frac: {exc}") from exc
    idx_atr_pctile = float(getattr(regime, 'index_atr_percentile', 0.5) or 0.5)
    mf_conf = getattr(regime, 'market_filter', {}) or {}
    if 'index_atr_soft_pct' not in mf_conf or 'index_atr_hard_pct' not in mf_conf:
        raise SystemExit("Missing market_filter.index_atr_soft_pct/hard_pct in policy")
    soft_thr = float(mf_conf['index_atr_soft_pct']); hard_thr = float(mf_conf['index_atr_hard_pct'])
    ou_conf = dict(getattr(regime, 'orders_ui', {}) or {})
    ttl_conf = dict(ou_conf.get('ttl_minutes', {}) or {})
    bucket_conf = ou_conf.get('ttl_bucket_minutes', {}) or {}
    bucket_state = ou_conf.get('ttl_bucket_state', {}) or {}
    bucket_label = bucket_state.get('current') if isinstance(bucket_state, dict) else None
    if bucket_label and isinstance(bucket_conf, dict) and bucket_label in bucket_conf:
        bucket_vals = bucket_conf.get(bucket_label) or {}
        for key in ('base', 'soft', 'hard'):
            if key in bucket_vals:
                ttl_conf[key] = bucket_vals[key]
    if not {'base','soft','hard'}.issubset(ttl_conf):
        raise SystemExit("Missing orders_ui.ttl_minutes.{base,soft,hard} in policy")
    ttl_base = int(ttl_conf['base']); ttl_soft = int(ttl_conf['soft']); ttl_hard = int(ttl_conf['hard'])
    with _open_for_write(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Ticker", "Side", "Quantity", "LimitPrice", "MarketPrice",
            "FillProb", "FillRateExp", "ExpR", "exp_r_net", "Priority", "priority_net", "TTL_Min",
            "SlipBps", "Signal", "LimitLock", "Notes",
        ])
        rows = []
        ticker_overrides = dict(getattr(regime, 'ticker_overrides', {}) or {})
        for o in orders:
            t = o.ticker; side = o.side; qty = int(o.quantity); limit = float(o.limit_price)
            snap_row = _as_series(snap, t)
            pre_row = _as_series(pre, t)
            market = None
            if snap_row is not None:
                market = to_float(snap_row.get('Price')) or to_float(snap_row.get('P'))
            # Adjust limit to market during pricing session
            limit_eff = _adjust_limit_to_market(side, limit, market)
            tick = hose_tick_size(market if market is not None else limit)
            ceil_tick = to_float(pre_row.get('BandCeiling_Tick')) or to_float(pre_row.get('BandCeilingRaw')) if pre_row is not None else None
            floor_tick = to_float(pre_row.get('BandFloor_Tick')) or to_float(pre_row.get('BandFloorRaw')) if pre_row is not None else None
            feats = feats_all.get(t, {}) if feats_all else {}
            metrics_exec = _compute_execution_metrics(
                ticker=t,
                side=side,
                qty=qty,
                limit=limit_eff,
                market=market,
                tick=tick,
                ceil_tick=ceil_tick,
                floor_tick=floor_tick,
                feats=feats,
                pricing_conf=pricing_conf,
                thresholds=th,
                snapshot_row=snap_row,
                ticker_overrides=ticker_overrides,
            )
            fill_prob_price = float(metrics_exec.get('fill_prob_price', 0.0))
            fill_rate = float(metrics_exec.get('fill_rate', 1.0))
            slip_bps = float(metrics_exec.get('slip_bps', 0.0))
            limit_lock = bool(metrics_exec.get('limit_lock', False))
            notes = list(metrics_exec.get('notes', []) or [])
            exp_r = 0.0
            exp_r_net = 0.0
            try:
                tp_pct = th.get('tp_pct'); sl_pct = th.get('sl_pct')
                tp_floor = float(th.get('tp_floor_pct') or 0.0)
                sl_floor = float(th.get('sl_floor_pct') or 0.0)
                tp_atr_mult = th.get('tp_atr_mult'); sl_atr_mult = th.get('sl_atr_mult')
                atr_pct = float(feats.get('atr_pct', 0.0) or 0.0)
                dyn_tp = (float(tp_atr_mult) * atr_pct) if (tp_atr_mult is not None and atr_pct > 0) else None
                dyn_sl = (float(sl_atr_mult) * atr_pct) if (sl_atr_mult is not None and atr_pct > 0) else None
                vals_tp = [v for v in [tp_pct, dyn_tp] if v is not None and float(v) > 0]
                vals_sl = [v for v in [sl_pct, dyn_sl] if v is not None and float(v) > 0]
                tp_eff = min(vals_tp) if vals_tp else (tp_floor if tp_floor > 0 else None)
                sl_eff = min(vals_sl) if vals_sl else (sl_floor if sl_floor > 0 else None)
                if tp_eff and sl_eff and sl_eff > 0:
                    tp_net = max(0.0, float(tp_eff) - 2.0 * tc)
                    sl_net = max(1e-9, float(sl_eff) + 2.0 * tc)
                    exp_r = float(tp_net) / float(sl_net)
                    slip_frac = max(0.0, slip_bps / 10000.0)
                    tp_net_adj = max(0.0, float(tp_eff) - 2.0 * tc - slip_frac)
                    sl_net_adj = max(1e-9, float(sl_eff) + 2.0 * tc + slip_frac)
                    if sl_net_adj > 0:
                        exp_r_net = max(0.0, tp_net_adj / sl_net_adj)
            except Exception:
                exp_r = 0.0
                exp_r_net = 0.0
            raw_score = max(0.0, float(scores.get(t, 0.0) if scores else 0.0))
            fill_combined = max(0.0, fill_prob_price * fill_rate)
            priority_raw = raw_score * fill_combined
            slip_frac = max(0.0, slip_bps / 10000.0)
            priority_net = priority_raw * max(0.0, 1.0 - tc)
            if slip_frac > 0.0:
                priority_net = priority_net / (1.0 + slip_frac)
            ttl = ttl_hard if idx_atr_pctile >= hard_thr else (ttl_soft if idx_atr_pctile >= soft_thr else ttl_base)
            sig = ''
            try:
                note = getattr(o, 'note', '') or ''
                s = str(note)
                if 'Chốt lời' in s or 'Take Profit' in s:
                    sig = 'TP'
                elif 'Bán toàn bộ' in s or 'Exit' in s:
                    sig = 'EXIT'
                elif 'Bán bớt' in s or 'Giảm' in s:
                    sig = 'TRIM'
                elif 'Mua mới' in s:
                    sig = 'NEW'
                elif 'Mua gia tăng' in s or 'bổ sung' in s:
                    sig = 'ADD'
            except Exception:
                sig = ''
            if limit_lock and 'limit_lock' not in notes:
                notes.append('limit_lock')
            rows.append([
                t,
                side,
                qty,
                f"{limit_eff:.2f}",
                (f"{market:.2f}" if market is not None else ''),
                f"{fill_prob_price:.3f}",
                f"{fill_rate:.3f}",
                f"{float(exp_r):.2f}",
                f"{float(exp_r_net):.2f}",
                f"{priority_raw:.3f}",
                f"{priority_net:.3f}",
                ttl,
                f"{slip_bps:.1f}",
                sig,
                'Y' if limit_lock else '',
                ';'.join(sorted(set(notes))),
            ])
        def _parse_prio(row):
            try:
                return float(row[10])
            except Exception:
                return 0.0
        rows.sort(key=lambda r: (r[1] != 'BUY', -_parse_prio(r)))
        for r in rows:
            w.writerow(r)





def write_orders_analysis(lines: list[str], path: Path) -> None:
    with _open_for_write(path, "w") as handle:
        handle.write("\n".join(lines))


def write_text_lines(lines: list[str], path: Path) -> None:
    """Write arbitrary suggestion/analysis text lines to a file.
    Fail fast on invalid input to keep pipeline behavior explicit.
    """
    if lines is None or not isinstance(lines, list):
        raise ValueError("lines must be a list[str]")
    for i, line in enumerate(lines):
        if not isinstance(line, str):
            raise ValueError(f"lines[{i}] must be str")
    with _open_for_write(path, "w") as handle:
        handle.write("\n".join(lines))
