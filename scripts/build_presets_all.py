from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from scripts.utils import detect_session_phase_now_vn, hose_tick_size

# Runtime-injected daily band (set once at engine start). Avoids per-module policy reads.
BAND_PCT_RUNTIME: float | None = None

def set_daily_band_pct(value: float) -> None:
    global BAND_PCT_RUNTIME
    BAND_PCT_RUNTIME = float(value)


VN_TZ = timezone(timedelta(hours=7))

# Copied/adapted from prep version; unchanged logic
def _hose_tick_thousand(p_thousand: float) -> float:
    if pd.isna(p_thousand):
        return np.nan
    return hose_tick_size(float(p_thousand))


def _round_down_tick_thousand(p_thousand: float) -> float:
    if pd.isna(p_thousand):
        return np.nan
    tick = _hose_tick_thousand(p_thousand)
    return float(np.floor(p_thousand / tick) * tick)


def _round_up_tick_thousand(p_thousand: float) -> float:
    if pd.isna(p_thousand):
        return np.nan
    tick = _hose_tick_thousand(p_thousand)
    return float(np.ceil(p_thousand / tick) * tick)


def _round_near_tick_thousand(p_thousand: float) -> float:
    if pd.isna(p_thousand):
        return np.nan
    tick = _hose_tick_thousand(p_thousand)
    lower = np.floor(p_thousand / tick) * tick
    upper = lower + tick
    return float(upper if (p_thousand - lower) >= (upper - p_thousand) else lower)


def _build_row(t: str, px: float, ref_px: float, ma10: float, ma20: float, ma50: float, bb_up: float, bb_low: float, atr: float, *, daily_band_pct: float):
    if daily_band_pct is None:
        raise SystemExit("daily_band_pct is required (provide from central policy at engine start)")
    band_pct = float(daily_band_pct)
    band_floor = ref_px * (1.0 - band_pct) if pd.notna(ref_px) else np.nan
    band_ceil = ref_px * (1.0 + band_pct) if pd.notna(ref_px) else np.nan
    band_floor_tick = _round_up_tick_thousand(band_floor) if pd.notna(band_floor) else np.nan
    band_ceil_tick = _round_down_tick_thousand(band_ceil) if pd.notna(band_ceil) else np.nan
    bullish = pd.notna(ma20) and pd.notna(ma50) and (ma20 > ma50)
    z = ((px - ma20) / atr) if (pd.notna(px) and pd.notna(ma20) and pd.notna(atr) and atr > 0) else np.nan

    def compute_buy(off1: float, off2: float, z_thresh: float):
        if bullish and pd.notna(z) and z > z_thresh and pd.notna(ma10):
            b1 = ma10 - off1
            b2 = ma20 - off2
        else:
            b1 = ma20 - off1 if pd.notna(ma20) else np.nan
            b2 = ma20 - off2 if pd.notna(ma20) else np.nan
        return b1, b2

    if pd.isna(atr) or atr <= 0:
        off_c1 = (ma20 * 0.005) if pd.notna(ma20) else np.nan
        off_c2 = (ma20 * 0.0125) if pd.notna(ma20) else np.nan
        off_cs1 = (bb_up * 0.005) if pd.notna(bb_up) else np.nan
        off_cs2 = (bb_up * 0.0125) if pd.notna(bb_up) else np.nan
    else:
        off_c1, off_c2 = 0.10 * atr, 0.25 * atr
        off_cs1, off_cs2 = 0.10 * atr, 0.25 * atr
    cons_b1_raw, cons_b2_raw = compute_buy(off_c1, off_c2, z_thresh=0.6)
    cons_s1_raw = (bb_up - off_cs1) if pd.notna(bb_up) else np.nan
    cons_s2_raw = (bb_up - off_cs2) if pd.notna(bb_up) else np.nan

    if pd.isna(atr) or atr <= 0:
        off_b1 = (ma20 * 0.008) if pd.notna(ma20) else np.nan
        off_b2 = (ma20 * 0.020) if pd.notna(ma20) else np.nan
        off_bs1 = (bb_up * 0.008) if pd.notna(bb_up) else np.nan
        off_bs2 = (bb_up * 0.020) if pd.notna(bb_up) else np.nan
    else:
        off_b1, off_b2 = 0.08 * atr, 0.20 * atr
        off_bs1, off_bs2 = 0.08 * atr, 0.20 * atr
    bal_b1_raw, bal_b2_raw = compute_buy(off_b1, off_b2, z_thresh=0.4)
    bal_s1_raw = (bb_up - off_bs1) if pd.notna(bb_up) else np.nan
    bal_s2_raw = (bb_up - off_bs2) if pd.notna(bb_up) else np.nan

    if pd.isna(atr) or atr <= 0:
        off_a1 = (ma20 * 0.005) if pd.notna(ma20) else np.nan
        off_a2 = (ma20 * 0.015) if pd.notna(ma20) else np.nan
        off_as1 = (bb_up * 0.005) if pd.notna(bb_up) else np.nan
        off_as2 = (bb_up * 0.015) if pd.notna(bb_up) else np.nan
    else:
        off_a1, off_a2 = 0.05 * atr, 0.15 * atr
        off_as1, off_as2 = 0.05 * atr, 0.15 * atr
    aggr_b1_raw, aggr_b2_raw = compute_buy(off_a1, off_a2, z_thresh=0.2)
    aggr_s1_raw = (bb_up - off_as1) if pd.notna(bb_up) else np.nan
    aggr_s2_raw = (bb_up - off_as2) if pd.notna(bb_up) else np.nan

    if pd.isna(atr) or atr <= 0:
        off_brk_b1 = (bb_up * 0.006) if pd.notna(bb_up) else np.nan
        off_brk_b2 = (ma20 * 0.01) if pd.notna(ma20) else np.nan
        off_brk_s1 = (bb_up * 0.015) if pd.notna(bb_up) else np.nan
        off_brk_s2 = (bb_up * 0.030) if pd.notna(bb_up) else np.nan
    else:
        off_brk_b1 = 0.05 * atr
        off_brk_b2 = 0.10 * atr
        off_brk_s1 = 0.25 * atr
        off_brk_s2 = 0.40 * atr
    brk_b1_raw = (bb_up + off_brk_b1) if pd.notna(bb_up) else (px + off_brk_b1 if pd.notna(px) else np.nan)
    brk_b2_raw = (ma20 + off_brk_b2) if pd.notna(ma20) else np.nan
    brk_s1_raw = (bb_up + off_brk_s1) if pd.notna(bb_up) else np.nan
    brk_s2_raw = (bb_up + off_brk_s2) if pd.notna(bb_up) else np.nan

    if pd.isna(atr) or atr <= 0:
        off_mr_b1 = (bb_low * 0.006) if pd.notna(bb_low) else np.nan
        off_mr_b2 = (bb_low * 0.015) if pd.notna(bb_low) else np.nan
        off_mr_s1 = (bb_up * 0.006) if pd.notna(bb_up) else np.nan
        off_mr_s2 = (bb_up * 0.015) if pd.notna(bb_up) else np.nan
    else:
        off_mr_b1 = 0.10 * atr
        off_mr_b2 = 0.25 * atr
        off_mr_s1 = 0.10 * atr
        off_mr_s2 = 0.25 * atr
    mr_b1_raw = (bb_low + off_mr_b1) if pd.notna(bb_low) else np.nan
    mr_b2_raw = (bb_low + off_mr_b2) if pd.notna(bb_low) else np.nan
    mr_s1_raw = (bb_up - off_mr_s1) if pd.notna(bb_up) else np.nan
    mr_s2_raw = (bb_up - off_mr_s2) if pd.notna(bb_up) else np.nan

    def pack(prefix: str, b1: float, b2: float, s1: float, s2: float):
        out = {}
        out[f'{prefix}_Buy1'] = b1
        out[f'{prefix}_Buy2'] = b2
        out[f'{prefix}_Sell1'] = s1
        out[f'{prefix}_Sell2'] = s2
        out[f'{prefix}_Buy1_Tick'] = _round_near_tick_thousand(b1)
        out[f'{prefix}_Buy2_Tick'] = _round_down_tick_thousand(b2)
        out[f'{prefix}_Sell1_Tick'] = _round_down_tick_thousand(s1)
        out[f'{prefix}_Sell2_Tick'] = _round_down_tick_thousand(s2)
        out[f'{prefix}_Buy1_OutOfBand'] = 1 if (pd.notna(out[f'{prefix}_Buy1_Tick']) and pd.notna(band_floor_tick) and out[f'{prefix}_Buy1_Tick'] < band_floor_tick) else 0
        out[f'{prefix}_Buy2_OutOfBand'] = 1 if (pd.notna(out[f'{prefix}_Buy2_Tick']) and pd.notna(band_floor_tick) and out[f'{prefix}_Buy2_Tick'] < band_floor_tick) else 0
        out[f'{prefix}_Sell1_OutOfBand'] = 1 if (pd.notna(out[f'{prefix}_Sell1_Tick']) and pd.notna(band_ceil_tick) and out[f'{prefix}_Sell1_Tick'] > band_ceil_tick) else 0
        out[f'{prefix}_Sell2_OutOfBand'] = 1 if (pd.notna(out[f'{prefix}_Sell2_Tick']) and pd.notna(band_ceil_tick) and out[f'{prefix}_Sell2_Tick'] > band_ceil_tick) else 0
        return out

    hint = 'Aggressive' if (pd.notna(z) and z >= 0.8) else ('Balanced' if (pd.notna(z) and z >= 0.3) else 'Conservative')
    row = {
        'Ticker': t,
        'Price': px,
        'RefPrice': ref_px,
        'MA10': ma10,
        'MA20': ma20,
        'MA50': ma50,
        'ATR14': atr,
        'BB20Upper': bb_up,
        'BB20Lower': bb_low,
        'BandFloorRaw': band_floor,
        'BandCeilingRaw': band_ceil,
        'BandFloor_Tick': band_floor_tick,
        'BandCeiling_Tick': band_ceil_tick,
        'PresetHint': hint,
    }
    row.update(pack('Cons', cons_b1_raw, cons_b2_raw, cons_s1_raw, cons_s2_raw))
    row.update(pack('Bal', bal_b1_raw, bal_b2_raw, bal_s1_raw, bal_s2_raw))
    row.update(pack('Aggr', aggr_b1_raw, aggr_b2_raw, aggr_s1_raw, aggr_s2_raw))
    row.update(pack('Break', brk_b1_raw, brk_b2_raw, brk_s1_raw, brk_s2_raw))
    row.update(pack('MR', mr_b1_raw, mr_b2_raw, mr_s1_raw, mr_s2_raw))
    return row


def _infer_in_session(session_in_progress: Optional[bool] = None, session_summary_df: Optional[pd.DataFrame] = None) -> bool:
    if session_in_progress is not None:
        return bool(session_in_progress)
    if session_summary_df is not None and not session_summary_df.empty and 'InVNSession' in session_summary_df.columns:
        raw = session_summary_df.iloc[0].get('InVNSession')
        if raw is not None and not pd.isna(raw):
            try:
                return bool(int(raw))
            except Exception as _exc:
                print(f"[warn] Invalid InVNSession value '{raw}' in session_summary; fallback to clock-based inference: {_exc}")
    phase = detect_session_phase_now_vn().lower()
    return phase not in ('pre', 'post')


def _compute_reference_maps(prices_history_df: pd.DataFrame, session_open: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
    ref_price: Dict[str, float] = {}
    prev_price: Dict[str, float] = {}
    if prices_history_df is None or prices_history_df.empty:
        return ref_price, prev_price
    if 'Date' not in prices_history_df.columns or 'Close' not in prices_history_df.columns:
        return ref_price, prev_price

    ph = prices_history_df.copy()
    ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
    ph = ph.dropna(subset=['Date'])
    if ph.empty:
        return ref_price, prev_price

    ph = ph.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    today_vn = datetime.now(VN_TZ).date()

    for ticker, grp in ph.groupby('Ticker'):
        closes = pd.to_numeric(grp['Close'], errors='coerce')
        dates = grp['Date'].dt.date
        mask = closes.notna()
        closes = closes[mask]
        dates = dates[mask]
        if closes.empty:
            continue
        idx = len(closes) - 1
        if session_open and idx >= 0 and dates.iloc[-1] == today_vn:
            idx -= 1
        if idx < 0:
            idx = len(closes) - 1
        ref_val = float(closes.iloc[idx])
        ref_price[ticker] = ref_val
        prev_idx = idx - 1 if idx - 1 >= 0 else idx
        prev_price[ticker] = float(closes.iloc[prev_idx])

    return ref_price, prev_price


def build_presets_all(precomputed_path: str = 'out/precomputed_indicators.csv',
                      snapshot_path: str = 'out/snapshot.csv',
                      out_path: str = 'out/presets_all.csv',
                      prices_history_path: str = 'out/prices_history.csv',
                      session_summary_path: str = 'out/session_summary.csv',
                      *,
                      daily_band_pct: float | None = None):
    pre = pd.read_csv(precomputed_path)
    snap = pd.read_csv(snapshot_path)
    snap_px = snap.set_index('Ticker')['Price'] if 'Ticker' in snap.columns and 'Price' in snap.columns else pd.Series(dtype=float)

    session_df = None
    session_p = Path(session_summary_path)
    if session_p.exists():
        try:
            session_df = pd.read_csv(session_p)
        except Exception:
            session_df = None
    session_open = _infer_in_session(session_summary_df=session_df)

    ref_price_map: Dict[str, float] = {}
    ref_prev_map: Dict[str, float] = {}
    ph_file = Path(prices_history_path)
    if ph_file.exists():
        ph = pd.read_csv(ph_file)
        ref_price_map, ref_prev_map = _compute_reference_maps(ph, session_open)

    for c in ['Ticker','MA10','MA20','MA50','BB20Upper','BB20Lower','ATR14']:
        if c not in pre.columns:
            pre[c] = np.nan
    cols = ['Ticker','MA10','MA20','MA50','BB20Upper','BB20Lower','ATR14']
    pre = pre[cols].copy()

    rows = []
    for _, r in pre.iterrows():
        t = str(r['Ticker']).upper()
        ma10 = float(r['MA10']) if pd.notna(r['MA10']) else np.nan
        ma20 = float(r['MA20']) if pd.notna(r['MA20']) else np.nan
        ma50 = float(r['MA50']) if pd.notna(r['MA50']) else np.nan
        bb_up = float(r['BB20Upper']) if pd.notna(r['BB20Upper']) else np.nan
        bb_low = float(r['BB20Lower']) if pd.notna(r['BB20Lower']) else np.nan
        atr = float(r['ATR14']) if pd.notna(r['ATR14']) else np.nan
        px = float(snap_px.get(t, np.nan))
        for candidate in (ref_price_map.get(t), ref_prev_map.get(t), px):
            if candidate is not None and pd.notna(candidate):
                ref_px = float(candidate)
                break
        else:
            ref_px = float('nan')
        band = daily_band_pct if daily_band_pct is not None else BAND_PCT_RUNTIME
        if band is None:
            raise SystemExit("daily_band_pct is required (set via set_daily_band_pct at engine start)")
        rows.append(_build_row(t, px, ref_px, ma10, ma20, ma50, bb_up, bb_low, atr, daily_band_pct=band))

    out_df = pd.DataFrame(rows)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(op, index=False)
    return op


def build_presets_all_df(pre_df: pd.DataFrame,
                         snapshot_df: pd.DataFrame,
                         prices_history_df: pd.DataFrame,
                         session_in_progress: Optional[bool] = None,
                         session_summary_df: Optional[pd.DataFrame] = None,
                         *,
                         daily_band_pct: float | None = None) -> pd.DataFrame:
    pre = pre_df.copy()
    snap = snapshot_df.copy()
    snap_px = snap.set_index('Ticker')['Price'] if 'Ticker' in snap.columns and 'Price' in snap.columns else pd.Series(dtype=float)

    for c in ['Ticker','MA10','MA20','MA50','BB20Upper','BB20Lower','ATR14']:
        if c not in pre.columns:
            pre[c] = np.nan
    cols = ['Ticker','MA10','MA20','MA50','BB20Upper','BB20Lower','ATR14']
    pre = pre[cols].copy()

    session_open = _infer_in_session(session_in_progress=session_in_progress, session_summary_df=session_summary_df)
    ref_price_map, ref_prev_map = _compute_reference_maps(prices_history_df, session_open)

    rows = []
    for _, r in pre.iterrows():
        t = str(r['Ticker']).upper()
        ma10 = float(r['MA10']) if pd.notna(r['MA10']) else np.nan
        ma20 = float(r['MA20']) if pd.notna(r['MA20']) else np.nan
        ma50 = float(r['MA50']) if pd.notna(r['MA50']) else np.nan
        bb_up = float(r['BB20Upper']) if pd.notna(r['BB20Upper']) else np.nan
        bb_low = float(r['BB20Lower']) if pd.notna(r['BB20Lower']) else np.nan
        atr = float(r['ATR14']) if pd.notna(r['ATR14']) else np.nan
        px = float(snap_px.get(t, np.nan))
        for candidate in (ref_price_map.get(t), ref_prev_map.get(t), px):
            if candidate is not None and pd.notna(candidate):
                ref_px = float(candidate)
                break
        else:
            ref_px = float('nan')
        band = daily_band_pct if daily_band_pct is not None else BAND_PCT_RUNTIME
        if band is None:
            raise SystemExit("daily_band_pct is required (set via set_daily_band_pct at engine start)")
        rows.append(_build_row(t, px, ref_px, ma10, ma20, ma50, bb_up, bb_low, atr, daily_band_pct=band))
    return pd.DataFrame(rows)
