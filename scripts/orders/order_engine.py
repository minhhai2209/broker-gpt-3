from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.build_metrics import build_metrics_df
from scripts.build_presets_all import build_presets_all_df, _infer_in_session
from scripts.build_snapshot import build_snapshot_df
from scripts.data_fetching.collect_intraday import ensure_intraday_latest
from scripts.compute_sector_strength import compute_sector_strength_df
from scripts.data_fetching.fetch_ticker_data import ensure_and_load_history_df
# Domain helpers: indicators and prep modules
from scripts.portfolio.ingest_auto import ingest_portfolio_df
from scripts.orders.orders_io import (
    print_orders,
    write_orders_csv,
    write_orders_reasoning,
    write_orders_analysis,
    write_text_lines,
    write_orders_quality,
    write_orders_csv_enriched,
)
from scripts.orders.ladder import (
    generate_ladder_levels,
    write_ladder_final_csv,
    build_ladder_quality_rows,
    write_ladder_quality_csv,
    write_ladder_log,
)
from scripts.indicators.precompute_indicators import precompute_indicators_from_history_df
from scripts.portfolio.report_pnl import build_portfolio_pnl_dfs
from scripts.utils import (clip_to_band, detect_session_phase_now_vn,
                           hose_tick_size, load_universe_from_files,
                           round_to_tick, to_float)
from scripts.engine.config_io import ensure_policy_override_file, suggest_tuning
from scripts.aggregate_patches import aggregate_to_runtime, PatchMergeError
from scripts.engine.pipeline import ensure_pipeline_artifacts
from scripts.engine.volatility import garman_klass_sigma, percentile_thresholds
from scripts.engine.schema import MarketFilter as _MarketFilter, NeutralAdaptive, Execution as _Execution
from scripts.portfolio.portfolio_risk import (
    ExpectedReturnInputs,
    compute_cov_matrix,
    compute_expected_returns,
    compute_risk_parity_weights,
    solve_mean_variance_weights,
    TRADING_DAYS_PER_YEAR,
)
from scripts.tuning.mean_variance_calibrator import calibrate_mean_variance_params
import json
from scripts.data.events.calendar_loader import load_events, in_event_window


def _relaxed_breadth_floor(
    raw_floor: float,
    mf_conf: Mapping[str, Any],
    *,
    risk_on_prob: float,
    atr_percentile: float,
) -> float:
    """Return an effective breadth floor after applying relaxation margin."""

    try:
        floor = float(raw_floor)
    except Exception:
        floor = 0.0
    floor = max(0.0, min(1.0, floor))

    try:
        margin = float((mf_conf or {}).get("breadth_relax_margin", 0.0) or 0.0)
    except Exception:
        margin = 0.0
    if margin <= 0.0:
        return floor

    prob = max(0.0, min(1.0, float(risk_on_prob)))
    if prob <= 0.0:
        return floor

    try:
        atr_pct = float(atr_percentile)
    except Exception:
        atr_pct = 0.0
    try:
        soft = float((mf_conf or {}).get("index_atr_soft_pct", 0.9) or 0.9)
    except Exception:
        soft = 0.9
    try:
        hard = float((mf_conf or {}).get("index_atr_hard_pct", 0.97) or 0.97)
    except Exception:
        hard = soft + 0.07
    if hard <= soft:
        hard = soft + 0.01

    span = max(hard - soft, 1e-6)
    vol_relax = 1.0 - max(0.0, min(1.0, (atr_pct - soft) / span))
    relax = margin * prob * max(vol_relax, 0.0)
    adjusted = floor - relax
    return max(0.0, min(1.0, adjusted))


# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"
OUT_ORDERS_DIR = OUT_DIR / "orders"
OVERRIDE_SRC = BASE_DIR / "config" / "policy_overrides.json"
DATA_DIR = BASE_DIR / "data"


"""Order Engine — stateless flow.
Non-core helpers are split into scripts/engine/* for readability.
"""


def _test_mode_enabled() -> bool:
    # Test mode removed; always behave normally (write outputs).
    return False


# ensure_policy_override_file moved to scripts.engine.config_io


# No persistent state


"""suggest_tuning moved to scripts.engine.config_io"""


# 2) Regime/scoring/classification (merged from policy_engine.py)
@dataclass
class MarketRegime:
    phase: str
    in_session: bool
    # Intraday/snapshot index change (%) from session_summary (raw, current day)
    index_change_pct: float
    breadth_hint: float
    risk_on: bool
    buy_budget_frac: float
    top_sectors: List[str]
    add_max: int
    new_max: int
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    sector_bias: Dict[str, float]
    ticker_bias: Dict[str, float]
    pricing: Dict[str, object]
    sizing: Dict[str, object]
    execution: Dict[str, object]
    # Per-ticker overrides for thresholds/sizing
    ticker_overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)
    # Scaled sector strength in [0..1] per sector (derived from sector_strength DF)
    sector_strength_rank: Dict[str, float] = field(default_factory=dict)
    index_vol_annualized: float = 0.0
    index_atr14_pct: float = 0.0
    index_atr_percentile: float = 0.5
    trend_strength: float = 0.0
    market_filter: Dict[str, object] = field(default_factory=dict)
    # Additional market-state diagnostics to propagate downstream
    momentum_63d: float = 0.0
    momentum_percentile: float = 0.5
    drawdown_pct: float = 0.0
    drawdown_percentile: float = 0.5
    vol_percentile: float = 0.5
    market_score: float = 0.0
    risk_on_probability: float = 0.0
    model_components: Dict[str, float] = field(default_factory=dict)
    model_zscores: Dict[str, float] = field(default_factory=dict)
    buy_budget_frac_effective: float = 0.0
    turnover_percentile: float = 0.5
    turnover_value: float = 0.0
    diag_warnings: List[str] = field(default_factory=list)
    # Optional global signals (diagnostics)
    epu_us_percentile: float = 0.0
    spx_drawdown_pct: float = 0.0
    dxy_percentile: float = 0.0
    brent_mom_63d: float = 0.0
    # Optional UI/pricing/microstructure/evaluation (dict-like) for downstream modules
    orders_ui: Dict[str, object] = field(default_factory=dict)
    evaluation: Dict[str, object] = field(default_factory=dict)
    # Microstructure daily band ± fraction (e.g., 0.07 for 7%)
    micro_daily_band_pct: float = 0.07
    # Smoothed daily index change (%) computed from history for diagnostics/UI
    index_change_pct_smoothed: float = 0.0
    gk_sigma: float = 0.0
    gk_percentile: float = 0.0
    ttl_bucket: str = 'medium'
    # Global lean (buy/observe) applied uniformly via ticker_sent weight
    market_bias: float = 0.0
    # Neutral-adaptive regime metadata
    neutral_state: Dict[str, object] = field(default_factory=dict)
    is_neutral: bool = False
    neutral_thresholds: Dict[str, float] = field(default_factory=dict)
    neutral_partial_tickers: List[str] = field(default_factory=list)
    neutral_override_tickers: List[str] = field(default_factory=list)
    neutral_accum_tickers: List[str] = field(default_factory=list)
    neutral_stats: Dict[str, object] = field(default_factory=dict)
    # Stateless sell metadata / execution overrides populated during action pass
    stateless_sell_meta: Dict[str, Dict[str, object]] = field(default_factory=dict)
    ttl_overrides: Dict[str, int] = field(default_factory=dict)
    tp_sl_map: Dict[str, Dict[str, object]] = field(default_factory=dict)
    position_state: Dict[str, Dict[str, object]] = field(default_factory=dict)
    new_buy_fill_diag: List[Dict[str, object]] = field(default_factory=list)


def get_market_regime(session_summary: pd.DataFrame, sector_strength: pd.DataFrame, tuning: Dict) -> MarketRegime:
    if session_summary is None:
        raise ValueError("session_summary is required to infer the market regime")
    if not isinstance(session_summary, pd.DataFrame):
        raise TypeError(
            "session_summary must be a pandas DataFrame with session metadata"
        )
    if session_summary.empty:
        raise ValueError("session_summary is empty; unable to infer market regime")
    required_cols = {"SessionPhase", "InVNSession", "IndexChangePct"}
    missing = required_cols - set(session_summary.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(
            f"session_summary is missing required columns: {missing_cols}"
        )
    phase_raw = session_summary.iloc[0].get("SessionPhase")
    if phase_raw is None or (isinstance(phase_raw, str) and not phase_raw.strip()):
        raise ValueError(
            "session_summary['SessionPhase'] is null or empty; cannot infer market phase"
        )
    phase = str(phase_raw)
    in_session = _infer_in_session(session_summary_df=session_summary)
    idx_chg_raw = session_summary.iloc[0].get("IndexChangePct")
    idx_chg = to_float(idx_chg_raw)
    if idx_chg is None:
        raise ValueError(
            "session_summary['IndexChangePct'] must be a finite numeric value"
        )
    breadth_session_value = None
    if not session_summary.empty and "BreadthPct" in session_summary.columns:
        breadth_raw = to_float(session_summary.loc[0, "BreadthPct"])
        if breadth_raw is not None:
            try:
                val = float(breadth_raw)
                if not math.isfinite(val):
                    raise ValueError
                # Session breadth sometimes reported in percent (0-100). Normalise to 0-1.
                if abs(val) > 1.0:
                    val = val / 100.0
                breadth_session_value = max(0.0, min(1.0, val))
            except Exception:
                breadth_session_value = None
    orders_ui = tuning.get('orders_ui') if isinstance(tuning, dict) else {}
    if not isinstance(orders_ui, dict):
        orders_ui = {}
        if isinstance(tuning, dict):
            tuning['orders_ui'] = orders_ui
    # Smooth index change and derive trend/volatility context from VNINDEX history
    trend_strength = 0.0
    index_vol_annualized = 0.0
    idx_chg_smoothed = idx_chg if idx_chg is not None else 0.0
    closes = None
    ph_path = Path('out') / 'prices_history.csv'
    if not ph_path.exists():
        raise SystemExit("Missing out/prices_history.csv — run broker.sh policy/orders pipeline first.")
    import pandas as _pd
    try:
        ph = _pd.read_csv(ph_path)
    except Exception as exc:  # pragma: no cover - fatal configuration issue
        raise SystemExit(f"Unable to read {ph_path}: {exc}") from exc
    required_cols = {"Date", "Ticker", "Close"}
    if not required_cols.issubset(ph.columns):
        missing = ", ".join(sorted(required_cols - set(ph.columns)))
        raise SystemExit(f"prices_history.csv is missing columns: {missing}")
    # Keep High/Low if available for ATR computation
    keep_cols = [c for c in ['Date','Ticker','Close','Open','High','Low'] if c in ph.columns]
    ph = ph[keep_cols].copy()
    ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
    vn = ph[ph['Ticker'] == 'VNINDEX'].copy()
    if vn.empty:
        raise SystemExit("prices_history.csv has no VNINDEX rows; cannot derive market regime.")
    vn['Date'] = _pd.to_datetime(vn['Date'], errors='coerce')
    vn = vn.dropna(subset=['Date']).sort_values('Date')
    if vn.empty:
        raise SystemExit("VNINDEX history is empty after cleaning; verify prices_history.csv.")
    vn['Close'] = _pd.to_numeric(vn['Close'], errors='coerce')
    vn = vn.dropna(subset=['Close'])
    if vn.empty:
        raise SystemExit("VNINDEX close series missing numeric data in prices_history.csv.")
    vn['Ret'] = vn['Close'].pct_change()
    ohlc = vn.reindex(columns=['Open','High','Low','Close'])
    gk_series = garman_klass_sigma(ohlc)
    gk_series = gk_series.replace([np.inf, -np.inf], np.nan).dropna()
    gk_latest = float(gk_series.iloc[-1]) if not gk_series.empty else 0.0
    gk_hist = gk_series.tail(260) if len(gk_series) > 260 else gk_series
    gk_percentile = float((gk_hist <= gk_latest).mean()) if not gk_hist.empty and np.isfinite(gk_latest) else 0.0
    gk_p75, gk_p95 = percentile_thresholds(gk_hist, (0.75, 0.95)) if not gk_hist.empty else (float('nan'), float('nan'))
    closes = vn['Close']
    bucket_conf = orders_ui.get('ttl_bucket_minutes', {}) if isinstance(orders_ui, dict) else {}
    threshold_conf = orders_ui.get('ttl_bucket_thresholds', {}) if isinstance(orders_ui, dict) else {}
    try:
        p75_cfg = float(threshold_conf.get('p75')) if threshold_conf and threshold_conf.get('p75') is not None else gk_p75
    except Exception:
        p75_cfg = gk_p75
    try:
        p95_cfg = float(threshold_conf.get('p95')) if threshold_conf and threshold_conf.get('p95') is not None else gk_p95
    except Exception:
        p95_cfg = gk_p95
    bucket_label = 'medium'
    if np.isfinite(gk_latest) and np.isfinite(p95_cfg) and gk_latest >= p95_cfg:
        bucket_label = 'high'
    elif np.isfinite(gk_latest) and np.isfinite(p75_cfg) and gk_latest >= p75_cfg:
        bucket_label = 'medium'
    elif np.isfinite(gk_latest):
        bucket_label = 'low'
    if isinstance(orders_ui, dict):
        bucket_state = dict(orders_ui.get('ttl_bucket_state', {}) or {})
        bucket_state.update({'current': bucket_label, 'sigma': gk_latest, 'percentile': gk_percentile})
        orders_ui['ttl_bucket_state'] = bucket_state
        thresh_state = dict(orders_ui.get('ttl_bucket_thresholds', {}) or {})
        if np.isfinite(p75_cfg): thresh_state['p75'] = float(p75_cfg)
        if np.isfinite(p95_cfg): thresh_state['p95'] = float(p95_cfg)
        thresh_state['latest'] = float(gk_latest)
        orders_ui['ttl_bucket_thresholds'] = thresh_state
        if isinstance(bucket_conf, dict) and bucket_label in bucket_conf:
            ttl_map = dict(orders_ui.get('ttl_minutes', {}) or {})
            for key in ('base','soft','hard'):
                if key in bucket_conf[bucket_label]:
                    ttl_map[key] = bucket_conf[bucket_label][key]
            orders_ui['ttl_minutes'] = ttl_map

    def _percentile_rank(series: pd.Series, value: float | None) -> float:
        arr = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
        if arr.size == 0 or value is None or pd.isna(value):
            return 0.5
        less = float(np.sum(arr < value))
        equal = float(np.sum(arr == value))
        rank = (less + 0.5 * equal) / float(arr.size)
        return max(0.0, min(1.0, rank))

    smooth = float((vn['Ret'].tail(6) * 100.0).dropna().mean()) if not vn.empty else None
    if smooth is not None and _pd.notna(smooth):
        idx_chg_smoothed = smooth

    momentum_63d = 0.0
    momentum_percentile = 0.5
    if closes.notna().sum() >= 63:
        mom_series = closes.pct_change(63)
        latest_mom = mom_series.iloc[-1]
        if pd.notna(latest_mom):
            momentum_63d = float(latest_mom)
        momentum_percentile = _percentile_rank(mom_series, latest_mom if pd.notna(latest_mom) else None)

    ma200_slope = 0.0
    uptrend = 0.0
    if closes.notna().sum() >= 200:
        last_close = float(closes.iloc[-1])
        ma50 = float(closes.tail(50).mean())
        ma200_series = closes.rolling(200).mean()
        ma200 = float(ma200_series.iloc[-1]) if pd.notna(ma200_series.iloc[-1]) else 0.0
        if ma200 != 0:
            trend_strength = (last_close - ma200) / ma200
        elif ma50 != 0:
            trend_strength = (last_close - ma50) / ma50
        # MA200 slope over last 20 sessions (normalized by MA200)
        if ma200_series.notna().sum() >= 220:
            prev = float(ma200_series.iloc[-20])
            cur = float(ma200_series.iloc[-1])
            if prev != 0 and math.isfinite(prev) and math.isfinite(cur):
                ma200_slope = (cur - prev) / abs(prev)
        if ma200 > 0 and last_close > ma200 and ma200_slope > 0:
            uptrend = 1.0

    rolling_max = closes.cummax()
    drawdown_series = 1.0 - closes / rolling_max.replace(0, np.nan)
    drawdown_series = drawdown_series.replace([np.inf, -np.inf], np.nan)
    drawdown_series = drawdown_series.dropna()
    drawdown_pct = float(drawdown_series.iloc[-1]) if not drawdown_series.empty else 0.0
    drawdown_pct = max(0.0, drawdown_pct)
    drawdown_percentile = _percentile_rank(drawdown_series, drawdown_pct if drawdown_series.size else None)

    ret_window = vn['Ret'].dropna()
    vol_percentile = 0.5
    index_atr14_pct = 0.0
    index_atr_percentile = 0.5
    diag_warnings: List[str] = []
    # Compute Index ATR14% if High/Low present
    if {'High','Low'}.issubset(set(ph.columns)):
        vn_hl = ph[ph['Ticker']=='VNINDEX'].copy()
        if not vn_hl.empty and {'High','Low','Close'}.issubset(set(vn_hl.columns)):
            vn_hl['High'] = _pd.to_numeric(vn_hl['High'], errors='coerce')
            vn_hl['Low'] = _pd.to_numeric(vn_hl['Low'], errors='coerce')
            vn_hl['Close'] = _pd.to_numeric(vn_hl['Close'], errors='coerce')
            vn_hl = vn_hl.dropna(subset=['High','Low','Close'])
            try:
                from scripts.indicators import atr_wilder as _atr_wilder
                if len(vn_hl) >= 14:
                    atr_series = _atr_wilder(vn_hl['High'], vn_hl['Low'], vn_hl['Close'], 14)
                    if len(atr_series):
                        last_close_idx = vn_hl['Close'].iloc[-1]
                        last_atr = float(atr_series.iloc[-1]) if _pd.notna(atr_series.iloc[-1]) else None
                        if last_atr is not None and _pd.notna(last_close_idx) and last_close_idx != 0:
                            index_atr14_pct = float(last_atr / last_close_idx)
                        # Percentile of ATR% over its own history
                        atr_pct_series = (atr_series / vn_hl['Close']).replace([np.inf,-np.inf], np.nan).dropna()
                        if not atr_pct_series.empty and index_atr14_pct > 0:
                            index_atr_percentile = _percentile_rank(atr_pct_series, index_atr14_pct)
                else:
                    diag_warnings.append('index_atr_insufficient_history')
            except Exception:
                diag_warnings.append('index_atr_compute_error')  # fallback handled by vol_ewm below
        else:
            diag_warnings.append('index_atr_missing_columns')
    else:
        diag_warnings.append('index_atr_missing_hilo')
    if not ret_window.empty:
        vol_ewm = ret_window.ewm(span=20, adjust=False).std().dropna()
        if not vol_ewm.empty:
            index_vol_annualized = float(vol_ewm.iloc[-1] * math.sqrt(252.0))
            vol_percentile = _percentile_rank(vol_ewm * math.sqrt(252.0), index_vol_annualized)
        elif len(ret_window) >= 21:
            index_vol_annualized = float(ret_window.tail(21).std() * math.sqrt(252.0))
    else:
        index_vol_annualized = 0.0

    # Turnover percentile (market trading value) over recent history
    turnover_percentile = 0.5
    turnover_value = 0.0
    if 'Volume' in ph.columns:
        ph_tv = ph.copy()
        ph_tv['Date'] = _pd.to_datetime(ph_tv['Date'], errors='coerce')
        ph_tv = ph_tv.dropna(subset=['Date'])
        non_index = ~ph_tv['Ticker'].astype(str).str.upper().isin(['VNINDEX','VN30','VN100'])
        ph_tv = ph_tv[non_index]
        if not ph_tv.empty and {'Close','Volume'}.issubset(set(ph_tv.columns)):
            tv_daily = ph_tv.assign(
                Close=_pd.to_numeric(ph_tv['Close'], errors='coerce'),
                Volume=_pd.to_numeric(ph_tv['Volume'], errors='coerce')
            ).dropna(subset=['Close','Volume'])
            tv_daily = tv_daily.groupby('Date', as_index=False).apply(lambda g: (g['Close']*g['Volume']).sum()).rename(columns={None:'TotalValue'})
            tv_daily = tv_daily.rename(columns={0:'TotalValue'}) if 0 in tv_daily.columns else tv_daily
            tv_daily = tv_daily.sort_values('Date')
            if not tv_daily.empty:
                turnover_value = float(tv_daily['TotalValue'].iloc[-1])
                # Lookback window 252 trading days (if available)
                series = tv_daily['TotalValue'].tail(252)
                turnover_percentile = _percentile_rank(series, turnover_value)

    breadth_hint = 0.0
    breadth_long_hint = 0.0
    if not sector_strength.empty and "breadth_above_ma50_pct" in sector_strength.columns:
        sec_col = next((cand for cand in ("sector", "Sector") if cand in sector_strength.columns), None)
        overall = None
        if sec_col is not None:
            labels = sector_strength[sec_col].astype(str).str.strip().str.lower()
            mask = labels.isin({"tất cả", "tat ca", "all"})
            if mask.any():
                overall = sector_strength.loc[mask, "breadth_above_ma50_pct"].iloc[0]
        val = overall if overall is not None else sector_strength["breadth_above_ma50_pct"].mean()
        numeric_val = pd.to_numeric(val, errors="coerce")
        if pd.isna(numeric_val):
            numeric_val = 0.0
        breadth_hint = float(numeric_val) / 100.0
    if breadth_session_value is not None:
        # Prefer session breadth when sector-strength snapshot missing; otherwise blend softly.
        if breadth_hint <= 0.0:
            breadth_hint = breadth_session_value
        else:
            breadth_hint = max(0.0, min(1.0, 0.5 * breadth_hint + 0.5 * breadth_session_value))
    # Long-term breadth (above MA200) if available
    if not sector_strength.empty and "breadth_above_ma200_pct" in sector_strength.columns:
        sec_col2 = next((cand for cand in ("sector", "Sector") if cand in sector_strength.columns), None)
        overall2 = None
        if sec_col2 is not None:
            labels2 = sector_strength[sec_col2].astype(str).str.strip().str.lower()
            mask2 = labels2.isin({"tất cả", "tat ca", "all"})
            if mask2.any():
                overall2 = sector_strength.loc[mask2, "breadth_above_ma200_pct"].iloc[0]
        val2 = overall2 if overall2 is not None else sector_strength["breadth_above_ma200_pct"].mean()
        numeric_val2 = pd.to_numeric(val2, errors="coerce")
        if pd.isna(numeric_val2):
            numeric_val2 = 0.0
        breadth_long_hint = float(numeric_val2) / 100.0

    def _clip01(val: float | None) -> float:
        if val is None:
            return 0.0
        return max(0.0, min(1.0, float(val)))

    # Scales for regime components — prefer runtime policy; fall back to schema-declared defaults
    def _get_scale(name: str) -> float:
        sc_cfg = dict(tuning.get('regime_scales', {}) or {}) if isinstance(tuning, dict) else {}
        if name in sc_cfg and sc_cfg.get(name) is not None:
            return float(sc_cfg.get(name))
        # Schema defaults (scripts.engine.schema.PolicyOverrides.RegimeScales)
        defaults = {
            'vol_ann_unit': 0.45,
            'trend_unit': 0.05,
            'idx_smoothed_unit': 1.5,
            'drawdown_unit': 0.20,
            'momentum_unit': 0.12,
        }
        if name in defaults:
            return float(defaults[name])
        raise SystemExit(f"Missing regime_scales.{name} in policy and no schema default available")

    TREND_UNIT = _get_scale('trend_unit')
    IDX_UNIT = _get_scale('idx_smoothed_unit')
    DD_UNIT = _get_scale('drawdown_unit')
    VOL_UNIT = _get_scale('vol_ann_unit')
    MOM_UNIT = _get_scale('momentum_unit')

    trend_norm_raw = max(-1.0, min(1.0, (trend_strength / TREND_UNIT) if trend_strength is not None else 0.0))
    idx_norm = max(-1.0, min(1.0, (idx_chg_smoothed or 0.0) / IDX_UNIT))
    breadth_norm = _clip01(breadth_hint)

    trend_up = _clip01(max(trend_strength, 0.0) / (2.0 * TREND_UNIT)) if trend_strength is not None else 0.0
    momentum_ratio = _clip01(max(momentum_63d, 0.0) / MOM_UNIT) if momentum_63d is not None else 0.0
    momentum_norm = 0.5 * _clip01(momentum_percentile) + 0.5 * momentum_ratio
    trend_momentum = 0.5 * trend_up + 0.5 * momentum_norm

    drawdown_level = _clip01(1.0 - ((drawdown_pct / DD_UNIT) if drawdown_pct is not None else 0.0))
    drawdown_component = min(drawdown_level, _clip01(1.0 - drawdown_percentile)) if drawdown_percentile is not None else drawdown_level

    vol_component = 0.0
    if index_vol_annualized is not None and index_vol_annualized > 0:
        vol_component = _clip01(1.0 - index_vol_annualized / VOL_UNIT)
    vol_component = min(vol_component, _clip01(1.0 - vol_percentile)) if vol_component > 0 else _clip01(1.0 - vol_percentile)

    component_map = {
        "trend": float(trend_momentum),
        "momentum": _clip01(momentum_percentile),
        "breadth": float(breadth_norm),
        "drawdown": float(drawdown_component),
        "volatility": float(vol_component),
        "index_return": float(idx_norm),
        "drawdown_pct": float(drawdown_pct),
        "vol_percentile": _clip01(vol_percentile),
        "trend_strength": float(trend_strength or 0.0),
        "ma200_slope": float(ma200_slope),
        "uptrend": float(uptrend),
        "breadth_long": _clip01(breadth_long_hint),
        "index_atr_pct": float(index_atr14_pct),
        "index_atr_percentile": _clip01(index_atr_percentile),
        "turnover": _clip01(turnover_percentile),
        # Advanced breadth components (optional): McClellan Oscillator percentile and NH-NL percentile
        # Computed below with robust fallbacks (set to 0.5 if insufficient data)
    }

    # Optional: read global factors snapshot if present
    epu_pct = None
    spx_dd = None
    dxy_pct = None
    brent_mom = None
    try:
        gf_path = Path('out') / 'global_factors_features.csv'
        if gf_path.exists():
            gf = _pd.read_csv(gf_path)
            if not gf.empty:
                last = gf.tail(1).iloc[0]
                epu_val = last.get('US_EPU_Percentile')
                if epu_val is not None and not _pd.isna(epu_val):
                    epu_pct = float(epu_val)
                dd_val = last.get('SPX_Drawdown_Pct')
                if dd_val is not None and not _pd.isna(dd_val):
                    spx_dd = float(dd_val)
                dxy_val = last.get('DXY_Percentile')
                if dxy_val is not None and not _pd.isna(dxy_val):
                    dxy_pct = float(dxy_val)
                br_mom = last.get('Brent_Mom_63d')
                if br_mom is not None and not _pd.isna(br_mom):
                    brent_mom = float(br_mom)
    except Exception:
        # Do not fail regime if optional file is malformed; tag diagnostic instead
        diag_warnings.append('global_factors_read_error')

    # Compute McClellan Oscillator percentile and NH-NL percentile from prices history
    try:
        ph2 = ph[['Date','Ticker','Close']].copy()
        ph2['Date'] = _pd.to_datetime(ph2['Date'], errors='coerce')
        ph2 = ph2.dropna(subset=['Date'])
        ph2['Ticker'] = ph2['Ticker'].astype(str).str.upper()
        non_index_mask = ~ph2['Ticker'].isin(['VNINDEX','VN30','VN100'])
        ph2 = ph2[non_index_mask]
        if not ph2.empty:
            ph2['Close'] = _pd.to_numeric(ph2['Close'], errors='coerce')
            ph2 = ph2.sort_values(['Ticker','Date'])
            ph2['PrevClose'] = ph2.groupby('Ticker')['Close'].shift(1)
            daily = ph2.dropna(subset=['PrevClose']).copy()
            if not daily.empty:
                daily['Adv'] = (daily['Close'] > daily['PrevClose']).astype(int)
                daily['Dec'] = (daily['Close'] < daily['PrevClose']).astype(int)
                ad = daily.groupby('Date').agg(Adv=('Adv','sum'), Dec=('Dec','sum')).reset_index()
                ad['AD'] = ad['Adv'] - ad['Dec']
                if len(ad) >= 40:
                    ema19 = ad['AD'].ewm(span=19, adjust=False).mean()
                    ema39 = ad['AD'].ewm(span=39, adjust=False).mean()
                    osc = ema19 - ema39
                    mcc_last = float(osc.iloc[-1]) if _pd.notna(osc.iloc[-1]) else 0.0
                    component_map['mcclellan'] = _clip01(_percentile_rank(osc, mcc_last))
                # NH-NL over 252d
                # Compute rolling 252-day highs/lows per ticker
                win = 252
                df_hl = ph2[['Date','Ticker','Close']].copy()
                df_hl = df_hl.sort_values(['Ticker','Date'])
                df_hl['RollHigh'] = df_hl.groupby('Ticker')['Close'].transform(lambda s: _pd.Series(s).rolling(win, min_periods=win).max())
                df_hl['RollLow'] = df_hl.groupby('Ticker')['Close'].transform(lambda s: _pd.Series(s).rolling(win, min_periods=win).min())
                df_hl = df_hl.dropna(subset=['RollHigh','RollLow'])
                if not df_hl.empty:
                    df_hl['IsNH'] = (df_hl['Close'] >= df_hl['RollHigh']).astype(int)
                    df_hl['IsNL'] = (df_hl['Close'] <= df_hl['RollLow']).astype(int)
                    nhnl = df_hl.groupby('Date').agg(NH=('IsNH','sum'), NL=('IsNL','sum'), N=('Ticker','nunique')).reset_index()
                    nhnl['DiffPct'] = (nhnl['NH'] - nhnl['NL']) / nhnl['N'].replace(0, np.nan)
                    nhnl = nhnl.dropna(subset=['DiffPct'])
                    if not nhnl.empty:
                        last = float(nhnl['DiffPct'].iloc[-1])
                        component_map['nhnl'] = _clip01(_percentile_rank(nhnl['DiffPct'], last))
    except Exception as _exc:
        # Keep defaults when data is insufficient but record a diagnostic tag
        diag_warnings.append('breadth_adv_error')
    if 'mcclellan' not in component_map:
        diag_warnings.append('mcclellan_unavailable')
    if 'nhnl' not in component_map:
        diag_warnings.append('nhnl_unavailable')

    # Enforce presence of a calibrated regime_model (logistic) instead of
    # relying on legacy fixed weights. This keeps the engine aligned with
    # accepted practice and avoids ad-hoc thresholds.
    if not isinstance(tuning, dict) or 'regime_model' not in tuning or not isinstance(tuning['regime_model'], dict):
        raise SystemExit("Missing 'regime_model' in policy_overrides: provide logistic components with mean/std/weight and threshold")
    regime_model_cfg = tuning['regime_model']
    if 'components' not in regime_model_cfg or not isinstance(regime_model_cfg['components'], dict):
        raise SystemExit("regime_model.components must be provided as an object of factor configs")

    # Compute a simple composite for reporting only (not gating)
    legacy_score = (
        0.30 * trend_momentum +
        0.25 * breadth_norm +
        0.20 * _clip01(max(0.0, idx_norm)) +
        0.15 * drawdown_component +
        0.10 * vol_component
    )

    base_score = legacy_score
    risk_on_probability = legacy_score
    risk_on = base_score >= 0.5
    model_zscores: Dict[str, float] = {}
    # Mandatory logistic regime model
    intercept = float(regime_model_cfg['intercept'])
    threshold = float(regime_model_cfg['threshold'])
    linear = intercept
    components_cfg = regime_model_cfg['components']
    for name, conf in components_cfg.items():
        if not isinstance(conf, dict):
            raise SystemExit(f"regime_model component '{name}' must be an object with mean/std/weight")
        if name not in component_map:
            raise SystemExit(f"regime_model references unknown component '{name}'")
        value = float(component_map[name])
        mean = float(conf['mean'])
        std = float(conf['std'])
        if not math.isfinite(std) or std <= 0.0:
            raise SystemExit(f"regime_model component '{name}' must specify std > 0")
        weight = float(conf['weight'])
        z = (value - mean) / std
        model_zscores[name] = z
        linear += weight * z
    prob = 1.0 / (1.0 + math.exp(-max(min(linear, 20.0), -20.0)))
    base_score = max(0.0, min(1.0, prob))
    risk_on_probability = base_score
    risk_on = base_score >= threshold

    # Parse market_filter early to apply hard guards via config
    def _require(obj: Dict, key: str, msg: str):
        if key not in obj:
            raise SystemExit(f"Missing tuning parameter '{key}': {msg}")
        return obj[key]

    market_filter_raw = dict(_require(tuning, "market_filter", "must define market_filter thresholds"))
    # Validate market_filter strictly (no baseline fallback here)
    try:
        mf_model = _MarketFilter.model_validate(market_filter_raw)
        market_filter_raw = mf_model.model_dump()
    except Exception as _exc_mf:
        raise SystemExit(f"Invalid market_filter: {_exc_mf}") from _exc_mf

    # Also prepare pricing and orders_ui by filling from baseline for missing nested keys
    import copy as _copy
    pricing = _copy.deepcopy(tuning.get('pricing', {}) or {})
    execution_conf = _copy.deepcopy(tuning.get('execution', {}) or {})
    orders_ui = _copy.deepcopy(tuning.get('orders_ui', {}) or {})

    # No baseline backfill for pricing/orders_ui — require completeness in runtime policy

    try:
        execution_conf = _Execution.model_validate(execution_conf).model_dump()
    except Exception as _exc_exec:
        raise SystemExit(f"Invalid execution config: {_exc_exec}") from _exc_exec

    def _mf_float(key: str) -> float:
        if key not in market_filter_raw:
            raise SystemExit(f"Missing market_filter['{key}'] in tuning")
        val = market_filter_raw[key]
        if val is None:
            raise SystemExit(f"Invalid market_filter['{key}']=null; provide a number or remove the key for defaulting")
        return float(val)

    def _mf_bool(key: str) -> bool:
        if key not in market_filter_raw:
            raise SystemExit(f"Missing market_filter['{key}'] in tuning")
        val = market_filter_raw[key]
        if isinstance(val, bool):
            return val
        return bool(int(val))

    # Fail fast if required keys missing — align with "no hidden defaults"
    market_filter_conf = {
        "risk_off_index_drop_pct": abs(_mf_float("risk_off_index_drop_pct")),
        "risk_off_trend_floor": _mf_float("risk_off_trend_floor"),
        "risk_off_breadth_floor": max(0.0, min(1.0, _mf_float("risk_off_breadth_floor"))),
        "breadth_relax_margin": max(0.0, min(0.2, _mf_float("breadth_relax_margin"))),
        "leader_min_rsi": _mf_float("leader_min_rsi"),
        "leader_min_mom_norm": max(0.0, min(1.0, _mf_float("leader_min_mom_norm"))),
        "leader_require_ma20": _mf_bool("leader_require_ma20"),
        "leader_require_ma50": _mf_bool("leader_require_ma50"),
        "market_score_soft_floor": max(0.0, min(1.0, _mf_float("market_score_soft_floor"))),
        "market_score_hard_floor": max(0.0, min(1.0, _mf_float("market_score_hard_floor"))),
        "leader_max": int(_mf_float("leader_max")),
        "risk_off_drawdown_floor": max(0.0, min(1.0, _mf_float("risk_off_drawdown_floor"))),
        "index_atr_soft_pct": max(0.0, min(1.0, _mf_float("index_atr_soft_pct"))),
        "index_atr_hard_pct": max(0.0, min(1.0, _mf_float("index_atr_hard_pct"))),
        # Parameterised hard guards
        "idx_chg_smoothed_hard_drop": abs(_mf_float("idx_chg_smoothed_hard_drop")),
        "trend_norm_hard_floor": float(_mf_float("trend_norm_hard_floor")),
        "vol_ann_hard_ceiling": abs(_mf_float("vol_ann_hard_ceiling")),
        # Scaling caps and severity multiplier
        "guard_new_scale_cap": max(0.0, min(1.0, float(_mf_float("guard_new_scale_cap")))),
        "atr_soft_scale_cap": max(0.0, min(1.0, float(_mf_float("atr_soft_scale_cap")))),
        "severe_drop_mult": max(1e-6, float(_mf_float("severe_drop_mult"))),
    }
    if market_filter_conf["market_score_soft_floor"] < market_filter_conf["market_score_hard_floor"]:
        raise SystemExit("market_filter.market_score_soft_floor must be >= market_score_hard_floor")

    # Apply configurable hard guards (replaces previous hard-coded values)
    try:
        hard_drop = float(market_filter_conf.get("idx_chg_smoothed_hard_drop", 0.5) or 0.5)
        if (idx_chg_smoothed or 0.0) <= -abs(hard_drop):
            risk_on = False
        trend_floor_hard = float(market_filter_conf.get("trend_norm_hard_floor", -0.25))
        if trend_norm_raw <= trend_floor_hard:
            risk_on = False
        vol_ceiling = float(market_filter_conf.get("vol_ann_hard_ceiling", 0.60) or 0.60)
        if index_vol_annualized and index_vol_annualized >= vol_ceiling:
            risk_on = False
    except Exception as _exc:
        # In case of misconfig, fall back to conservative guards and tag diagnostics
        if (idx_chg_smoothed or 0.0) <= -0.5 or trend_norm_raw <= -0.25:
            risk_on = False
        if index_vol_annualized and index_vol_annualized >= 0.60:
            risk_on = False
        diag_warnings.append('market_filter_misconfig')
    # Drawdown hard guard will be enforced after market_filter parsed; keep flag here via component

    top = sector_strength.sort_values(["breadth_above_ma50_pct", "avg_rsi14"], ascending=False)["sector"].tolist() if not sector_strength.empty else []
    # Filter out non-actionable pseudo-sectors
    def _ok_sector(s: object) -> bool:
        ss = '' if s is None else str(s).strip()
        low = ss.lower()
        if ss == '' or ss is None:
            return False
        if ss in ('Tất cả',):
            return False
        if low in ('all','tat ca','nan','none'):
            return False
        if ss == 'Index':
            return False
        return True
    top = [s for s in top if _ok_sector(s)][:4]
    # Build sector strength rank in [0..1] for smoother boosting
    sector_strength_rank: Dict[str, float] = {}
    if not sector_strength.empty and {"sector", "breadth_above_ma50_pct", "avg_rsi14"}.issubset(sector_strength.columns):
        # Prefer a composite sector strength that includes long-term breadth if available
        has_ma200 = "breadth_above_ma200_pct" in sector_strength.columns
        cols = ["sector", "breadth_above_ma50_pct", "avg_rsi14"] + (["breadth_above_ma200_pct"] if has_ma200 else [])
        df = sector_strength[cols].copy()
        df = df[df["sector"].apply(_ok_sector)]
        if not df.empty:
            b50 = pd.to_numeric(df["breadth_above_ma50_pct"], errors="coerce")
            rsi = pd.to_numeric(df["avg_rsi14"], errors="coerce")
            if has_ma200:
                b200 = pd.to_numeric(df["breadth_above_ma200_pct"], errors="coerce")
                df["raw"] = 0.4 * b50 + 0.3 * b200 + 0.3 * rsi
            else:
                df["raw"] = 0.5 * b50 + 0.5 * rsi
            if df["raw"].notna().any():
                vmin = float(df["raw"].min())
                vmax = float(df["raw"].max())
                if vmax > vmin:
                    df["rank01"] = (df["raw"] - vmin) / (vmax - vmin)
                else:
                    df["rank01"] = 0.0
                sector_strength_rank = {str(r["sector"]): float(r["rank01"]) for _, r in df.iterrows()}

    def _require(obj: Dict, key: str, msg: str):
        if key not in obj:
            raise SystemExit(f"Missing tuning parameter '{key}': {msg}")
        return obj[key]

    bb = float(_require(tuning, "buy_budget_frac", "set 0..0.30 in policy_overrides.json"))
    if not (0.0 <= bb <= 0.30):
        raise SystemExit("buy_budget_frac out of range (0..0.30)")
    budget_map: Dict[str, float] = {}
    if isinstance(tuning, dict) and isinstance(tuning.get('buy_budget_by_regime'), dict):
        for key, value in tuning['buy_budget_by_regime'].items():
            if value is None:
                continue
            try:
                budget_map[str(key)] = float(value)
            except Exception:
                continue
    def _clip_budget(val: float) -> float:
        return max(0.0, min(0.30, float(val)))
    if budget_map:
        regime_key = 'risk_on' if risk_on else 'risk_off'
        if regime_key in budget_map:
            bb = _clip_budget(budget_map[regime_key])
    add_max = int(_require(tuning, "add_max", "integer 0..50"))
    new_max = int(_require(tuning, "new_max", "integer 0..50"))

    weights = dict(_require(tuning, "weights", "must define all scoring weights"))
    for k in ("w_trend", "w_momo", "w_liq", "w_vol_guard", "w_beta", "w_sector", "w_sector_sent"):
        if k not in weights:
            raise SystemExit(f"Missing weights['{k}'] in tuning")

    # Allow profile-based thresholds by regime; fallback to flat thresholds for compatibility
    thresholds: Dict[str, float]
    if "thresholds_profiles" in tuning and isinstance(tuning.get("thresholds_profiles"), dict):
        tp = tuning["thresholds_profiles"] or {}
        prof_key = "risk_on" if risk_on else "risk_off"
        if prof_key in tp and isinstance(tp[prof_key], dict):
            thresholds = dict(tp[prof_key])
        else:
            thresholds = dict(_require(tuning, "thresholds", "must define thresholds (profiles missing keys)"))
    else:
        thresholds = dict(_require(tuning, "thresholds", "must define thresholds"))
    for k in ("base_add", "base_new", "trim_th", "q_add", "q_new", "min_liq_norm", "near_ceiling_pct", "tp_pct", "sl_pct", "tp_trim_frac", "exit_on_ma_break",
              # New thresholds to parameterize trim/exit heuristics
              "exit_ma_break_rsi", "trim_rsi_below_ma20", "trim_rsi_macdh_neg", "exit_ma_break_score_gate", "tilt_exit_downgrade_min"):
        if k not in thresholds:
            raise SystemExit(f"Missing thresholds['{k}'] in tuning")

    sector_bias = dict(_require(tuning, "sector_bias", "must provide sector_bias map (can be empty)"))
    ticker_bias = dict(_require(tuning, "ticker_bias", "must provide ticker_bias map (can be empty)"))
    # pricing already merged with baseline above
    sizing = dict(_require(tuning, "sizing", "must define sizing object"))
    # market_filter_conf already parsed above

    # Optional: global-driven sector tilt (Brent -> Energy). Respect schema/no-hidden-defaults.
    # If enabled and condition holds, boost sector_strength_rank minimum for the configured sector.
    try:
        gt = dict(tuning.get('global_tilts', {}) or {})
        if gt and bool(int(gt.get('brent_energy_enable', 0))):
            thr = gt.get('brent_mom_soft')
            sector_label = str(gt.get('brent_energy_sector_label', 'Năng lượng'))
            boost_min = float(gt.get('brent_boost_min', 0.20) or 0.20)
            if brent_mom is not None and thr is not None and float(brent_mom) >= float(thr):
                # Boost energy sector rank to at least boost_min
                cur = float(sector_strength_rank.get(sector_label, 0.0))
                sector_strength_rank[sector_label] = max(cur, max(0.0, min(1.0, boost_min)))
    except Exception:
        diag_warnings.append('global_tilts_parse_error')

    mr = MarketRegime(
        phase=phase,
        in_session=in_session,
        # Use raw intraday change for guards/gating to match session context
        index_change_pct=idx_chg or 0.0,
        index_change_pct_smoothed=idx_chg_smoothed or 0.0,
        breadth_hint=breadth_hint,
        risk_on=risk_on,
        buy_budget_frac=_clip_budget(bb),
        top_sectors=top,
        add_max=add_max,
        new_max=new_max,
        weights=weights,
        thresholds=thresholds,
        sector_bias=sector_bias,
        ticker_bias=ticker_bias,
        market_bias=float(tuning.get('market_bias', 0.0) or 0.0),
        pricing=pricing,
        execution=execution_conf,
        sizing=sizing,
        ticker_overrides=dict(tuning.get('ticker_overrides', {}) or {}),
        sector_strength_rank=sector_strength_rank,
        index_vol_annualized=index_vol_annualized or 0.0,
        index_atr14_pct=float(index_atr14_pct or 0.0),
        index_atr_percentile=_clip01(index_atr_percentile),
        trend_strength=trend_strength or 0.0,
        market_filter=market_filter_conf,
        momentum_63d=momentum_63d,
        momentum_percentile=_clip01(momentum_percentile),
        drawdown_pct=drawdown_pct,
        drawdown_percentile=_clip01(drawdown_percentile),
        vol_percentile=_clip01(vol_percentile),
        market_score=_clip01(base_score),
        risk_on_probability=_clip01(risk_on_probability),
        model_components=component_map,
        model_zscores=model_zscores,
        buy_budget_frac_effective=0.0,
        turnover_percentile=_clip01(turnover_percentile),
        turnover_value=float(turnover_value or 0.0),
        diag_warnings=diag_warnings,
        epu_us_percentile=float(epu_pct) if epu_pct is not None else 0.0,
        spx_drawdown_pct=float(spx_dd) if spx_dd is not None else 0.0,
        dxy_percentile=float(dxy_pct) if dxy_pct is not None else 0.0,
        brent_mom_63d=float(brent_mom) if brent_mom is not None else 0.0,
        orders_ui=orders_ui,
    )
    mr.gk_sigma = float(gk_latest)
    mr.gk_percentile = float(gk_percentile)
    mr.ttl_bucket = bucket_label
    # Attach optional UI/evaluation configs and microstructure band
    try:
        ou = dict(tuning.get('orders_ui', {}) or {}) if isinstance(tuning, dict) else {}
        eva = dict(tuning.get('evaluation', {}) or {}) if isinstance(tuning, dict) else {}
        setattr(mr, 'orders_ui', ou)
        setattr(mr, 'evaluation', eva)
    except Exception:
        pass
    try:
        mkt = dict(tuning.get('market', {}) or {}) if isinstance(tuning, dict) else {}
        micro = dict(mkt.get('microstructure', {}) or {})
        band = micro.get('daily_band_pct')
        if band is not None:
            mr.micro_daily_band_pct = float(band)
    except Exception:
        pass
    # Preserve calibration targets (for notional guard, etc.)
    try:
        ct = tuning.get('calibration_targets') if isinstance(tuning, dict) else None
        if isinstance(ct, dict):
            setattr(mr, 'calibration_targets', ct)
    except Exception:
        pass

    # Neutral/sideways adaptive regime detection (stateless)
    neutral_conf_raw = dict(tuning.get('neutral_adaptive', {}) or {}) if isinstance(tuning, dict) else {}
    try:
        neutral_conf = NeutralAdaptive.model_validate(neutral_conf_raw).model_dump()
    except Exception:
        neutral_conf = NeutralAdaptive().model_dump()
    neutral_state: Dict[str, object] = {'config': neutral_conf}
    is_neutral = False
    guard_active = False
    guard_flags: Dict[str, object] = {}
    reasons: List[str] = []
    if bool(int(neutral_conf.get('neutral_enable', 0))):
        prob = float(risk_on_probability)
        low = float(neutral_conf.get('neutral_risk_on_prob_low', 0.35) or 0.35)
        high = float(neutral_conf.get('neutral_risk_on_prob_high', 0.65) or 0.65)
        if prob < low - 1e-9:
            reasons.append(f"risk_on_prob {prob:.3f} < low {low:.3f}")
        if prob > high + 1e-9:
            reasons.append(f"risk_on_prob {prob:.3f} > high {high:.3f}")

        breadth_val = float(breadth_hint)
        neutral_state.update({'breadth_value': breadth_val})
        if breadth_session_value is not None:
            neutral_state.update({'breadth_session_value': float(breadth_session_value)})
        center = neutral_conf.get('neutral_breadth_center')
        try:
            center_val = float(center) if center is not None else 0.5
        except Exception:
            center_val = 0.5
        center_val = max(0.0, min(1.0, center_val))
        band = float(neutral_conf.get('neutral_breadth_band', 0.05) or 0.0)
        band = max(0.0, min(1.0, band))
        if breadth_val < center_val - band - 1e-9 or breadth_val > center_val + band + 1e-9:
            reasons.append(f"breadth {breadth_val:.3f} outside neutral band [{center_val-band:.3f},{center_val+band:.3f}]")

        atr_cap_cfg = neutral_conf.get('neutral_index_atr_soft_cap')
        atr_metric = float(getattr(mr, 'index_atr_percentile', 0.0) or 0.0)
        atr_cap_mode = 'percentile'
        if atr_cap_cfg is None:
            try:
                mf_cap = market_filter_conf.get('index_atr_soft_pct')
                if mf_cap is not None:
                    atr_cap_cfg = float(mf_cap)
            except Exception:
                atr_cap_cfg = None
        else:
            try:
                atr_cap_cfg = float(atr_cap_cfg)
            except Exception:
                atr_cap_cfg = None
        if atr_cap_cfg is not None:
            atr_cap_val = float(atr_cap_cfg)
            if atr_cap_val > 1.0:
                atr_cap_mode = 'absolute'
                atr_metric = float(getattr(mr, 'index_atr14_pct', 0.0) or 0.0)
                atr_cap_val = atr_cap_val / 100.0 if atr_cap_val > 10.0 else atr_cap_val
            if atr_metric > atr_cap_val + 1e-9:
                if atr_cap_mode == 'absolute':
                    reasons.append(f"index_atr_pct {atr_metric:.4f} > cap {atr_cap_val:.4f}")
                else:
                    reasons.append(f"index_atr_pctile {atr_metric:.3f} > cap {atr_cap_val:.3f}")
        neutral_state.update({
            'atr_metric': atr_metric,
            'atr_cap_mode': atr_cap_mode,
            'atr_cap_value': atr_cap_cfg,
        })

        # Evaluate market guards to avoid overrides under stress
        try:
            idx_drop_thr = float(market_filter_conf.get('risk_off_index_drop_pct', 0.0) or 0.0)
        except Exception:
            idx_drop_thr = 0.0
        try:
            trend_floor = float(market_filter_conf.get('risk_off_trend_floor', 0.0) or 0.0)
        except Exception:
            trend_floor = 0.0
        try:
            breadth_floor_raw = float(market_filter_conf.get('risk_off_breadth_floor', 0.0) or 0.0)
        except Exception:
            breadth_floor_raw = 0.0
        breadth_floor = _relaxed_breadth_floor(
            breadth_floor_raw,
            market_filter_conf,
            risk_on_prob=getattr(mr, 'risk_on_probability', 0.0) or 0.0,
            atr_percentile=getattr(mr, 'index_atr_percentile', 0.0) or 0.0,
        )
        try:
            score_hard = float(market_filter_conf.get('market_score_hard_floor', 0.0) or 0.0)
        except Exception:
            score_hard = 0.0
        idx_chg_now = float(idx_chg or 0.0)
        market_score_now = float(base_score or 0.0)
        trend_now = float(trend_strength or 0.0)
        guard_new = (
            (idx_drop_thr > 0.0 and idx_chg_now <= -abs(idx_drop_thr))
            or (trend_now <= trend_floor)
            or (breadth_val < breadth_floor)
            or (market_score_now <= score_hard)
        )
        atr_pctile = float(getattr(mr, 'index_atr_percentile', 0.0) or 0.0)
        atr_soft = float(market_filter_conf.get('index_atr_soft_pct', 0.8) or 0.8)
        atr_hard = float(market_filter_conf.get('index_atr_hard_pct', 0.95) or 0.95)
        atr_soft_hit = atr_pctile >= atr_soft - 1e-9
        atr_hard_hit = atr_pctile >= atr_hard - 1e-9
        vol_ceiling = float(market_filter_conf.get('vol_ann_hard_ceiling', 0.0) or 0.0)
        vol_hard_hit = vol_ceiling > 0.0 and float(index_vol_annualized or 0.0) >= vol_ceiling - 1e-9
        global_hard = False
        try:
            epu_hard = market_filter_conf.get('us_epu_hard_pct')
            dxy_hard = market_filter_conf.get('dxy_hard_pct')
            spx_dd_hard = market_filter_conf.get('spx_drawdown_hard_pct')
            if epu_hard is not None and float(getattr(mr, 'epu_us_percentile', 0.0) or 0.0) >= float(epu_hard):
                global_hard = True
            if dxy_hard is not None and float(getattr(mr, 'dxy_percentile', 0.0) or 0.0) >= float(dxy_hard):
                global_hard = True
            if spx_dd_hard is not None and float(getattr(mr, 'spx_drawdown_pct', 0.0) or 0.0) >= float(spx_dd_hard):
                global_hard = True
        except Exception:
            global_hard = True
        guard_flags = {
            'guard_new': bool(guard_new),
            'atr_soft': bool(atr_soft_hit),
            'atr_hard': bool(atr_hard_hit),
            'vol_hard': bool(vol_hard_hit),
            'global_hard': bool(global_hard),
        }
        guard_active = guard_new or atr_soft_hit or atr_hard_hit or vol_hard_hit or global_hard
        if guard_active:
            reasons.append('guard_active')
        neutral_state.update({'guard_flags': guard_flags, 'reasons': reasons})
        is_neutral = not reasons

    mr.neutral_state = neutral_state
    mr.is_neutral = bool(is_neutral)
    if is_neutral:
        th_orig = dict(thresholds)
        base_new = float(th_orig.get('base_new', 0.0))
        base_add = float(th_orig.get('base_add', 0.0))
        new_scale = float(neutral_conf.get('neutral_base_new_scale', 1.0) or 1.0)
        new_floor = float(neutral_conf.get('neutral_base_new_floor', 0.80) or 0.80)
        add_scale = float(neutral_conf.get('neutral_base_add_scale', 1.0) or 1.0)
        add_floor = float(neutral_conf.get('neutral_base_add_floor', 0.80) or 0.80)
        scaled_new = max(base_new * new_scale, base_new * new_floor)
        scaled_add = max(base_add * add_scale, base_add * add_floor)
        scaled_new = max(0.0, min(1.0, scaled_new))
        scaled_add = max(0.0, min(1.0, scaled_add))
        thresholds = dict(thresholds)
        thresholds['base_new'] = scaled_new
        thresholds['base_add'] = scaled_add
        mr.thresholds = thresholds
        mr.neutral_thresholds = {'base_new': scaled_new, 'base_add': scaled_add}
        neutral_state.update({'thresholds_scaled': {'base_new': scaled_new, 'base_add': scaled_add}})
    else:
        mr.neutral_thresholds = {}
        if 'reasons' not in neutral_state:
            neutral_state.update({'reasons': list(reasons)})

    if budget_map:
        if is_neutral and 'neutral' in budget_map:
            mr.buy_budget_frac = _clip_budget(budget_map['neutral'])
        elif not risk_on and 'risk_off' in budget_map:
            mr.buy_budget_frac = _clip_budget(budget_map['risk_off'])
        elif risk_on and 'risk_on' in budget_map:
            mr.buy_budget_frac = _clip_budget(budget_map['risk_on'])

    return mr


def compute_features(
    ticker: str,
    snapshot_row: pd.Series,
    metrics_row: Optional[pd.Series],
    normalizers: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    r = snapshot_row
    def g(col):
        return to_float(r.get(col))
    price = g("Price") or g("P")
    ma20 = g("MA20")
    ma50 = g("MA50")
    rsi_raw = g("RSI14")
    rsi = float(rsi_raw) if rsi_raw is not None else 50.0
    macdh = to_float(metrics_row.get("MACDHist")) if metrics_row is not None else g("MACDHist")
    atr_pct_raw = to_float(metrics_row.get("ATR14_Pct")) if metrics_row is not None else None
    liq_norm_in = to_float(metrics_row.get("LiqNorm")) if metrics_row is not None else None
    beta_val = to_float(metrics_row.get("Beta60D")) if metrics_row is not None else None
    mom_norm = to_float(metrics_row.get("MomRetNorm")) if metrics_row is not None else None
    rs_trend = to_float(metrics_row.get("RS_Trend50")) if metrics_row is not None else None

    def _normalise(key: str, value: Optional[float]) -> Optional[float]:
        if normalizers is None:
            return None
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return None
        stats = normalizers.get(key)
        if not stats:
            return None
        std = float(stats.get("std", 0.0))
        if std <= 0.0:
            return None
        mean = float(stats.get("mean", 0.0))
        val = float(value)
        if not math.isfinite(val):
            return None
        z = (val - mean) / std
        # Allow per-metric z clipping hint; fallback to 8.0 for legacy behaviour
        z_clip = float(stats.get("z_clip", 8.0) or 8.0)
        return max(-z_clip, min(z_clip, z))

    def _sigmoid(z: Optional[float]) -> float:
        if z is None:
            return 0.5
        return 1.0 / (1.0 + math.exp(-float(z)))

    atr_pct = 0.0
    atr_vol_norm = 0.0
    if atr_pct_raw is not None and atr_pct_raw > 0:
        atr_pct = atr_pct_raw / 100.0
        # Normalise daily ATR% so that ~5% move maps to 1.0 (very high volatility)
        atr_vol_norm = max(0.0, min(atr_pct / 0.05, 1.0))

    adtv_k = to_float(metrics_row.get("AvgTurnover20D_k")) if metrics_row is not None else None
    avg_volume_20 = 0.0
    adtv_vnd = 0.0
    if adtv_k is not None and adtv_k > 0:
        adtv_vnd = float(adtv_k) * 1000.0
        if price is not None and price > 0:
            avg_volume_20 = float(adtv_k) / float(price)

    feats = {
        "above_ma20": 1.0 if (price is not None and ma20 is not None and price > ma20) else 0.0,
        "above_ma50": 1.0 if (price is not None and ma50 is not None and price > ma50) else 0.0,
        "rsi": rsi,
        "macdh_pos": 1.0 if (macdh is not None and macdh > 0) else 0.0,
        "atr_pct": atr_pct,
        "atr_vol_norm": atr_vol_norm,
        # Liquidity normalization is expected to be a percentile rank in [0..1]
        # computed upstream (metrics). If missing, default to 0.0 (no boost),
        # avoiding heuristic turnover scaling.
        "liq_norm": (liq_norm_in if liq_norm_in is not None else 0.0),
        "beta": beta_val if beta_val is not None else 1.0,
    }
    feats["adtv20_k"] = float(adtv_k) if adtv_k is not None and adtv_k > 0 else 0.0
    feats["avg_volume_20"] = float(avg_volume_20) if avg_volume_20 else 0.0
    feats["adtv20_vnd"] = float(adtv_vnd)
    if mom_norm is not None:
        feats["mom_norm"] = max(0.0, min(1.0, float(mom_norm)))
    else:
        feats["mom_norm"] = 0.0
    feats["rsi60_score"] = max(0.0, 1.0 - abs((feats["rsi"] - 60.0) / 20.0))
    # Fundamental features (latest values merged via metrics)
    fund_roe = to_float(metrics_row.get("Fund_ROE")) if metrics_row is not None else None
    if fund_roe is not None:
        feats["fund_roe_norm"] = max(-0.5, min(1.0, fund_roe / 0.25))  # allow negative ROE to penalise
    else:
        feats["fund_roe_norm"] = 0.0
    earnings_yield = to_float(metrics_row.get("Fund_EarningsYield")) if metrics_row is not None else None
    if earnings_yield is not None and earnings_yield > 0:
        feats["fund_earnings_yield"] = max(0.0, min(1.0, earnings_yield))
    else:
        feats["fund_earnings_yield"] = -0.5

    z_rsi = _normalise("RSI14", rsi)
    z_liq = _normalise("LiqNorm", liq_norm_in)
    z_mom = _normalise("MomRetNorm", mom_norm)
    z_atr = _normalise("ATR14_Pct", atr_pct_raw)
    z_beta = _normalise("Beta60D", beta_val)
    z_roe = _normalise("Fund_ROE", fund_roe)
    z_ey = _normalise("Fund_EarningsYield", earnings_yield)
    z_rs = _normalise("RS_Trend50", rs_trend)

    feats["trend_component"] = 0.6 * feats["above_ma50"] + 0.4 * feats["above_ma20"]
    feats["momo_component"] = 0.6 * _sigmoid(z_rsi) + 0.4 * feats["macdh_pos"]
    feats["mom_ret_component"] = _sigmoid(z_mom)
    feats["liq_component"] = _sigmoid(z_liq)
    # Volatility guard: higher ATR -> higher component -> penalised by negative weight
    feats["vol_component"] = _sigmoid(z_atr)
    feats["beta_component"] = _sigmoid(z_beta)
    feats["fund_roe_component"] = _sigmoid(z_roe)
    feats["fund_ey_component"] = _sigmoid(z_ey)
    feats["rs_component"] = _sigmoid(z_rs)

    return feats


def conviction_score(feats: Dict[str, float], sector: str, regime: MarketRegime, ticker: Optional[str] = None) -> float:
    w = regime.weights
    # Optional scaling for fundamentals to align horizon — sourced from policy only
    fund_scale = 1.0
    try:
        if isinstance(w, dict) and w.get('fund_scale') is not None:
            fund_scale = float(w.get('fund_scale'))
    except Exception:
        fund_scale = 1.0
    for k in ("w_trend", "w_momo", "w_mom_ret", "w_liq", "w_vol_guard", "w_beta", "w_sector", "w_sector_sent"):
        if k not in w:
            raise SystemExit(f"Missing weight '{k}' in regime.weights")
    for k in ("w_roe", "w_earnings_yield"):
        if k not in w:
            raise SystemExit(f"Missing weight '{k}' in regime.weights")
    trend = feats.get("trend_component")
    if trend is None:
        trend = 0.6 * feats.get("above_ma50", 0.0) + 0.4 * feats.get("above_ma20", 0.0)
    momo = feats.get("momo_component")
    if momo is None:
        momo = 0.6 * feats.get("rsi60_score", 0.0) + 0.4 * feats.get("macdh_pos", 0.0)
    mom_ret = feats.get("mom_ret_component", feats.get("mom_norm", 0.0))
    liq = feats.get("liq_component", feats.get("liq_norm", 0.0))
    vol_guard = feats.get("vol_component")
    if vol_guard is None:
        vol_guard = max(0.0, min(feats.get("atr_vol_norm", 0.0), 1.0))
    beta_pen = feats.get("beta_component")
    if beta_pen is None:
        # Fallback: reconstruct beta_component via z from regime.normalizers if available
        try:
            import math as _m
            beta_raw = feats.get("beta", None)
            norms = getattr(regime, 'normalizers', {}) or {}
            st = norms.get('Beta60D') if isinstance(norms, dict) else None
            if st and beta_raw is not None and _m.isfinite(float(beta_raw)):
                std = float(st.get('std', 0.0))
                if std > 0:
                    mean = float(st.get('mean', 0.0))
                    zc = float(st.get('z_clip', 8.0) or 8.0)
                    z = (float(beta_raw) - mean) / std
                    z = max(-zc, min(zc, z))
                    beta_pen = 1.0 / (1.0 + _m.exp(-z))
        except Exception:
            beta_pen = None
        if beta_pen is None:
            # Last-resort smooth fallback (centred at beta=1.0, scale 0.2)
            try:
                import math as _m2
                b = float(feats.get('beta', 1.0) or 1.0)
                z = (b - 1.0) / 0.2
                beta_pen = 1.0 / (1.0 + _m2.exp(-z))
            except Exception:
                beta_pen = 0.5
    # Use smoothed sector strength in [0..1]; fallback to binary top-sector if rank missing
    sector_boost = float(regime.sector_strength_rank.get(sector, 0.0))
    if sector_boost == 0.0 and sector in set(regime.top_sectors):
        sector_boost = 1.0
    sector_sent = float(regime.sector_bias.get(sector, 0.0))
    w_ticker_sent = float(w.get("w_ticker_sent", 0.0))
    global_sent = 0.0
    try:
        global_sent = float(getattr(regime, 'market_bias', 0.0) or 0.0)
    except Exception:
        global_sent = 0.0
    ticker_sent = float(regime.ticker_bias.get(ticker, 0.0)) if ticker else 0.0
    ticker_sent = ticker_sent + global_sent
    fund_roe = feats.get("fund_roe_component", feats.get("fund_roe_norm", 0.0))
    fund_ey = feats.get("fund_ey_component", feats.get("fund_earnings_yield", 0.0))
    return (
        w["w_trend"]*trend +
        w["w_momo"]*momo +
        w["w_mom_ret"]*mom_ret +
        w["w_liq"]*liq +
        w["w_vol_guard"]*vol_guard +
        w["w_beta"]*beta_pen +
        w["w_sector"]*sector_boost +
        w["w_sector_sent"]*sector_sent +
        w_ticker_sent*ticker_sent +
        fund_scale * w["w_roe"]*fund_roe +
        fund_scale * w["w_earnings_yield"]*fund_ey +
        w.get("w_rs", 0.0)*feats.get("rs_component", 0.5)
    )


def classify_action(is_holding: bool, score: float, feats: Dict[str, float], regime: MarketRegime, thresholds_override: Optional[Dict[str, object]] = None, ticker: Optional[str] = None) -> str:
    th = dict(regime.thresholds)
    if thresholds_override:
        for k, v in thresholds_override.items():
            if k in th and v is not None:
                th[k] = v
    for k in ("base_add", "base_new", "trim_th", "tp_pct", "sl_pct", "exit_on_ma_break"):
        if k not in th:
            raise SystemExit(f"Missing thresholds['{k}'] in regime")
    base_add = float(th["base_add"])
    base_new = float(th["base_new"])
    # Transaction cost penalty on action gates
    try:
        tc = 0.0
        pr = getattr(regime, 'pricing', {}) or {}
        if 'tc_roundtrip_frac' in pr and pr['tc_roundtrip_frac'] is not None:
            tc = float(pr['tc_roundtrip_frac'])
        tc_gate_scale = float(th.get('tc_gate_scale', 0.0) or 0.0)
        if tc_gate_scale > 0.0 and tc > 0.0:
            penalty = tc_gate_scale * tc
            base_add = min(1.0, base_add + penalty)
            base_new = min(1.0, base_new + penalty)
    except Exception:
        pass
    trim_th = float(th["trim_th"])
    tp_pct_eff, sl_pct_eff, tp_sl_info = resolve_tp_sl(th, feats)
    if feats is not None:
        feats['tp_pct_eff'] = tp_pct_eff
        feats['sl_pct_eff'] = sl_pct_eff
        if ticker:
            feats['_tp_sl_info'] = tp_sl_info
    e = th["exit_on_ma_break"]
    exit_on_ma_break = bool(int(e)) if not isinstance(e, bool) else bool(e)
    # Parameterised RSI thresholds (validated in policy)
    exit_rsi_th = float(th["exit_ma_break_rsi"])
    trim_rsi_below_ma20 = float(th["trim_rsi_below_ma20"])
    trim_rsi_macdh_neg = float(th["trim_rsi_macdh_neg"])
    tilt_downgrade_min = float(th["tilt_exit_downgrade_min"])
    # Minimum R-multiple required to allow MA-break downgrade EXIT->TRIM
    try:
        min_r_for_downgrade = float(th.get("exit_downgrade_min_r", 0.0) or 0.0)
    except Exception:
        min_r_for_downgrade = 0.0
    r_mult = 0.0
    try:
        if tp_pct_eff is not None and sl_pct_eff is not None and float(sl_pct_eff) > 0:
            r_mult = float(tp_pct_eff) / float(sl_pct_eff)
    except Exception:
        r_mult = 0.0
    # Session-aware deferral for MA-break exits
    min_phase = str(th.get("exit_ma_break_min_phase", "morning")).strip().lower()
    _phase_order = {"pre": 0, "morning": 1, "lunch": 2, "afternoon": 3, "atc": 4, "post": 5}
    cur_phase = str(getattr(regime, 'phase', 'pre')).strip().lower()
    cur_rank = _phase_order.get(cur_phase, 0)
    min_rank = _phase_order.get(min_phase, 1)

    if is_holding:
        pnl = feats.get("pnl_pct")
        # Helper for quality/tilt gating
        gate = float(th["exit_ma_break_score_gate"]) if "exit_ma_break_score_gate" in th else 0.0
        tilt_val = 0.0
        try:
            if ticker is not None and isinstance(regime.ticker_bias, dict):
                tilt_val = float(regime.ticker_bias.get(ticker, 0.0) or 0.0)
        except Exception:
            tilt_val = 0.0
        if sl_pct_eff is not None and pnl is not None and pnl <= -abs(sl_pct_eff):
            # Hard stop-loss: do not downgrade by bias or conviction
            return "exit"
        if tp_pct_eff is not None and pnl is not None and pnl >= abs(tp_pct_eff):
            return "take_profit"
        if exit_on_ma_break and feats.get("above_ma50", 0.0) == 0 and feats.get("rsi", 50.0) < exit_rsi_th:
            # Volatility gating: when market ATR percentile is elevated, do not downgrade — exit decisively
            try:
                idx_atr_pctile = float(getattr(regime, 'index_atr_percentile', 0.5) or 0.5)
                mf_conf = getattr(regime, 'market_filter', {}) or {}
                atr_soft = float(mf_conf.get('index_atr_soft_pct', 0.80) or 0.80)
                if idx_atr_pctile >= atr_soft:
                    return "exit"
            except Exception:
                pass
            # Soft gate by conviction score: if strong, downgrade to TRIM instead of EXIT
            if gate > 0.0 and score >= gate and r_mult >= min_r_for_downgrade:
                return "trim"
            # If user supplies a positive ticker tilt above threshold, respect tilt by trimming instead of exiting
            if tilt_val >= tilt_downgrade_min > 0.0 and r_mult >= min_r_for_downgrade:
                return "trim"
            # Session-phase deferral: avoid panic exits early in the day unless hard SL hit
            if cur_rank < min_rank and r_mult >= min_r_for_downgrade:
                return "trim"
            return "exit"
        if score >= base_add:
            return "add"
        if score <= trim_th or (feats.get("above_ma20", 0.0) == 0 and feats.get("rsi", 50.0) < trim_rsi_below_ma20) or (feats.get("macdh_pos", 0.0) == 0 and feats.get("rsi", 50.0) < trim_rsi_macdh_neg):
            return "trim"
        return "hold"
    else:
        if score >= base_new:
            return "new"
        return "ignore"


# 3) Engine main flow (orchestration + order decisions)
# No required-files assertion; engine builds all artifacts itself


# no wrapper: use utils.load_universe_from_files directly


"""ensure_pipeline_artifacts moved to scripts.engine.pipeline"""


@dataclass
class Order:
    ticker: str
    side: str
    quantity: int
    limit_price: float
    note: str = ""

    # IO/price utils moved to scripts.order_price and scripts.orders.orders_io


def pick_limit_price(ticker: str, side: str, snap_row: pd.Series, preset_row: Optional[pd.Series], metrics_row: Optional[pd.Series], regime: Optional[MarketRegime] = None) -> float:
    """Pick LO price based on a richer set of presets, with robust fallbacks.

    Strategy:
    - Collect candidates across Cons/Bal/Aggr/Break/MR levels (1/2),
      filter out-of-band, then choose closest sensible price w.r.t market price.
    - BUY prefers the highest candidate <= market price; SELL prefers the lowest
      candidate >= market price. If none, fall back to closest candidate.
    - Finalize with tick rounding and band clipping. Units are nghìn đồng/cp.
    """
    price = to_float(snap_row.get("Price")) or to_float(snap_row.get("P")) or 0.0

    def _infer_atr_thousand(price_k: float, pr: Optional[pd.Series], mr: Optional[pd.Series]) -> float:
        # 1) Prefer ATR14 from presets (already in thousand VND per share if present)
        if pr is not None:
            atr_val = to_float(pr.get("ATR14"))
            if atr_val is not None and atr_val > 0:
                return atr_val
        # 2) Fallback: convert ATR14_Pct from metrics into price units (thousand)
        if mr is not None and price_k is not None and price_k > 0:
            ap = to_float(mr.get("ATR14_Pct"))
            if ap is not None and ap > 0:
                return (ap / 100.0) * price_k
        return 0.0

    atr = _infer_atr_thousand(price, preset_row, metrics_row)

    floor_tick = to_float(preset_row.get("BandFloor_Tick")) if preset_row is not None else None
    ceil_tick = to_float(preset_row.get("BandCeiling_Tick")) if preset_row is not None else None

    def _cand(prefix: str, bs: str, n: int) -> tuple[float | None, int]:
        if preset_row is None:
            return None, 0
        val = to_float(preset_row.get(f"{prefix}_{bs}{n}_Tick"))
        ob = int(to_float(preset_row.get(f"{prefix}_{bs}{n}_OutOfBand")) or 0)
        if val is None or (bs == 'Buy' and floor_tick is not None and val < floor_tick) or (bs == 'Sell' and ceil_tick is not None and val > ceil_tick):
            ob = 1
        return val, ob

    # Choose preset priority per widely-used TA heuristics:
    # - Risk-on: prefer momentum pullback buys (Aggr/Bal), target higher sells (Cons/Bal)
    # - Risk-off: prefer conservative/mean-reversion buys (Cons/MR), quicker sells (MR/Cons)
    risk_on = bool(getattr(regime, 'risk_on', False)) if regime is not None else False
    pricing_conf = getattr(regime, 'pricing', {}) if regime is not None else {}
    pref_key = ("risk_on_buy" if (risk_on and side=="BUY") else
                "risk_on_sell" if (risk_on and side=="SELL") else
                "risk_off_buy" if (not risk_on and side=="BUY") else
                "risk_off_sell")
    if not isinstance(pricing_conf, dict) or pref_key not in pricing_conf:
        raise SystemExit(f"Missing pricing['{pref_key}'] in regime.pricing")
    v = pricing_conf[pref_key]
    if not (isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) > 0):
        raise SystemExit(f"pricing['{pref_key}'] must be a non-empty array of strings")
    prefixes = tuple(v)

    candidates: list[float] = []
    if side == "BUY":
        for prefix in prefixes:
            for n in (1, 2):
                v, ob = _cand(prefix, 'Buy', n)
                if v is not None and ob == 0:
                    candidates.append(v)
    else:
        for prefix in prefixes:
            for n in (1, 2):
                v, ob = _cand(prefix, 'Sell', n)
                if v is not None and ob == 0:
                    candidates.append(v)

    limit_price: float | None = None
    used_fallback = False
    if candidates:
        if side == "BUY":
            le = [c for c in candidates if c <= price]
            limit_price = max(le) if le else min(candidates)
        else:
            ge = [c for c in candidates if c >= price]
            limit_price = min(ge) if ge else max(candidates)

    if limit_price is None:
        if not (isinstance(pricing_conf, dict) and 'atr_fallback_buy_mult' in pricing_conf and 'atr_fallback_sell_mult' in pricing_conf):
            raise SystemExit("Missing pricing['atr_fallback_buy_mult'/'atr_fallback_sell_mult'] in regime.pricing")
        buy_mult = float(pricing_conf['atr_fallback_buy_mult'])
        sell_mult = float(pricing_conf['atr_fallback_sell_mult'])
        limit_price = price - buy_mult * atr if side == "BUY" else price + sell_mult * atr
        used_fallback = True

    limit_price = clip_to_band(limit_price, floor_tick, ceil_tick)
    tick = to_float(metrics_row.get("TickSizeHOSE_Thousand")) if metrics_row is not None else hose_tick_size(limit_price)
    tick = tick or hose_tick_size(limit_price)
    if used_fallback and tick and tick > 0:
        import math
        if side == "BUY":
            limit_price = math.floor(limit_price / tick) * tick
        else:
            limit_price = math.ceil(limit_price / tick) * tick
    else:
        limit_price = round_to_tick(limit_price, tick)
    limit_price = clip_to_band(limit_price, floor_tick, ceil_tick)
    # Enforce near-ceiling guard: cap BUY limit to near_ceiling_pct * ceiling
    try:
        if side == 'BUY' and ceil_tick is not None and regime is not None:
            th_conf_nc = dict(getattr(regime, 'thresholds', {}) or {})
            nc_val = th_conf_nc.get('near_ceiling_pct')
            if nc_val is not None:
                nc = float(nc_val)
                if nc > 0.0:
                    cap_px = nc * float(ceil_tick)
                    if limit_price > cap_px:
                        import math as _mm
                        step_nc = tick if tick else 0.0
                        limit_price = (_mm.floor(cap_px / step_nc) * step_nc) if step_nc > 0 else cap_px
                        limit_price = clip_to_band(limit_price, floor_tick, ceil_tick)
    except Exception:
        pass
    return float(f"{limit_price:.2f}")


_VN_SESSION_SECONDS = 4.25 * 3600.0
_DEPTH_RATIO = 0.015  # fraction of average daily volume assumed at top of book


def _estimate_new_buy_fill_env(
    snap_row: pd.Series,
    metrics_row: Optional[pd.Series],
    lot: int,
    fill_cfg: Dict[str, object],
) -> Optional[Dict[str, float]]:
    price = to_float(snap_row.get("Price")) or to_float(snap_row.get("P"))
    if price is None or price <= 0:
        return None
    tick = None
    if metrics_row is not None:
        tick = to_float(metrics_row.get("TickSizeHOSE_Thousand"))
    if tick is None or tick <= 0:
        tick = hose_tick_size(price)
    if tick is None or tick <= 0:
        return None
    bid = max(tick, price - tick)
    ask = max(bid + tick, price + tick)
    avg_turnover_k = None
    atr_pct = None
    if metrics_row is not None:
        avg_turnover_k = to_float(metrics_row.get("AvgTurnover20D_k"))
        atr_pct = to_float(metrics_row.get("ATR14_Pct"))
    avg_volume_shares = 0.0
    if avg_turnover_k is not None and price > 0:
        avg_volume_shares = max(float(avg_turnover_k) / float(price), 0.0)
    vol_rate = 0.0
    if _VN_SESSION_SECONDS > 0:
        vol_rate = max(0.0, avg_volume_shares / _VN_SESSION_SECONDS)
    base_depth = max(float(lot), avg_volume_shares * _DEPTH_RATIO)
    bid_vol1 = max(base_depth, float(lot))
    ask_vol1 = max(base_depth, float(lot))
    atr_ticks = 0.0
    if atr_pct is not None and price > 0:
        try:
            atr_thousand = float(atr_pct) / 100.0 * float(price)
            atr_ticks = atr_thousand / tick if tick > 0 else 0.0
        except Exception:
            atr_ticks = 0.0
    sigma_per_sec = 0.0
    if atr_ticks > 0 and _VN_SESSION_SECONDS > 0:
        sigma_per_sec = atr_ticks / math.sqrt(_VN_SESSION_SECONDS)
    window_sigma = max(int(fill_cfg.get("window_sigma_s", 45) or 45), 1)
    floor_sigma = tick / math.sqrt(float(window_sigma))
    sigma_ticks = max(sigma_per_sec, floor_sigma)
    if sigma_ticks <= 0:
        sigma_ticks = floor_sigma if floor_sigma > 0 else tick
    cancel_ratio = float(fill_cfg.get("cancel_ratio_per_min", 0.30) or 0.0)
    joiner_factor = float(fill_cfg.get("joiner_factor", 0.05) or 0.0)
    horizon_s = max(int(fill_cfg.get("horizon_s", 60) or 60), 1)
    return {
        "price": float(price),
        "tick": float(tick),
        "bid": float(bid),
        "ask": float(ask),
        "bidVol1": float(bid_vol1),
        "askVol1": float(ask_vol1),
        "sigma_ticks": float(sigma_ticks),
        "vol_rate": float(vol_rate),
        "cancel_ratio": float(cancel_ratio),
        "joiner_factor": float(joiner_factor),
        "horizon_s": float(horizon_s),
    }


def _pof_new_buy(limit_price: float, env: Dict[str, float]) -> Tuple[float, float, float, float, float]:
    tick = env["tick"]
    horizon = max(env.get("horizon_s", 60.0), 1.0)
    sigma_ticks = max(env.get("sigma_ticks", tick), 1e-9)
    bid = env["bid"]
    ask = env["ask"]
    mid = (bid + ask) / 2.0
    d_ticks = max(0.0, (mid - limit_price) / max(tick, 1e-9))
    denom = sigma_ticks * math.sqrt(max(horizon, 1e-6))
    if denom <= 0:
        hit = 0.0
    else:
        z = d_ticks / denom
        phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        hit = max(0.0, min(1.0, 2.0 * (1.0 - phi_z)))
    bid_vol1 = max(env.get("bidVol1", 0.0), 0.0)
    ask_vol1 = max(env.get("askVol1", 0.0), 0.0)
    depth_sum = max(bid_vol1 + ask_vol1, 1e-6)
    obi = max(-1.0, min(1.0, (bid_vol1 - ask_vol1) / depth_sum))
    phi_bid = max(0.1, min(0.9, 0.5 + 0.4 * obi))
    vol_rate = max(env.get("vol_rate", 0.0), 0.0)
    evol_bid = max(0.0, vol_rate * horizon * phi_bid)
    cancel_ratio = max(0.0, min(1.0, env.get("cancel_ratio", 0.0)))
    joiner_factor = max(0.0, env.get("joiner_factor", 0.0))
    cancel_factor = max(0.0, min(1.0, cancel_ratio * horizon / 60.0))
    queue_ahead = bid_vol1 * max(0.0, 1.0 - cancel_factor) + (vol_rate * joiner_factor * horizon)
    queue_ahead = max(queue_ahead, 1.0)
    queue_prob = max(0.0, min(1.0, evol_bid / queue_ahead))
    return hit * queue_prob, hit, queue_prob, d_ticks, obi


def _apply_new_buy_execution(
    ticker: str,
    base_limit: float,
    market_price: Optional[float],
    snap_row: pd.Series,
    metrics_row: Optional[pd.Series],
    lot: int,
    fill_cfg: Dict[str, object],
) -> Tuple[Optional[float], Optional[Dict[str, object]]]:
    if not fill_cfg:
        return base_limit, None
    env = _estimate_new_buy_fill_env(snap_row, metrics_row, lot, fill_cfg)
    diag: Dict[str, object] = {
        "ticker": ticker,
        "base_limit": float(base_limit),
        "market_price": None if market_price is None else float(market_price),
        "candidates": [],
    }
    target_prob = float(fill_cfg.get("target_prob", 0.0) or 0.0)
    diag["target_prob"] = target_prob
    if env is None:
        diag["status"] = "insufficient_data"
        return base_limit, diag
    diag.update({
        "tick": env["tick"],
        "bid": env["bid"],
        "ask": env["ask"],
        "sigma_ticks": env["sigma_ticks"],
        "vol_rate": env["vol_rate"],
        "cancel_ratio": env["cancel_ratio"],
        "joiner_factor": env["joiner_factor"],
        "horizon_s": env["horizon_s"],
    })
    start_limit = min(base_limit, env["bid"])
    if market_price is not None:
        start_limit = min(start_limit, float(market_price))
    tick = env["tick"]
    max_steps = int(fill_cfg.get("max_chase_ticks", 0) or 0)
    no_cross = bool(fill_cfg.get("no_cross", True))
    best = None
    for step in range(max_steps + 1):
        candidate = start_limit + step * tick
        if market_price is not None and candidate > float(market_price):
            candidate = float(market_price)
        if no_cross and candidate >= env["ask"]:
            if step == 0:
                candidate = min(start_limit, env["bid"])
            else:
                break
        candidate = round_to_tick(candidate, tick)
        pof, hit, queue, d_ticks, obi = _pof_new_buy(candidate, env)
        cand_diag = {
            "step": step,
            "limit": candidate,
            "pof": pof,
            "hit": hit,
            "queue": queue,
            "d_ticks": d_ticks,
            "obi": obi,
        }
        diag["candidates"].append(cand_diag)
        if best is None or pof > best["pof"]:
            best = {"step": step, "limit": candidate, "pof": pof}
        if pof >= target_prob and target_prob > 0:
            diag["status"] = "accepted"
            diag["selected_step"] = step
            diag["selected_limit"] = candidate
            diag["selected_pof"] = pof
            return candidate, diag
    if best is not None:
        diag["best_step"] = best["step"]
        diag["best_limit"] = best["limit"]
        diag["best_pof"] = best["pof"]
    if target_prob > 0:
        diag["status"] = "skipped"
        return None, diag
    diag["status"] = "fallback"
    return base_limit, diag

def _market_tone_sentence(regime: MarketRegime) -> str:
    """Construct a qualitative market tone based on regime diagnostics."""

    idx_chg = float(getattr(regime, 'index_change_pct', 0.0) or 0.0)
    trend = float(getattr(regime, 'trend_strength', 0.0) or 0.0)
    prob = float(getattr(regime, 'risk_on_probability', 0.0) or 0.0)
    breadth_short = float(getattr(regime, 'breadth_hint', 0.0) or 0.0)
    breadth_long = float((getattr(regime, 'model_components', {}) or {}).get('breadth_long', breadth_short) or 0.0)
    atr_pct = float(getattr(regime, 'index_atr14_pct', 0.0) or 0.0) * 100.0

    abs_delta = abs(idx_chg)
    atr_ratio = abs_delta / atr_pct if atr_pct > 0 else None

    strong_breadth = breadth_short >= 0.55 and breadth_long >= 0.50
    weak_breadth = breadth_short <= 0.45 and breadth_long <= 0.45
    decisively_positive = prob >= 0.60 and trend > 0 and idx_chg >= 0
    decisively_negative = prob <= 0.40 or trend < 0 or idx_chg < 0

    if decisively_positive and strong_breadth:
        return "Thị trường đang khỏe, dòng tiền nghiêng về bên mua"
    if atr_ratio is not None and idx_chg < 0 and atr_ratio >= 0.8 and weak_breadth:
        return "Thị trường đang bị bán mạnh; ưu tiên phòng thủ và kiểm soát biên độ lỗ"
    if decisively_negative and weak_breadth:
        return "Thị trường đang yếu, áp lực bán chiếm ưu thế"
    if atr_ratio is not None and atr_ratio <= 0.4 and abs(trend) < 0.01:
        return "Thị trường đi ngang, xu hướng chưa rõ ràng"
    return "Thị trường cân bằng; cần đọc tín hiệu dòng tiền theo từng nhịp"


def _format_market_metrics(regime: MarketRegime) -> str:
    idx_chg = float(getattr(regime, 'index_change_pct', 0.0) or 0.0)
    breadth = float(getattr(regime, 'breadth_hint', 0.0) or 0.0)
    trend = float(getattr(regime, 'trend_strength', 0.0) or 0.0)
    prob = float(getattr(regime, 'risk_on_probability', 0.0) or 0.0)
    atr_pct = float(getattr(regime, 'index_atr14_pct', 0.0) or 0.0) * 100.0

    diag = [f"VNINDEX {idx_chg:+.2f}%", f"Breadth>MA50 {breadth*100:.0f}%", f"Risk-on {prob*100:.0f}%"]
    if atr_pct > 0:
        diag.append(f"ATR14 {atr_pct:.2f}%")
    diag.append(f"Trend vs MA200 {trend*100:.1f}%")
    return "; ".join(diag)


def _session_flow_sentence(phase: Optional[str], regime: MarketRegime) -> str:
    risk_on = bool(getattr(regime, 'risk_on', False))
    if phase in (None, '', 'pre'):
        return "Phiên chưa mở; rà soát danh mục và chờ tín hiệu khớp rõ ràng."
    if phase == 'morning':
        return "Đang trong phiên sáng; giải ngân từng bước và quan sát lực cung-cầu đầu phiên." if risk_on else "Phiên sáng thận trọng; ưu tiên quan sát, chỉ giải ngân khi xuất hiện lực cầu ổn định."
    if phase == 'lunch':
        return "Thị trường tạm nghỉ trưa; chuẩn bị kịch bản cho phiên chiều dựa trên nhịp sáng."
    if phase == 'afternoon':
        return "Phiên chiều đang diễn ra; tận dụng nhịp hồi nhưng luôn giữ kỷ luật chốt lời." if risk_on else "Phiên chiều thiếu lực mua; cân nhắc hạ tỷ trọng và chờ tín hiệu cải thiện."
    if phase == 'atc':
        return "Đang vào phiên ATC; khối lượng có thể biến động mạnh, đặt lệnh cần bám sát khối lượng sổ lệnh."
    if phase == 'post':
        return "Phiên đã đóng; cập nhật kế hoạch cho phiên kế tiếp dựa trên bối cảnh hiện tại."
    return "Theo dõi sát biến động và linh hoạt điều chỉnh khối lượng."


def _suggest_execution_window(regime: MarketRegime, session_summary: pd.DataFrame, session_now: str) -> str:
    phase: Optional[str] = None
    if session_summary is not None and not session_summary.empty and 'SessionPhase' in session_summary.columns:
        val = session_summary.loc[0, 'SessionPhase']
        if pd.notna(val):
            phase = str(val).strip().lower()
    if not phase and session_now:
        phase = session_now.strip().lower()

    tone = _market_tone_sentence(regime)
    session_hint = _session_flow_sentence(phase, regime)
    diagnostics = _format_market_metrics(regime)
    return f"{tone}. {session_hint} ({diagnostics})."


def build_trade_suggestions(
    actions: Dict[str, str],
    scores: Dict[str, float],
    feats_all: Dict[str, Dict[str, float]],
    regime: MarketRegime,
    top_n: int = 3,
) -> List[str]:
    """Build concise, risk-aware trade suggestions from engine outputs.

    The content relies only on existing signals/thresholds decided upstream
    (trend/breadth/volatility/quantile gates) and does not introduce ad‑hoc
    formulas. This keeps logic aligned with accepted practices: follow the
    regime gate, rank by conviction score, and surface risk management.
    """
    if actions is None or scores is None or feats_all is None or regime is None:
        raise ValueError("build_trade_suggestions requires actions, scores, feats_all, regime")
    try:
        n = int(top_n)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("top_n must be int") from exc
    if n <= 0:
        raise ValueError("top_n must be > 0")

    def _sorted_by_score(items: List[str]) -> List[str]:
        pairs = [(t, float(scores.get(t, 0.0) or 0.0)) for t in items]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in pairs[:n]]

    adds = _sorted_by_score([t for t, a in actions.items() if a == "add"])
    news = _sorted_by_score([t for t, a in actions.items() if a in {"new", "new_partial"}])
    trims = [t for t, a in actions.items() if a == "trim"]
    takes = [t for t, a in actions.items() if a == "take_profit"]
    exits = [t for t, a in actions.items() if a == "exit"]

    # Market context
    tone = _market_tone_sentence(regime)
    diag = _format_market_metrics(regime)
    risk_on = bool(getattr(regime, 'risk_on', False))
    buy_frac_eff = float(getattr(regime, 'buy_budget_frac_effective', 0.0) or getattr(regime, 'buy_budget_frac', 0.0))
    top_sectors = list(getattr(regime, 'top_sectors', []) or [])
    atr_pctile = float(getattr(regime, 'index_atr_percentile', 0.0) or 0.0)
    mf = dict(getattr(regime, 'market_filter', {}) or {})
    atr_soft = float(mf.get('index_atr_soft_pct', 0.8) or 0.8)
    atr_hard = float(mf.get('index_atr_hard_pct', 0.95) or 0.95)

    lines: List[str] = []
    lines.append("GỢI Ý GIAO DỊCH (tự động)")
    lines.append(f"- Bối cảnh: {tone} ({diag}).")
    if risk_on and (adds or news):
        add_txt = ", ".join(adds) if adds else "-"
        new_txt = ", ".join(news) if news else "-"
        lines.append(f"- Ưu tiên mua: ADD [{add_txt}] | NEW [{new_txt}].")
    elif (trims or takes or exits):
        lines.append("- Ưu tiên phòng thủ; hạn chế mở vị thế mới nếu chưa có tín hiệu cải thiện.")

    if trims or takes:
        trim_txt = ", ".join(sorted(set(trims))) if trims else "-"
        take_txt = ", ".join(sorted(set(takes))) if takes else "-"
        lines.append(f"- Giảm tỷ trọng/Chốt lời: TRIM [{trim_txt}] | TAKE_PROFIT [{take_txt}].")
    if exits:
        exit_txt = ", ".join(sorted(set(exits)))
        lines.append(f"- Dừng lỗ/Thoát: EXIT [{exit_txt}].")

    if top_sectors:
        lines.append(f"- Ngành nên ưu tiên (breadth/momentum): {', '.join(top_sectors)}.")

    lines.append(f"- Khối lượng mua dự kiến: ~{buy_frac_eff*100:.0f}% NAV (hiệu lực theo regime).")

    # Risk management guidance derived from policy thresholds
    th = dict(getattr(regime, 'thresholds', {}) or {})
    sl_pct = th.get('sl_pct'); tp_pct = th.get('tp_pct')
    try:
        sl_val = float(sl_pct) if sl_pct is not None else None
        tp_val = float(tp_pct) if tp_pct is not None else None
    except Exception:
        sl_val = None; tp_val = None
    rm_bits: List[str] = []
    if tp_val is not None and tp_val > 0:
        rm_bits.append(f"TP≈{tp_val*100:.0f}%")
    if sl_val is not None and sl_val > 0:
        rm_bits.append(f"SL≈{sl_val*100:.0f}%")
    if rm_bits:
        lines.append("- Quản trị rủi ro: " + ", ".join(rm_bits) + " (tham chiếu policy).")

    # Volatility caution using index ATR percentile thresholds
    if atr_pctile >= atr_hard:
        lines.append("- Cảnh báo biến động: ATR của VNINDEX đang rất cao (>= hard). Giảm quy mô lệnh về 0 và chờ bình ổn.")
    elif atr_pctile >= atr_soft:
        lines.append("- Cảnh báo biến động: ATR của VNINDEX cao (>= soft). Giảm khối lượng/giãn giá và đặt SL chặt hơn.")

    return lines


def decide_actions(
    portfolio: pd.DataFrame,
    snapshot: pd.DataFrame,
    metrics: pd.DataFrame,
    presets: pd.DataFrame,
    industry: pd.DataFrame,
    sector_strength: pd.DataFrame,
    session_summary: pd.DataFrame,
    tuning: Dict,
    prices_history: Optional[pd.DataFrame] = None,
    *,
    runtime_overrides: Optional[Dict[str, object]] = None,
):
    # Ensure tuning has centrally-provided policy sections when tests call this helper directly
    # Avoid any baseline reads; only use the runtime merged copy if augmentation is needed.
    try:
        need_scales = not isinstance(tuning, dict) or 'regime_scales' not in tuning or not tuning.get('regime_scales')
    except Exception:
        need_scales = True
    if need_scales:
        try:
            import json as _json, re as _re
            pol_path = OUT_ORDERS_DIR / 'policy_overrides.json'
            if pol_path.exists():
                raw = pol_path.read_text(encoding='utf-8')
                raw = _re.sub(r"/\*.*?\*/", "", raw, flags=_re.S)
                raw = _re.sub(r"(^|\s)//.*$", "", raw, flags=_re.M)
                raw = _re.sub(r"(^|\s)#.*$", "", raw, flags=_re.M)
                obj = _json.loads(raw)
                rs = obj.get('regime_scales') or {}
                if rs:
                    tuning = dict(tuning or {})
                    tuning['regime_scales'] = rs
                # Also backfill market_filter keys from runtime policy when absent/null
                mf_src = (obj.get('market_filter') or {})
                if mf_src:
                    mf_dest = dict((tuning.get('market_filter') or {}))
                    required_keys = (
                        'risk_off_index_drop_pct','risk_off_trend_floor','risk_off_breadth_floor','leader_min_rsi',
                        'leader_min_mom_norm','leader_require_ma20','leader_require_ma50','market_score_soft_floor',
                        'market_score_hard_floor','leader_max','risk_off_drawdown_floor','index_atr_soft_pct','index_atr_hard_pct',
                        'idx_chg_smoothed_hard_drop','trend_norm_hard_floor','vol_ann_hard_ceiling','guard_new_scale_cap',
                        'atr_soft_scale_cap','severe_drop_mult'
                    )
                    for k in required_keys:
                        if k not in mf_dest or mf_dest.get(k) is None:
                            if k in mf_src and mf_src.get(k) is not None:
                                mf_dest[k] = mf_src.get(k)
                    if mf_dest:
                        tuning['market_filter'] = mf_dest
        except Exception:
            pass
    regime = get_market_regime(session_summary, sector_strength, tuning)
    # Apply runtime execution-style overrides (from policy patches) broadly:
    # We allow exec.* keys to override fields in execution/market_filter/orders_ui
    # without mutating the baseline policy. This avoids editing overrides files.
    try:
        if isinstance(runtime_overrides, dict):
            exec_patch = runtime_overrides.get('exec')
            if isinstance(exec_patch, dict) and exec_patch:
                # Make local copies
                exec_conf = dict(getattr(regime, 'execution', {}) or {})
                mf_conf = dict(getattr(regime, 'market_filter', {}) or {})
                ou_conf = dict(getattr(regime, 'orders_ui', {}) or {})
                event_pause_new = False
                event_budget_scale = None
                for k, v in exec_patch.items():
                    key = str(k)
                    if key == 'filter_buy_limit_gt_market':
                        try:
                            exec_conf[key] = int(float(v))
                        except Exception:
                            pass
                    elif key == 'leader_fallback_topk_if_empty':
                        try:
                            mf_conf['leader_fallback_topk_if_empty'] = int(float(v))
                        except Exception:
                            pass
                    elif key == 'write_pre_candidates':
                        try:
                            ou_conf['write_pre_candidates'] = int(float(v))
                        except Exception:
                            pass
                    elif key == 'event_pause_new':
                        try:
                            event_pause_new = bool(int(float(v)))
                        except Exception:
                            event_pause_new = bool(v)
                    elif key == 'event_budget_scale':
                        try:
                            event_budget_scale = max(0.0, min(1.0, float(v)))
                        except Exception:
                            event_budget_scale = None
                try:
                    regime.execution = exec_conf
                    regime.market_filter = mf_conf
                    regime.orders_ui = ou_conf
                except Exception:
                    pass
                # Apply event overrides
                try:
                    if event_budget_scale is not None:
                        base = float(getattr(regime, 'buy_budget_frac', 0.0) or 0.0)
                        eff = base * float(event_budget_scale)
                        regime.buy_budget_frac = eff
                        regime.buy_budget_frac_effective = eff
                except Exception:
                    pass
                try:
                    if event_pause_new:
                        # Set hard cap for NEW entries to 0
                        regime.new_max = 0
                except Exception:
                    pass
    except Exception:
        pass
    snap = snapshot.set_index("Ticker")
    pre = presets.set_index("Ticker") if not presets.empty else pd.DataFrame()
    if not metrics.empty and "AvgTurnover20D_k" in metrics.columns:
        tmp = metrics[["Ticker", "AvgTurnover20D_k"]].dropna().sort_values("AvgTurnover20D_k")
        if not tmp.empty:
            n = float(len(tmp))
            rank_map = {t: i / max(n - 1.0, 1.0) for i, t in enumerate(tmp["Ticker"].tolist())}
            metrics = metrics.copy()
            metrics["LiqNorm"] = metrics["Ticker"].map(rank_map).fillna(0.0)
    if not metrics.empty and "MomRet_12_1" in metrics.columns:
        tmp2 = metrics[["Ticker", "MomRet_12_1"]].dropna().sort_values("MomRet_12_1")
        if not tmp2.empty:
            n2 = float(len(tmp2))
            rank_map2 = {t: i / max(n2 - 1.0, 1.0) for i, t in enumerate(tmp2["Ticker"].tolist())}
            if "MomRetNorm" not in metrics.columns:
                metrics["MomRetNorm"] = 0.0
            metrics["MomRetNorm"] = metrics["Ticker"].map(rank_map2).fillna(metrics.get("MomRetNorm", 0.0))
    met = metrics.set_index("Ticker")
    held = set(portfolio["Ticker"].tolist())
    # Build universe from industry map (preferred), else metrics
    universe = set(industry["Ticker"].tolist()) if not industry.empty else set(metrics["Ticker"].tolist())
    sector_by_ticker = {row["Ticker"]: row.get("Sector") for _, row in industry.iterrows()}
    thresholds_conf = dict(regime.thresholds)

    position_state: Dict[str, Dict[str, object]] = {}
    tp_sl_map: Dict[str, Dict[str, object]] = {}
    state_path = OUT_ORDERS_DIR / 'position_state.csv'
    if state_path.exists():
        try:
            state_df = pd.read_csv(state_path)
            if not state_df.empty and 'Ticker' in state_df.columns:
                for _, row in state_df.iterrows():
                    ticker = str(row.get('Ticker', '')).strip().upper()
                    if not ticker:
                        continue
                    entry: Dict[str, object] = {}
                    for col in ('tp1_done', 'sl_step_hit_50', 'sl_step_hit_80'):
                        if col in row and not pd.isna(row[col]):
                            try:
                                entry[col] = bool(int(row[col]))
                            except Exception:
                                entry[col] = bool(row[col])
                    if 'cooldown_until' in row and isinstance(row['cooldown_until'], str):
                        entry['cooldown_until'] = row['cooldown_until']
                    if entry:
                        position_state[ticker] = entry
        except Exception:
            position_state = {}
    try:
        regime.position_state = position_state
    except Exception:
        pass

    def _as_int(value: object, default: int) -> int:
        try:
            if value is None:
                return int(default)
            return int(float(value))
        except Exception:
            return int(default)

    tp1_lookback = max(1, _as_int(thresholds_conf.get('tp1_hh_lookback', 10), 10))
    trail_lookback = max(1, _as_int(thresholds_conf.get('trail_hh_lookback', 22), 22))

    price_history_df: pd.DataFrame
    if prices_history is not None:
        price_history_df = prices_history.copy()
    else:
        ph_path = OUT_DIR / "prices_history.csv"
        if ph_path.exists():
            try:
                price_history_df = pd.read_csv(ph_path)
            except Exception:
                price_history_df = pd.DataFrame()
        else:
            price_history_df = pd.DataFrame()

    hh_cache: Dict[str, Dict[str, float]] = {}
    if not price_history_df.empty and 'Ticker' in price_history_df.columns:
        ph = price_history_df.copy()
        ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
        if 'Date' in ph.columns:
            ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
            ph = ph.dropna(subset=['Date']).sort_values('Date')
        high_col = None
        for candidate in ('High', 'HighPrice', 'Close'):
            if candidate in ph.columns:
                high_col = candidate
                break
        if high_col is not None:
            ph[high_col] = pd.to_numeric(ph[high_col], errors='coerce')
            tickers_needed = {str(t).upper() for t in held if isinstance(t, str)}
            max_lookback = max(tp1_lookback, trail_lookback)
            for ticker_up in tickers_needed:
                sub = ph[ph['Ticker'] == ticker_up]
                if sub.empty:
                    continue
                series = sub[high_col].dropna()
                if series.empty:
                    continue
                tail = series.tail(max_lookback)
                hh_tp1 = tail.tail(tp1_lookback).max() if tp1_lookback > 0 else float('nan')
                hh_trail = tail.tail(trail_lookback).max() if trail_lookback > 0 else float('nan')
                hh_cache[ticker_up] = {
                    'hh_tp1': float(hh_tp1) if pd.notna(hh_tp1) else None,
                    'hh_trail': float(hh_trail) if pd.notna(hh_trail) else None,
                }

    exec_conf = dict(getattr(regime, 'execution', {}) or {})
    stop_ttl_min = max(1, _as_int(exec_conf.get('stop_ttl_min', 3), 3))
    slip_pct_min = float(exec_conf.get('slip_pct_min', 0.0) or 0.0)
    slip_atr_mult = float(exec_conf.get('slip_atr_mult', 0.0) or 0.0)
    slip_ticks_min = max(0, _as_int(exec_conf.get('slip_ticks_min', 1), 1))
    flash_k_atr = float(exec_conf.get('flash_k_atr', 0.0) or 0.0)

    sl_pct_conf = float(thresholds_conf.get('sl_pct', 0.0) or 0.0)
    sl_atr_mult_conf = thresholds_conf.get('sl_atr_mult')
    sl_atr_mult_conf = float(sl_atr_mult_conf) if sl_atr_mult_conf is not None else 0.0
    tp1_frac = float(thresholds_conf.get('tp1_frac', thresholds_conf.get('tp_trim_frac', 0.4)) or 0.0)
    be_buffer_pct = float(thresholds_conf.get('be_buffer_pct', 0.0) or 0.0)
    trail_atr_mult = float(thresholds_conf.get('trail_atr_mult', 0.0) or 0.0)
    trim_rsi_threshold = float(thresholds_conf.get('trim_rsi_below_ma20', 45.0) or 45.0)

    sell_meta: Dict[str, Dict[str, object]] = {}
    ttl_overrides: Dict[str, int] = {}

    def _snap_val(row: pd.Series, cols: tuple[str, ...]) -> Optional[float]:
        for col in cols:
            if col in row.index:
                val = to_float(row.get(col))
                if val is not None:
                    return float(val)
        return None

    # Strict data validation: required metrics must exist and be non-null for all candidates
    # This surfaces upstream data issues instead of silently defaulting to neutral values.
    non_index = {t for t in snap.index.astype(str) if str(t).upper() not in {'VNINDEX','VN30','VN100'}}
    required_cols_base = []
    # Require LiqNorm only if policy enforces a positive floor
    try:
        th_conf = (tuning.get('thresholds', {}) if isinstance(tuning, dict) else {}) or {}
        min_liq_floor = float(th_conf.get('min_liq_norm', 0.0) or 0.0)
    except Exception:
        min_liq_floor = 0.0
    required_cols = list(required_cols_base) + (['LiqNorm'] if min_liq_floor > 0.0 else [])
    missing_cols = [c for c in required_cols if c not in metrics.columns]
    if missing_cols:
        raise SystemExit(f"metrics.csv missing required columns: {', '.join(missing_cols)}")
    missing_cells: list[str] = []
    for t in sorted(non_index):
        if t not in met.index:
            missing_cells.append(f"{t}:<row>")
            continue
        mrow = met.loc[t]
        for c in required_cols:
            val = mrow.get(c)
            if pd.isna(val):
                missing_cells.append(f"{t}:{c}")
    if missing_cells:
        preview = ", ".join(missing_cells[:10]) + (" ..." if len(missing_cells) > 10 else "")
        raise SystemExit(f"Missing numeric metric values for tickers/fields: {preview}")

    normalizers: Dict[str, Dict[str, float]] = {}
    if not metrics.empty:
        # Robust normalization flag from policy (optional, default False)
        robust = False
        try:
            feats_cfg = dict(tuning.get('features', {}) or {}) if isinstance(tuning, dict) else {}
            robust = bool(int(feats_cfg.get('normalization_robust', 0)))
        except Exception:
            robust = False
        def _collect_stats(col: str) -> None:
            if col not in metrics.columns:
                return
            series = pd.to_numeric(metrics[col], errors='coerce')
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                return
            if robust:
                med = float(series.median())
                mad = float((series - med).abs().median())
                # Consistent MAD estimate for normal dist
                stdr = 1.4826 * mad
                if stdr <= 0.0 or not math.isfinite(stdr):
                    # Fallback to population std if MAD degenerate
                    stdr = float(series.std(ddof=0))
                    mu = float(series.mean())
                    zc = 8.0
                else:
                    mu = med
                    zc = 4.0
                if stdr <= 0.0 or not math.isfinite(stdr):
                    return
                normalizers[col] = {"mean": mu, "std": max(stdr, 1e-6), "z_clip": zc}
            else:
                std = float(series.std(ddof=0))
                if std <= 0.0 or not math.isfinite(std):
                    return
                normalizers[col] = {"mean": float(series.mean()), "std": max(std, 1e-6), "z_clip": 8.0}

        for column in ("RSI14", "LiqNorm", "MomRetNorm", "ATR14_Pct", "Beta60D", "Fund_ROE", "Fund_EarningsYield", "RS_Trend50"):
            _collect_stats(column)
        # Expose normalizers for downstream fallbacks (e.g., beta percentile)
        try:
            setattr(regime, 'normalizers', normalizers)
        except Exception:
            pass

    act: Dict[str, str] = {}
    score: Dict[str, float] = {}
    feats_all: Dict[str, Dict[str, float]] = {}
    debug_filters: Dict[str, List[str]] = {
        "market": [],
        "liquidity": [],
        "near_ceiling": [],
        "limit_gt_market": [],
        "ml_gate": [],
    }
    filtered_records: List[Dict[str, object]] = []

    def _note_filter(ticker: str, reason: str, note: str = "") -> None:
        """Record a filtered candidate with audit context.

        Adds extra fields to support operator audit even when filtered:
        - Side: inferred as BUY for pre-order filters (market/liquidity/near_ceiling)
        - MarketPrice: current price from snapshot (thousand VND/share)
        - LimitPrice: indicative limit from presets if available; otherwise 0.0
        - Quantity: not sized at this stage; set to 0
        """
        side = "BUY"
        market_px = 0.0
        limit_px = 0.0
        try:
            if ticker in snap.index:
                srow = snap.loc[ticker]
                market_px = (to_float(srow.get("Price")) or to_float(srow.get("P")) or 0.0)
            prow = pre.loc[ticker] if ticker in pre.index else None
            mrow = met.loc[ticker] if ticker in met.index else None
            if ticker in snap.index:
                limit_px = float(pick_limit_price(ticker, "BUY", snap.loc[ticker], prow, mrow, regime))
                if not math.isfinite(limit_px):
                    limit_px = 0.0
        except Exception:
            limit_px = float(limit_px) if isinstance(limit_px, (int, float)) else 0.0
        filtered_records.append({
            "Ticker": ticker,
            "Reason": reason,
            "Side": side,
            "Quantity": 0,
            "LimitPrice": float(limit_px) if limit_px is not None else 0.0,
            "MarketPrice": float(market_px) if market_px is not None else 0.0,
            "Note": note,
        })

    # Diagnostics: prefer RSI14 present to avoid hidden defaults
    if 'RSI14' not in metrics.columns:
        try:
            regime.diag_warnings.append('metrics_rsi_column_missing')
        except Exception:
            pass

    for _, row in portfolio.iterrows():
        t = row["Ticker"]
        if t not in snap.index:
            act[t] = "hold"; score[t] = 0.0; feats_all[t] = {}; continue
        s = snap.loc[t]
        m = met.loc[t] if t in met.index else None
        # Enrich snapshot row with TA features from presets/metrics for correct TA computation
        s2 = s.copy()
        if m is not None and "RSI14" in m.index and pd.notna(m.get("RSI14")):
            s2["RSI14"] = m.get("RSI14")
        if t in pre.index:
            pr = pre.loc[t]
            for k in ("MA20", "MA50"):
                val = pr.get(k)
                if pd.notna(val):
                    s2[k] = val
        feats = compute_features(t, s2, m, normalizers)
        # Diagnostics for missing key metrics to avoid hidden defaults
        try:
            if m is None or pd.isna(m.get('RSI14')):
                regime.diag_warnings.append(f'metrics_rsi_missing:{t}')
            if ('LiqNorm' not in metrics.columns) or m is None or pd.isna(m.get('LiqNorm')):
                regime.diag_warnings.append(f'metrics_liqnorm_missing:{t}')
            if ('ATR14_Pct' not in metrics.columns) or m is None or pd.isna(m.get('ATR14_Pct')):
                regime.diag_warnings.append(f'metrics_atr_missing:{t}')
            if ('Beta60D' not in metrics.columns) or m is None or pd.isna(m.get('Beta60D')):
                regime.diag_warnings.append(f'metrics_beta_missing:{t}')
            # Momentum diagnostic: require MomRetNorm or MomRet_12_1 to derive percentile
            has_mom_col = ('MomRetNorm' in metrics.columns) or ('MomRet_12_1' in metrics.columns)
            mom_val = None
            if m is not None:
                if 'MomRetNorm' in metrics.columns and not pd.isna(m.get('MomRetNorm')):
                    mom_val = m.get('MomRetNorm')
                elif 'MomRet_12_1' in metrics.columns and not pd.isna(m.get('MomRet_12_1')):
                    mom_val = m.get('MomRet_12_1')
            if (not has_mom_col) or (mom_val is None or pd.isna(mom_val)):
                regime.diag_warnings.append(f'metrics_mom_missing:{t}')
        except Exception:
            pass
        avg_cost = to_float(row.get("AvgCost"))
        cur_px = to_float(s.get("Price")) or to_float(s.get("P"))
        if avg_cost and avg_cost > 0 and cur_px is not None:
            feats["pnl_pct"] = (cur_px - avg_cost) / avg_cost
        feats_all[t] = feats
        sc = conviction_score(feats, sector_by_ticker.get(t, ""), regime, t)
        score[t] = sc
        cur_qty = int(float(row.get("Quantity") or 0)) if row.get("Quantity") is not None else 0
        ticker_up = str(t).upper()
        hh_info = hh_cache.get(ticker_up, {}) if ticker_up else {}
        hh_tp1 = hh_info.get('hh_tp1') if isinstance(hh_info, dict) else None
        hh_trail = hh_info.get('hh_trail') if isinstance(hh_info, dict) else None
        th_ov = {}
        try:
            raw_ov = regime.ticker_overrides.get(t, {}) if hasattr(regime, 'ticker_overrides') else {}
            if isinstance(raw_ov, dict):
                for key in (
                    "base_add","base_new","trim_th","tp_pct","sl_pct","tp_atr_mult","sl_atr_mult",
                    "tp_floor_pct","tp_cap_pct","sl_floor_pct","sl_cap_pct","tp_trim_frac","exit_on_ma_break",
                    "exit_ma_break_rsi","trim_rsi_below_ma20","trim_rsi_macdh_neg","tp_sl_mode",
                    "sl_trim_step_1_trigger","sl_trim_step_1_frac","sl_trim_step_2_trigger","sl_trim_step_2_frac",
                    "tp1_atr_mult","tp2_atr_mult","trailing_atr_mult","trim_frac_tp1","trim_frac_tp2",
                    "breakeven_after_tp1","time_stop_days","trim_rsi_gate","cooldown_days_after_exit",
                    "partial_entry_enabled","partial_entry_frac","partial_entry_floor_lot","new_partial_buffer"
                ):
                    if key in raw_ov and raw_ov.get(key) is not None:
                        th_ov[key] = raw_ov.get(key)
        except Exception:
            th_ov = {}
        th_local = dict(thresholds_conf)
        if th_ov:
            th_local.update(th_ov)
        tp_pct_eff, sl_pct_eff, tp_sl_info = resolve_tp_sl(th_local, feats)
        if isinstance(feats, dict):
            feats['tp_pct_eff'] = tp_pct_eff
            feats['sl_pct_eff'] = sl_pct_eff
            feats['_tp_sl_info'] = tp_sl_info
        tp_sl_map[str(t).upper()] = dict(tp_sl_info)
        atr_pct_val = float(feats.get('atr_pct', 0.0) or 0.0)
        atr_abs = float(cur_px) * atr_pct_val if (cur_px is not None) else None
        tp_trim_frac_conf = float(th_local.get('tp_trim_frac', 0.4) or 0.4)
        tp1_frac = float(th_local.get('tp1_frac', tp_trim_frac_conf) or tp_trim_frac_conf)
        sl_step1_trigger = float(th_local.get('sl_trim_step_1_trigger', thresholds_conf.get('sl_trim_step_1_trigger', 0.5)) or 0.5)
        sl_step1_frac = float(th_local.get('sl_trim_step_1_frac', thresholds_conf.get('sl_trim_step_1_frac', 0.25)) or 0.25)
        sl_step2_trigger = float(th_local.get('sl_trim_step_2_trigger', thresholds_conf.get('sl_trim_step_2_trigger', 0.8)) or 0.8)
        sl_step2_frac = float(th_local.get('sl_trim_step_2_frac', thresholds_conf.get('sl_trim_step_2_frac', 0.35)) or 0.35)
        state_flags = position_state.get(ticker_up, {}) if ticker_up else {}
        tp1_done_state = bool(state_flags.get('tp1_done'))
        sl_step1_done = bool(state_flags.get('sl_step_hit_50'))
        sl_step2_done = bool(state_flags.get('sl_step_hit_80'))
        trailing_level = None
        if tp1_done_state and hh_trail is not None and atr_abs is not None and trail_atr_mult > 0.0:
            try:
                trailing_level = float(hh_trail) - trail_atr_mult * float(atr_abs)
            except Exception:
                trailing_level = None
        hard_stop_level = None
        if sl_pct_eff is not None and avg_cost and avg_cost > 0:
            try:
                hard_stop_level = float(avg_cost) * (1.0 - float(sl_pct_eff))
            except Exception:
                hard_stop_level = None
        pnl_now = feats.get('pnl_pct')
        loss_pct = 0.0
        if pnl_now is not None and float(pnl_now) < 0:
            loss_pct = abs(float(pnl_now))
        meta_decision: Optional[Dict[str, object]] = None
        state_updates: Dict[str, object] = {}
        telemetry_entry = {
            'tp_pct': tp_pct_eff,
            'sl_pct': sl_pct_eff,
            'atr_pct': atr_pct_val,
            'trailing_level': trailing_level,
            'tp1_done': tp1_done_state,
            'sl_step_hit_50': sl_step1_done,
            'sl_step_hit_80': sl_step2_done,
        }
        if ticker_up in tp_sl_map:
            tp_sl_map[ticker_up]['trailing_level'] = trailing_level
            tp_sl_map[ticker_up]['tp1_done'] = tp1_done_state
        default_action = classify_action(
            True,
            sc,
            feats,
            regime,
            thresholds_override=th_ov if th_ov else None,
            ticker=t,
        )
        exit_reason = None
        if cur_qty > 0 and sl_pct_eff is not None and pnl_now is not None and float(pnl_now) <= -abs(float(sl_pct_eff)) + 1e-9:
            exit_reason = 'HARD_SL'
        if exit_reason is None and cur_qty > 0 and trailing_level is not None and cur_px is not None and float(cur_px) <= float(trailing_level) + 1e-9:
            exit_reason = 'TRAILING'
        if exit_reason and cur_qty > 0:
            stop_level = None
            if exit_reason == 'HARD_SL' and hard_stop_level is not None:
                stop_level = float(hard_stop_level)
            elif exit_reason == 'TRAILING' and trailing_level is not None:
                stop_level = float(trailing_level)
            base_price = float(cur_px) if cur_px is not None else float(stop_level or 0.0)
            slip_val = slip_pct_min
            if atr_pct_val > 0.0 and slip_atr_mult > 0.0:
                slip_val = max(slip_pct_min, slip_atr_mult * atr_pct_val)
            meta_decision = {
                'action': 'exit',
                'note': exit_reason,
                'stop_order': True,
                'exit_reason': exit_reason,
                'stop_level': float(stop_level) if stop_level is not None else float(cur_px or 0.0),
                'stop_components': {
                    'hard': hard_stop_level,
                    'trail': trailing_level,
                },
                'base_price': base_price,
                'slip_pct': max(slip_val, 0.0),
                'slip_ticks_min': slip_ticks_min,
                'stop_ttl': stop_ttl_min,
            }
        elif cur_qty > 0 and sl_pct_eff is not None and loss_pct > 0.0:
            step2_threshold = float(sl_pct_eff) * float(sl_step2_trigger)
            step1_threshold = float(sl_pct_eff) * float(sl_step1_trigger)
            if not sl_step2_done and step2_threshold > 0.0 and loss_pct >= step2_threshold - 1e-9:
                meta_decision = {
                    'action': 'trim',
                    'trim_frac': max(0.0, min(1.0, sl_step2_frac)),
                    'note': 'SL_STEP2',
                }
                state_updates['sl_step_hit_80'] = True
            elif not sl_step1_done and step1_threshold > 0.0 and loss_pct >= step1_threshold - 1e-9:
                meta_decision = {
                    'action': 'trim',
                    'trim_frac': max(0.0, min(1.0, sl_step1_frac)),
                    'note': 'SL_STEP1',
                }
                state_updates['sl_step_hit_50'] = True
        if meta_decision is None and cur_qty > 0 and tp_pct_eff is not None and pnl_now is not None and float(pnl_now) >= float(tp_pct_eff) - 1e-9:
            if not tp1_done_state and tp1_frac > 0.0:
                meta_decision = {
                    'action': 'take_profit',
                    'tp_frac': max(0.0, min(1.0, tp1_frac)),
                    'note': 'TP1_ATR',
                }
                state_updates['tp1_done'] = True
        if meta_decision is None and cur_qty > 0:
            rsi_now = feats.get('rsi')
            above_ma20 = feats.get('above_ma20', 1.0)
            above_ma50 = feats.get('above_ma50', 1.0)
            try:
                if (
                    rsi_now is not None
                    and float(rsi_now) < trim_rsi_threshold
                    and (float(above_ma20) <= 0.0 or float(above_ma50) <= 0.0)
                ) and default_action not in ('exit', 'take_profit'):
                    meta_decision = {
                        'action': 'trim',
                        'trim_frac': 0.3,
                        'note': 'MOM_WEAK',
                    }
            except Exception:
                pass
        if meta_decision is not None:
            if state_updates:
                meta_decision['state_updates'] = state_updates
            meta_decision['telemetry'] = telemetry_entry
            sell_meta[t] = meta_decision
            if meta_decision.get('stop_order'):
                ttl_overrides[t] = stop_ttl_min
            act[t] = str(meta_decision.get('action', default_action))
            continue
        act[t] = default_action

    for t, srow in snap.iterrows():
        if t in held or t not in universe:
            continue
        m = met.loc[t] if t in met.index else None
        s2 = srow.copy()
        if m is not None and "RSI14" in m.index and pd.notna(m.get("RSI14")):
            s2["RSI14"] = m.get("RSI14")
        if t in pre.index:
            pr = pre.loc[t]
            for k in ("MA20", "MA50"):
                val = pr.get(k)
                if pd.notna(val):
                    s2[k] = val
        feats = compute_features(t, s2, m, normalizers)
        try:
            if m is None or pd.isna(m.get('RSI14')):
                regime.diag_warnings.append(f'metrics_rsi_missing:{t}')
            if ('LiqNorm' not in metrics.columns) or m is None or pd.isna(m.get('LiqNorm')):
                regime.diag_warnings.append(f'metrics_liqnorm_missing:{t}')
            if ('ATR14_Pct' not in metrics.columns) or m is None or pd.isna(m.get('ATR14_Pct')):
                regime.diag_warnings.append(f'metrics_atr_missing:{t}')
            if ('Beta60D' not in metrics.columns) or m is None or pd.isna(m.get('Beta60D')):
                regime.diag_warnings.append(f'metrics_beta_missing:{t}')
            has_mom_col = ('MomRetNorm' in metrics.columns) or ('MomRet_12_1' in metrics.columns)
            mom_val = None
            if m is not None:
                if 'MomRetNorm' in metrics.columns and not pd.isna(m.get('MomRetNorm')):
                    mom_val = m.get('MomRetNorm')
                elif 'MomRet_12_1' in metrics.columns and not pd.isna(m.get('MomRet_12_1')):
                    mom_val = m.get('MomRet_12_1')
            if (not has_mom_col) or (mom_val is None or pd.isna(mom_val)):
                regime.diag_warnings.append(f'metrics_mom_missing:{t}')
        except Exception:
            pass
        sc = conviction_score(feats, sector_by_ticker.get(t, ""), regime, t)
        score[t] = sc
        feats_all[t] = feats
        th_ov = {}
        try:
            raw_ov = regime.ticker_overrides.get(t, {}) if hasattr(regime, 'ticker_overrides') else {}
            if isinstance(raw_ov, dict):
                for key in (
                    "base_add","base_new","trim_th","tp_pct","sl_pct","tp_atr_mult","sl_atr_mult",
                    "tp_floor_pct","tp_cap_pct","sl_floor_pct","sl_cap_pct","tp_trim_frac","exit_on_ma_break",
                    "exit_ma_break_rsi","trim_rsi_below_ma20","trim_rsi_macdh_neg","tp_sl_mode",
                    "sl_trim_step_1_trigger","sl_trim_step_1_frac","sl_trim_step_2_trigger","sl_trim_step_2_frac"
                ):
                    if key in raw_ov and raw_ov.get(key) is not None:
                        th_ov[key] = raw_ov.get(key)
        except Exception:
            th_ov = {}
        action = classify_action(False, sc, feats, regime, thresholds_override=th_ov if th_ov else None, ticker=t)
        if action == "new":
            act[t] = "new"
        info_new = feats.get('_tp_sl_info') if isinstance(feats, dict) else None
        if info_new:
            try:
                tp_sl_map[str(t).upper()] = dict(info_new)
            except Exception:
                tp_sl_map[str(t).upper()] = {'tp_pct': info_new.get('tp_pct'), 'sl_pct': info_new.get('sl_pct')}

    th = dict(regime.thresholds)
    q_add = float(th["q_add"]); q_new = float(th["q_new"]); base_add = float(th["base_add"]); base_new = float(th["base_new"])
    # Snapshot pre-filter BUY candidates (before quantile/safety/market guard)
    try:
        pre_candidates = []
        for t_key, action0 in act.items():
            if action0 in {"new", "new_partial", "add"}:
                pre_candidates.append((t_key, action0))
        try:
            regime.pre_buy_candidates = pre_candidates
        except Exception:
            pass
    except Exception:
        try:
            regime.pre_buy_candidates = []
        except Exception:
            pass

    # Apply transaction cost penalty consistently to gating thresholds
    try:
        tc = 0.0
        pr = getattr(regime, 'pricing', {}) or {}
        if 'tc_roundtrip_frac' in pr and pr['tc_roundtrip_frac'] is not None:
            tc = float(pr['tc_roundtrip_frac'])
        tc_gate_scale = float(th.get('tc_gate_scale', 0.0) or 0.0)
        if tc_gate_scale > 0.0 and tc > 0.0:
            penalty = tc_gate_scale * tc
            base_add = min(1.0, base_add + penalty)
            base_new = min(1.0, base_new + penalty)
    except Exception:
        pass

    state_local = dict(getattr(regime, 'neutral_state', {}) or {})
    if 'config' in state_local and isinstance(state_local['config'], dict):
        neutral_conf = dict(state_local['config'])
    else:
        neutral_conf = {}
    neutral_active = bool(getattr(regime, 'is_neutral', False))
    partial_ratio = float(neutral_conf.get('partial_threshold_ratio', 0.0) or 0.0)
    partial_entry_enabled_base = bool(int(th.get('partial_entry_enabled', 0)))
    if neutral_active and not partial_entry_enabled_base:
        # Neutral mode historically enabled partial entries even when baseline flag omitted.
        fallback_flag = neutral_conf.get('partial_entry_enabled')
        if fallback_flag is not None:
            try:
                partial_entry_enabled_base = bool(int(fallback_flag))
            except Exception:
                partial_entry_enabled_base = bool(fallback_flag)
        elif partial_ratio > 0.0 or float(neutral_conf.get('partial_entry_frac', 0.0) or 0.0) > 0.0:
            partial_entry_enabled_base = True
    partial_entry_frac_conf = th.get('partial_entry_frac')
    if partial_entry_frac_conf is None:
        partial_entry_frac_conf = neutral_conf.get('partial_entry_frac', 0.30)
    partial_entry_frac = float(partial_entry_frac_conf or 0.0)
    partial_entry_frac = max(0.0, min(1.0, partial_entry_frac))
    partial_allow_leftover = bool(int(neutral_conf.get('partial_allow_leftover', 0)))
    partial_entry_floor_lot = int(float(th.get('partial_entry_floor_lot', 1) or 1))
    partial_entry_floor_lot = max(1, partial_entry_floor_lot)
    buffer_conf = th.get('new_partial_buffer')
    if buffer_conf is None:
        buffer_conf = neutral_conf.get('new_partial_buffer')
    if buffer_conf is None:
        buffer_conf = 0.05
    try:
        new_partial_buffer_base = float(buffer_conf)
    except Exception:
        new_partial_buffer_base = 0.05
    new_partial_buffer_base = max(0.0, new_partial_buffer_base)
    min_new_per_day = int(neutral_conf.get('min_new_per_day', 0) or 0)
    max_new_overrides_per_day = int(neutral_conf.get('max_new_overrides_per_day', 0) or 0)
    add_max_neutral_cap = int(neutral_conf.get('add_max_neutral_cap', 0) or 0)
    partial_entry_set: set[str] = set()
    partial_frac_map: Dict[str, float] = dict(state_local.get('partial_frac_map', {}) or {})
    partial_floor_map: Dict[str, int] = dict(state_local.get('partial_floor_map', {}) or {})
    partial_buffer_map: Dict[str, float] = dict(state_local.get('partial_buffer_map', {}) or {})
    partial_enabled_map: Dict[str, bool] = dict(state_local.get('partial_enabled_map', {}) or {})
    neutral_override_set: set[str] = set()
    neutral_accum_set: set[str] = set()

    ticker_overrides_map = getattr(regime, 'ticker_overrides', {}) if hasattr(regime, 'ticker_overrides') else {}

    def _partial_conf_for(ticker: str) -> tuple[bool, float, float, int]:
        enabled_local = partial_entry_enabled_base
        frac_local = partial_entry_frac
        buffer_local = new_partial_buffer_base
        floor_local = partial_entry_floor_lot
        try:
            raw_conf = ticker_overrides_map.get(ticker, {}) if isinstance(ticker_overrides_map, dict) else {}
            if isinstance(raw_conf, dict):
                if raw_conf.get('partial_entry_enabled') is not None:
                    try:
                        enabled_local = bool(int(raw_conf.get('partial_entry_enabled')))
                    except Exception:
                        enabled_local = bool(raw_conf.get('partial_entry_enabled'))
                if raw_conf.get('partial_entry_frac') is not None:
                    try:
                        frac_local = float(raw_conf.get('partial_entry_frac'))
                    except Exception:
                        pass
                if raw_conf.get('new_partial_buffer') is not None:
                    try:
                        buffer_local = float(raw_conf.get('new_partial_buffer'))
                    except Exception:
                        pass
                if raw_conf.get('partial_entry_floor_lot') is not None:
                    try:
                        floor_local = max(1, int(raw_conf.get('partial_entry_floor_lot')))
                    except Exception:
                        pass
        except Exception:
            pass
        frac_local = max(0.0, min(1.0, float(frac_local)))
        buffer_local = max(0.0, float(buffer_local))
        floor_local = max(1, int(floor_local))
        return enabled_local, frac_local, buffer_local, floor_local

    try:
        state_local.update({
            'partial_entry_frac': partial_entry_frac,
            'partial_allow_leftover': partial_allow_leftover,
            'partial_entry_enabled': partial_entry_enabled_base,
            'partial_entry_floor_lot': partial_entry_floor_lot,
            'new_partial_buffer': new_partial_buffer_base,
        })
        state_local['partial_frac_map'] = partial_frac_map
        state_local['partial_floor_map'] = partial_floor_map
        state_local['partial_buffer_map'] = partial_buffer_map
        state_local['partial_enabled_map'] = partial_enabled_map
        regime.neutral_state = state_local
    except Exception:
        pass

    min_liq = float(th["min_liq_norm"]); near_ceil_pct = float(th["near_ceiling_pct"])
    try:
        cd_days = int(float(th['cooldown_days']))
    except Exception:
        cd_days = 0
    cooldown_block: Dict[str, str] = {}
    if cd_days and cd_days > 0:
        last_actions_path = OUT_ORDERS_DIR / 'last_actions.csv'
        try:
            if last_actions_path.exists():
                hist = pd.read_csv(last_actions_path)
                if not hist.empty and {'Ticker','LastAction','Date'}.issubset(hist.columns):
                    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
                    cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=cd_days)
                    recent = hist[(hist['Date'].notna()) & (hist['Date'] >= cutoff)]
                    cool = recent['Ticker'].astype(str).str.upper().tolist()
                    for ticker_up in cool:
                        cooldown_block[ticker_up] = f"cooldown active ({cd_days}d) after recent exit/take_profit"
        except Exception:
            cooldown_block = {}

    safety_cache: Dict[str, Tuple[bool, Optional[str], Optional[str]]] = {}
    # Optional corporate events gating (Ex-rights/Record/Execution dates)
    # Configured via policy.market_filter.events; disabled by default.
    events_cfg = None
    try:
        mf_conf_local = getattr(regime, 'market_filter', {}) or {}
        events_cfg = mf_conf_local.get('events') if isinstance(mf_conf_local, dict) else None
    except Exception:
        events_cfg = None
    events_enabled = False
    events_no_new = True
    events_t_minus = 1
    events_t_plus = 1
    if isinstance(events_cfg, dict):
        try:
            events_enabled = bool(int(events_cfg.get('enable', 0))) if not isinstance(events_cfg.get('enable', 0), bool) else bool(events_cfg.get('enable', 0))
        except Exception:
            events_enabled = False
        try:
            events_no_new = bool(int(events_cfg.get('no_new_on_event', 1))) if not isinstance(events_cfg.get('no_new_on_event', 1), bool) else bool(events_cfg.get('no_new_on_event', 1))
        except Exception:
            events_no_new = True
        try:
            events_t_minus = max(0, int(events_cfg.get('t_minus', 1) or 1))
        except Exception:
            events_t_minus = 1
        try:
            events_t_plus = max(0, int(events_cfg.get('t_plus', 1) or 1))
        except Exception:
            events_t_plus = 1
    events_df = load_events() if events_enabled else None
    today_date = datetime.now().date()

    def _check_new_safety(ticker: str) -> tuple[bool, Optional[str], Optional[str]]:
        if ticker in safety_cache:
            return safety_cache[ticker]
        ticker_up = str(ticker).upper()
        mi = met.loc[ticker] if ticker in met.index else None
        liq_norm = to_float(mi.get("LiqNorm")) if mi is not None else None
        if min_liq > 0.0:
            if liq_norm is None or liq_norm < min_liq:
                note = f"liq_norm {liq_norm:.3f} < min {min_liq:.3f}" if liq_norm is not None else "liq_norm missing"
                safety_cache[ticker] = (False, "liquidity", note)
                return safety_cache[ticker]
        ceil_px = None
        ref = None
        if ticker in pre.index:
            prow = pre.loc[ticker]
            ceil_px = to_float(prow.get("BandCeiling_Tick"))
            if ceil_px is None:
                ceil_px = to_float(prow.get("BandCeilingRaw"))
            ref = to_float(prow.get("RefPrice"))
            if ceil_px is None and ref is not None:
                try:
                    band = float(getattr(regime, 'micro_daily_band_pct', 0.07) or 0.07)
                except Exception:
                    band = 0.07
                ceil_px = ref * (1.0 + band)
        px = to_float(snap.loc[ticker].get("Price")) if ticker in snap.index else None
        if ceil_px is not None and px is not None and px >= near_ceil_pct * ceil_px:
            safety_cache[ticker] = (False, "near_ceiling", f"price {px:.2f} within {near_ceil_pct:.2f} of ceiling {ceil_px:.2f}")
            return safety_cache[ticker]
        if ticker_up in cooldown_block:
            safety_cache[ticker] = (False, "market", cooldown_block[ticker_up])
            return safety_cache[ticker]
        # Optional: block NEW around corporate event window when enabled
        if events_df is not None and events_no_new:
            try:
                if in_event_window(ticker_up, today_date, t_minus=events_t_minus, t_plus=events_t_plus, df=events_df):
                    note = f"within event window ±{events_t_minus}/{events_t_plus} days"
                    safety_cache[ticker] = (False, "events", note)
                    return safety_cache[ticker]
            except Exception:
                # On any parsing error, do not block
                pass
        safety_cache[ticker] = (True, None, None)
        return safety_cache[ticker]

    # Allow partial entries for near-miss candidates when neutral
    if neutral_active and base_new > 0.0 and partial_ratio > 0.0:
        partial_threshold = base_new * partial_ratio
        candidates = [
            (t, float(score.get(t, 0.0)))
            for t in score.keys()
            if t not in held and t in snap.index and act.get(t) not in {"new", "new_partial"}
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        for t, sc in candidates:
            if sc < partial_threshold - 1e-9 or sc >= base_new - 1e-9:
                continue
            ok, reason, note = _check_new_safety(t)
            if ok:
                enabled_local, frac_local, buffer_local, floor_local = _partial_conf_for(t)
                if not enabled_local:
                    continue
                act[t] = "new_partial"
                partial_entry_set.add(t)
                partial_frac_map[t] = frac_local
                partial_floor_map[t] = floor_local
                partial_buffer_map[t] = buffer_local
                partial_enabled_map[t] = enabled_local
            else:
                if reason:
                    debug_filters.setdefault(reason, []).append(t)
                    _note_filter(t, reason, note or "")

    # thresholds.quantile_pool is validated by schema; default 'subset'
    q_mode = str(th.get('quantile_pool', 'subset')).strip().lower()
    if q_mode in ('full','universe','all'):
        add_pool_scores = [score[t] for t in score.keys()]
        new_pool_scores = [score[t] for t in score.keys()]
    else:
        add_pool_scores = [score[t] for t in held if t in score]
        new_pool_scores = [score[t] for t in act.keys() if t not in held and act.get(t) in {"new", "new_partial"}]
    if add_pool_scores:
        qv_add = float(np.quantile(np.array(add_pool_scores), q_add))
        add_gate = max(base_add, qv_add)
        for t in list(held):
            if act.get(t) == "add" and score.get(t, 0.0) < add_gate:
                act[t] = "hold"
    quantile_filtered: List[Tuple[str, float]] = []
    if new_pool_scores:
        qv_new = float(np.quantile(np.array(new_pool_scores), q_new))
        new_gate = max(base_new, qv_new)
        for t in list(act.keys()):
            if t in held:
                continue
            sc_val = float(score.get(t, 0.0) or 0.0)
            action_now = act.get(t)
            if action_now not in {"new", "new_partial"}:
                continue
            enabled_local, frac_local, buffer_local, floor_local = _partial_conf_for(t)
            lower_bound = new_gate
            if enabled_local and new_gate > 0.0:
                if buffer_local >= 1.0:
                    lower_bound = max(base_new, new_gate - buffer_local)
                else:
                    lower_bound = max(base_new, new_gate * (1.0 - buffer_local))
            if action_now == "new":
                if sc_val >= new_gate - 1e-9:
                    # keep as new; record per-ticker configs
                    partial_enabled_map[t] = enabled_local
                    partial_frac_map[t] = frac_local
                    partial_floor_map[t] = floor_local
                    partial_buffer_map[t] = buffer_local
                    continue
                if enabled_local and sc_val >= lower_bound - 1e-9:
                    act[t] = "new_partial"
                    partial_entry_set.add(t)
                    partial_frac_map[t] = frac_local
                    partial_floor_map[t] = floor_local
                    partial_buffer_map[t] = buffer_local
                    partial_enabled_map[t] = enabled_local
                    continue
                quantile_filtered.append((t, sc_val))
                del act[t]
            elif action_now == "new_partial":
                if not enabled_local or sc_val < lower_bound - 1e-9:
                    quantile_filtered.append((t, sc_val))
                    del act[t]
                    partial_entry_set.discard(t)
                else:
                    partial_frac_map[t] = frac_local
                    partial_floor_map[t] = floor_local
                    partial_buffer_map[t] = buffer_local
                    partial_enabled_map[t] = enabled_local

    # Snapshot pre-filter BUY candidates (before safety/market guard/budget)
    try:
        pre_candidates = []
        for t, action_now in act.items():
            if action_now in {"new", "new_partial", "add"}:
                pre_candidates.append((t, action_now))
        try:
            regime.pre_buy_candidates = pre_candidates
        except Exception:
            pass
    except Exception:
        try:
            regime.pre_buy_candidates = []
        except Exception:
            pass

    # Safety filters for NEW and NEW_PARTIAL candidates
    new_safety_pass: set[str] = set()
    for t in list(act.keys()):
        action = act.get(t)
        if action not in {"new", "new_partial"}:
            continue
        ok, reason, note = _check_new_safety(t)
        if not ok:
            del act[t]
            if action == "new_partial":
                partial_entry_set.discard(t)
                partial_frac_map.pop(t, None)
                partial_floor_map.pop(t, None)
                partial_buffer_map.pop(t, None)
                partial_enabled_map.pop(t, None)
            if reason:
                debug_filters.setdefault(reason, []).append(t)
                _note_filter(t, reason, note or "")
        else:
            enabled_local, frac_local, buffer_local, floor_local = _partial_conf_for(t)
            partial_frac_map[t] = frac_local
            partial_floor_map[t] = floor_local
            partial_buffer_map[t] = buffer_local
            partial_enabled_map[t] = enabled_local
            if action == "new":
                new_safety_pass.add(t)

    # Quantile override when neutral to ensure minimum NEW orders
    if neutral_active and min_new_per_day > 0 and max_new_overrides_per_day > 0:
        current_new = sum(1 for a in act.values() if a == "new")
        needed = max(0, min_new_per_day - current_new)
        overrides_budget = min(needed, max_new_overrides_per_day)
        if overrides_budget > 0 and quantile_filtered:
            candidates = []
            for t, sc in sorted(quantile_filtered, key=lambda x: x[1], reverse=True):
                if t in act or t in held or t not in snap.index:
                    continue
                ok, reason, note = _check_new_safety(t)
                if not ok:
                    if reason:
                        debug_filters.setdefault(reason, []).append(t)
                        _note_filter(t, reason, note or "")
                    continue
                candidates.append((t, sc))
            for t, _ in candidates:
                if overrides_budget <= 0:
                    break
                enabled_local, frac_local, buffer_local, floor_local = _partial_conf_for(t)
                lower_bound = base_new
                if enabled_local and base_new > 0.0:
                    if partial_ratio > 0.0:
                        lower_bound = max(0.0, base_new * partial_ratio)
                    elif buffer_local >= 1.0:
                        lower_bound = max(base_new, base_new - buffer_local)
                    else:
                        lower_bound = max(base_new, base_new * (1.0 - buffer_local))
                sc_val = float(score.get(t, 0.0) or 0.0)
                if enabled_local and sc_val < base_new - 1e-9 and sc_val >= lower_bound - 1e-9:
                    act[t] = "new_partial"
                    partial_entry_set.add(t)
                    partial_frac_map[t] = frac_local
                    partial_floor_map[t] = floor_local
                    partial_buffer_map[t] = buffer_local
                    partial_enabled_map[t] = enabled_local
                else:
                    act[t] = "new"
                neutral_override_set.add(t)
                new_safety_pass.add(t)
                overrides_budget -= 1

    # Also apply near-ceiling prohibition for ADD
    for t in list(act.keys()):
        if act.get(t) != "add":
            continue
        ceil_px = None
        ref = None
        if t in pre.index:
            prow = pre.loc[t]
            ceil_px = to_float(prow.get("BandCeiling_Tick"))
            if ceil_px is None:
                ceil_px = to_float(prow.get("BandCeilingRaw"))
            ref = to_float(prow.get("RefPrice"))
            if ceil_px is None and ref is not None:
                try:
                    band = float(getattr(regime, 'micro_daily_band_pct', 0.07) or 0.07)
                except Exception:
                    band = 0.07
                ceil_px = ref * (1.0 + band)
        px = to_float(snap.loc[t].get("Price")) if t in snap.index else None
        if ceil_px is not None and px is not None and px >= near_ceil_pct * ceil_px:
            act[t] = "hold"
            debug_filters["near_ceiling"].append(t)
            _note_filter(t, "near_ceiling", f"(ADD) price {px:.2f} within {near_ceil_pct:.2f} of ceiling {ceil_px:.2f}")

    mf_conf = getattr(regime, 'market_filter', {}) or {}
    idx_drop_thr = float(mf_conf.get('risk_off_index_drop_pct', 0.5) or 0.0)
    trend_floor = float(mf_conf.get('risk_off_trend_floor', 0.0) or 0.0)
    breadth_floor_raw = float(mf_conf.get('risk_off_breadth_floor', 0.4) or 0.0)
    score_soft = float(mf_conf.get('market_score_soft_floor', 1.0) or 1.0)
    score_hard = float(mf_conf.get('market_score_hard_floor', 0.0) or 0.0)
    idx_chg = float(getattr(regime, 'index_change_pct', 0.0) or 0.0)
    trend_strength = float(getattr(regime, 'trend_strength', 0.0) or 0.0)
    breadth_hint = float(getattr(regime, 'breadth_hint', 0.0) or 0.0)
    market_score = float(getattr(regime, 'market_score', 0.0) or 0.0)
    breadth_floor = _relaxed_breadth_floor(
        breadth_floor_raw,
        mf_conf,
        risk_on_prob=getattr(regime, 'risk_on_probability', 0.0) or 0.0,
        atr_percentile=getattr(regime, 'index_atr_percentile', 0.0) or 0.0,
    )
    guard_new = (
        (idx_drop_thr > 0.0 and idx_chg <= -abs(idx_drop_thr))
        or (trend_strength <= trend_floor)
        or (breadth_hint < breadth_floor)
        or (market_score <= score_hard)
    )

    if guard_new:
        # Build explicit diagnostic note with triggered conditions
        triggers = []
        if idx_drop_thr > 0.0 and idx_chg <= -abs(idx_drop_thr):
            triggers.append(f"smoothed drop {idx_chg:.2%} ≤ floor {(-abs(idx_drop_thr)):.2%}")
        if trend_strength <= trend_floor:
            triggers.append(f"trend {trend_strength:.3f} ≤ floor {trend_floor:.3f}")
        if breadth_hint < breadth_floor:
            triggers.append(f"breadth {breadth_hint:.3f} < floor {breadth_floor:.3f}")
        # ATR context if available
        try:
            atr_pctile = float(getattr(regime, 'index_atr_percentile', 0.0) or 0.0)
            soft = float(mf_conf.get('index_atr_soft_pct', 0.0) or 0.0)
            if atr_pctile >= soft and soft > 0.0:
                triggers.append(f"ATR%ile {atr_pctile:.1%} ≥ soft {soft:.1%} (budget cap active)")
        except Exception:
            pass
        reason_core = "; ".join(triggers) if triggers else "guard conditions met"
        reason_note = f"market filter active – pause new entries ({reason_core})"
        if market_score <= score_hard:
            reason_note += f"; market_score {market_score:.2f} ≤ hard floor {score_hard:.2f}"
        # Always defer ADD when market weak
        for t in list(act.keys()):
            if act.get(t) == "add":
                act[t] = "hold"
                debug_filters["market"].append(t)
                _note_filter(t, "market", f"ADD deferred: {reason_core}")
        # Leader bypass for NEW
        leader_min_rsi = float(mf_conf.get('leader_min_rsi', 1e9))
        leader_min_mom = float(mf_conf.get('leader_min_mom_norm', 1.0))
        req_ma20 = bool(int(mf_conf.get('leader_require_ma20', 0)))
        req_ma50 = bool(int(mf_conf.get('leader_require_ma50', 0)))
        leader_max = int(mf_conf.get('leader_max', 0))
        # Disable bypass under severe market stress conditions
        risk_off_dd_floor = float(mf_conf.get('risk_off_drawdown_floor', 0.0) or 0.0)
        idx_drop_thr = float(mf_conf.get('risk_off_index_drop_pct', 0.0) or 0.0)
        score_hard = float(mf_conf.get('market_score_hard_floor', 0.0) or 0.0)
        bypass_allowed = True
        try:
            dd = float(getattr(regime, 'drawdown_pct', 0.0) or 0.0)
            if risk_off_dd_floor > 0.0 and dd >= risk_off_dd_floor:
                bypass_allowed = False
            # Also disable when smoothed index drop breaches threshold or market_score at/below hard floor
            # For leader-bypass disable check, use raw intraday change (session context)
            idx_sm = float(getattr(regime, 'index_change_pct', 0.0) or 0.0)
            if idx_drop_thr > 0.0 and idx_sm <= -abs(idx_drop_thr):
                bypass_allowed = False
            ms = float(getattr(regime, 'market_score', 0.0) or 0.0)
            if ms <= score_hard:
                bypass_allowed = False
        except Exception:
            bypass_allowed = True
        new_pool = [(t, score.get(t, 0.0)) for t, a in act.items() if a == 'new']
        keep: set[str] = set()
        if bypass_allowed and leader_max > 0 and new_pool:
            # filter leaders
            leaders: list[tuple[str, float]] = []
            for t, sc in new_pool:
                f = feats_all.get(t, {})
                rsi = float(f.get('rsi', 0.0) or 0.0)
                momn = float(f.get('mom_norm', 0.0) or 0.0)
                if rsi < leader_min_rsi or momn < leader_min_mom:
                    continue
                if req_ma20 and float(f.get('above_ma20', 0.0) or 0.0) < 1.0:
                    continue
                if req_ma50 and float(f.get('above_ma50', 0.0) or 0.0) < 1.0:
                    continue
                leaders.append((t, sc))
            leaders = sorted(leaders, key=lambda x: x[1], reverse=True)[:leader_max]
            keep = {t for t, _ in leaders}
            # Fallback: if no leaders and a fallback K is configured, keep top-K NEW by score
            try:
                k_fallback = int(mf_conf.get('leader_fallback_topk_if_empty', 0) or 0)
            except Exception:
                k_fallback = 0
            if not keep and k_fallback > 0:
                # Choose from the original new_pool, highest scores first
                fallback_pick = [t for t, _ in sorted(new_pool, key=lambda x: x[1], reverse=True)[:k_fallback]]
                keep = set(fallback_pick)
        # Remove other NEW
        for t, _ in list(new_pool):
            if t not in keep:
                del act[t]
                debug_filters["market"].append(t)
                _note_filter(t, "market", reason_note if keep else reason_note + "; leader bypass not active")
        # Policy: Under market guard, do not add to existing positions; only allow
        # constrained NEW via leader bypass when bypass is allowed. Budgets are
        # subsequently scaled by ATR/market_score gates downstream.
        regime.add_max = 0
        if bypass_allowed:
            regime.new_max = min(len(keep), leader_max) if keep else 0
        else:
            regime.new_max = 0
            # Severe conditions: zero out buy budget
            try:
                regime.buy_budget_frac = 0.0
            except Exception:
                pass

    # Apply runtime gating overrides (e.g., ML calibrator)
    gate_context = runtime_overrides if isinstance(runtime_overrides, dict) else {}
    gate_map = gate_context.get("gate") if isinstance(gate_context, dict) else {}
    meta_map = gate_context.get("meta") if isinstance(gate_context, dict) else {}
    ml_meta = None
    if isinstance(meta_map, dict):
        for candidate_key in ("cal_ml", "ml", "patch_ml"):
            meta_candidate = meta_map.get(candidate_key)
            if isinstance(meta_candidate, dict):
                ml_meta = meta_candidate
                break
    gate_threshold = 0.65
    if isinstance(ml_meta, dict) and ml_meta.get("p_gate") is not None:
        try:
            gate_threshold = float(ml_meta.get("p_gate"))
        except Exception:
            gate_threshold = 0.65
    if isinstance(gate_map, dict) and gate_map:
        for key, info in gate_map.items():
            try:
                ticker_up = str(key).upper()
            except Exception:
                continue
            matched = [t for t in list(act.keys()) if str(t).upper() == ticker_up]
            if not matched:
                continue
            if not isinstance(info, dict):
                continue
            block = False
            reason = None
            if "decision" in info:
                decision = str(info.get("decision", "")).strip().lower()
                if decision == "block":
                    block = True
                    reason = info.get("reason") or "decision_block"
                elif decision == "allow":
                    block = False
            if not block and "allow" in info and isinstance(info.get("allow"), bool):
                if not info.get("allow"):
                    block = True
                    reason = info.get("reason") or "allow_flag_false"
            if not block and "p_succ" in info:
                try:
                    prob = float(info.get("p_succ"))
                except Exception:
                    prob = None
                if prob is not None:
                    try:
                        local_gate = float(info.get("p_gate")) if info.get("p_gate") is not None else gate_threshold
                    except Exception:
                        local_gate = gate_threshold
                    if prob < local_gate:
                        block = True
                        reason = f"p_succ {prob:.2f} < p_gate {local_gate:.2f}"
            if block and not reason:
                reason = "ml_gate_block"
            if not block:
                continue
            blocked_now: List[str] = []
            for ticker_actual in matched:
                action_now = act.get(ticker_actual)
                if action_now in {"new", "new_partial"}:
                    del act[ticker_actual]
                    partial_entry_set.discard(ticker_actual)
                    partial_frac_map.pop(ticker_actual, None)
                    partial_floor_map.pop(ticker_actual, None)
                    partial_buffer_map.pop(ticker_actual, None)
                    partial_enabled_map.pop(ticker_actual, None)
                    blocked_now.append(ticker_actual)
                elif action_now == "add":
                    act[ticker_actual] = "hold"
                    blocked_now.append(ticker_actual)
            if blocked_now and reason:
                for ticker_actual in blocked_now:
                    debug_filters.setdefault("ml_gate", []).append(ticker_actual)
                    _note_filter(ticker_actual, "ml_gate", reason)

    # Focus top-N add/new (respect neutral caps when active)
    add_names = [t for t, a in act.items() if a == "add"]
    new_names = [t for t, a in act.items() if a == "new"]
    effective_add_cap = int(regime.add_max)
    if neutral_active and add_max_neutral_cap > 0:
        effective_add_cap = min(effective_add_cap, add_max_neutral_cap)
    add_sorted = sorted(add_names, key=lambda x: score.get(x, 0.0), reverse=True)[: effective_add_cap]
    add_capped_count = 0
    for t in add_names:
        if t not in add_sorted:
            act[t] = "hold"
            add_capped_count += 1
    neutral_accum_set = set(add_sorted) if neutral_active else set()
    new_sorted = sorted(new_names, key=lambda x: score.get(x, 0.0), reverse=True)[: int(regime.new_max)]
    for t in new_names:
        if t not in new_sorted:
            del act[t]
    neutral_partial_final = sorted(t for t in partial_entry_set if act.get(t) == "new_partial")
    neutral_override_final = sorted(t for t in neutral_override_set if act.get(t) == "new")
    neutral_accum_final = sorted(neutral_accum_set)
    try:
        regime.neutral_partial_tickers = neutral_partial_final
        regime.neutral_override_tickers = neutral_override_final
        regime.neutral_accum_tickers = neutral_accum_final
        regime.neutral_stats = {
            'is_neutral': neutral_active,
            'partial_count': len(neutral_partial_final),
            'override_count': len(neutral_override_final),
            'add_capped_count': add_capped_count if neutral_active else 0,
            'min_new_per_day': min_new_per_day,
            'max_new_overrides_per_day': max_new_overrides_per_day,
            'partial_entry_frac': partial_entry_frac,
            'partial_allow_leftover': partial_allow_leftover,
        }
        state_local.update({
            'partial_final': neutral_partial_final,
            'override_final': neutral_override_final,
            'add_capped_count': add_capped_count if neutral_active else 0,
        })
        regime.neutral_state = state_local
    except Exception:
        pass
    try:
        regime.stateless_sell_meta = sell_meta
        regime.ttl_overrides = ttl_overrides
        regime.tp_sl_map = tp_sl_map
    except Exception:
        pass
    regime.debug_filters = debug_filters
    regime.filtered_records = filtered_records
    return act, score, feats_all, regime


def allocate_proportional(budget_k: float, items: List[Tuple[str, float]], tau: float = 0.6) -> Dict[str, float]:
    """Distribute `budget_k` (nghìn đồng) theo điểm số mềm (softmax).

    - Bỏ các ứng viên điểm <= 0 và chuẩn hoá điểm về [0,1] theo min-max.
    - Dùng hàm mũ với hệ số "nhiệt" tau=0.6 để tăng tương phản giữa
      các điểm, rồi phân bổ ngân sách theo tỷ trọng e_i / sum(e).
    - Trả về map {ticker -> ngân sách_k}.
    """
    pos = [(t, max(0.0, float(s))) for t, s in items if s > 0]
    if budget_k <= 0 or not pos:
        return {t: 0.0 for t, _ in pos}
    scores = [s for _, s in pos]
    s_min = min(scores); s_max = max(scores)
    import math
    norm = [(s - s_min) / (s_max - s_min) if (s_max - s_min) > 1e-9 else 1.0 for s in scores]
    exps = [math.exp(n / max(tau, 1e-6)) for n in norm]
    z = sum(exps) or 1.0
    return {t: budget_k * (e / z) for (t, _), e in zip(pos, exps)}


def build_orders(
    actions: Dict[str, str],
    portfolio: pd.DataFrame,
    snapshot: pd.DataFrame,
    metrics: pd.DataFrame,
    presets: pd.DataFrame,
    pnl_summary: pd.DataFrame,
    scores: Dict[str, float],
    regime: MarketRegime,
    prices_history: pd.DataFrame,
):
    snap = snapshot.set_index("Ticker"); met = metrics.set_index("Ticker")
    pre = presets.set_index("Ticker") if not presets.empty else pd.DataFrame()
    held_qty = dict(zip(portfolio["Ticker"], portfolio["Quantity"]))
    total_mkt = float(pnl_summary.loc[0, "TotalMarket"]) if not pnl_summary.empty else 0.0
    stateless_meta = dict(getattr(regime, 'stateless_sell_meta', {}) or {})
    ttl_override_map = dict(getattr(regime, 'ttl_overrides', {}) or {})
    tp_sl_context = {str(k).upper(): v for k, v in (getattr(regime, 'tp_sl_map', {}) or {}).items()}

    def _stop_limit_price(
        ticker: str,
        meta: Dict[str, object],
        snap_row: pd.Series,
        preset_row: Optional[pd.Series],
    ) -> float:
        base_price = float(meta.get('base_price', 0.0) or 0.0)
        if base_price <= 0.0:
            base_price = to_float(snap_row.get('Price')) or to_float(snap_row.get('P')) or 0.0
        slip_pct_val = float(meta.get('slip_pct', 0.0) or 0.0)
        slip_pct_val = max(0.0, min(slip_pct_val, 0.99))
        slip_ticks = int(meta.get('slip_ticks_min', 0) or 0)
        ref_price = base_price if base_price > 0.0 else (to_float(snap_row.get('Price')) or 0.0)
        tick = hose_tick_size(ref_price if ref_price > 0.0 else base_price)
        limit_raw = base_price * (1.0 - slip_pct_val)
        limit = round_to_tick(limit_raw, tick)
        if tick > 0.0:
            steps = max(slip_ticks, 1)
            if limit >= base_price or abs(limit - base_price) < 1e-9:
                limit = base_price - steps * tick
        floor = None
        ceil = None
        if preset_row is not None:
            floor = to_float(preset_row.get('BandFloor_Tick')) or to_float(preset_row.get('BandFloorRaw'))
            ceil = to_float(preset_row.get('BandCeiling_Tick')) or to_float(preset_row.get('BandCeilingRaw'))
        limit = clip_to_band(limit, floor, ceil)
        if tick > 0.0 and limit >= base_price:
            limit = base_price - tick
        if floor is not None:
            limit = max(limit, float(floor))
        if limit <= 0.0:
            limit = max(limit, 0.0)
        return float(limit)

    def _estimate_nav() -> float:
        nav = total_mkt if total_mkt and total_mkt > 0 else 0.0
        if nav <= 0.0 and not pnl_summary.empty and "TotalCost" in pnl_summary.columns:
            total_cost = to_float(pnl_summary.loc[0, "TotalCost"])
            if total_cost is not None and total_cost > 0:
                nav = total_cost
        if nav <= 0.0 and not portfolio.empty and "CostValue" in portfolio.columns:
            cost_sum = pd.to_numeric(portfolio["CostValue"], errors='coerce').fillna(0.0).sum()
            if cost_sum > 0:
                nav = float(cost_sum)
        if nav <= 0.0 and held_qty:
            total = 0.0
            for t, q in held_qty.items():
                if q <= 0:
                    continue
                if t in snap.index:
                    px = to_float(snap.loc[t].get('Price')) or to_float(snap.loc[t].get('P'))
                    if px is not None:
                        total += float(q) * float(px)
            nav = total
        return max(float(nav), 0.0)

    nav_reference = _estimate_nav()
    nav_for_caps = max(nav_reference, total_mkt, 0.0)

    # Configurable sizing and risk options (validated by schema; no fallbacks)
    sizing = getattr(regime, 'sizing', None)
    if not isinstance(sizing, dict):
        raise SystemExit("Invalid regime.sizing (expected object validated by schema)")
    # Normalize/validate sizing with schema to ensure defaults (e.g., allocation_model)
    try:
        from scripts.engine.schema import Sizing as _SizingModel
        sizing = _SizingModel.model_validate(sizing).model_dump()
    except Exception as _exc_sz:
        raise SystemExit(f"Invalid sizing config: {_exc_sz}") from _exc_sz
    add_share = float(sizing['add_share'])
    new_share = float(sizing['new_share'])
    tau = float(sizing['softmax_tau'])
    risk_weighting = str(sizing['risk_weighting'])
    risk_alpha = float(sizing['risk_alpha'])
    allocation_model = str(sizing['allocation_model'])
    if allocation_model not in {'softmax', 'risk_budget', 'mean_variance'}:
        raise SystemExit(
            f"Unsupported sizing.allocation_model '{allocation_model}' "
            "(expected 'softmax', 'risk_budget', or 'mean_variance')"
        )
    cov_lookback = int(float(sizing['cov_lookback_days']))
    cov_reg = float(sizing['cov_reg'])
    risk_parity_floor = float(sizing['risk_parity_floor'])
    max_pos_frac = float(sizing['max_pos_frac'])
    max_sector_frac = float(sizing['max_sector_frac'])
    reuse_sell_proceeds_frac = float(sizing['reuse_sell_proceeds_frac'])
    risk_blend_eta = float(sizing.get('risk_blend_eta', 0.0))
    min_names_target = int(float(sizing.get('min_names_target', 0)))
    bl_rf_annual = float(sizing.get('bl_rf_annual', 0.0))
    bl_mkt_prem_annual = float(sizing.get('bl_mkt_prem_annual', 0.0))
    bl_alpha_scale = float(sizing.get('bl_alpha_scale', 0.0))
    market_index_symbol = str(sizing.get('market_index_symbol', 'VNINDEX') or 'VNINDEX').strip().upper()
    tranche_frac = float(sizing.get('tranche_frac', 1.0) or 1.0)

    allocation_diagnostics: Dict[str, Dict[str, object]] = {}

    # Fail-fast: if fundamentals are used in weights, require data presence for relevant tickers
    try:
        w_cfg = dict(getattr(regime, 'weights', {}) or {})
    except Exception:
        w_cfg = {}
    use_fund = (abs(float(w_cfg.get('w_roe', 0.0) or 0.0)) > 1e-12) or (abs(float(w_cfg.get('w_earnings_yield', 0.0) or 0.0)) > 1e-12)
    if use_fund:
        for col in ('Fund_ROE', 'Fund_EarningsYield'):
            if col not in metrics.columns:
                raise SystemExit(f"Missing fundamentals column '{col}' in metrics while weights.fund are non-zero")
        idx_labels = {'VNINDEX','VN30','VN100'}
        missing_list: list[str] = []
        # Only enforce fundamentals for tickers actually held in the portfolio (not the whole universe)
        for t, q in held_qty.items():
            if q is None or int(q) <= 0:
                continue
            if str(t).upper() in idx_labels:
                continue
            if t not in met.index:
                missing_list.append(str(t))
                continue
            row = met.loc[t]
            fr = pd.to_numeric(pd.Series([row.get('Fund_ROE')]), errors='coerce').iloc[0]
            ey = pd.to_numeric(pd.Series([row.get('Fund_EarningsYield')]), errors='coerce').iloc[0]
            if pd.isna(fr) or pd.isna(ey):
                missing_list.append(str(t))
        if missing_list:
            sample = ', '.join(missing_list[:10]) + (f" … (+{len(missing_list)-10})" if len(missing_list) > 10 else '')
            raise SystemExit(f"Fundamentals missing for tickers: {sample}. Provide Fund_ROE and Fund_EarningsYield or set weights.w_roe/w_earnings_yield=0 in policy.")

    # Optional: dynamic caps driven by regime/breadth (exists in sizing; enable flag decides)
    dyn_caps = sizing['dynamic_caps']
    if isinstance(dyn_caps, dict) and str(dyn_caps['enable']) not in ('0', '', 'False', 'false', 'no', 'None'):
        # Compute risk score in [0..1]
        idx = to_float(getattr(regime, 'index_change_pct', 0.0)) or 0.0  # percent units
        br = to_float(getattr(regime, 'breadth_hint', 0.0)) or 0.0  # 0..1
        ro = bool(getattr(regime, 'risk_on', False))
        def _clip01(x: float) -> float:
            return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        idx_norm = _clip01((idx / 1.0 + 1.0) / 2.0)  # map −1%→0, +1%→1
        risk_score = 0.5 * idx_norm + 0.5 * _clip01(br) + (0.1 if ro else -0.1)
        risk_score = _clip01(risk_score)
        # Bounds & blend
        def _fget(name: str) -> float:
            v = float(dyn_caps[name])
            return max(0.0, min(1.0, v))
        pos_min = _fget('pos_min'); pos_max = _fget('pos_max')
        sec_min = _fget('sector_min'); sec_max = _fget('sector_max')
        blend = _fget('blend')
        override_static = str(dyn_caps['override_static']) in ('1','true','True','yes','on')
        dyn_pos = pos_min + (pos_max - pos_min) * risk_score
        dyn_sec = sec_min + (sec_max - sec_min) * risk_score
        if override_static:
            max_pos_frac = dyn_pos
            max_sector_frac = dyn_sec
        else:
            # Mix with static caps if present; otherwise purely dynamic
            max_pos_frac = (1.0 - blend) * (max_pos_frac if max_pos_frac is not None else dyn_pos) + blend * dyn_pos
            max_sector_frac = (1.0 - blend) * (max_sector_frac if max_sector_frac is not None else dyn_sec) + blend * dyn_sec

    require_sector_caps = max_sector_frac > 0.0

    # Sector map and current exposures
    sector_by_ticker: Dict[str, Optional[str]] = {}
    # Prefer sector from metrics (already merged with industry map), fallback to snapshot if present
    def _normalise_sector(val) -> Optional[str]:
        if val is None:
            return None
        if pd.isna(val):
            return None
        text = str(val).strip()
        return text if text else None

    if 'Sector' in metrics.columns:
        sector_by_ticker = {str(t): _normalise_sector(s) for t, s in zip(metrics['Ticker'], metrics['Sector'])}
    elif 'Sector' in snapshot.columns:
        sector_by_ticker = {str(r['Ticker']): _normalise_sector(r.get('Sector')) for _, r in snapshot.iterrows()}
    if 'Sector' in portfolio.columns:
        for _, row in portfolio.iterrows():
            ticker = str(row.get('Ticker'))
            if ticker and sector_by_ticker.get(ticker) is None:
                sector_by_ticker[ticker] = _normalise_sector(row.get('Sector'))
    if require_sector_caps:
        missing_holdings = sorted(
            {
                str(t)
                for t, qty in held_qty.items()
                if qty and qty > 0 and sector_by_ticker.get(str(t)) is None
            }
        )
        if missing_holdings:
            joined = ", ".join(missing_holdings)
            raise SystemExit(
                f"Missing sector metadata for holdings [{joined}] while enforcing sizing.max_sector_frac; "
                "ensure metrics or portfolio data include Sector."
            )
    def _price_of(t: str) -> float:
        if t in snap.index:
            return to_float(snap.loc[t].get('Price')) or to_float(snap.loc[t].get('P')) or 0.0
        return 0.0
    sector_expo_k: Dict[str, float] = {}
    for t, q in held_qty.items():
        px = _price_of(t)
        sec = sector_by_ticker.get(str(t))
        if sec:
            sector_expo_k[sec] = sector_expo_k.get(sec, 0.0) + float(q) * float(px)

    # 1) Pre-compute SELL orders to estimate proceeds (optional reuse of proceeds)
    sell_candidates: List[Order] = []
    expected_proceeds_k = 0.0
    # Sell tax (Thuế bán) fraction is read from policy.pricing.tc_sell_tax_frac (no hidden default)
    SELL_TAX_RATE = 0.0
    try:
        pr_conf = dict(getattr(regime, 'pricing', {}) or {})
    except Exception:
        pr_conf = {}
    if 'tc_sell_tax_frac' in pr_conf and pr_conf['tc_sell_tax_frac'] is not None:
        SELL_TAX_RATE = float(pr_conf['tc_sell_tax_frac'])
    else:
        # Fallback to runtime policy file (centralized source) when tests bypass full tuning
        try:
            import json as _json, re as _re
            pol_path = OUT_ORDERS_DIR / 'policy_overrides.json'
            if pol_path.exists():
                raw = pol_path.read_text(encoding='utf-8')
                raw = _re.sub(r"/\*.*?\*/", "", raw, flags=_re.S)
                raw = _re.sub(r"(^|\s)//.*$", "", raw, flags=_re.M)
                raw = _re.sub(r"(^|\s)#.*$", "", raw, flags=_re.M)
                obj = _json.loads(raw)
                base_tax = (((obj.get('pricing') or {})).get('tc_sell_tax_frac'))
                if base_tax is not None:
                    SELL_TAX_RATE = float(base_tax)
                else:
                    raise SystemExit("Missing pricing.tc_sell_tax_frac in runtime policy")
            else:
                raise SystemExit("Missing pricing.tc_sell_tax_frac in runtime policy")
        except Exception as _exc_tax:
            raise SystemExit(f"Missing pricing.tc_sell_tax_frac in runtime policy: {_exc_tax}")
    for t, a in actions.items():
        if a not in ("trim", "take_profit", "exit") or t not in snap.index:
            continue
        s = snap.loc[t]; m = met.loc[t] if t in met.index else None; p = pre.loc[t] if t in pre.index else None
        meta = stateless_meta.get(t, {}) if isinstance(stateless_meta, dict) else {}
        stop_order = bool(meta.get('stop_order')) if isinstance(meta, dict) else False
        if stop_order:
            limit = _stop_limit_price(t, meta, s, p)
        else:
            limit = pick_limit_price(t, "SELL", s, p, m, regime)
        cur_qty = int(held_qty.get(t, 0) or 0)
        if cur_qty <= 0:
            continue
        th = dict(regime.thresholds)
        if a == "exit":
            qty = (cur_qty // 100) * 100; note = "Bán toàn bộ (Exit)"
        elif a == "take_profit":
            tp_frac = float(meta.get('tp_frac', th.get("tp_trim_frac", 0.3))) if isinstance(meta, dict) else float(th.get("tp_trim_frac", 0.3))
            tp_frac = max(0.1, min(0.9, tp_frac))
            qty = max(int(cur_qty * tp_frac) // 100 * 100, 100)
            qty = min(qty, (cur_qty // 100) * 100); note = "Chốt lời (Take Profit)"
        else:
            trim_frac_meta = meta.get('trim_frac') if isinstance(meta, dict) else None
            if trim_frac_meta is not None:
                try:
                    trim_frac = max(0.1, min(0.5, float(trim_frac_meta)))
                except Exception:
                    trim_frac = 0.3
            else:
                weakness = max(0.0, 0.5 - scores.get(t, 0.0))
                trim_frac = min(0.4, 0.2 + 0.4 * weakness)
            qty = max(int(cur_qty * trim_frac) // 100 * 100, 100)
            qty = min(qty, (cur_qty // 100) * 100); note = "Giảm - Bán bớt"
            # Annotate MA-break downgrade reason when applicable (for operator transparency)
            try:
                th_loc = dict(regime.thresholds)
                e_loc = th_loc.get('exit_on_ma_break')
                exit_ma_on = bool(int(e_loc)) if not isinstance(e_loc, bool) else bool(e_loc)
                rsi_th = float(th_loc.get('exit_ma_break_rsi', 45.0) or 45.0)
                # above_ma50 and RSI from snapshot/metrics/presets
                price_now = to_float(s.get('Price')) or to_float(s.get('P'))
                ma50_val = None
                if p is not None:
                    ma50_val = to_float(p.get('MA50'))
                if ma50_val is None:
                    ma50_val = to_float(s.get('MA50'))
                rsi_now = to_float(m.get('RSI14')) if m is not None else None
                if exit_ma_on and ma50_val is not None and price_now is not None and rsi_now is not None:
                    if price_now <= ma50_val and float(rsi_now) < rsi_th:
                        note += " (MA-break downgrade)"
            except Exception:
                pass
        if qty > 0:
            note_parts: List[str] = [note]
            if stop_order:
                note_parts.append("STOP_FINAL")
            meta_note = meta.get('note') if isinstance(meta, dict) else None
            if meta_note:
                note_parts.append(str(meta_note))
            note = " | ".join(note_parts)
            if stop_order and meta.get('stop_level') is not None:
                try:
                    note = f"{note} @ {float(meta.get('stop_level')):.2f}"
                except Exception:
                    note = note
            sell_candidates.append(Order(ticker=t, side="SELL", quantity=qty, limit_price=limit, note=note))
            gross = float(qty) * float(limit)
            net = gross * (1.0 - SELL_TAX_RATE)
            expected_proceeds_k += net
            new_qty = max(cur_qty - qty, 0)
            held_qty[t] = new_qty
            sec = sector_by_ticker.get(str(t))
            if sec:
                sector_expo_k[sec] = max(0.0, sector_expo_k.get(sec, 0.0) - gross)
            if stop_order and isinstance(meta, dict):
                ttl_val = meta.get('stop_ttl')
                try:
                    if ttl_val is not None:
                        ttl_override_map[t] = int(float(ttl_val))
                except Exception:
                    pass

    # 2) Compute gross buy budget (base + optional reuse from sells)
    target_gross_buy = float(regime.buy_budget_frac) * nav_reference
    if reuse_sell_proceeds_frac > 0.0 and expected_proceeds_k > 0.0:
        target_gross_buy += reuse_sell_proceeds_frac * expected_proceeds_k

    # Optional pre-size writing moved below after filtered_records is initialized

    # 3) Build alloc by score softmax, then optionally risk-adjust budgets
    add_names = [t for t, a in actions.items() if a == "add"]
    new_names = [t for t, a in actions.items() if a in {"new", "new_partial"}]
    new_partial_names = {t for t, a in actions.items() if a == "new_partial"}
    neutral_override_names = {t for t in (getattr(regime, 'neutral_override_tickers', []) or []) if actions.get(t) == "new"}
    neutral_partial_names = {t for t in (getattr(regime, 'neutral_partial_tickers', []) or []) if actions.get(t) == "new_partial"}
    neutral_accum_names = {t for t in (getattr(regime, 'neutral_accum_tickers', []) or []) if actions.get(t) == "add"}
    neutral_active = bool(getattr(regime, 'is_neutral', False))
    state_local = dict(getattr(regime, 'neutral_state', {}) or {})
    partial_entry_frac = float(state_local.get('partial_entry_frac', 0.30) or 0.30)
    partial_entry_frac = max(0.0, min(1.0, partial_entry_frac))
    partial_allow_leftover = bool(state_local.get('partial_allow_leftover', False))
    partial_entry_enabled_state = bool(state_local.get('partial_entry_enabled', False))
    partial_frac_map_state = dict(state_local.get('partial_frac_map', {}) or {})
    partial_floor_map_state = dict(state_local.get('partial_floor_map', {}) or {})
    partial_buffer_map_state = dict(state_local.get('partial_buffer_map', {}) or {})
    partial_enabled_map_state = dict(state_local.get('partial_enabled_map', {}) or {})
    if require_sector_caps:
        missing_candidates = sorted(
            {
                str(t)
                for t in (list(add_names) + list(new_names))
                if sector_by_ticker.get(str(t)) is None
            }
        )
        if missing_candidates:
            joined = ", ".join(missing_candidates)
            raise SystemExit(
                f"Missing sector metadata for tickers [{joined}] required by sizing.max_sector_frac; "
                "ensure metrics/snapshot supply Sector."
            )

    calibration_conf = dict(sizing.get('mean_variance_calibration', {}) or {})
    if (
        allocation_model == 'mean_variance'
        and calibration_conf.get('enable')
        and (add_names or new_names)
    ):
        try:
            outcome = calibrate_mean_variance_params(
                prices_history=prices_history,
                scores=scores,
                tickers=list(set(add_names + new_names)),
                market_symbol=market_index_symbol,
                sector_by_ticker=sector_by_ticker,
                sizing_conf=sizing,
                calibration_conf=calibration_conf,
            )
            risk_alpha = float(outcome.params['risk_alpha'])
            cov_reg = float(outcome.params['cov_reg'])
            bl_alpha_scale = float(outcome.params['bl_alpha_scale'])
            allocation_diagnostics['calibration'] = outcome.diagnostics
        except SystemExit as _exc_cal:
            allocation_diagnostics['calibration'] = {
                'status': 'fallback',
                'reason': str(_exc_cal),
                'params': 'baseline',
            }
        except Exception as _exc_cal2:
            allocation_diagnostics['calibration'] = {
                'status': 'fallback',
                'reason': f'unexpected_error: {_exc_cal2}',
                'params': 'baseline',
            }
    def _allocate_mean_variance(
        budget_k: float,
        items: List[Tuple[str, float]],
        *,
        group_name: str,
    ) -> Dict[str, float]:
        base: Dict[str, float] = {t: 0.0 for t, _ in items}
        if budget_k <= 0.0 or not items:
            return base
        usable = [(t, float(scores.get(t, 0.0))) for t, _ in items if float(scores.get(t, 0.0)) > 0.0]
        if not usable:
            return base
        tickers = [t for t, _ in usable]
        required_cols = {"Date", "Ticker", "Close"}
        if prices_history is None or prices_history.empty:
            raise SystemExit("prices_history is required for mean-variance allocation")
        missing_cols = required_cols - set(prices_history.columns)
        if missing_cols:
            cols_txt = ", ".join(sorted(missing_cols))
            raise SystemExit(f"prices_history missing required columns for mean-variance allocation: {cols_txt}")
        symbols = set(tickers)
        symbols.add(market_index_symbol)
        hist = prices_history[prices_history['Ticker'].isin(symbols)].copy()
        if hist.empty:
            raise SystemExit(
                f"No price history found for tickers {sorted(tickers)} (allocation model mean_variance)"
            )
        hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
        hist = hist.dropna(subset=['Date']).sort_values('Date')
        pivot = hist.pivot(index='Date', columns='Ticker', values='Close')
        pivot = pivot.replace([np.inf, -np.inf], np.nan)
        pivot = pivot.dropna(axis=0, how='all')
        missing_tickers = [t for t in tickers if t not in pivot.columns]
        if missing_tickers:
            raise SystemExit(
                f"Missing price history columns for tickers {missing_tickers} in allocation model mean_variance"
            )
        if market_index_symbol not in pivot.columns:
            raise SystemExit(
                f"Market index '{market_index_symbol}' not found in price history for mean-variance allocation"
            )
        prices_wide = pivot[tickers].copy().dropna(axis=1, how='all')
        prices_wide = prices_wide.dropna(axis=0, how='any')
        if cov_lookback > 0 and len(prices_wide) > cov_lookback + 1:
            prices_wide = prices_wide.tail(cov_lookback + 1)
        if prices_wide.shape[0] < 60:
            raise SystemExit("Need at least 60 aligned price observations for mean-variance allocation")
        market_series = pivot[market_index_symbol].reindex(prices_wide.index).dropna()
        joint_index = prices_wide.index.intersection(market_series.index)
        prices_wide = prices_wide.loc[joint_index]
        market_series = market_series.loc[joint_index]
        if len(market_series) < 60:
            raise SystemExit("Market index history too short for mean-variance allocation (need >= 60 observations)")
        returns = np.log(prices_wide).diff().dropna(how='any')
        if cov_lookback > 0 and len(returns) > cov_lookback:
            returns = returns.tail(cov_lookback)
        if returns.empty:
            raise SystemExit("Insufficient return history for mean-variance allocation")
        cov = compute_cov_matrix(returns, reg=cov_reg).loc[:, returns.columns]

        score_view = pd.Series({t: float(scores.get(t, 0.0)) for t in cov.index}, dtype=float)
        mu_inputs = ExpectedReturnInputs(
            prices=prices_wide[cov.index],
            market_index=market_series,
            rf_annual=bl_rf_annual,
            market_premium_annual=bl_mkt_prem_annual,
            score_view=score_view,
            alpha_scale=bl_alpha_scale,
        )
        mu_annual = compute_expected_returns(mu_inputs)
        try:
            weight_series = solve_mean_variance_weights(
                mu_annual=mu_annual,
                cov=cov,
                risk_alpha=risk_alpha,
                max_pos_frac=max_pos_frac,
                max_sector_frac=max_sector_frac,
                min_names_target=min_names_target,
                sector_by_ticker=sector_by_ticker,
            )
        except Exception as exc:
            raise SystemExit(f"Mean-variance solver failed: {exc}") from exc

        weights = weight_series.copy()
        if risk_blend_eta > 0.0:
            eta = max(0.0, min(1.0, risk_blend_eta))
            try:
                rp_weights = compute_risk_parity_weights(cov)
                rp_weights = rp_weights.reindex(weights.index).fillna(0.0)
                weights = (1.0 - eta) * weights + eta * rp_weights
                weights = weights.clip(lower=0.0)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
            except Exception as exc:
                raise SystemExit(f"Risk parity blend failed: {exc}") from exc

        cov_local = cov.reindex(index=weights.index, columns=weights.index)
        alloc = {ticker: budget_k * float(weights.get(ticker, 0.0)) for ticker in weights.index}
        allocation_diagnostics[group_name] = {
            'weights': {ticker: float(weights.get(ticker, 0.0)) for ticker in weights.index},
            'expected_return_annual': {ticker: float(mu_annual.get(ticker, 0.0)) for ticker in weights.index},
            'portfolio_mu_annual': float(np.dot(weights.to_numpy(), mu_annual.reindex(weights.index).fillna(mu_annual.mean()).to_numpy())),
            'portfolio_vol_annual': float(np.sqrt(252.0 * (weights.to_numpy() @ cov_local.to_numpy() @ weights.to_numpy()))),
            'tickers': list(weights.index),
            'sectors': {ticker: sector_by_ticker.get(str(ticker)) for ticker in weights.index},
        }
        return alloc

    def _allocate_risk_budget(budget_k: float, items: List[Tuple[str, float]]) -> Dict[str, float]:
        base: Dict[str, float] = {t: 0.0 for t, _ in items}
        if budget_k <= 0 or not items:
            return base
        usable: List[Tuple[str, float, Optional[float]]] = []
        vols: List[float] = []
        for t, raw_score in items:
            score_val = max(0.0, float(raw_score))
            if score_val <= 0.0:
                continue
            vol = None
            if t in met.index:
                vol_pct = to_float(met.loc[t].get('ATR14_Pct'))
                if vol_pct is not None and vol_pct > 0:
                    vol = vol_pct / 100.0
            if vol is not None and vol > 0:
                vols.append(vol)
            usable.append((t, score_val, vol))
        if not usable:
            return base
        fallback_vol = float(np.median(vols)) if vols else 0.04
        fallback_vol = max(fallback_vol, 0.01)
        gamma = max(risk_alpha, 1e-3)
        weights: Dict[str, float] = {}
        for t, score_val, vol in usable:
            sigma = vol if (vol is not None and vol > 0) else fallback_vol
            sigma = max(sigma, 1e-4)
            weight = score_val / (gamma * (sigma ** 2))
            weights[t] = max(weight, 0.0)
        total = sum(weights.values())
        if total <= 0.0:
            return base
        for t in base.keys():
            if t in weights:
                base[t] = budget_k * (weights[t] / total)
        return base

    def _allocate_candidates(budget_k: float, items: List[Tuple[str, float]], group_name: str) -> Dict[str, float]:
        if allocation_model == 'mean_variance':
            return _allocate_mean_variance(budget_k, items, group_name=group_name)
        if allocation_model == 'risk_budget':
            return _allocate_risk_budget(budget_k, items)
        return allocate_proportional(budget_k, items, tau=tau)

    add_alloc = _allocate_candidates(
        (add_share * target_gross_buy) if new_names else target_gross_buy,
        [(t, scores.get(t, 0.0)) for t in add_names],
        'add',
    )
    new_alloc = _allocate_candidates(
        (new_share * target_gross_buy) if add_names else target_gross_buy,
        [(t, scores.get(t, 0.0)) for t in new_names],
        'new',
    )

    if partial_entry_enabled_state and partial_entry_frac > 0.0:
        for t in list(new_alloc.keys()):
            if t in new_partial_names or t in neutral_override_names:
                frac_local = float(partial_frac_map_state.get(t, partial_entry_frac) or partial_entry_frac)
                frac_local = max(0.0, min(1.0, frac_local))
                new_alloc[t] = frac_local * float(new_alloc.get(t, 0.0))

    if allocation_model == 'mean_variance' and allocation_diagnostics:
        try:
            eval_map = dict(getattr(regime, 'evaluation', {}) or {})
            eval_map['mean_variance'] = allocation_diagnostics
            regime.evaluation = eval_map
        except Exception:
            pass

    debug_filters = dict(getattr(regime, 'debug_filters', {}) or {})
    for key in ("market", "liquidity", "near_ceiling", "limit_gt_market"):
        debug_filters.setdefault(key, [])
    filtered_records: List[Dict[str, object]] = list(getattr(regime, 'filtered_records', []) or [])

    # Optional: write pre-sized BUY candidates (before quantile/safety/market filters)
    try:
        ou_conf = dict(getattr(regime, 'orders_ui', {}) or {})
        write_pre = bool(int(ou_conf.get('write_pre_candidates', 0)))
    except Exception:
        write_pre = False
    if write_pre:
        try:
            pre_list = list(getattr(regime, 'pre_buy_candidates', []) or [])
            if pre_list:
                ou_conf = dict(getattr(regime, 'orders_ui', {}) or {})
                mode = str(ou_conf.get('pre_size_mode', 'budget_softmax')).strip().lower()
                # Collect scores for candidates
                cand = [(t, float(scores.get(t, 0.0) or 0.0), a) for t, a in pre_list]
                cand = [(t, s if s > 0 else 0.0, a) for (t, s, a) in cand]
                try:
                    lot_sz = int(float(regime.sizing.get('min_lot', 100) or 100))
                except Exception:
                    lot_sz = 100
                lot_sz = max(1, lot_sz)
                if mode == 'score_min_lot':
                    # Map min positive score → 1 lot; others proportional
                    pos_scores = [s for (_, s, _) in cand if s > 0]
                    if pos_scores:
                        s_min = min(pos_scores)
                        for t, s, a in cand:
                            if t not in snap.index:
                                continue
                            srow = snap.loc[t]; mrow = met.loc[t] if t in met.index else None; prow = pre.loc[t] if t in pre.index else None
                            limit0 = pick_limit_price(t, 'BUY', srow, prow, mrow, regime)
                            market_px = to_float(srow.get('Price')) or to_float(srow.get('P'))
                            lots = 1 if s <= 0 or s_min <= 0 else int(round(s / s_min))
                            lots = max(lots, 1)
                            qty_est = lots * lot_sz
                            filtered_records.append({
                                'Ticker': t,
                                'Reason': 'candidate_pre',
                                'Side': 'BUY',
                                'Quantity': int(qty_est),
                                'LimitPrice': float(limit0),
                                'MarketPrice': float(market_px) if market_px is not None else 0.0,
                                'Note': 'pre-sized score_min_lot',
                            })
                    else:
                        # All scores ≤ 0 → default to 1 lot each
                        for t, s, a in cand:
                            if t not in snap.index:
                                continue
                            srow = snap.loc[t]; mrow = met.loc[t] if t in met.index else None; prow = pre.loc[t] if t in pre.index else None
                            limit0 = pick_limit_price(t, 'BUY', srow, prow, mrow, regime)
                            market_px = to_float(srow.get('Price')) or to_float(srow.get('P'))
                            qty_est = lot_sz
                            filtered_records.append({
                                'Ticker': t,
                                'Reason': 'candidate_pre',
                                'Side': 'BUY',
                                'Quantity': int(qty_est),
                                'LimitPrice': float(limit0),
                                'MarketPrice': float(market_px) if market_px is not None else 0.0,
                                'Note': 'pre-sized score_min_lot (all<=0)',
                            })
                else:
                    # Legacy budget_softmax behavior
                    try:
                        add_share = float(regime.sizing.get('add_share', 0.5) or 0.5)
                        new_share = float(regime.sizing.get('new_share', 0.5) or 0.5)
                        tau = float(regime.sizing.get('softmax_tau', 0.6) or 0.6)
                    except Exception:
                        add_share, new_share, tau = 0.5, 0.5, 0.6
                    budget_add_k = max(0.0, target_gross_buy * add_share)
                    budget_new_k = max(0.0, target_gross_buy * new_share)
                    pre_add = [t for t, a in pre_list if a == 'add']
                    pre_new = [t for t, a in pre_list if a in {'new','new_partial'}]
                    items_add = [(t, float(scores.get(t, 0.0) or 0.0)) for t in pre_add]
                    items_new = [(t, float(scores.get(t, 0.0) or 0.0)) for t in pre_new]
                    alloc_add_pre = allocate_proportional(budget_add_k, items_add, tau=tau) if items_add else {}
                    alloc_new_pre = allocate_proportional(budget_new_k, items_new, tau=tau) if items_new else {}
                    for t, s, a in cand:
                        if t not in snap.index:
                            continue
                        srow = snap.loc[t]; mrow = met.loc[t] if t in met.index else None; prow = pre.loc[t] if t in pre.index else None
                        limit0 = pick_limit_price(t, 'BUY', srow, prow, mrow, regime)
                        market_px = to_float(srow.get('Price')) or to_float(srow.get('P'))
                        budget_k = float(alloc_add_pre.get(t, 0.0) if a == 'add' else alloc_new_pre.get(t, 0.0))
                        qty_est = int(max(budget_k / max(limit0, 1e-9) / lot_sz, 0)) * lot_sz
                        filtered_records.append({
                            'Ticker': t,
                            'Reason': 'candidate_pre',
                            'Side': 'BUY',
                            'Quantity': int(qty_est),
                            'LimitPrice': float(limit0),
                            'MarketPrice': float(market_px) if market_px is not None else 0.0,
                            'Note': 'pre-sized budget_softmax',
                        })
        except Exception as _exc_pre:
            print(f"[warn] write_pre_candidates failed: {_exc_pre}")

    def _track_filter(ticker: str, reason: str, note: str = "", *, side: str = "", quantity: int = 0, limit_price: float = 0.0, market_price: float = 0.0) -> None:
        """Track filters during order construction with richer audit fields."""
        debug_filters.setdefault(reason, []).append(ticker)
        try:
            q = int(quantity) if quantity is not None else 0
        except Exception:
            q = 0
        try:
            lp = float(limit_price) if limit_price is not None and math.isfinite(float(limit_price)) else 0.0
        except Exception:
            lp = 0.0
        try:
            mp = float(market_price) if market_price is not None and math.isfinite(float(market_price)) else 0.0
        except Exception:
            mp = 0.0
        filtered_records.append({
            "Ticker": ticker,
            "Reason": reason,
            "Side": side or "",
            "Quantity": q,
            "LimitPrice": lp,
            "MarketPrice": mp,
            "Note": note,
        })

    mf_conf = getattr(regime, 'market_filter', {}) or {}
    idx_drop_thr = float(mf_conf.get('risk_off_index_drop_pct', 0.5) or 0.0)
    trend_floor = float(mf_conf.get('risk_off_trend_floor', 0.0) or 0.0)
    breadth_floor_raw = float(mf_conf.get('risk_off_breadth_floor', 0.4) or 0.0)
    score_soft = float(mf_conf.get('market_score_soft_floor', 1.0) or 1.0)
    score_hard = float(mf_conf.get('market_score_hard_floor', 0.0) or 0.0)
    idx_chg = float(getattr(regime, 'index_change_pct', 0.0) or 0.0)
    trend_strength = float(getattr(regime, 'trend_strength', 0.0) or 0.0)
    breadth_hint = float(getattr(regime, 'breadth_hint', 0.0) or 0.0)
    market_score = float(getattr(regime, 'market_score', 0.0) or 0.0)
    breadth_floor = _relaxed_breadth_floor(
        breadth_floor_raw,
        mf_conf,
        risk_on_prob=getattr(regime, 'risk_on_probability', 0.0) or 0.0,
        atr_percentile=getattr(regime, 'index_atr_percentile', 0.0) or 0.0,
    )
    # Severity multiplier configurable via policy (default 1.5)
    severe_mult = float(mf_conf.get('severe_drop_mult', 1.5) or 1.5)
    severe = idx_drop_thr > 0.0 and idx_chg <= -abs(idx_drop_thr) * severe_mult
    severe = severe or (market_score <= score_hard)
    guard_new = (
        (idx_drop_thr > 0.0 and idx_chg <= -abs(idx_drop_thr))
        or (trend_strength <= trend_floor)
        or (breadth_hint < breadth_floor)
        or (market_score <= score_hard)
    )

    denom = max(score_soft - score_hard, 1e-6)
    score_scale = 1.0
    if market_score < score_soft:
        score_scale = max(0.0, min(1.0, (market_score - score_hard) / denom))

    scale = 1.0
    if guard_new:
        # When guard_new active but not severe, cap budget by guard_new_scale_cap (default 0.4)
        cap = float(mf_conf.get('guard_new_scale_cap', 0.4) or 0.4)
        scale = 0.0 if severe else min(cap, score_scale)
    else:
        scale = min(scale, score_scale)

    # Additional scaling/guard by Index ATR percentile
    atr_pctile = float(getattr(regime, 'index_atr_percentile', 0.0) or 0.0)
    atr_soft = float(mf_conf.get('index_atr_soft_pct', 0.8) or 0.8)
    atr_hard = float(mf_conf.get('index_atr_hard_pct', 0.95) or 0.95)
    atr_soft_cap = float(mf_conf.get('atr_soft_scale_cap', 0.5) or 0.5)
    if atr_pctile >= atr_hard:
        scale = 0.0
    elif atr_pctile >= atr_soft:
        # Cap budget further when ATR is high but below hard stop
        scale = min(scale, atr_soft_cap)

    # Annualized index volatility hard ceiling (if configured): treat as severe risk-off
    try:
        vol_hard = mf_conf.get('vol_ann_hard_ceiling', None)
        if vol_hard is not None:
            vol_now = float(getattr(regime, 'index_vol_annualized', 0.0) or 0.0)
            if vol_now >= float(vol_hard):
                scale = 0.0
    except Exception:
        pass

    # Optional additional guard by global factors (US EPU, DXY, SPX drawdown)
    try:
        epu_soft = mf_conf.get('us_epu_soft_pct', None)
        epu_hard = mf_conf.get('us_epu_hard_pct', None)
        dxy_soft = mf_conf.get('dxy_soft_pct', None)
        dxy_hard = mf_conf.get('dxy_hard_pct', None)
        spx_dd_hard = mf_conf.get('spx_drawdown_hard_pct', None)
        epu_val = getattr(regime, 'epu_us_percentile', 0.0)
        dxy_val = getattr(regime, 'dxy_percentile', 0.0)
        dd_val = getattr(regime, 'spx_drawdown_pct', 0.0)
        if epu_hard is not None and float(epu_val) >= float(epu_hard):
            scale = 0.0
        elif epu_soft is not None and float(epu_val) >= float(epu_soft):
            cap = float(mf_conf.get('guard_new_scale_cap', 0.4) or 0.4)
            scale = min(scale, cap)
        if dxy_hard is not None and float(dxy_val) >= float(dxy_hard):
            scale = 0.0
        elif dxy_soft is not None and float(dxy_val) >= float(dxy_soft):
            cap = float(mf_conf.get('guard_new_scale_cap', 0.4) or 0.4)
            scale = min(scale, cap)
        if spx_dd_hard is not None and float(dd_val) >= float(spx_dd_hard):
            scale = 0.0
    except Exception:
        # Non-fatal; leave scale as-is and continue
        pass

    if scale < 1.0:
        add_alloc = {t: scale * v for t, v in add_alloc.items()}
        new_alloc = {t: scale * v for t, v in new_alloc.items()}

    # Expose effective buy budget fraction for reporting
    try:
        regime.buy_budget_frac_effective = float(regime.buy_budget_frac) * float(scale)
    except Exception:
        regime.buy_budget_frac_effective = float(regime.buy_budget_frac)

    def _risk_parity_alloc(alloc: Dict[str, float]) -> Dict[str, float]:
        total = sum(alloc.values()) or 0.0
        if total <= 0:
            return alloc
        tickers = [t for t, v in alloc.items() if v > 0]
        if len(tickers) <= 1 or prices_history is None or prices_history.empty:
            return alloc
        required_cols = {"Date", "Ticker", "Close"}
        if not required_cols.issubset(prices_history.columns):
            return alloc
        hist = prices_history[prices_history['Ticker'].isin(tickers)].copy()
        if hist.empty:
            return alloc
        hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
        hist = hist.dropna(subset=['Date']).sort_values('Date')
        pivot = hist.pivot(index='Date', columns='Ticker', values='Close')
        pivot = pivot.replace([np.inf, -np.inf], np.nan)
        pivot = pivot.dropna(axis=0, how='all')
        returns = np.log(pivot).diff().dropna(how='any')
        if returns.empty:
            return alloc
        if cov_lookback > 0 and len(returns) > cov_lookback:
            returns = returns.tail(cov_lookback)
        try:
            cov = compute_cov_matrix(returns, reg=cov_reg)
            rp_weights = compute_risk_parity_weights(cov)
        except Exception:
            return alloc
        rp_weights = rp_weights.reindex(tickers).fillna(0.0)
        if rp_weights.sum() <= 0:
            return alloc
        rp_weights = rp_weights / rp_weights.sum()
        base_weights = {t: alloc[t] / total for t in alloc if total > 0}
        floor = max(0.0, min(1.0, risk_parity_floor))
        out: Dict[str, float] = {}
        for t in alloc:
            rp_w = float(rp_weights.get(t, 0.0))
            baseline = float(base_weights.get(t, 0.0))
            final_w = (1.0 - floor) * rp_w + floor * baseline
            out[t] = total * final_w
        return out

    def _risk_adjust_alloc(alloc: Dict[str, float]) -> Dict[str, float]:
        if not alloc:
            return alloc
        if allocation_model == 'mean_variance' and risk_weighting == 'risk_parity' and risk_blend_eta > 0.0:
            # Already blended with risk parity inside the mean-variance allocator.
            return alloc
        if risk_weighting == 'risk_parity':
            return _risk_parity_alloc(alloc)
        if risk_weighting not in ("inverse_atr", "hybrid", "inverse_sigma"):
            return alloc
        # Compute inverse volatility weights using annualised ATR-based proxy
        eps = 1e-6
        factors: Dict[str, float] = {}
        atr_cache: Dict[str, float] = {}
        atr_values: List[float] = []
        # Optional: inverse std of returns (log) instead of ATR% proxy
        sigma_map: Dict[str, float] = {}
        if risk_weighting == 'inverse_sigma':
            try:
                if prices_history is not None and not prices_history.empty:
                    ph = prices_history[prices_history['Ticker'].isin(list(alloc.keys()))].copy()
                    ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
                    ph = ph.dropna(subset=['Date'])
                    piv = ph.pivot(index='Date', columns='Ticker', values='Close').sort_index()
                    rets = (np.log(piv).diff()).dropna(how='all')
                    lb = cov_lookback if cov_lookback > 0 else None
                    if lb and len(rets) > lb:
                        rets = rets.tail(lb)
                    # Winsorize at 1%
                    wl = 0.01
                    lo = rets.quantile(wl)
                    hi = rets.quantile(1.0 - wl)
                    rets = rets.clip(lower=lo, upper=hi, axis=1)
                    sigma_map = rets.std().replace([np.inf, -np.inf], np.nan).dropna().to_dict()
            except Exception:
                sigma_map = {}
        for t in list(alloc.keys()):
            if sigma_map:
                s = sigma_map.get(t)
                if s is not None and s > 0:
                    atr_cache[t] = float(s)  # reuse cache naming for unified flow
                    atr_values.append(float(s))
            else:
                atr_ann = None
                if t in met.index:
                    ap = to_float(met.loc[t].get('ATR14_Pct'))
                    if ap is not None and ap > 0:
                        atr_ann = ap / 100.0
                if atr_ann is not None and atr_ann > 0:
                    atr_cache[t] = atr_ann
                    atr_values.append(atr_ann)
        fallback_atr = float(np.median(atr_values)) if atr_values else 0.04
        for t in list(alloc.keys()):
            atr = atr_cache.get(t)
            if atr is None or atr <= 0:
                atr = fallback_atr
            atr = max(atr, eps)
            base = 1.0 / atr
            f = base ** max(0.0, risk_alpha)
            factors[t] = f
        total = sum(alloc.values()) or 0.0
        if total <= 0:
            return alloc
        # Apply and renormalize to preserve original sum
        weighted = {t: alloc[t] * factors.get(t, 1.0) for t in alloc}
        z = sum(weighted.values()) or 1.0
        invatr_alloc = {t: total * (weighted[t] / z) for t in alloc}
        if risk_weighting == 'hybrid':
            blend = float(sizing['risk_blend'])
            blend = max(0.0, min(1.0, blend))
            return {t: (1.0 - blend) * alloc.get(t, 0.0) + blend * invatr_alloc.get(t, 0.0) for t in alloc}
        return invatr_alloc

    add_alloc = _risk_adjust_alloc(add_alloc)
    new_alloc = _risk_adjust_alloc(new_alloc)

    # After risk-budgeting, ensure each candidate has at least one lot worth of budget when feasible.
    def _ensure_min_lot_alloc(alloc: Dict[str, float]) -> Dict[str, float]:
        if not alloc:
            return alloc
        # Require pricing context to compute lot cost
        try:
            min_lot_local = int(float(sizing['min_lot']))
        except Exception:
            min_lot_local = 100
        lot_sz = max(min_lot_local, 1)
        # Helper: current price
        def _px(t: str) -> float:
            if t in snap.index:
                return to_float(snap.loc[t].get('Price')) or to_float(snap.loc[t].get('P')) or 0.0
            return 0.0
        # Compute required minimum cash per ticker
        need = {t: lot_sz * max(_px(t), 0.0) for t in alloc}
        total = float(sum(alloc.values()))
        if total <= 0.0:
            return alloc
        short = {t: max(0.0, need.get(t, 0.0) - alloc.get(t, 0.0)) for t in alloc}
        # If total available surplus cannot cover all shorts, leave as-is
        surplus = {t: max(0.0, alloc.get(t, 0.0) - need.get(t, 0.0)) for t in alloc}
        if sum(short.values()) <= 1e-6:
            return alloc
        if sum(surplus.values()) < (sum(short.values()) - 1e-6):
            return alloc
        # Greedy redistribution: take from largest surplus to fill smallest short first
        alloc2 = dict(alloc)
        shorts = sorted([k for k,v in short.items() if v > 1e-6], key=lambda k: short[k])
        surpl = sorted([k for k,v in surplus.items() if v > 1e-6], key=lambda k: surplus[k], reverse=True)
        for t in shorts:
            need_amt = short[t]
            i = 0
            while need_amt > 1e-6 and i < len(surpl):
                s_name = surpl[i]
                s_avail = max(0.0, alloc2.get(s_name, 0.0) - need.get(s_name, 0.0))
                take = min(s_avail, need_amt)
                if take > 0:
                    alloc2[s_name] = alloc2.get(s_name, 0.0) - take
                    alloc2[t] = alloc2.get(t, 0.0) + take
                    need_amt -= take
                i += 1
        return alloc2

    # Apply min-lot guarantee post allocation to improve practical fill even when
    # overall allocation model is softmax, as long as risk weighting may shrink
    # some names below one lot. Only applies when there are multiple candidates
    # and total budget can cover at least one lot per name.
    if len(add_alloc) > 1:
        add_alloc = _ensure_min_lot_alloc(add_alloc)
    if len(new_alloc) > 1:
        new_alloc = _ensure_min_lot_alloc(new_alloc)

    # 4) Build BUY orders with optional caps (per-position, per-sector)
    orders: List[Order] = []
    notes: Dict[str, str] = {}
    min_lot = int(float(sizing['min_lot']))
    lot = max(min_lot, 1)
    fill_cfg = dict((getattr(regime, 'execution', {}) or {}).get('fill') or {})
    # Time-of-day: if ATC and target_prob high, allow crossing (temporarily disable no_cross)
    try:
        exec_conf_local = dict(getattr(regime, 'execution', {}) or {})
        tod_conf = exec_conf_local.get('time_of_day') if isinstance(exec_conf_local, dict) else None
        if isinstance(tod_conf, dict):
            prules = tod_conf.get('phase_rules') if isinstance(tod_conf.get('phase_rules'), dict) else None
            phase_now = str(getattr(regime, 'phase', '') or '').strip().upper()
            if prules and phase_now == 'ATC':
                atc_rule = prules.get('ATC') if isinstance(prules.get('ATC'), dict) else None
                thr = atc_rule.get('allow_cross_if_target_prob_gte') if atc_rule else None
                if thr is not None:
                    try:
                        thr_val = float(thr)
                    except Exception:
                        thr_val = None
                    if thr_val is not None:
                        try:
                            tp_local = float(fill_cfg.get('target_prob', 0.0) or 0.0)
                        except Exception:
                            tp_local = 0.0
                        if tp_local >= thr_val:
                            fill_cfg['no_cross'] = False
    except Exception:
        pass

    def _apply_caps_and_qty(t: str, budget_k: float, limit_price: float) -> int:
        # Base qty from budget
        qty0 = max(int(budget_k / max(limit_price, 1e-9) / lot) * lot, 0)
        if qty0 <= 0:
            return 0
        qty_cap = qty0
        risk_cap_qty: Optional[int] = None
        # Position cap
        # Allow per-ticker override of max_pos_frac
        mpf_override = None
        try:
            if hasattr(regime, 'ticker_overrides') and isinstance(regime.ticker_overrides, dict):
                ov = regime.ticker_overrides.get(t, {}) or {}
                if isinstance(ov, dict) and ov.get('max_pos_frac') is not None:
                    mpf_override = float(ov.get('max_pos_frac'))
        except Exception:
            mpf_override = None
        _max_pos_frac = float(mpf_override) if mpf_override is not None else float(max_pos_frac)
        if _max_pos_frac > 0:
            cur_val = float(held_qty.get(t, 0) or 0) * float(_price_of(t))
            max_val = float(_max_pos_frac) * float(nav_for_caps)
            headroom = max(0.0, max_val - cur_val)
            cap_qty = int(max(headroom / max(limit_price, 1e-9) / lot, 0)) * lot
            qty_cap = min(qty_cap, cap_qty)
        # Sector cap
        if max_sector_frac > 0:
            sec = sector_by_ticker.get(str(t))
            if sec:
                cur_sec = float(sector_expo_k.get(sec, 0.0))
                max_sec = float(max_sector_frac) * float(nav_for_caps)
                headroom_sec = max(0.0, max_sec - cur_sec)
                cap_qty2 = int(max(headroom_sec / max(limit_price, 1e-9) / lot, 0)) * lot
                qty_cap = min(qty_cap, cap_qty2)
        # Notional guard vs ADTV: cap quantity so that order notional ≤ multiple × ADTV
        try:
            adtv_k = None
            if t in met.index:
                adtv_k = to_float(met.loc[t].get('AvgTurnover20D_k'))
            mult = 20.0
            try:
                ct = getattr(regime, 'calibration_targets', {}) or {}
                liq = ct.get('liquidity', {}) if isinstance(ct, dict) else {}
                mv = liq.get('adtv_multiple') if isinstance(liq, dict) else None
                if mv is not None:
                    mult = float(mv)
            except Exception:
                pass
            if adtv_k is not None and adtv_k > 0 and limit_price > 0 and mult > 0:
                max_qty = int((mult * float(adtv_k) / float(limit_price)) / lot) * lot
                qty_cap = min(qty_cap, max(0, max_qty))
        except Exception:
            pass
        # Risk-per-trade cap (optional) — estimate stop distance via ATR
        # Per-ticker override for risk_per_trade_frac and stop multiple
        rpt = float(sizing['risk_per_trade_frac'])
        stop_mult = float(sizing['default_stop_atr_mult'])
        try:
            if hasattr(regime, 'ticker_overrides') and isinstance(regime.ticker_overrides, dict):
                ov = regime.ticker_overrides.get(t, {}) or {}
                if isinstance(ov, dict):
                    if ov.get('risk_per_trade_frac') is not None:
                        rpt = float(ov.get('risk_per_trade_frac'))
                    if ov.get('default_stop_atr_mult') is not None:
                        stop_mult = float(ov.get('default_stop_atr_mult'))
        except Exception:
            pass
        if rpt and rpt > 0.0:
            # Helper: infer ATR in thousand
            def _infer_atr_thousand(price_k: float, ticker: str) -> float:
                atr_val = 0.0
                if ticker in pre.index:
                    atrv = to_float(pre.loc[ticker].get('ATR14'))
                    if atrv is not None and atrv > 0:
                        atr_val = float(atrv)
                if atr_val <= 0.0 and ticker in met.index and price_k is not None and price_k > 0:
                    ap = to_float(met.loc[ticker].get('ATR14_Pct'))
                    if ap is not None and ap > 0:
                        atr_val = float(ap) / 100.0 * float(price_k)
                return max(0.0, atr_val)
            px_now = float(_price_of(t))
            atr_k = _infer_atr_thousand(px_now, t)
            stop_dist_k = max(0.0, stop_mult * atr_k)
            tp_sl_entry = tp_sl_context.get(str(t).upper())
            if tp_sl_entry and px_now > 0.0:
                sl_pct_eff = to_float(tp_sl_entry.get('sl_pct'))
                if sl_pct_eff is not None and sl_pct_eff > 0:
                    sl_pct_val = float(sl_pct_eff)
                    # Ignore extreme placeholders (>=50%) which typically mean "no static stop";
                    # otherwise blend with ATR-derived distance by taking the larger distance.
                    if sl_pct_val < 0.5:
                        stop_dist_k = max(stop_dist_k, sl_pct_val * float(px_now))
            allowed_risk_k = float(rpt) * float(nav_for_caps)
            if stop_dist_k > 0 and allowed_risk_k > 0:
                cap_qty3 = int(max(allowed_risk_k / stop_dist_k / lot, 0)) * lot
                risk_cap_qty = cap_qty3
                qty_cap = min(qty_cap, cap_qty3)
        if tranche_frac > 0.0 and tranche_frac < 1.0 and _max_pos_frac > 0.0 and limit_price > 0.0:
            max_val = float(_max_pos_frac) * float(nav_for_caps)
            cur_val = float(held_qty.get(t, 0) or 0) * float(_price_of(t))
            headroom_val = max(0.0, max_val - cur_val)
            tranche_cap_qty = int(max(headroom_val * tranche_frac / max(limit_price, 1e-9) / lot, 0)) * lot
            if tranche_cap_qty >= 0:
                qty_cap = min(qty_cap, tranche_cap_qty)
        min_notional_vnd = float(sizing.get('min_notional_per_order', 0.0) or 0.0)
        if min_notional_vnd > 0.0 and limit_price > 0.0:
            min_notional_k = min_notional_vnd / 1000.0
            required_lots = int(math.ceil(min_notional_k / max(limit_price, 1e-9) / lot))
            required_qty = max(required_lots * lot, 0)
            if required_qty > 0 and qty_cap < required_qty:
                if risk_cap_qty is not None and risk_cap_qty == qty_cap and qty_cap >= lot:
                    pass
                else:
                    return 0
        return max(qty_cap, 0)

    spent_buy_k = 0.0

    # Config: clamp vs filter when BUY limit > market
    try:
        exec_conf = dict(getattr(regime, 'execution', {}) or {})
        filter_buy_cross = bool(int(exec_conf.get('filter_buy_limit_gt_market', 1)))
    except Exception:
        filter_buy_cross = True

    for t in add_names:
        if t not in snap.index:
            continue
        s = snap.loc[t]; m = met.loc[t] if t in met.index else None; p = pre.loc[t] if t in pre.index else None
        limit = pick_limit_price(t, "BUY", s, p, m, regime)
        market_price = to_float(s.get("Price")) or to_float(s.get("P"))
        budget = float(add_alloc.get(t, 0.0))
        # Optionally clamp BUY limit down to market to avoid filtering
        if market_price is not None and limit > float(market_price) + 1e-9:
            if filter_buy_cross:
                qty_probe = _apply_caps_and_qty(t, budget, limit)
                _track_filter(t, "limit_gt_market", f"limit {limit:.2f} > market {market_price:.2f}", side="BUY", quantity=qty_probe, limit_price=limit, market_price=float(market_price))
                continue
            else:
                limit = float(market_price)
        qty = _apply_caps_and_qty(t, budget, limit)
        if qty > 0:
            base_note = "Mua gia tăng"
            tags: List[str] = []
            if neutral_active:
                tags.append("NEUTRAL_ADAPT")
            if t in neutral_accum_names:
                tags.append("NEUTRAL_ACCUM")
            note_text = base_note + (" | " + "; ".join(tags) if tags else "")
            orders.append(Order(ticker=t, side="BUY", quantity=qty, limit_price=limit, note=note_text))
            notes[t] = note_text
            # Update exposures for subsequent cap checks
            sec = sector_by_ticker.get(str(t))
            if sec:
                sector_expo_k[sec] = sector_expo_k.get(sec, 0.0) + float(qty) * float(limit)
            held_qty[t] = int(held_qty.get(t, 0) or 0) + int(qty)
            spent_buy_k += float(qty) * float(limit)
    for t in new_names:
        if t not in snap.index:
            continue
        s = snap.loc[t]; m = met.loc[t] if t in met.index else None; p = pre.loc[t] if t in pre.index else None
        limit = pick_limit_price(t, "BUY", s, p, m, regime)
        market_price = to_float(s.get("Price")) or to_float(s.get("P"))
        budget = float(new_alloc.get(t, 0.0))
        limit_adj, diag = _apply_new_buy_execution(t, limit, market_price, s, m, lot, fill_cfg)
        if diag:
            regime.new_buy_fill_diag.append(diag)
        if limit_adj is None:
            note = "fill prob below target"
            if isinstance(diag, dict) and 'best_pof' in diag:
                note = f"bestPOF={diag['best_pof']:.2f}<target={diag.get('target_prob',0.0):.2f}"
            _track_filter(t, "fill_prob_below_target", note, side="BUY")
            continue
        limit = float(limit_adj)
        # Optionally clamp BUY limit down to market to avoid filtering
        if market_price is not None and limit > float(market_price) + 1e-9:
            if filter_buy_cross:
                qty_probe = _apply_caps_and_qty(t, budget, limit)
                _track_filter(t, "limit_gt_market", f"limit {limit:.2f} > market {market_price:.2f}", side="BUY", quantity=qty_probe, limit_price=limit, market_price=float(market_price))
                continue
            else:
                limit = float(market_price)
        qty = _apply_caps_and_qty(t, budget, limit)
        is_partial = t in new_partial_names
        floor_lot_local = int(partial_floor_map_state.get(t, state_local.get('partial_entry_floor_lot', 1)) or 1)
        floor_lot_local = max(1, floor_lot_local)
        min_qty = lot if not is_partial else max(lot, floor_lot_local * lot)
        if is_partial:
            if qty < min_qty:
                required = min_qty * float(limit)
                if required <= budget + 1e-6:
                    qty = min_qty
                else:
                    qty = 0
            else:
                qty = max(min_qty, qty)
        else:
            # Cap first tranche size for non-partial NEW entries by configured lots
            try:
                first_tranche_lots = int(float(sizing.get('new_first_tranche_lots', 1) or 1))
            except Exception:
                first_tranche_lots = 1
            first_tranche_lots = max(1, first_tranche_lots)
            if qty >= lot:
                qty = min(qty, first_tranche_lots * lot)
            elif qty > 0:
                qty = 0
        if market_price is not None and limit > float(market_price) + 1e-9:
            if filter_buy_cross:
                _track_filter(t, "limit_gt_market", f"limit {limit:.2f} > market {market_price:.2f}", side="BUY", quantity=qty, limit_price=limit, market_price=float(market_price))
                continue
            else:
                limit = float(market_price)
        if qty > 0:
            base_note = "Mua mới"
            tags: List[str] = []
            if neutral_active:
                tags.append("NEUTRAL_ADAPT")
            if t in neutral_partial_names:
                tags.append("PARTIAL_ENTRY")
            if t in neutral_override_names:
                tags.append("NEUTRAL_GATE_OVERRIDE")
            note_text = base_note + (" | " + "; ".join(tags) if tags else "")
            orders.append(Order(ticker=t, side="BUY", quantity=qty, limit_price=limit, note=note_text))
            notes[t] = note_text
            sec = sector_by_ticker.get(t)
            if sec:
                sector_expo_k[sec] = sector_expo_k.get(sec, 0.0) + float(qty) * float(limit)
            held_qty[t] = int(held_qty.get(t, 0) or 0) + int(qty)
            spent_buy_k += float(qty) * float(limit)

    # 4b) Leftover redistribution after lot rounding and caps (optional, default enabled)
    enable_leftover = bool(int(sizing['leftover_redistribute']))
    if enable_leftover:
        total_alloc_k = float(sum(add_alloc.values()) + sum(new_alloc.values()))
        leftover_k = max(0.0, total_alloc_k - spent_buy_k)
        min_ticket_k = float(sizing['min_ticket_k'])
        if leftover_k > max(0.0, min_ticket_k):
            cand_names = [t for t in (add_names + new_names) if t in snap.index]
            cand_names = sorted(set(cand_names), key=lambda x: scores.get(x, 0.0), reverse=True)
            if not partial_allow_leftover:
                cand_names = [t for t in cand_names if t not in new_partial_names and t not in neutral_override_names]
            max_iters = 32
            iters = 0
            progressed = True
            while leftover_k > 0.0 and iters < max_iters and progressed:
                progressed = False
                for t in cand_names:
                    if leftover_k <= 0.0:
                        break
                    s = snap.loc[t]
                    m = met.loc[t] if t in met.index else None
                    p = pre.loc[t] if t in pre.index else None
                    limit = pick_limit_price(t, "BUY", s, p, m, regime)
                    market_price = to_float(s.get("Price")) or to_float(s.get("P"))
                    if market_price is not None and limit > float(market_price) + 1e-9:
                        if filter_buy_cross:
                            qty_try_est = _apply_caps_and_qty(t, leftover_k, limit)
                            _track_filter(t, "limit_gt_market", f"limit {limit:.2f} > market {market_price:.2f}", side="BUY", quantity=int(qty_try_est), limit_price=limit, market_price=float(market_price))
                            continue
                        else:
                            limit = float(market_price)
                    qty_try = _apply_caps_and_qty(t, leftover_k, limit)
                    if qty_try <= 0:
                        continue
                    cost_k = float(qty_try) * float(limit)
                    if cost_k <= 0.0 or cost_k > leftover_k + 1e-6:
                        continue
                    merged = False
                    for o in orders:
                        if o.ticker == t and o.side == "BUY" and abs(o.limit_price - limit) < 1e-6:
                            o.quantity += int(qty_try)
                            merged = True
                            break
                    if not merged:
                        orders.append(Order(ticker=t, side="BUY", quantity=int(qty_try), limit_price=limit, note=notes.get(t, "Mua bổ sung")))
                    sec = sector_by_ticker.get(str(t))
                    if sec:
                        sector_expo_k[sec] = sector_expo_k.get(sec, 0.0) + float(qty_try) * float(limit)
                    held_qty[t] = int(held_qty.get(t, 0) or 0) + int(qty_try)
                    leftover_k -= cost_k
                    spent_buy_k += cost_k
                    progressed = True
                iters += 1

    # 5) Append SELL orders (computed earlier)
    for o in sell_candidates:
        orders.append(o); notes[o.ticker] = o.note
    try:
        regime.ttl_overrides = ttl_override_map
    except Exception:
        pass
    regime.debug_filters = debug_filters
    regime.filtered_records = filtered_records
    # Minimal analysis output when build_orders is called directly (outside run())
    try:
        analysis_lines = [
            f"Regime risk_on: {regime.risk_on}",
        ]
        if getattr(regime, 'is_neutral', False):
            stats = dict(getattr(regime, 'neutral_stats', {}) or {})
            analysis_lines.append(
                "[neutral] active: partial={partial} override={override} add_capped={capped}".format(
                    partial=stats.get('partial_count', 0),
                    override=stats.get('override_count', 0),
                    capped=stats.get('add_capped_count', 0),
                )
            )
        dw = list(getattr(regime, 'diag_warnings', []) or [])
        if dw:
            # Summary counts per missing metric
            from collections import Counter
            kinds = []
            for x in dw:
                try:
                    k = str(x).split(':', 1)[0]
                except Exception:
                    k = str(x)
                kinds.append(k)
            cnt = Counter(kinds)
            summary_chunks = []
            for key in ('metrics_rsi_missing','metrics_liqnorm_missing','metrics_atr_missing','metrics_beta_missing'):
                if cnt.get(key, 0) > 0:
                    summary_chunks.append(f"{key}={cnt[key]}")
            if summary_chunks:
                analysis_lines.append("Diagnostics summary: " + ", ".join(summary_chunks))
        if getattr(regime, 'new_buy_fill_diag', None):
            diag_list = list(regime.new_buy_fill_diag)
            skipped = sum(1 for d in diag_list if isinstance(d, dict) and d.get('status') == 'skipped')
            accepted = sum(1 for d in diag_list if isinstance(d, dict) and d.get('status') == 'accepted')
            analysis_lines.append(
                f"[execution] new_buy fill diag: accepted={accepted} skipped={skipped} total={len(diag_list)}"
            )
            analysis_lines.append("Diagnostics warnings: " + ", ".join(sorted(set(str(x) for x in dw))))
        else:
            analysis_lines.append("Diagnostics warnings: -")
        write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")
    except Exception:
        pass
    # Persist regime components snapshot for audit (compatible with tests)
    try:
        comp_out = OUT_ORDERS_DIR / 'regime_components.json'
        payload = dict(getattr(regime, 'model_components', {}) or {})
        payload.update({
            'market_score': float(getattr(regime, 'market_score', 0.0) or 0.0),
            'risk_on_probability': float(getattr(regime, 'risk_on_probability', 0.0) or 0.0),
            'index_vol_annualized': float(getattr(regime, 'index_vol_annualized', 0.0) or 0.0),
            'index_atr14_pct': float(getattr(regime, 'index_atr14_pct', 0.0) or 0.0),
            'index_atr_percentile': float(getattr(regime, 'index_atr_percentile', 0.0) or 0.0),
            'turnover_percentile': float(getattr(regime, 'turnover_percentile', 0.5) or 0.5),
            'breadth_long': float(getattr(regime, 'model_components', {}).get('breadth_long', 0.0) if getattr(regime, 'model_components', {}) else 0.0),
            'diag_warnings': list(getattr(regime, 'diag_warnings', []) or []),
            'index_change_pct_intraday': float(getattr(regime, 'index_change_pct', 0.0) or 0.0),
            'index_change_pct_smoothed': float(getattr(regime, 'index_change_pct_smoothed', 0.0) or 0.0),
        })
        comp_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
    return orders, notes, regime


    # IO moved to scripts.orders.orders_io


def run(simulate: bool = False, *, context: Optional[Dict[str, Any]] = None, flags: Optional[Dict[str, Any]] = None):
    OUT_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure we have a runtime copy of the policy first; then derive static knobs (e.g., band)
    ensure_policy_override_file()
    try:
        runtime_path = aggregate_to_runtime()
    except PatchMergeError as _exc_merge:
        raise SystemExit(f"Failed to aggregate policy patches: {_exc_merge}") from _exc_merge
    # Read single policy source once (runtime merged copy)
    import json as _json, re as _re
    pol_path = runtime_path if runtime_path.exists() else OUT_ORDERS_DIR / 'policy_overrides.json'
    if not pol_path.exists():
        raise SystemExit('Missing runtime policy after aggregation')
    try:
        raw = pol_path.read_text(encoding='utf-8')
        raw = _re.sub(r"/\*.*?\*/", "", raw, flags=_re.S)
        raw = _re.sub(r"(^|\s)//.*$", "", raw, flags=_re.M)
        raw = _re.sub(r"(^|\s)#.*$", "", raw, flags=_re.M)
        pol_obj = _json.loads(raw)
    except Exception as _exc_pol:
        raise SystemExit(f'Invalid runtime policy JSON: {_exc_pol}') from _exc_pol
    try:
        _band = float((((pol_obj.get('market') or {}).get('microstructure') or {}).get('daily_band_pct')))
    except Exception as _exc_band:
        raise SystemExit(f"Missing or invalid market.microstructure.daily_band_pct in runtime policy: {_exc_band}") from _exc_band

    # Inject band into presets builder module once before building artifacts
    try:
        from scripts.build_presets_all import set_daily_band_pct as _set_band
        _set_band(_band)
    except Exception as _exc_set:
        raise SystemExit(f"Failed to set daily_band_pct into presets module: {_exc_set}") from _exc_set
    # Build fresh artifacts
    portfolio, prices_history, snapshot, metrics, sector_strength, presets, session_summary = ensure_pipeline_artifacts()
    # Runtime calibrations have been removed from the order generation flow. Calibrated
    # policy overrides must be produced ahead of time via the scheduled GitHub Actions
    # workflows. Guard against legacy toggles that would attempt to re-enable
    # calibration during order runs so we fail fast with a clear remediation path.
    import os as _os

    def _truthy(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        try:
            text_val = str(value).strip().lower()
        except Exception:
            return False
        return text_val not in ("", "0", "false", "no", "off", "none")

    cal_conf = dict(pol_obj.get('calibration', {}) or {})
    cfg_violations: list[str] = []
    if _truthy(cal_conf.get('on_run')):
        cfg_violations.append('calibration.on_run')
        for _key in sorted(k for k in cal_conf.keys() if k != 'on_run'):
            if _truthy(cal_conf.get(_key)):
                cfg_violations.append(f'calibration.{_key}')

    if cfg_violations:
        lines = [
            'Runtime calibration during order generation has been removed.',
            'Run the GitHub Actions tuning workflows to refresh policy overrides before generating orders.',
        ]
        lines.append('Disable the following config toggles: ' + ', '.join(cfg_violations))
        raise SystemExit('\n'.join(lines))

    # Policy runtime copy should already be updated by the tuning workflows upstream
    session_now = detect_session_phase_now_vn()
    industry = pd.read_csv(DATA_DIR / "industry_map.csv") if (DATA_DIR / "industry_map.csv").exists() else pd.DataFrame(columns=['Ticker','Sector'])
    pnl_summary = pd.read_csv(OUT_DIR / "portfolio_pnl_summary.csv") if (OUT_DIR / "portfolio_pnl_summary.csv").exists() else pd.DataFrame(columns=['TotalCost','TotalMarket','TotalPnL','ReturnPct'])

    # Metrics/snapshot already prepared; skip recomputation

    tuning = suggest_tuning(session_summary, sector_strength)
    runtime_overrides = dict(tuning.pop("_runtime_overrides", {}) or {})
    actions, scores, feats_all, regime = decide_actions(
        portfolio,
        snapshot,
        metrics,
        presets,
        industry,
        sector_strength,
        session_summary,
        tuning,
        runtime_overrides=runtime_overrides,
        prices_history=prices_history,
    )
    orders, notes, regime = build_orders(
        actions, portfolio, snapshot, metrics, presets, pnl_summary, scores, regime, prices_history
    )
    if simulate or (flags and bool(flags.get("simulate"))):
        return {
            "orders": orders,
            "notes": notes,
            "regime": regime,
            "snapshot": snapshot,
            "metrics": metrics,
            "presets": presets,
        }
    # Always persist outputs; test mode removed.
    # Entry gating: move low-probability/weak micro-tape BUYs to watchlist (configurable via orders_ui)
    ou_conf = {}
    try:
        ou_conf = dict(getattr(regime, 'orders_ui', {}) or {})
    except Exception:
        ou_conf = {}
    wl_conf = dict(ou_conf.get('watchlist', {}) or {})
    wl_enable = bool(int(wl_conf.get('enable', 1)))
    wl_min_prio = float(wl_conf.get('min_priority', 0.25) or 0.25)
    wl_win = int(wl_conf.get('micro_window', 3) or 3)
    def _micro_up(ticker: str) -> bool:
        try:
            p = OUT_DIR / 'intraday' / f'{ticker}_intraday.csv'
            if not p.exists():
                return True
            df = pd.read_csv(p)
            if df.empty or 'price' not in df.columns:
                return True
            last = float(pd.to_numeric(df['price'], errors='coerce').dropna().iloc[-1])
            prev_idx = -wl_win if len(df) >= wl_win else -1
            prev = float(pd.to_numeric(df['price'], errors='coerce').dropna().iloc[prev_idx])
            return last >= prev
        except Exception:
            return True
    def _fill_prob(o) -> float:
        try:
            t = o.ticker
            market = None
            if t in snapshot.set_index('Ticker').index:
                s = snapshot.set_index('Ticker').loc[t]
                market = to_float(s.get('Price')) or to_float(s.get('P'))
            tick = hose_tick_size(market if market else o.limit_price)
            ceil_tick = None
            if t in presets.set_index('Ticker').index:
                pr = presets.set_index('Ticker').loc[t]
                ceil_tick = to_float(pr.get('BandCeiling_Tick')) or to_float(pr.get('BandCeilingRaw'))
            # Pricing heuristics from policy
            pr_conf = {}
            try:
                pr_conf = dict(getattr(regime, 'pricing', {}) or {})
            except Exception:
                pr_conf = {}
            fp = dict(pr_conf.get('fill_prob', {}) or {})
            fp_base = float(fp.get('base', 0.30) or 0.30)
            fp_cross = float(fp.get('cross', 0.90) or 0.90)
            fp_near = float(fp.get('near_ceiling', 0.05) or 0.05)
            fp_min = float(fp.get('min', 0.05) or 0.05)
            fp_scale_min = float(fp.get('decay_scale_min_ticks', 5.0) or 5.0)
            if market is None:
                return fp_base
            th = dict(getattr(regime, 'thresholds', {}) or {})
            if th.get('near_ceiling_pct') is None:
                raise SystemExit("Missing thresholds.near_ceiling_pct in policy")
            near_thr = float(th.get('near_ceiling_pct'))
            if ceil_tick is not None and market >= (near_thr * ceil_tick):
                return fp_near
            if o.side == 'BUY':
                if o.limit_price >= market:
                    return fp_cross
                dist = max(0.0, (market - o.limit_price) / max(tick, 1e-6))
            else:
                if o.limit_price <= market:
                    return fp_cross
                dist = max(0.0, (o.limit_price - market) / max(tick, 1e-6))
            f = feats_all.get(t, {}) or {}
            atr_pct = float(f.get('atr_pct', 0.0) or 0.0)
            atr_ticks = (atr_pct * (market if market else 0.0)) / max(tick, 1e-6)
            import math
            scale = max(fp_scale_min, atr_ticks)
            return max(fp_min, math.exp(-dist / scale))
        except Exception:
            return fp_base
    buy_kept: list[Order] = []
    buy_watch: list[Order] = []
    idx_micro_ok = _micro_up('VNINDEX')
    if wl_enable:
        for o in list(orders):
            if o.side != 'BUY':
                continue
            prio = max(0.0, float(scores.get(o.ticker, 0.0) or 0.0)) * float(_fill_prob(o))
            if (not idx_micro_ok) or (not _micro_up(o.ticker)) or prio < wl_min_prio:
                buy_watch.append(o)
            else:
                buy_kept.append(o)
    else:
        buy_kept = [o for o in orders if o.side == 'BUY']
    if buy_watch:
        # Remove watchlist buys from final orders
        kept_set = {id(x) for x in buy_kept}
        final_orders: list[Order] = []
        for o in orders:
            if o.side == 'BUY' and id(o) not in kept_set:
                continue
            final_orders.append(o)
        orders = final_orders
        # Write watchlist
        try:
            write_orders_csv_enriched(
                buy_watch,
                OUT_ORDERS_DIR / 'orders_watchlist.csv',
                snapshot=snapshot,
                presets=presets,
                regime=regime,
                feats_all=feats_all,
                scores=scores,
            )
        except Exception as _exc_wl:
            print(f"[warn] failed to write watchlist: {_exc_wl}")
    prev_orders_df = None
    prev_orders_path = OUT_ORDERS_DIR / "orders_final.csv"
    if prev_orders_path.exists():
        try:
            prev_orders_df = pd.read_csv(prev_orders_path)
        except Exception:
            prev_orders_df = None
    exec_hint = _suggest_execution_window(regime, session_summary, session_now)
    # Execution Ladder integration (after sizing/pricing, before writing CSVs)
    levels: list = []
    ladder_dropped: list = []
    ladder_debug: list = []
    try:
        levels, ladder_dropped, ladder_debug = generate_ladder_levels(
            orders, regime=regime, snapshot=snapshot, metrics=metrics, presets=presets
        )
    except SystemExit:
        # Hard fail as per policy if config is malformed
        raise
    except Exception as _exc_lad:
        # If ladder fails unexpectedly, keep legacy output to avoid blackouts
        print(f"[warn] ladder_generation_failed: {_exc_lad}")
        levels = []

    if levels:
        write_ladder_final_csv(levels, OUT_ORDERS_DIR / "orders_final.csv")
        try:
            q_rows = build_ladder_quality_rows(levels, snapshot=snapshot, presets=presets, metrics=metrics, regime=regime, feats_all=feats_all)
            write_ladder_quality_csv(q_rows, OUT_ORDERS_DIR / "orders_quality.csv")
        except Exception as _exc_q:
            print(f"[warn] ladder_quality_failed: {_exc_q}")
        try:
            write_ladder_log(ladder_debug, OUT_DIR / 'debug' / 'ladder.log')
        except Exception as _exc_log:
            print(f"[warn] ladder_log_failed: {_exc_log}")
    else:
        # Fallback to legacy minimal CSV when ladder disabled or produced no levels
        write_orders_csv(
            orders,
            OUT_ORDERS_DIR / "orders_final.csv",
            snapshot=snapshot,
            presets=presets,
            regime=regime,
            feats_all=feats_all,
            scores=scores,
        )
    (OUT_ORDERS_DIR / "orders_print.txt").write_text(print_orders(orders, snapshot), encoding="utf-8")
    diff_lines: List[str] = []
    if prev_orders_df is not None and not prev_orders_df.empty:
        prev_map = {
            (str(row["Ticker"]).strip().upper(), str(row["Side"]).strip().upper()): (
                int(row["Quantity"]), float(row["LimitPrice"])
            )
            for _, row in prev_orders_df.iterrows()
        }
        new_map = {
            (o.ticker, o.side): (int(o.quantity), float(o.limit_price))
            for o in orders
        }
        prev_keys = set(prev_map.keys())
        new_keys = set(new_map.keys())
        added = sorted(new_keys - prev_keys)
        removed = sorted(prev_keys - new_keys)
        changed = sorted(
            k for k in (new_keys & prev_keys)
            if prev_map[k] != new_map[k]
        )
        if added:
            diff_lines.append("Added: " + ", ".join(f"{t}/{s}" for t, s in added))
        if removed:
            diff_lines.append("Removed: " + ", ".join(f"{t}/{s}" for t, s in removed))
        if changed:
            diff_lines.extend(
                [
                    f"Changed: {t}/{s} qty {prev_map[(t, s)][0]}→{new_map[(t, s)][0]} price {prev_map[(t, s)][1]:.2f}→{new_map[(t, s)][1]:.2f}"
                    for (t, s) in changed
                ]
            )
        if not diff_lines:
            diff_lines.append("No order changes vs previous run")
    else:
        diff_lines.append("No previous orders snapshot found (first run or file missing)")
    for line in diff_lines:
        print(f"[diff] {line}")

    filters = getattr(regime, 'debug_filters', {}) or {}
    filtered_records = getattr(regime, 'filtered_records', []) or []
    def _fmt_filter(key: str) -> str:
        items = sorted(set(filters.get(key, []) or []))
        return ", ".join(items) if items else "-"

    filter_lines = {
        "market": _fmt_filter("market"),
        "liquidity": _fmt_filter("liquidity"),
        "near_ceiling": _fmt_filter("near_ceiling"),
    }
    for tag, txt in filter_lines.items():
        print(f"[filter] {tag}: {txt}")

    analysis_lines = [
        f"Session now (clock): {session_now}",
        f"Session (file): {session_summary.loc[0, 'SessionPhase'] if not session_summary.empty else ''}",
        f"Regime risk_on: {regime.risk_on}",
        f"Buy budget frac: {regime.buy_budget_frac}",
        f"Buy budget frac (effective): {getattr(regime, 'buy_budget_frac_effective', 0.0)}",
        f"Top sectors: {', '.join(regime.top_sectors)}",
        f"Holdings: {len(portfolio)} tickers",
        f"Proposed orders: {len(orders)}",
        f"Execution window hint: {exec_hint}",
        f"VNINDEX Δ% (smoothed): {float(regime.index_change_pct):.3f}",
        f"Trend strength vs MA: {float(regime.trend_strength):.3f}",
        f"Breadth (>MA50): {float(regime.breadth_hint):.3f}",
        f"Breadth (>MA200): {float(regime.model_components.get('breadth_long', 0.0)):.3f}",
        f"Risk-on probability: {float(regime.risk_on_probability):.2f}",
        f"MA200 slope (20d/MA200): {float(regime.model_components.get('ma200_slope', 0.0)):.4f}",
        f"Uptrend (Close>MA200 & slope>0): {bool(regime.model_components.get('uptrend', 0.0) >= 0.5)}",
        f"Drawdown from peak: {float(regime.drawdown_pct)*100.0:.1f}%",
        f"Turnover pct (252d): {float(getattr(regime, 'turnover_percentile', 0.5))*100.0:.1f}",
        f"VNINDEX ATR14%: {float(getattr(regime, 'index_atr14_pct', 0.0))*100.0:.2f}",
        f"Index ATR percentile: {float(getattr(regime, 'index_atr_percentile', 0.5))*100.0:.1f}%",
        f"US EPU percentile: {float(getattr(regime, 'epu_us_percentile', 0.0) or 0.0)*100.0:.1f}%",
        f"S&P 500 drawdown: {float(getattr(regime, 'spx_drawdown_pct', 0.0) or 0.0)*100.0:.1f}%",
        f"DXY percentile: {float(getattr(regime, 'dxy_percentile', 0.0) or 0.0)*100.0:.1f}%",
        f"Brent mom 63d: {float(getattr(regime, 'brent_mom_63d', 0.0) or 0.0)*100.0:.1f}%",
        f"Filtered (market guard): {filter_lines['market']}",
        f"Filtered (liquidity): {filter_lines['liquidity']}",
        f"Filtered (near ceiling): {filter_lines['near_ceiling']}",
    ]
    alloc_model_for_diag = locals().get('allocation_model', None)
    if alloc_model_for_diag == 'mean_variance' and allocation_diagnostics:
        calib_diag = allocation_diagnostics.get('calibration') if isinstance(allocation_diagnostics, dict) else None
        for group_name, diag in allocation_diagnostics.items():
            if group_name == 'calibration':
                continue
            weights_map = diag.get('weights', {}) or {}
            mu_port = float(diag.get('portfolio_mu_annual', 0.0))
            vol_port = float(diag.get('portfolio_vol_annual', 0.0))
            sorted_weights = sorted(weights_map.items(), key=lambda kv: kv[1], reverse=True)
            top_weights = ", ".join(f"{t}:{w:.3f}" for t, w in sorted_weights[:5]) if sorted_weights else "-"
            sectors = diag.get('sectors', {}) or {}
            sector_w: Dict[str, float] = {}
            for ticker, weight in weights_map.items():
                sector = sectors.get(ticker) or 'Unknown'
                sector_w[sector] = sector_w.get(sector, 0.0) + float(weight)
            sector_str = ", ".join(f"{sec}:{weight:.3f}" for sec, weight in sorted(sector_w.items())) if sector_w else "-"
            analysis_lines.append(
                f"[mean_variance] {group_name}: mu={mu_port:.3%}, vol={vol_port:.3%}, names={len(weights_map)}"
            )
            analysis_lines.append(f"[mean_variance] {group_name} weights: {top_weights}")
            analysis_lines.append(f"[mean_variance] {group_name} sectors: {sector_str}")
        if calib_diag:
            best_row = (calib_diag.get('results') or [{}])[0]
            analysis_lines.append(
                "[mean_variance] calibration: "
                f"risk_alpha={best_row.get('risk_alpha')}, cov_reg={best_row.get('cov_reg')}, "
                f"bl_alpha_scale={best_row.get('bl_alpha_scale')}, sharpe={best_row.get('sharpe', 0):.2f}, "
                f"drawdown={best_row.get('max_drawdown', 0):.2%}, turnover={best_row.get('avg_turnover', 0):.2f}, "
                f"grid={calib_diag.get('grid_size')}"
            )
    analysis_lines.extend(diff_lines)
    # Append diagnostic warnings if present
    try:
        dw = list(getattr(regime, 'diag_warnings', []) or [])
        if dw:
            analysis_lines.append("Diagnostics warnings: " + ", ".join(sorted(set(str(x) for x in dw))))
    except Exception as _exc:
        analysis_lines.append(f"[warn] failed to read diag_warnings: {_exc}")
    write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")
    write_orders_reasoning(actions, scores, feats_all, OUT_ORDERS_DIR / "orders_reasoning.csv")
    # Per-order execution/profitability heuristics to help operator prioritize input
    # If ladder used, orders_quality.csv already written above; otherwise, keep legacy writer
    if not levels:
        try:
            write_orders_quality(orders, snapshot, presets, regime, feats_all, scores, OUT_ORDERS_DIR / "orders_quality.csv")
        except Exception as _exc_q:
            analysis_lines.append(f"[warn] write_orders_quality failed: {_exc_q}")
            write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")
    # Trade suggestions (human-readable summary)
    try:
        # Use policy knob for suggestions length if provided
        top_n = 3
        try:
            ou_conf = dict(getattr(regime, 'orders_ui', {}) or {})
            top_n = int(ou_conf.get('suggestions_top_n', 3) or 3)
        except Exception:
            top_n = 3
        sugg_lines = build_trade_suggestions(actions, scores, feats_all, regime, top_n=top_n)
        write_text_lines(sugg_lines, OUT_ORDERS_DIR / "trade_suggestions.txt")
    except Exception as _exc:
        # Do not swallow silently; surface error via analysis file while keeping orders intact
        analysis_lines.append(f"[error] build_trade_suggestions failed: {_exc}")
        write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")
    # Portfolio evaluation (exposures, concentration, liquidity, risk)
    try:
        from scripts.portfolio.evaluation import build_portfolio_evaluation as _build_eval
        _build_eval(portfolio, snapshot, metrics, OUT_ORDERS_DIR, regime=regime)
    except Exception as _exc_eval:
        analysis_lines.append(f"[warn] portfolio_evaluation failed: {_exc_eval}")
        write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")
    # Write regime components snapshot for audit
    try:
        comp_out = OUT_ORDERS_DIR / 'regime_components.json'
        payload = dict(getattr(regime, 'model_components', {}) or {})
        # add key diagnostics
        payload.update({
            'market_score': float(getattr(regime, 'market_score', 0.0) or 0.0),
            'risk_on_probability': float(getattr(regime, 'risk_on_probability', 0.0) or 0.0),
            'index_vol_annualized': float(getattr(regime, 'index_vol_annualized', 0.0) or 0.0),
            'index_atr14_pct': float(getattr(regime, 'index_atr14_pct', 0.0) or 0.0),
            'index_atr_percentile': float(getattr(regime, 'index_atr_percentile', 0.0) or 0.0),
            'turnover_percentile': float(getattr(regime, 'turnover_percentile', 0.5) or 0.5),
            'breadth_long': float(getattr(regime, 'model_components', {}).get('breadth_long', 0.0) if getattr(regime, 'model_components', {}) else 0.0),
            'diag_warnings': list(getattr(regime, 'diag_warnings', []) or []),
            'epu_us_percentile': float(getattr(regime, 'epu_us_percentile', 0.0) or 0.0),
            'spx_drawdown_pct': float(getattr(regime, 'spx_drawdown_pct', 0.0) or 0.0),
            'dxy_percentile': float(getattr(regime, 'dxy_percentile', 0.0) or 0.0),
            'brent_mom_63d': float(getattr(regime, 'brent_mom_63d', 0.0) or 0.0),
            'index_change_pct_intraday': float(getattr(regime, 'index_change_pct', 0.0) or 0.0),
            'index_change_pct_smoothed': float(getattr(regime, 'index_change_pct_smoothed', 0.0) or 0.0),
        })
        comp_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as _exc:
        analysis_lines.append(f"[warn] failed to write regime_components.json: {_exc}")
        write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")
    if filtered_records:
        filtered_df = pd.DataFrame(filtered_records)
        filtered_df.to_csv(OUT_ORDERS_DIR / "orders_filtered.csv", index=False)
    else:
        filtered_path = OUT_ORDERS_DIR / "orders_filtered.csv"
        if filtered_path.exists():
            filtered_path.unlink()
    # Append ladder‑dropped levels to watchlist (enriched) if any
    if ladder_dropped:
        dropped_orders = []
        for d in ladder_dropped:
            note = f"LADDER_DROPPED:{d.get('Reason','')};Bundle={d.get('BundleId','')};Level={d.get('Level','')}"
            try:
                dropped_orders.append(Order(
                    ticker=str(d.get('Ticker')), side=str(d.get('Side')).upper(),
                    quantity=int(d.get('Quantity') or 0), limit_price=float(d.get('LimitPrice') or 0.0), note=note
                ))
            except Exception:
                continue
        wl_path = OUT_ORDERS_DIR / 'orders_watchlist.csv'
        if wl_path.exists():
            tmp_path = OUT_ORDERS_DIR / 'orders_watchlist._ladder_tmp.csv'
            write_orders_csv_enriched(dropped_orders, tmp_path, snapshot=snapshot, presets=presets, regime=regime, feats_all=feats_all, scores=scores)
            try:
                df_old = pd.read_csv(wl_path)
                df_new = pd.read_csv(tmp_path)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
                df_all.to_csv(wl_path, index=False)
                tmp_path.unlink(missing_ok=True)
            except Exception as _exc_merge:
                print(f"[warn] merge_watchlist_failed: {_exc_merge}")
        else:
            write_orders_csv_enriched(dropped_orders, wl_path, snapshot=snapshot, presets=presets, regime=regime, feats_all=feats_all, scores=scores)
    print(f"[hint] {exec_hint}")

    # Update cooldown ledger based on today actions (exit/take_profit/new actually placed)
    try:
        today = datetime.now().date().isoformat()
        ledger_path = OUT_ORDERS_DIR / 'last_actions.csv'
        records = []
        for t, a in actions.items():
            if a in ('exit','take_profit'):
                records.append({'Ticker': t, 'LastAction': a, 'Date': today})
        if records:
            # merge/update by ticker (keep latest date)
            df_new = pd.DataFrame(records)
            if ledger_path.exists():
                try:
                    df_old = pd.read_csv(ledger_path)
                except Exception as _exc:
                    df_old = pd.DataFrame(columns=['Ticker','LastAction','Date'])
                    analysis_lines.append(f"[warn] failed to read last_actions.csv, recreating: {_exc}")
            else:
                df_old = pd.DataFrame(columns=['Ticker','LastAction','Date'])
            df = pd.concat([df_old, df_new], ignore_index=True)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values(['Ticker','Date']).dropna(subset=['Ticker','Date'])
            df = df.drop_duplicates(subset=['Ticker'], keep='last')
            df['Ticker'] = df['Ticker'].astype(str).str.upper()
            df.to_csv(ledger_path, index=False)
    except Exception as _exc:
        analysis_lines.append(f"[warn] failed to update cooldown ledger: {_exc}")
        write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")

    # Persist per-position state (TP1/trailing/SL steps) for future sessions
    try:
        state_path = OUT_ORDERS_DIR / 'position_state.csv'
        existing_state = dict(getattr(regime, 'position_state', {}) or {})
        # Be robust to missing local 'stateless_meta' symbol in rare paths
        _stateless = locals().get('stateless_meta', None)
        if _stateless is None:
            _stateless = dict(getattr(regime, 'stateless_sell_meta', {}) or {})
        stateless_meta_upper = {str(k).upper(): v for k, v in (_stateless.items() if isinstance(_stateless, dict) else [])}
        actions_upper = {str(k).upper(): v for k, v in actions.items()}
        updated_state: Dict[str, Dict[str, object]] = {}
        cd_days = 0
        try:
            cd_days = int(float(regime.thresholds.get('cooldown_days', 0)))
        except Exception:
            cd_days = 0
        for _, row in portfolio.iterrows():
            ticker_up = str(row.get('Ticker', '')).strip().upper()
            if not ticker_up:
                continue
            state_entry = dict(existing_state.get(ticker_up, {}))
            meta = stateless_meta_upper.get(ticker_up)
            if isinstance(meta, dict) and meta.get('state_updates'):
                for key, val in meta.get('state_updates', {}).items():
                    if key in {'tp1_done', 'sl_step_hit_50', 'sl_step_hit_80'}:
                        state_entry[key] = bool(val)
            if ticker_up in actions_upper and actions_upper[ticker_up] == 'exit':
                # Will be handled below (cooldown entry)
                continue
            updated_state[ticker_up] = state_entry
        if cd_days > 0:
            from datetime import timedelta
            cooldown_until = (datetime.now().date() + timedelta(days=cd_days)).isoformat()
        else:
            cooldown_until = None
        for ticker_up, act in actions_upper.items():
            if act in ('exit', 'take_profit'):
                entry = dict(updated_state.get(ticker_up, {}))
                if cooldown_until:
                    entry['cooldown_until'] = cooldown_until
                else:
                    entry.pop('cooldown_until', None)
                entry.pop('tp1_done', None)
                entry.pop('sl_step_hit_50', None)
                entry.pop('sl_step_hit_80', None)
                updated_state[ticker_up] = entry
        if updated_state:
            rows = []
            for ticker_up, state in sorted(updated_state.items()):
                row = {'Ticker': ticker_up}
                for key in ('tp1_done', 'sl_step_hit_50', 'sl_step_hit_80'):
                    if key in state:
                        row[key] = 1 if bool(state.get(key)) else 0
                if state.get('cooldown_until'):
                    row['cooldown_until'] = state['cooldown_until']
                rows.append(row)
            pd.DataFrame(rows).to_csv(state_path, index=False)
        elif state_path.exists():
            state_path.unlink()
    except Exception as _exc:
        analysis_lines.append(f"[warn] failed to update position_state.csv: {_exc}")
        write_orders_analysis(analysis_lines, OUT_ORDERS_DIR / "orders_analysis.txt")
def resolve_tp_sl(thresholds: Dict[str, object], feats: Optional[Dict[str, float]] = None) -> tuple[Optional[float], Optional[float], Dict[str, float]]:
    atr_pct_val = None
    if feats is not None and feats.get('atr_pct') is not None:
        try:
            atr_pct_val = float(feats.get('atr_pct'))
        except Exception:
            atr_pct_val = None
    atr_dec = float(atr_pct_val) if atr_pct_val is not None else None

    def _apply_floor_cap(value: Optional[float], floor: float, cap: float) -> Optional[float]:
        if value is None:
            return None
        try:
            val = float(value)
        except Exception:
            return None
        if floor > 0.0 and val < floor:
            val = floor
        if cap > 0.0 and val > cap:
            val = cap
        if val <= 0.0:
            return None
        return val

    tp_pct_static = float(thresholds.get('tp_pct')) if thresholds.get('tp_pct') is not None else None
    sl_pct_static = float(thresholds.get('sl_pct')) if thresholds.get('sl_pct') is not None else None
    tp_atr_mult = thresholds.get('tp_atr_mult')
    sl_atr_mult = thresholds.get('sl_atr_mult')
    tp_floor = float(thresholds.get('tp_floor_pct') or 0.0)
    sl_floor = float(thresholds.get('sl_floor_pct') or 0.0)
    tp_cap = float(thresholds.get('tp_cap_pct') or 0.0)
    sl_cap = float(thresholds.get('sl_cap_pct') or 0.0)
    mode = str(thresholds.get('tp_sl_mode', 'legacy')).strip().lower()
    tp_rule = str(thresholds.get('tp_rule', 'min')).strip().lower()
    sl_rule = str(thresholds.get('sl_rule', 'min')).strip().lower()

    def _combine_with_rule(static_val: Optional[float], dyn_val: Optional[float], floor_val: float, cap_val: float, rule: str) -> tuple[Optional[float], str]:
        vals = [float(v) for v in (static_val, dyn_val) if v is not None and float(v) > 0]
        source = 'none'
        eff: Optional[float]
        if rule == 'static_only':
            eff = float(static_val) if (static_val is not None and float(static_val) > 0) else None
            source = 'static'
        elif rule == 'dynamic_only':
            eff = float(dyn_val) if (dyn_val is not None and float(dyn_val) > 0) else None
            source = 'dynamic'
        elif rule == 'max':
            eff = max(vals) if vals else None
            source = 'max'
        else:
            eff = min(vals) if vals else None
            source = 'min'
        floor_pos = float(floor_val) if floor_val is not None else 0.0
        eff = _apply_floor_cap(eff, floor_pos, cap_val)
        if eff is None and floor_pos > 0.0:
            eff = _apply_floor_cap(floor_pos, floor_pos, cap_val)
            source = 'floor'
        return eff, source

    def _compute_dynamic(static_val: Optional[float], atr_mult: Optional[float], floor_val: float, cap_val: float, rule: str) -> tuple[Optional[float], str, Optional[float]]:
        dyn_val = None
        if atr_mult is not None and atr_dec is not None and atr_dec > 0.0:
            try:
                dyn_val = float(atr_mult) * float(atr_dec)
            except Exception:
                dyn_val = None
        if mode == 'atr_per_ticker':
            eff = _apply_floor_cap(dyn_val, floor_val, cap_val)
            source = 'dynamic'
            if eff is None:
                eff = _apply_floor_cap(static_val, floor_val, cap_val)
                source = 'static'
            if eff is None and floor_val > 0.0:
                eff = _apply_floor_cap(floor_val, floor_val, cap_val)
                source = 'floor'
            return eff, source, dyn_val
        eff, source = _combine_with_rule(static_val, dyn_val, floor_val, cap_val, rule)
        return eff, source, dyn_val

    tp_eff, tp_source, dyn_tp = _compute_dynamic(tp_pct_static, tp_atr_mult, tp_floor, tp_cap, tp_rule)
    sl_eff, sl_source, dyn_sl = _compute_dynamic(sl_pct_static, sl_atr_mult, sl_floor, sl_cap, sl_rule)

    info = {
        'mode': mode,
        'atr_pct': atr_dec,
        'tp_pct': tp_eff,
        'sl_pct': sl_eff,
        'tp_floor_pct': tp_floor,
        'tp_cap_pct': tp_cap,
        'sl_floor_pct': sl_floor,
        'sl_cap_pct': sl_cap,
        'tp_source': tp_source,
        'sl_source': sl_source,
        'tp_dynamic_raw': dyn_tp,
        'sl_dynamic_raw': dyn_sl,
    }
    return tp_eff, sl_eff, info
