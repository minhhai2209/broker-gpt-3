from __future__ import annotations

"""Pipeline orchestration to prepare all required artifacts for the engine.
Separated from order_engine.py for readability.
"""

import os
from pathlib import Path
from typing import List, Tuple
import re

import pandas as pd

from scripts.build_metrics import build_metrics_df
from scripts.build_presets_all import build_presets_all_df, _infer_in_session
from scripts.build_snapshot import build_snapshot_df
from scripts.data_fetching.collect_intraday import ensure_intraday_latest
from scripts.compute_sector_strength import compute_sector_strength_df
from scripts.data_fetching.fetch_ticker_data import ensure_and_load_history_df
from scripts.portfolio.ingest_auto import ingest_portfolio_df
from scripts.fundamentals import load_latest_fundamentals, merge_fundamentals
from scripts.indicators.precompute_indicators import precompute_indicators_from_history_df
from scripts.portfolio.report_pnl import build_portfolio_pnl_dfs
from scripts.utils import load_universe_from_files
from scripts.data_fetching.collect_global_factors import ensure_global_factors


BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "out"
DATA_DIR = BASE_DIR / "data"


def ensure_pipeline_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1) Portfolio (DF) + write for inspection
    portfolio_df = ingest_portfolio_df('in/portfolio')
    # 2) Universe
    uni = load_universe_from_files(str(DATA_DIR / 'industry_map.csv'))
    for t in (portfolio_df.get('Ticker', pd.Series(dtype=str)).astype(str).str.upper().tolist() if not portfolio_df.empty else []):
        if t and t not in uni: uni.append(t)
    for t in ['VNINDEX','VN30','VN100']:
        if t not in uni:
            uni.append(t)
    if not uni:
        raise SystemExit('No tickers to fetch (need data/industry_map.csv or in/portfolio/*.csv)')
    # Optional: limit universe size for development/testing
    lim_env = os.getenv('BROKER_UNI_LIMIT', '0').strip()
    if lim_env:
        lim = int(lim_env)
        if lim > 0:
            uni = uni[:lim]
    # 3) Ensure OHLC caches and load merged history DF
    # Use a longer calendar window to ensure >= ~400 trading sessions for calibrations
    prices_history_df = ensure_and_load_history_df(uni, outdir=str(OUT_DIR / 'data'), min_days=700, resolution='D')
    prices_history_df.to_csv(OUT_DIR / 'prices_history.csv', index=False)
    portfolio_df.to_csv(OUT_DIR / 'portfolio_clean.csv', index=False)
    # 4) Intraday snapshot must exist before metrics/session computations
    ensure_intraday_latest(uni, outdir=str(OUT_DIR / 'intraday'), window_minutes=12*60)
    # 5) Snapshot (DF) + metrics DF + session summary DF
    snapshot_df = build_snapshot_df(portfolio_df, data_dir=str(OUT_DIR / 'data'))
    snapshot_df.to_csv(OUT_DIR / 'snapshot.csv', index=False)
    industry_df = pd.read_csv(DATA_DIR / 'industry_map.csv') if (DATA_DIR / 'industry_map.csv').exists() else pd.DataFrame(columns=['Ticker','Sector'])
    metrics_df, session_summary_df = build_metrics_df(snapshot_df, industry_df, prices_history_df)
    fundamentals_df = load_latest_fundamentals(BASE_DIR / "data" / "fundamentals_vietstock.csv")
    metrics_df = merge_fundamentals(metrics_df, fundamentals_df)
    metrics_df.to_csv(OUT_DIR / 'metrics.csv', index=False)
    session_summary_df.to_csv(OUT_DIR / 'session_summary.csv', index=False)
    if not fundamentals_df.empty:
        fundamentals_df.to_csv(OUT_DIR / 'fundamentals_snapshot.csv', index=False)
    # 6) Sector strength DF
    sector_strength_df = compute_sector_strength_df(snapshot_df, industry_df, prices_history_df)
    sector_strength_df.to_csv(OUT_DIR / 'sector_strength.csv', index=False)
    # 7) Precompute + presets DF
    precomp_df = precompute_indicators_from_history_df(prices_history_df)
    precomp_df.to_csv(OUT_DIR / 'precomputed_indicators.csv', index=False)
    # 7b) Optional global factors features (best-effort; non-fatal)
    ensure_global_factors(OUT_DIR)
    session_in_progress = _infer_in_session(session_summary_df=session_summary_df)
    presets_df = build_presets_all_df(
        precomp_df,
        snapshot_df,
        prices_history_df,
        session_in_progress=session_in_progress,
        session_summary_df=session_summary_df,
    )
    presets_df.to_csv(OUT_DIR / 'presets_all.csv', index=False)
    # 8) PnL (best-effort write only)
    summary_df, by_sector_df = build_portfolio_pnl_dfs(portfolio_df, snapshot_df)
    summary_df.to_csv(OUT_DIR / 'portfolio_pnl_summary.csv', index=False)
    by_sector_df.to_csv(OUT_DIR / 'portfolio_pnl_by_sector.csv', index=False)
    return portfolio_df, prices_history_df, snapshot_df, metrics_df, sector_strength_df, presets_df, session_summary_df
