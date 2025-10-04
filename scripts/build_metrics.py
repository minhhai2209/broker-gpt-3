from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone, time
import pandas as pd
import numpy as np


VN_TZ = timezone(timedelta(hours=7))


def _phase_from_dt(dt: datetime) -> str:
    local_dt = dt.astimezone(VN_TZ)
    t = local_dt.time()
    if t < time(9, 0):
        return 'pre'
    if time(9, 0) <= t < time(11, 30):
        return 'morning'
    if time(11, 30) <= t < time(13, 0):
        return 'lunch'
    if time(13, 0) <= t < time(14, 30):
        return 'afternoon'
    if time(14, 30) <= t < time(14, 45):
        return 'ATC'
    return 'post'


def _latest_intraday_timestamp(latest_path: Path = Path('out/intraday/latest.csv')) -> datetime | None:
    if not latest_path.exists():
        raise FileNotFoundError(f"Missing intraday snapshot: {latest_path}")
    df = pd.read_csv(latest_path)
    if df.empty:
        raise ValueError(f"Intraday snapshot is empty: {latest_path}")

    def _ensure_vn(dt_val: pd.Timestamp | datetime | None) -> datetime | None:
        if dt_val is None or pd.isna(dt_val):
            return None
        if isinstance(dt_val, pd.Timestamp):
            if dt_val.tzinfo is None:
                localized = dt_val.tz_localize(VN_TZ)
                return localized.to_pydatetime()
            return dt_val.tz_convert(VN_TZ).to_pydatetime()
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=VN_TZ)
        return dt_val.astimezone(VN_TZ)

    if 'TimeVN' in df.columns:
        for value in reversed(df['TimeVN'].dropna().tolist()):
            ts = pd.to_datetime(value, errors='coerce')
            candidate = _ensure_vn(ts)
            if candidate is not None:
                return candidate
    if 'Ts' in df.columns:
        ts_values = pd.to_numeric(df['Ts'], errors='coerce').dropna().tolist()
        for value in reversed(ts_values):
            ts = pd.to_datetime(float(value), unit='s', utc=True)
            candidate = _ensure_vn(ts)
            if candidate is not None:
                return candidate
    return None


def infer_session_context(intraday_dir: Path = Path('out/intraday')) -> tuple[str, int]:
    now = datetime.now(VN_TZ)
    phase_now = _phase_from_dt(now)
    session_active_now = phase_now in ('morning', 'afternoon', 'ATC')
    latest_path = intraday_dir / 'latest.csv'
    latest_ts = _latest_intraday_timestamp(latest_path)
    if latest_ts is None:
        if session_active_now:
            raise RuntimeError(
                f"Intraday snapshot {latest_path} has no usable timestamp while the Vietnam session is active at "
                f"{now.strftime('%Y-%m-%d %H:%M:%S')} (VN). Refresh intraday caches before running the pipeline."
            )
        return phase_now, int(session_active_now)
    if latest_ts.date() != now.date():
        if session_active_now:
            raise RuntimeError(
                f"Intraday snapshot {latest_path} is stale: last update {latest_ts.strftime('%Y-%m-%d %H:%M:%S')} (VN) "
                f"while current Vietnam time is {now.strftime('%Y-%m-%d %H:%M:%S')}. Refresh intraday data before "
                "generating orders."
            )
        return phase_now, int(session_active_now)
    phase = _phase_from_dt(latest_ts)
    return phase, int(phase in ('morning', 'afternoon', 'ATC'))


def _latest_prev_close_map(prices_history_path: Path) -> dict[str, float]:
    ph = pd.read_csv(prices_history_path, usecols=['Date','Ticker','Close'])
    if ph.empty:
        raise ValueError(f"Prices history CSV has no rows: {prices_history_path}")
    ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
    ph = ph.dropna(subset=['Date'])
    ph = ph.sort_values(['Ticker','Date']).groupby('Ticker', as_index=False).tail(1)
    ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
    return {t: float(c) for t, c in zip(ph['Ticker'], ph['Close']) if pd.notna(c)}


def _compute_objective_addons(ph_path: Path) -> pd.DataFrame:
    cols = ['Ticker','Date','Close','High','Low','Volume']
    ph = pd.read_csv(ph_path, usecols=cols)
    if ph.empty:
        raise ValueError(f"Prices history CSV has no data: {ph_path}")
    ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
    ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
    ph = ph.dropna(subset=['Date'])
    ph = ph.sort_values(['Ticker','Date'])
    ph['Ret'] = ph.groupby('Ticker')['Close'].transform(lambda s: pd.to_numeric(s, errors='coerce').pct_change())
    mkt = ph[ph['Ticker'] == 'VNINDEX'][['Date','Ret']].rename(columns={'Ret':'MktRet'})

    from scripts.indicators import rsi_wilder, atr_wilder, avg_turnover_k, beta_rolling
    from scripts.indicators.macd import macd_hist
    rows = []
    for tkr, g in ph.groupby('Ticker'):
        if tkr in ('VNINDEX','VN30','VN100'):
            rows.append({'Ticker': tkr,
                         'AvgTurnover20D_k': np.nan,
                         'Beta60D': np.nan,
                         'ATR14_Pct': np.nan,
                         'RSI14': np.nan})
            continue
        g = g[['Date','Close','High','Low','Volume','Ret']].copy()
        close = pd.to_numeric(g['Close'], errors='coerce')
        high = pd.to_numeric(g['High'], errors='coerce')
        low = pd.to_numeric(g['Low'], errors='coerce')
        vol = pd.to_numeric(g['Volume'], errors='coerce')
        liq20 = np.nan
        if len(close) >= 20:
            liq_series = avg_turnover_k(close, vol, 20)
            if not liq_series.empty:
                liq20 = liq_series.iloc[-1]
        atr14_pct = np.nan
        if len(close) >= 14:
            atr14 = atr_wilder(high, low, close, 14)
            if len(atr14) and len(close):
                last_close = close.iloc[-1]
                last_atr = atr14.iloc[-1]
                if np.isfinite(last_close) and np.isfinite(last_atr) and last_close != 0:
                    atr14_pct = float((last_atr / last_close) * 100.0)
        rsi14 = np.nan
        if len(close) >= 14:
            rsi_vals = rsi_wilder(close, 14)
            if len(rsi_vals):
                rsi_last = rsi_vals.iloc[-1]
                if np.isfinite(rsi_last):
                    rsi14 = float(rsi_last)
        beta60 = np.nan
        gm = g.merge(mkt, on='Date', how='left').dropna(subset=['Ret','MktRet'])
        if len(gm) >= 60:
            b = beta_rolling(gm['Ret'], gm['MktRet'], window=60)
            if b is not None and np.isfinite(b):
                beta60 = float(b)
        macdh = np.nan
        if len(close) >= 35:
            mh = macd_hist(close)
            if len(mh):
                macd_last = mh.iloc[-1]
                if np.isfinite(macd_last):
                    macdh = float(macd_last)
        mom_12_1 = np.nan
        if len(close) >= 252:
            px_21 = close.iloc[-21]
            px_252 = close.iloc[-252]
            if np.isfinite(px_21) and np.isfinite(px_252) and px_252 != 0:
                mom_12_1 = float(px_21 / px_252 - 1.0)
        mom_6_1 = np.nan
        if len(close) >= 126:
            px_126 = close.iloc[-126]
            px_21 = close.iloc[-21]
            if np.isfinite(px_21) and np.isfinite(px_126) and px_126 != 0:
                mom_6_1 = float(px_21 / px_126 - 1.0)
        rows.append({'Ticker': tkr,
                     'AvgTurnover20D_k': round(liq20, 2) if np.isfinite(liq20) else np.nan,
                     'Beta60D': round(beta60, 4) if np.isfinite(beta60) else np.nan,
                     'ATR14_Pct': round(atr14_pct, 2) if np.isfinite(atr14_pct) else np.nan,
                     'RSI14': round(rsi14, 2) if np.isfinite(rsi14) else np.nan,
                     'MACDHist': round(macdh, 4) if np.isfinite(macdh) else np.nan,
                     'MomRet_12_1': round(mom_12_1, 4) if np.isfinite(mom_12_1) else np.nan,
                     'MomRet_6_1': round(mom_6_1, 4) if np.isfinite(mom_6_1) else np.nan})
    return pd.DataFrame(rows)


def build_metrics(snapshot_path: str = 'out/snapshot.csv', industry_map_path: str = 'data/industry_map.csv', out_path: str = 'out/metrics.csv') -> None:
    snap_p = Path(snapshot_path)
    imap_p = Path(industry_map_path)
    out_p = Path(out_path)
    ph_p = Path('out/prices_history.csv')
    if not snap_p.exists():
        raise SystemExit('snapshot not found: out/snapshot.csv')
    df = pd.read_csv(snap_p)
    for col in ('AsOfVN', 'AsOfPrint'):
        if col in df.columns:
            df = df.drop(columns=col)
    sector_map = {}
    if imap_p.exists():
        m = pd.read_csv(imap_p)
        if {'Ticker', 'Sector'}.issubset(m.columns):
            sector_map = {str(t).upper(): s for t, s in zip(m['Ticker'], m['Sector'])}
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    index_labels = {"VNINDEX", "VN30", "VN100"}
    def _sector_of(t: str) -> str:
        t = str(t).upper()
        if t in sector_map:
            return sector_map[t]
        if t in index_labels:
            return 'Index'
        return ''
    df['Sector'] = df['Ticker'].map(_sector_of)
    session_phase, in_session = infer_session_context()
    df['SessionPhase'] = session_phase
    df['InVNSession'] = int(in_session)

    prev_close = _latest_prev_close_map(ph_p)
    curr_price = {str(t).upper(): p for t, p in zip(df['Ticker'].astype(str).str.upper(), pd.to_numeric(df.get('Price', np.nan), errors='coerce'))}
    index_labels = {"VNINDEX", "VN30", "VN100"}
    adv = 0
    dec = 0
    for t, px in curr_price.items():
        if t in index_labels:
            continue
        pc = prev_close.get(t)
        if pc is None or not np.isfinite(px) or not np.isfinite(pc):
            continue
        if px > pc:
            adv += 1
        elif px < pc:
            dec += 1
    pc = prev_close.get('VNINDEX')
    px = curr_price.get('VNINDEX')
    idx_chg = float((px - pc) / pc * 100.0) if (pc is not None and px is not None and np.isfinite(pc) and np.isfinite(px) and pc != 0) else np.nan

    ph = pd.read_csv(ph_p, usecols=['Date','Ticker','Close','Volume'])
    if ph.empty:
        raise ValueError(f"Prices history CSV has no data: {ph_p}")
    ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
    g = ph[ph['Date'] == ph['Date'].max()]
    g = g[~g['Ticker'].astype(str).str.upper().isin(['VNINDEX','VN30','VN100'])]
    tv = (pd.to_numeric(g['Close'], errors='coerce') * pd.to_numeric(g['Volume'], errors='coerce')).sum()
    total_value = float(tv) if np.isfinite(tv) else 0.0

    df['Advancers'] = int(adv)
    df['Decliners'] = int(dec)
    df['TotalValue'] = total_value
    df['IndexChangePct'] = float(idx_chg) if np.isfinite(idx_chg) else 0.0
    vnindex_raw = curr_price.get('VNINDEX', np.nan)
    vnindex_val = float(vnindex_raw) if np.isfinite(vnindex_raw) else np.nan
    df['VNIndex'] = vnindex_val if np.isfinite(vnindex_val) else ''

    addons = _compute_objective_addons(ph_p)
    df = df.merge(addons, on='Ticker', how='left')
    def _hose_tick_thousand(p):
        value = pd.to_numeric(pd.Series([p]), errors='coerce').iloc[0]
        if pd.isna(value):
            return ''
        if value < 10.0:
            return 0.01
        if value < 49.95:
            return 0.05
        return 0.10
    df['TickSizeHOSE_Thousand'] = df['Price'].apply(_hose_tick_thousand)

    cols = ['Ticker','Sector',
            'Price',
            'SessionPhase','InVNSession','Advancers','Decliners','TotalValue','IndexChangePct','VNIndex',
            'RSI14','ATR14_Pct','AvgTurnover20D_k','Beta60D','TickSizeHOSE_Thousand','MACDHist','MomRet_12_1','MomRet_6_1']
    have = [c for c in cols if c in df.columns]
    out = df.reindex(columns=have)

    session_summary = pd.DataFrame([{
        'SessionPhase': session_phase,
        'InVNSession': int(in_session),
        'VNIndex': vnindex_val if np.isfinite(vnindex_val) else '',
        'IndexChangePct': float(idx_chg) if np.isfinite(idx_chg) else 0.0,
        'Advancers': int(adv),
        'Decliners': int(dec),
        'TotalValue': float(total_value),
    }])
    out_dir = Path('out')
    out_dir.mkdir(parents=True, exist_ok=True)
    session_summary.to_csv(out_dir / 'session_summary.csv', index=False)
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_p, index=False)
    print(f'Wrote {out_p}')


def build_metrics_df(snapshot_df: pd.DataFrame, industry_map_df: pd.DataFrame, prices_history_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if prices_history_df is None or prices_history_df.empty:
        raise ValueError('prices_history_df must be provided with data')
    if snapshot_df is None or snapshot_df.empty:
        raise ValueError('snapshot_df must be provided with data')
    df = snapshot_df.copy()
    if 'Price' not in df.columns:
        raise ValueError('snapshot_df missing required column: Price')
    df['Ticker'] = df['Ticker'].astype(str).str.upper()
    for col in ('AsOfVN', 'AsOfPrint'):
        if col in df.columns:
            df = df.drop(columns=col)
    # Sector map
    sector_map = {}
    if not industry_map_df.empty and {'Ticker','Sector'}.issubset(industry_map_df.columns):
        tickers = industry_map_df['Ticker'].astype(str)
        sectors = industry_map_df['Sector'].astype(str).str.strip()
        sector_map = {str(t).upper(): s for t, s in zip(tickers, sectors)}
    index_labels = {"VNINDEX", "VN30", "VN100"}
    def _sector_of(t: str) -> str:
        t = str(t).upper()
        if t in sector_map:
            return sector_map[t]
        if t in index_labels:
            return 'Index'
        return ''
    df['Sector'] = df['Ticker'].map(_sector_of)
    session_phase, in_session = infer_session_context()
    df['SessionPhase'] = session_phase
    df['InVNSession'] = int(in_session)
    # Breadth and index change
    # Build latest prev close map from prices_history_df
    prev_close = {}
    if not prices_history_df.empty:
        ph = prices_history_df[['Date','Ticker','Close']].copy()
        ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
        ph = ph.dropna(subset=['Date']).sort_values(['Ticker','Date']).groupby('Ticker', as_index=False).tail(1)
        ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
        prev_close = {t: float(c) for t, c in zip(ph['Ticker'], ph['Close']) if pd.notna(c)}
    curr_price = {str(t).upper(): p for t, p in zip(df['Ticker'].astype(str).str.upper(), pd.to_numeric(df.get('Price', np.nan), errors='coerce'))}
    adv = dec = 0
    for t, px in curr_price.items():
        if t in index_labels: continue
        pc = prev_close.get(t)
        if pc is None or not np.isfinite(px) or not np.isfinite(pc): continue
        if px > pc: adv += 1
        elif px < pc: dec += 1
    idx_chg = 0.0
    pc_idx = prev_close.get('VNINDEX')
    px_idx = curr_price.get('VNINDEX')
    if pc_idx is not None and px_idx is not None and np.isfinite(pc_idx) and np.isfinite(px_idx) and pc_idx != 0:
        idx_chg = float((px_idx - pc_idx) / pc_idx * 100.0)
    else:
        idx_chg = 0.0
    # TotalValue approx
    if prices_history_df.empty:
        raise ValueError('prices_history_df must contain data')
    ph_total = prices_history_df[['Date','Ticker','Close','Volume']].copy()
    ph_total['Date'] = pd.to_datetime(ph_total['Date'], errors='coerce')
    g = ph_total[ph_total['Date'] == ph_total['Date'].max()]
    g = g[~g['Ticker'].astype(str).str.upper().isin(['VNINDEX','VN30','VN100'])]
    tv = (pd.to_numeric(g['Close'], errors='coerce') * pd.to_numeric(g['Volume'], errors='coerce')).sum()
    if not np.isfinite(tv):
        raise ValueError('Unable to compute total trading value from prices_history_df')
    total_value = float(tv)
    df['Advancers'] = int(adv)
    df['Decliners'] = int(dec)
    df['TotalValue'] = total_value
    df['IndexChangePct'] = float(idx_chg)
    vnindex_raw = curr_price.get('VNINDEX', np.nan)
    vnindex_val = float(vnindex_raw) if np.isfinite(vnindex_raw) else 0.0
    df['VNIndex'] = vnindex_val if np.isfinite(vnindex_val) else ''
    if prices_history_df is None:
        addons_from_disk = _compute_objective_addons(Path('out/prices_history.csv'))
        df = df.merge(addons_from_disk, on='Ticker', how='left')
    elif not prices_history_df.empty:
        ph = prices_history_df.copy()
        ph['Ticker'] = ph['Ticker'].astype(str).str.upper()
        ph['Date'] = pd.to_datetime(ph['Date'], errors='coerce')
        ph = ph.dropna(subset=['Date']).sort_values(['Ticker','Date'])
        ph['Ret'] = ph.groupby('Ticker')['Close'].transform(lambda s: pd.to_numeric(s, errors='coerce').pct_change())
        from scripts.indicators import rsi_wilder, atr_wilder, avg_turnover_k, beta_rolling
        from scripts.indicators.macd import macd_hist
        rows = []
        mkt = ph[ph['Ticker'] == 'VNINDEX'][['Date','Ret','Close']].rename(columns={'Ret':'MktRet','Close':'MktClose'})
        for tkr, g in ph.groupby('Ticker'):
            if tkr in ('VNINDEX','VN30','VN100'):
                rows.append({'Ticker': tkr, 'AvgTurnover20D_k': np.nan, 'Beta60D': np.nan, 'ATR14_Pct': np.nan, 'RSI14': np.nan,
                             'MomRet_12_1': np.nan, 'MomRet_6_1': np.nan})
                continue
            g = g[['Date','Close','High','Low','Volume','Ret']].copy()
            close = pd.to_numeric(g['Close'], errors='coerce')
            high = pd.to_numeric(g['High'], errors='coerce')
            low = pd.to_numeric(g['Low'], errors='coerce')
            vol = pd.to_numeric(g['Volume'], errors='coerce')
            liq20 = avg_turnover_k(close, vol, 20).iloc[-1] if len(close) else np.nan
            atr14 = atr_wilder(high, low, close, 14)
            atr14_pct = float((atr14.iloc[-1] / close.iloc[-1]) * 100.0) if len(atr14) and len(close) and np.isfinite(close.iloc[-1]) and close.iloc[-1] != 0 else np.nan
            rsi14 = float(rsi_wilder(close, 14).iloc[-1]) if len(close) else np.nan
            gm = g.merge(mkt, on='Date', how='left').dropna(subset=['Ret','MktRet'])
            beta60 = np.nan
            if len(gm) >= 60:
                b = beta_rolling(gm['Ret'], gm['MktRet'], window=60)
                if b is not None and np.isfinite(b):
                    beta60 = float(b)
            macdh = np.nan
            if len(close) >= 35:
                mh = macd_hist(close)
                if len(mh):
                    macd_last = mh.iloc[-1]
                    if np.isfinite(macd_last):
                        macdh = float(macd_last)
            # RS slope 50d vs VNINDEX: percent change of RS ratio over 50 sessions
            rs_trend50 = np.nan
            gm_rs = g[['Date','Close']].merge(mkt[['Date','MktClose']], on='Date', how='left').dropna(subset=['Close','MktClose'])
            if len(gm_rs) >= 51:
                rs = pd.to_numeric(gm_rs['Close'], errors='coerce') / pd.to_numeric(gm_rs['MktClose'], errors='coerce')
                rs_last = rs.iloc[-1]
                rs_prev = rs.iloc[-51]
                if np.isfinite(rs_last) and np.isfinite(rs_prev) and rs_prev != 0:
                    rs_trend50 = float(rs_last / rs_prev - 1.0)
            # Momentum windows for leader-bypass gates
            mom_12_1 = np.nan
            if len(close) >= 252:
                px_21 = close.iloc[-21]
                px_252 = close.iloc[-252]
                if np.isfinite(px_21) and np.isfinite(px_252) and px_252 != 0:
                    mom_12_1 = float(px_21 / px_252 - 1.0)
            mom_6_1 = np.nan
            if len(close) >= 126:
                px_126 = close.iloc[-126]
                px_21b = close.iloc[-21]
                if np.isfinite(px_21b) and np.isfinite(px_126) and px_126 != 0:
                    mom_6_1 = float(px_21b / px_126 - 1.0)
            rows.append({'Ticker': tkr,
                         'AvgTurnover20D_k': round(liq20, 2) if np.isfinite(liq20) else np.nan,
                        'Beta60D': round(beta60, 4) if np.isfinite(beta60) else np.nan,
                        'ATR14_Pct': round(atr14_pct, 2) if np.isfinite(atr14_pct) else np.nan,
                        'RSI14': round(rsi14, 2) if np.isfinite(rsi14) else np.nan,
                        'MACDHist': round(macdh, 4) if np.isfinite(macdh) else np.nan,
                        'MomRet_12_1': round(mom_12_1, 4) if np.isfinite(mom_12_1) else np.nan,
                        'MomRet_6_1': round(mom_6_1, 4) if np.isfinite(mom_6_1) else np.nan,
                        'RS_Trend50': round(rs_trend50, 4) if np.isfinite(rs_trend50) else np.nan})
        addons_df = pd.DataFrame(rows)
        df = df.merge(addons_df, on='Ticker', how='left')
    # Compute HOSE tick size (thousand VND) based on current price
    def _hose_tick_thousand(p: float):
        # Pricing step sizes on HOSE: <10k -> 10 VND (0.01), 10k-49.95k -> 50 VND (0.05), >=50k -> 100 VND (0.10)
        val = pd.to_numeric(pd.Series([p]), errors='coerce').iloc[0]
        if pd.isna(val):
            return ''
        if val < 10.0:
            return 0.01
        if val < 49.95:
            return 0.05
        return 0.10
    df['TickSizeHOSE_Thousand'] = df['Price'].apply(_hose_tick_thousand)

    # Session summary DF
    session_summary = pd.DataFrame([{
        'SessionPhase': session_phase,
        'InVNSession': int(in_session),
        'VNIndex': float(curr_price.get('VNINDEX')) if curr_price.get('VNINDEX') is not None else '',
        'IndexChangePct': float(idx_chg),
        'Advancers': int(adv),
        'Decliners': int(dec),
        'TotalValue': float(total_value),
    }])
    cols = ['Ticker','Sector','Price','SessionPhase','InVNSession','Advancers','Decliners','TotalValue','IndexChangePct','VNIndex','RSI14','ATR14_Pct','AvgTurnover20D_k','Beta60D','TickSizeHOSE_Thousand','MACDHist','MomRet_12_1','MomRet_6_1','RS_Trend50']
    out = df.reindex(columns=[c for c in cols if c in df.columns])
    return out, session_summary
